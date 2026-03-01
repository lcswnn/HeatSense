"""
TomorrowLand Heat — Phase 2: Train Heat Prediction Model
=========================================================
This script trains a gradient boosting model (LightGBM or XGBoost) to predict
land surface temperature from urban features like vegetation, impervious surface,
and location.

Once trained, this model enables:
  1. FORECASTING: Predict which areas will be hottest on a given day
  2. INTERVENTION SIMULATION: "What if we planted trees here?" — modify NDVI
     inputs and see the predicted temperature change

The model learns: given a 100m grid cell's characteristics, what is its
expected summer surface temperature?

Prerequisites:
  - process_grid.py completed (data/grid/chicago_grid.csv exists)
  - pip install lightgbm scikit-learn

Usage:
  python train_heat_model.py

Output:
  - model/models/chicago_heat_model.pkl (trained model)
  - model/models/chicago_model_metadata.json (feature info, metrics)
  - output/model_feature_importance.png
  - output/model_performance.png
  - output/model_predicted_vs_actual.png
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Try LightGBM first, fall back to XGBoost, then to sklearn GradientBoosting
MODEL_TYPE = None
try:
    import lightgbm as lgb
    MODEL_TYPE = "lightgbm"
    print("Using LightGBM")
except ImportError:
    try:
        import xgboost as xgb
        MODEL_TYPE = "xgboost"
        print("Using XGBoost")
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        MODEL_TYPE = "sklearn"
        print("Using sklearn GradientBoostingRegressor (install lightgbm for better results)")


# ============================================================
# CONFIGURATION
# ============================================================

GRID_CSV_PATH = "data/grid/chicago_grid.csv"
MODEL_OUTPUT_DIR = "model/models"
PLOT_OUTPUT_DIR = "output"

# Features to use for training
# These are the columns from chicago_grid.csv that the model will learn from
FEATURE_COLUMNS = [
    "ndvi",              # Vegetation index (strongest predictor)
    "impervious_pct",    # Impervious surface percentage
    "lat",               # Latitude (captures north-south gradient + lake effect)
    "lon",               # Longitude (captures east-west / lake proximity)
    # These will be included when OSM data is available:
    # "building_count",
    # "building_density",
    # "avg_building_height_m",
    # "road_density_km",
    # "distance_to_park_m",
    # "distance_to_water_m",
    # "park_area_pct",
]

TARGET_COLUMN = "mean_lst_f"  # Land Surface Temperature in Fahrenheit

# Model hyperparameters
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "verbosity": 0,
}

SKLEARN_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 20,
    "random_state": 42,
}


# ============================================================
# DATA LOADING & PREPARATION
# ============================================================

def load_and_prepare_data():
    """Load the grid CSV and prepare it for training."""
    print("=" * 50)
    print("  Loading Data")
    print("=" * 50)

    # Load grid
    df = pd.read_csv(GRID_CSV_PATH)
    print(f"  Loaded {len(df):,} grid cells from {GRID_CSV_PATH}")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Check which features are available
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]

    if missing_features:
        print(f"  Note: Missing features (will skip): {missing_features}")
    print(f"  Using features: {available_features}")
    print(f"  Target: {TARGET_COLUMN}")
    print()

    # Filter to rows that have target + all features
    required_cols = available_features + [TARGET_COLUMN]
    df_clean = df[required_cols].dropna()

    # Remove extreme outliers (sensor errors, cloud artifacts)
    # LST below 50°F or above 150°F in summer is almost certainly an error
    before = len(df_clean)
    df_clean = df_clean[
        (df_clean[TARGET_COLUMN] >= 50) &
        (df_clean[TARGET_COLUMN] <= 150)
    ]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"  Removed {removed:,} rows with extreme temperature values")

    # Remove water pixels for training (NDVI < -0.1 is almost always water)
    # Water temperatures behave completely differently from land
    before = len(df_clean)
    df_clean = df_clean[df_clean["ndvi"] > -0.1]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"  Removed {removed:,} water pixels (NDVI < -0.1)")

    print(f"  Clean dataset: {len(df_clean):,} rows")
    print()

    # Summary statistics
    print("  Feature Statistics:")
    print("  " + "-" * 60)
    for col in available_features + [TARGET_COLUMN]:
        vals = df_clean[col]
        print(f"  {col:<25} min={vals.min():>8.2f}  mean={vals.mean():>8.2f}  max={vals.max():>8.2f}")
    print()

    # Split features and target
    X = df_clean[available_features]
    y = df_clean[TARGET_COLUMN]

    return X, y, available_features, df


def create_train_test_split(X, y):
    """Split data into train/validation/test sets."""
    print("  Splitting data:")

    # 70% train, 15% validation, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42  # 0.176 of 85% ≈ 15%
    )

    print(f"    Train:      {len(X_train):>7,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    Validation: {len(X_val):>7,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    Test:       {len(X_test):>7,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# MODEL TRAINING
# ============================================================

def train_lightgbm(X_train, X_val, y_train, y_val, features):
    """Train a LightGBM model."""
    print("  Training LightGBM model...")

    model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
    )

    best_iter = model.best_iteration_
    print(f"  Best iteration: {best_iter}")

    return model


def train_xgboost(X_train, X_val, y_train, y_val, features):
    """Train an XGBoost model."""
    print("  Training XGBoost model...")

    model = xgb.XGBRegressor(**XGBOOST_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    return model


def train_sklearn(X_train, X_val, y_train, y_val, features):
    """Train a sklearn GradientBoostingRegressor."""
    print("  Training sklearn GradientBoostingRegressor...")
    print("  (This may take a few minutes — install lightgbm for 10x speed)")

    model = GradientBoostingRegressor(**SKLEARN_PARAMS)
    model.fit(X_train, y_train)

    return model


def train_model(X_train, X_val, y_train, y_val, features):
    """Train the appropriate model based on available library."""
    if MODEL_TYPE == "lightgbm":
        return train_lightgbm(X_train, X_val, y_train, y_val, features)
    elif MODEL_TYPE == "xgboost":
        return train_xgboost(X_train, X_val, y_train, y_val, features)
    else:
        return train_sklearn(X_train, X_val, y_train, y_val, features)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features):
    """Comprehensive model evaluation."""
    print("=" * 50)
    print("  Model Evaluation")
    print("=" * 50)

    results = {}

    for name, X, y in [("Train", X_train, y_train),
                         ("Validation", X_val, y_val),
                         ("Test", X_test, y_test)]:
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100

        results[name.lower()] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
            "n_samples": len(y)
        }

        print(f"\n  {name} Set ({len(y):,} samples):")
        print(f"    MAE:  {mae:.2f} F  (average prediction error)")
        print(f"    RMSE: {rmse:.2f} F  (penalizes large errors)")
        print(f"    R2:   {r2:.4f}     (variance explained)")
        print(f"    MAPE: {mape:.2f}%   (percentage error)")

    # Cross-validation on full training data
    print(f"\n  5-Fold Cross-Validation:")
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

    if MODEL_TYPE == "lightgbm":
        cv_model = lgb.LGBMRegressor(
            **{k: v for k, v in LIGHTGBM_PARAMS.items() if k != "early_stopping_rounds"}
        )
    elif MODEL_TYPE == "xgboost":
        cv_model = xgb.XGBRegressor(
            **{k: v for k, v in XGBOOST_PARAMS.items() if k != "early_stopping_rounds"}
        )
    else:
        cv_model = GradientBoostingRegressor(**SKLEARN_PARAMS)

    cv_scores = cross_val_score(
        cv_model, X_full, y_full,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"    CV MAE: {cv_mae:.2f} +/- {cv_std:.2f} F")
    results["cv_mae"] = round(cv_mae, 3)
    results["cv_std"] = round(cv_std, 3)

    return results


def get_feature_importance(model, features, X_test, y_test):
    """Extract and display feature importance."""
    print("\n  Feature Importance:")
    print("  " + "-" * 50)

    # Model-native importance
    if MODEL_TYPE == "lightgbm":
        importance = model.feature_importances_
    elif MODEL_TYPE == "xgboost":
        importance = model.feature_importances_
    else:
        importance = model.feature_importances_

    # Normalize to percentages
    importance_pct = importance / importance.sum() * 100

    # Sort by importance
    sorted_idx = np.argsort(importance_pct)[::-1]

    importance_dict = {}
    for idx in sorted_idx:
        feat = features[idx]
        pct = importance_pct[idx]
        bar = "█" * int(pct / 2)
        print(f"    {feat:<25} {pct:>6.1f}%  {bar}")
        importance_dict[feat] = round(float(pct), 2)

    # Permutation importance (more reliable, model-agnostic)
    print("\n  Permutation Importance (more reliable):")
    print("  " + "-" * 50)

    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    perm_sorted_idx = perm_result.importances_mean.argsort()[::-1]
    perm_importance_dict = {}

    for idx in perm_sorted_idx:
        feat = features[idx]
        mean_imp = perm_result.importances_mean[idx]
        std_imp = perm_result.importances_std[idx]
        print(f"    {feat:<25} {mean_imp:>8.3f} +/- {std_imp:.3f} F (MAE increase)")
        perm_importance_dict[feat] = {
            "mean": round(float(mean_imp), 4),
            "std": round(float(std_imp), 4)
        }

    return importance_dict, perm_importance_dict


# ============================================================
# INTERVENTION SIMULATION DEMO
# ============================================================

def demo_intervention_simulation(model, X_test, y_test, features):
    """
    Demonstrate the intervention simulator — the Tomorrowland feature.
    Shows what happens when you increase NDVI (simulate planting trees).
    """
    print("\n" + "=" * 50)
    print("  Intervention Simulation Demo")
    print("  'What if we planted trees here?'")
    print("=" * 50)

    if "ndvi" not in features:
        print("  ⚠ NDVI not in features, cannot simulate tree planting")
        return {}

    # Find the hottest grid cells (top 5%)
    y_pred = model.predict(X_test)
    hot_threshold = np.percentile(y_pred, 95)
    hot_mask = y_pred >= hot_threshold
    X_hot = X_test[hot_mask].copy()
    y_hot_pred = y_pred[hot_mask]

    print(f"\n  Analyzing the hottest {hot_mask.sum():,} grid cells "
          f"(predicted temp >= {hot_threshold:.1f} F)")
    print(f"  Current avg predicted temp: {y_hot_pred.mean():.1f} F")
    print(f"  Current avg NDVI: {X_hot['ndvi'].mean():.3f}")

    # Simulate interventions
    simulations = [
        ("Light tree planting", 0.15, "scattered street trees"),
        ("Moderate greening", 0.30, "park-like coverage"),
        ("Heavy reforestation", 0.45, "urban forest canopy"),
    ]

    results = {}

    print(f"\n  Simulated interventions on hottest areas:")
    print("  " + "-" * 60)

    for name, ndvi_increase, description in simulations:
        X_modified = X_hot.copy()
        new_ndvi = np.clip(X_modified["ndvi"] + ndvi_increase, -1, 1)
        X_modified["ndvi"] = new_ndvi

        # Also adjust impervious surface (more green = less pavement)
        if "impervious_pct" in features:
            # Rough heuristic: each 0.1 NDVI increase reduces impervious by ~10%
            imperv_reduction = ndvi_increase * 100
            X_modified["impervious_pct"] = np.clip(
                X_modified["impervious_pct"] - imperv_reduction, 0, 100
            )

        y_new_pred = model.predict(X_modified)
        temp_reduction = y_hot_pred.mean() - y_new_pred.mean()

        results[name] = {
            "ndvi_increase": ndvi_increase,
            "description": description,
            "avg_temp_reduction_f": round(float(temp_reduction), 2),
            "new_avg_temp_f": round(float(y_new_pred.mean()), 2),
        }

        print(f"\n    {name} (NDVI +{ndvi_increase:.2f} — {description}):")
        print(f"      New avg NDVI: {new_ndvi.mean():.3f}")
        print(f"      Predicted temp: {y_new_pred.mean():.1f} F")
        print(f"      Temperature reduction: {temp_reduction:.1f} F")

    print(f"\n  Key insight: Even light tree planting on the hottest blocks")
    print(f"  could reduce surface temperatures by several degrees.")

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_feature_importance(importance_dict, features, output_path):
    """Bar chart of feature importance."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), values, color="#E74C3C", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title("Feature Importance — What Drives Urban Heat?",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


def plot_predicted_vs_actual(model, X_test, y_test, output_path):
    """Scatter plot of predicted vs actual temperatures."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Density scatter
    hb = ax.hexbin(y_test, y_pred, gridsize=50, cmap="YlOrRd",
                    mincnt=1, alpha=0.9)
    plt.colorbar(hb, ax=ax, label="Number of grid cells")

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "b--", linewidth=2,
            label="Perfect prediction", alpha=0.7)

    # Metrics annotation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f"MAE: {mae:.2f} F\nR2: {r2:.4f}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Actual Surface Temperature (F)", fontsize=12)
    ax.set_ylabel("Predicted Surface Temperature (F)", fontsize=12)
    ax.set_title("Predicted vs Actual — Heat Model Performance",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


def plot_residual_map(model, X, y, full_df, output_path):
    """
    Map showing where the model over/under-predicts.
    Helps identify areas where additional features might help.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_pred = model.predict(X)
    residuals = y.values - y_pred  # Positive = model under-predicts (hotter than expected)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Predicted temperature map
    scatter1 = axes[0].scatter(
        X["lon"], X["lat"], c=y_pred, cmap="RdYlBu_r",
        s=0.5, alpha=0.5, vmin=75, vmax=115
    )
    plt.colorbar(scatter1, ax=axes[0], label="Predicted Temp (F)", shrink=0.7)
    axes[0].set_title("Predicted Surface Temperature", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Right: Residual map (where model is wrong)
    scatter2 = axes[1].scatter(
        X["lon"], X["lat"], c=residuals, cmap="RdBu_r",
        s=0.5, alpha=0.5, vmin=-10, vmax=10
    )
    plt.colorbar(scatter2, ax=axes[1], label="Residual: Actual - Predicted (F)", shrink=0.7)
    axes[1].set_title("Model Residuals\n(Red = hotter than predicted, Blue = cooler)",
                      fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    fig.suptitle("Heat Model — Spatial Performance",
                 fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


def plot_intervention_demo(model, X_test, y_test, features, output_path):
    """
    Visualize the intervention simulation — before and after tree planting
    for the hottest grid cells.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "ndvi" not in features:
        return

    y_pred = model.predict(X_test)
    hot_threshold = np.percentile(y_pred, 90)
    hot_mask = y_pred >= hot_threshold
    X_hot = X_test[hot_mask].copy()
    y_hot_pred = y_pred[hot_mask]

    # Simulate moderate greening (NDVI + 0.3)
    X_greened = X_hot.copy()
    X_greened["ndvi"] = np.clip(X_greened["ndvi"] + 0.3, -1, 1)
    if "impervious_pct" in features:
        X_greened["impervious_pct"] = np.clip(X_greened["impervious_pct"] - 30, 0, 100)

    y_greened_pred = model.predict(X_greened)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Before
    scatter1 = axes[0].scatter(
        X_hot["lon"], X_hot["lat"], c=y_hot_pred,
        cmap="RdYlBu_r", s=3, vmin=85, vmax=120
    )
    plt.colorbar(scatter1, ax=axes[0], label="Temp (F)", shrink=0.7)
    axes[0].set_title("BEFORE: Hottest Areas\n(Current Conditions)",
                      fontsize=13, fontweight="bold")

    # After
    scatter2 = axes[1].scatter(
        X_hot["lon"], X_hot["lat"], c=y_greened_pred,
        cmap="RdYlBu_r", s=3, vmin=85, vmax=120
    )
    plt.colorbar(scatter2, ax=axes[1], label="Temp (F)", shrink=0.7)
    axes[1].set_title("AFTER: With Moderate Greening\n(NDVI +0.3, Impervious -30%)",
                      fontsize=13, fontweight="bold")

    # Temperature reduction
    temp_diff = y_hot_pred - y_greened_pred
    scatter3 = axes[2].scatter(
        X_hot["lon"], X_hot["lat"], c=temp_diff,
        cmap="YlOrRd", s=3, vmin=0, vmax=15
    )
    plt.colorbar(scatter3, ax=axes[2], label="Cooling Effect (F)", shrink=0.7)
    axes[2].set_title(f"COOLING EFFECT\nAvg reduction: {temp_diff.mean():.1f} F",
                      fontsize=13, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(
        "Intervention Simulation — What If We Added Green Space?",
        fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, features, metrics, intervention_results):
    """Save the trained model and metadata."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_OUTPUT_DIR, "chicago_heat_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model: {model_path}")

    # Save metadata
    metadata = {
        "model_type": MODEL_TYPE,
        "features": features,
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "intervention_simulations": intervention_results,
        "trained_at": datetime.now().isoformat(),
        "grid_csv": GRID_CSV_PATH,
        "n_features": len(features),
        "description": "Gradient boosting model predicting land surface temperature "
                       "from urban characteristics. Trained on Landsat 8/9 thermal data "
                       "for Chicago summers (2022-2025)."
    }

    metadata_path = os.path.join(MODEL_OUTPUT_DIR, "chicago_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("  TomorrowLand Heat — ML Model Training")
    print("=" * 60)
    print()

    # ---- Load data ----
    X, y, features, full_df = load_and_prepare_data()

    # ---- Split ----
    print("=" * 50)
    print("  Splitting Data")
    print("=" * 50)
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)

    # ---- Train ----
    print("=" * 50)
    print("  Training Model")
    print("=" * 50)
    model = train_model(X_train, X_val, y_train, y_val, features)
    print("  ✓ Model trained")
    print()

    # ---- Evaluate ----
    metrics = evaluate_model(model, X_train, X_val, X_test,
                             y_train, y_val, y_test, features)

    # ---- Feature importance ----
    importance_dict, perm_importance = get_feature_importance(
        model, features, X_test, y_test
    )
    metrics["feature_importance"] = importance_dict
    metrics["permutation_importance"] = perm_importance

    # ---- Intervention simulation ----
    intervention_results = demo_intervention_simulation(model, X_test, y_test, features)

    # ---- Visualizations ----
    print("\n" + "=" * 50)
    print("  Generating Visualizations")
    print("=" * 50)

    plot_feature_importance(
        importance_dict, features,
        os.path.join(PLOT_OUTPUT_DIR, "model_feature_importance.png")
    )

    plot_predicted_vs_actual(
        model, X_test, y_test,
        os.path.join(PLOT_OUTPUT_DIR, "model_predicted_vs_actual.png")
    )

    plot_residual_map(
        model, X, y, full_df,
        os.path.join(PLOT_OUTPUT_DIR, "model_residual_map.png")
    )

    plot_intervention_demo(
        model, X_test, y_test, features,
        os.path.join(PLOT_OUTPUT_DIR, "model_intervention_demo.png")
    )

    # ---- Save ----
    print("\n" + "=" * 50)
    print("  Saving Model")
    print("=" * 50)
    save_model(model, features, metrics, intervention_results)

    # ---- Summary ----
    test_metrics = metrics["test"]
    print()
    print("=" * 60)
    print("  MODEL TRAINING COMPLETE")
    print()
    print(f"  Model type: {MODEL_TYPE}")
    print(f"  Features: {len(features)}")
    print(f"  Test MAE: {test_metrics['mae']:.2f} F")
    print(f"  Test R2:  {test_metrics['r2']:.4f}")
    print()
    if intervention_results:
        for name, result in intervention_results.items():
            print(f"  {name}: ~{result['avg_temp_reduction_f']:.1f} F cooling")
    print()
    print("  Output files:")
    print(f"  - {MODEL_OUTPUT_DIR}/chicago_heat_model.pkl")
    print(f"  - {MODEL_OUTPUT_DIR}/chicago_model_metadata.json")
    print(f"  - {PLOT_OUTPUT_DIR}/model_feature_importance.png")
    print(f"  - {PLOT_OUTPUT_DIR}/model_predicted_vs_actual.png")
    print(f"  - {PLOT_OUTPUT_DIR}/model_residual_map.png")
    print(f"  - {PLOT_OUTPUT_DIR}/model_intervention_demo.png")
    print()
    print("  NEXT STEPS:")
    print("  - Add OSM data (buildings, parks, water) to improve accuracy")
    print("  - Build the forecasting pipeline (forecast.py)")
    print("  - Build the FastAPI backend (api/main.py)")
    print("  - Build the interactive frontend")
    print("=" * 60)


if __name__ == "__main__":
    main()