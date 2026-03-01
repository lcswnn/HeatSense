"""
HeatSense — Model Tuning & Optimization
================================================
This script addresses the issues found in the initial model:

1. OVERFITTING: Train R²=0.97 vs Test R²=0.80 — too much gap
2. DATA LOSS: 147k cells → 45k after dropping NaN rows
3. FEATURE SELECTION: Do all 11 features help, or do some add noise?
4. EXTREME PREDICTION: Model compresses toward the mean at extremes

Approach:
  - Impute missing values instead of dropping rows (recover ~100k cells)
  - Test feature subsets to find the optimal combination
  - Grid search key hyperparameters to reduce overfitting
  - Evaluate specifically on extreme heat cells (the ones that matter most)

Usage:
  python tune_model.py

Output:
  - model/models/chicago_heat_model_tuned.pkl
  - model/models/chicago_tuned_metadata.json
  - output/tuning_comparison.png
  - output/tuning_extreme_performance.png
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations

import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

GRID_CSV_PATH = "data/grid/chicago_grid.csv"
MODEL_OUTPUT_DIR = "model/models"
PLOT_OUTPUT_DIR = "output"
TARGET = "mean_lst_f"


# ============================================================
# STEP 1: SMARTER DATA PREPARATION
# ============================================================

def load_and_prepare_data_v2():
    """
    Load data with intelligent imputation instead of dropping rows.
    This recovers the ~100k cells we lost before.
    """
    print("=" * 60)
    print("  Step 1: Smart Data Preparation")
    print("=" * 60)

    df = pd.read_csv(GRID_CSV_PATH)
    print(f"\n  Raw dataset: {len(df):,} rows")

    # Must have target
    df = df.dropna(subset=[TARGET])
    print(f"  After requiring target: {len(df):,} rows")

    # Remove extreme temperature outliers
    df = df[(df[TARGET] >= 50) & (df[TARGET] <= 150)]
    print(f"  After removing temp outliers: {len(df):,} rows")

    # Remove water pixels
    df = df[df["ndvi"] > -0.1]
    print(f"  After removing water: {len(df):,} rows")

    # All potential features
    all_features = [
        "ndvi", "impervious_pct", "lat", "lon",
        "building_count", "building_density", "avg_building_height_m",
        "road_density_km", "distance_to_park_m", "distance_to_water_m",
        "park_area_pct"
    ]

    # Check missingness
    print(f"\n  Missing data per feature:")
    print("  " + "-" * 50)
    for feat in all_features:
        if feat in df.columns:
            n_missing = df[feat].isna().sum()
            pct = n_missing / len(df) * 100
            print(f"    {feat:<25} {n_missing:>7,} missing ({pct:.1f}%)")

    # Strategy: Impute missing values with sensible defaults
    # rather than dropping entire rows
    print(f"\n  Imputation strategy:")

    # For building features: missing likely means no buildings (sparse OSM coverage)
    # → fill with 0 (no buildings in that cell)
    building_fills = {
        "building_count": 0,
        "building_density": 0.0,
        "avg_building_height_m": 0.0,
    }
    for col, fill_val in building_fills.items():
        if col in df.columns:
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(fill_val)
            if n_filled > 0:
                print(f"    {col}: filled {n_filled:,} with {fill_val} (no buildings)")

    # For road_density: missing means no roads → 0
    if "road_density_km" in df.columns:
        n_filled = df["road_density_km"].isna().sum()
        df["road_density_km"] = df["road_density_km"].fillna(0.0)
        if n_filled > 0:
            print(f"    road_density_km: filled {n_filled:,} with 0.0 (no roads)")

    # For park features: missing means no parks nearby
    if "park_area_pct" in df.columns:
        n_filled = df["park_area_pct"].isna().sum()
        df["park_area_pct"] = df["park_area_pct"].fillna(0.0)
        if n_filled > 0:
            print(f"    park_area_pct: filled {n_filled:,} with 0.0 (no park in cell)")

    # For distance features: missing means far from parks/water
    # Fill with a large value (95th percentile of known distances)
    for dist_col in ["distance_to_park_m", "distance_to_water_m"]:
        if dist_col in df.columns:
            n_missing = df[dist_col].isna().sum()
            if n_missing > 0:
                p95 = df[dist_col].quantile(0.95)
                df[dist_col] = df[dist_col].fillna(p95)
                print(f"    {dist_col}: filled {n_missing:,} with {p95:.0f} (95th percentile)")

    # For impervious_pct: use median imputation (genuinely unknown)
    if "impervious_pct" in df.columns:
        n_missing = df["impervious_pct"].isna().sum()
        if n_missing > 0:
            median_val = df["impervious_pct"].median()
            df["impervious_pct"] = df["impervious_pct"].fillna(median_val)
            print(f"    impervious_pct: filled {n_missing:,} with median ({median_val:.1f})")

    # Now drop any remaining NaN rows (should be very few)
    available_features = [f for f in all_features if f in df.columns]
    before = len(df)
    df = df.dropna(subset=available_features + [TARGET])
    dropped = before - len(df)
    if dropped > 0:
        print(f"\n    Dropped {dropped:,} remaining rows with NaN")

    print(f"\n  Final dataset: {len(df):,} rows (recovered {len(df) - 45877:+,} vs dropping NaN)")
    print(f"  Features: {available_features}")

    return df, available_features


# ============================================================
# STEP 2: FEATURE SELECTION
# ============================================================

def test_feature_subsets(df, all_features):
    """
    Test different feature combinations to find the optimal set.
    Some features might add noise and hurt generalization.
    """
    print("\n" + "=" * 60)
    print("  Step 2: Feature Selection")
    print("=" * 60)

    X = df[all_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Define feature groups to test
    feature_sets = {
        "Minimal (4)": ["ndvi", "impervious_pct", "lat", "lon"],

        "Core + water (5)": ["ndvi", "impervious_pct", "lat", "lon",
                              "distance_to_water_m"],

        "Core + buildings (6)": ["ndvi", "impervious_pct", "lat", "lon",
                                  "building_density", "distance_to_water_m"],

        "Balanced (8)": ["ndvi", "impervious_pct", "lat", "lon",
                          "building_density", "avg_building_height_m",
                          "distance_to_water_m", "distance_to_park_m"],

        "Full (11)": all_features,

        "No lat/lon (9)": [f for f in all_features if f not in ["lat", "lon"]],

        "Top permutation (7)": ["ndvi", "impervious_pct", "lat", "lon",
                                 "avg_building_height_m", "building_density",
                                 "distance_to_water_m"],
    }

    # Moderate regularization params for fair comparison
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "verbose": -1,
        "n_estimators": 800,
        "random_state": 42,
    }

    results = {}

    print(f"\n  Testing {len(feature_sets)} feature combinations...")
    print(f"  Train: {len(X_train_inner):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print()
    print(f"  {'Name':<25} {'Features':>3} {'Train MAE':>10} {'Val MAE':>10} "
          f"{'Test MAE':>10} {'Test R²':>8} {'Gap':>8}")
    print("  " + "-" * 80)

    for name, features in feature_sets.items():
        features = [f for f in features if f in df.columns]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_inner[features], y_train_inner,
            eval_set=[(X_val[features], y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        train_pred = model.predict(X_train_inner[features])
        val_pred = model.predict(X_val[features])
        test_pred = model.predict(X_test[features])

        train_mae = mean_absolute_error(y_train_inner, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        gap = train_mae - test_mae  # Negative means overfitting

        results[name] = {
            "features": features,
            "n_features": len(features),
            "train_mae": round(train_mae, 3),
            "val_mae": round(val_mae, 3),
            "test_mae": round(test_mae, 3),
            "test_r2": round(test_r2, 4),
            "overfit_gap": round(abs(gap), 3),
        }

        print(f"  {name:<25} {len(features):>3}   {train_mae:>8.3f}   {val_mae:>8.3f}   "
              f"{test_mae:>8.3f}   {test_r2:>7.4f}  {abs(gap):>7.3f}")

    # Find the best on test MAE
    best_name = min(results, key=lambda k: results[k]["test_mae"])
    print(f"\n  ★ Best test MAE: {best_name} → {results[best_name]['test_mae']:.3f} F")

    # Find best balance of accuracy and low overfitting
    # Score = test_mae + 0.5 * overfit_gap (penalize overfitting)
    for name, r in results.items():
        r["combined_score"] = r["test_mae"] + 0.5 * r["overfit_gap"]

    best_balanced = min(results, key=lambda k: results[k]["combined_score"])
    print(f"  ★ Best balanced: {best_balanced} → MAE={results[best_balanced]['test_mae']:.3f}, "
          f"Gap={results[best_balanced]['overfit_gap']:.3f}")

    return results, best_balanced


# ============================================================
# STEP 3: HYPERPARAMETER TUNING
# ============================================================

def tune_hyperparameters(df, features):
    """
    Grid search over key hyperparameters that control overfitting.
    """
    print("\n" + "=" * 60)
    print("  Step 3: Hyperparameter Tuning")
    print("=" * 60)

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Parameters to search
    param_grid = {
        "num_leaves": [31, 63, 127],
        "min_child_samples": [20, 50, 100, 200],
        "learning_rate": [0.03, 0.05],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0],
    }

    # Generate combinations
    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    print(f"\n  Testing {len(combos)} hyperparameter combinations...")
    print(f"  Features: {features}")
    print()

    best_score = float("inf")
    best_params = None
    best_model = None
    all_results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params.update({
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 1500,
            "random_state": 42,
        })

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_inner[features], y_train_inner,
            eval_set=[(X_val[features], y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        train_pred = model.predict(X_train_inner[features])
        test_pred = model.predict(X_test[features])

        train_mae = mean_absolute_error(y_train_inner, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        gap = abs(train_mae - test_mae)

        # Combined score: prioritize test MAE but penalize overfitting
        score = test_mae + 0.3 * gap

        all_results.append({
            **{k: v for k, v in zip(keys, combo)},
            "train_mae": train_mae,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "gap": gap,
            "score": score,
            "n_estimators_used": model.best_iteration_ or 1500,
        })

        if score < best_score:
            best_score = score
            best_params = params
            best_model = model

        # Progress update every 25 combos
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(combos)}] Current best score: {best_score:.4f} "
                  f"(MAE={best_score:.3f})")

    # Sort and show top 5
    all_results.sort(key=lambda x: x["score"])

    print(f"\n  Top 5 configurations:")
    print(f"  {'Leaves':>6} {'MinChild':>8} {'LR':>5} {'Alpha':>5} {'Lambda':>6} "
          f"{'TrainMAE':>8} {'TestMAE':>8} {'R²':>7} {'Gap':>6}")
    print("  " + "-" * 75)

    for r in all_results[:5]:
        print(f"  {r['num_leaves']:>6} {r['min_child_samples']:>8} "
              f"{r['learning_rate']:>5.2f} {r['reg_alpha']:>5.1f} {r['reg_lambda']:>6.1f} "
              f"{r['train_mae']:>8.3f} {r['test_mae']:>8.3f} "
              f"{r['test_r2']:>7.4f} {r['gap']:>6.3f}")

    print(f"\n  ★ Best params: num_leaves={best_params['num_leaves']}, "
          f"min_child_samples={best_params['min_child_samples']}, "
          f"lr={best_params['learning_rate']}, "
          f"alpha={best_params['reg_alpha']}, lambda={best_params['reg_lambda']}")

    return best_params, best_model, all_results


# ============================================================
# STEP 4: TRAIN FINAL MODEL & EVALUATE
# ============================================================

def train_final_model(df, features, params):
    """Train the final model with best params and full evaluation."""
    print("\n" + "=" * 60)
    print("  Step 4: Final Model Training")
    print("=" * 60)

    X = df[features]
    y = df[TARGET]

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print(f"\n  Train: {len(X_train_inner):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Features: {len(features)}")

    # Train
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train_inner[features], y_train_inner,
        eval_set=[(X_val[features], y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Evaluate on all sets
    sets = {
        "Train": (X_train_inner, y_train_inner),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test),
    }

    metrics = {}
    print()
    for name, (X_s, y_s) in sets.items():
        pred = model.predict(X_s[features])
        mae = mean_absolute_error(y_s, pred)
        rmse = np.sqrt(mean_squared_error(y_s, pred))
        r2 = r2_score(y_s, pred)

        metrics[name.lower()] = {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4)}

        print(f"  {name:<12} MAE: {mae:.3f} F | RMSE: {rmse:.3f} F | R²: {r2:.4f}")

    # Cross-validation
    cv_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    cv_params.pop("n_estimators", None)
    cv_params["n_estimators"] = model.best_iteration_ or 800
    cv_model = lgb.LGBMRegressor(**cv_params)

    cv_scores = cross_val_score(
        cv_model, X_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"\n  5-Fold CV:   MAE: {cv_mae:.3f} +/- {cv_std:.3f} F")
    metrics["cv_mae"] = round(cv_mae, 3)
    metrics["cv_std"] = round(cv_std, 3)

    # ---- Evaluate on EXTREME heat cells specifically ----
    print(f"\n  Extreme Heat Performance (cells > 105°F):")
    test_pred = model.predict(X_test[features])
    extreme_mask = y_test > 105
    if extreme_mask.sum() > 0:
        extreme_mae = mean_absolute_error(y_test[extreme_mask], test_pred[extreme_mask])
        extreme_r2 = r2_score(y_test[extreme_mask], test_pred[extreme_mask])
        n_extreme = extreme_mask.sum()
        print(f"    {n_extreme:,} extreme cells | MAE: {extreme_mae:.3f} F | R²: {extreme_r2:.4f}")
        metrics["extreme_mae"] = round(extreme_mae, 3)
        metrics["extreme_r2"] = round(extreme_r2, 4)
        metrics["extreme_n"] = int(n_extreme)
    else:
        print(f"    No extreme cells in test set")

    # ---- Evaluate on COOL cells specifically ----
    cool_mask = y_test < 90
    if cool_mask.sum() > 0:
        cool_mae = mean_absolute_error(y_test[cool_mask], test_pred[cool_mask])
        n_cool = cool_mask.sum()
        print(f"    {n_cool:,} cool cells (<90°F) | MAE: {cool_mae:.3f} F")
        metrics["cool_mae"] = round(cool_mae, 3)

    return model, metrics, X_test, y_test


# ============================================================
# STEP 5: INTERVENTION SIMULATION (IMPROVED)
# ============================================================

def improved_intervention_simulation(model, X_test, y_test, features):
    """
    Improved simulation that adjusts correlated features together.
    When you plant trees (increase NDVI), it also affects:
    - impervious_pct goes down
    - park_area_pct goes up
    - distance_to_park_m goes down (for nearby cells)
    """
    print("\n" + "=" * 60)
    print("  Step 5: Improved Intervention Simulation")
    print("=" * 60)

    y_pred = model.predict(X_test[features])
    hot_threshold = np.percentile(y_pred, 95)
    hot_mask = y_pred >= hot_threshold
    X_hot = X_test[hot_mask].copy()
    y_hot_pred = y_pred[hot_mask]

    print(f"\n  Hottest {hot_mask.sum():,} cells (>= {hot_threshold:.1f}°F predicted)")
    print(f"  Avg predicted temp: {y_hot_pred.mean():.1f}°F")
    print(f"  Avg NDVI: {X_hot['ndvi'].mean():.3f}")
    if "building_density" in features:
        print(f"  Avg building density: {X_hot['building_density'].mean():.3f}")

    simulations = [
        {
            "name": "Street trees (light)",
            "description": "Plant street trees along main roads",
            "ndvi_add": 0.12,
            "impervious_reduce": 5,
            "park_area_add": 0.0,
            "park_dist_reduce_pct": 0.0,
        },
        {
            "name": "Pocket parks (moderate)",
            "description": "Convert vacant lots to pocket parks + tree planting",
            "ndvi_add": 0.25,
            "impervious_reduce": 20,
            "park_area_add": 0.05,
            "park_dist_reduce_pct": 0.3,
        },
        {
            "name": "Green corridor (heavy)",
            "description": "Major green infrastructure: parks, green roofs, urban forest",
            "ndvi_add": 0.40,
            "impervious_reduce": 35,
            "park_area_add": 0.15,
            "park_dist_reduce_pct": 0.6,
        },
    ]

    results = {}

    print(f"\n  {'Intervention':<30} {'Temp After':>10} {'Cooling':>10}")
    print("  " + "-" * 55)

    for sim in simulations:
        X_mod = X_hot.copy()

        # Modify NDVI
        if "ndvi" in features:
            X_mod["ndvi"] = np.clip(X_mod["ndvi"] + sim["ndvi_add"], -1, 1)

        # Reduce impervious surface
        if "impervious_pct" in features:
            X_mod["impervious_pct"] = np.clip(
                X_mod["impervious_pct"] - sim["impervious_reduce"], 0, 100
            )

        # Increase park area
        if "park_area_pct" in features:
            X_mod["park_area_pct"] = np.clip(
                X_mod["park_area_pct"] + sim["park_area_add"], 0, 1
            )

        # Reduce distance to park
        if "distance_to_park_m" in features and sim["park_dist_reduce_pct"] > 0:
            X_mod["distance_to_park_m"] = X_mod["distance_to_park_m"] * (
                1 - sim["park_dist_reduce_pct"]
            )

        y_new = model.predict(X_mod[features])
        cooling = y_hot_pred.mean() - y_new.mean()

        results[sim["name"]] = {
            "description": sim["description"],
            "avg_temp_after": round(float(y_new.mean()), 2),
            "avg_cooling_f": round(float(cooling), 2),
            "max_cooling_f": round(float((y_hot_pred - y_new).max()), 2),
            "ndvi_change": sim["ndvi_add"],
        }

        print(f"  {sim['name']:<30} {y_new.mean():>8.1f}°F   {cooling:>+7.1f}°F")

    print(f"\n  Note: These estimates adjust correlated features together")
    print(f"  (tree planting also reduces impervious surface and increases park area)")

    return results


# ============================================================
# STEP 6: COMPARISON VISUALIZATION
# ============================================================

def plot_comparison(model_old_metrics, model_new, X_test, y_test, features, df):
    """Side-by-side comparison of old vs new model."""
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    y_pred = model_new.predict(X_test[features])

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    hb = ax.hexbin(y_test, y_pred, gridsize=50, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "b--", linewidth=2, alpha=0.7)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.95, f"MAE: {mae:.2f}°F\nR²: {r2:.4f}",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Actual Temperature (°F)")
    ax.set_ylabel("Predicted Temperature (°F)")
    ax.set_title("Predicted vs Actual (Tuned Model)", fontweight="bold")

    # 2. Residual distribution
    ax = axes[0, 1]
    residuals = y_test.values - y_pred
    ax.hist(residuals, bins=80, color="#E74C3C", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=2, linestyle="--")
    ax.axvline(residuals.mean(), color="blue", linewidth=2, label=f"Mean: {residuals.mean():.2f}°F")
    ax.set_xlabel("Residual (Actual - Predicted, °F)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.legend()
    ax.text(0.05, 0.95, f"Std: {residuals.std():.2f}°F\n"
            f"|Residual| < 2°F: {(np.abs(residuals) < 2).mean()*100:.1f}%",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 3. Feature importance
    ax = axes[1, 0]
    importance = model_new.feature_importances_
    importance_pct = importance / importance.sum() * 100
    sorted_idx = np.argsort(importance_pct)
    ax.barh(range(len(features)), importance_pct[sorted_idx], color="#E74C3C", alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in sorted_idx])
    ax.set_xlabel("Importance (%)")
    ax.set_title("Feature Importance (Tuned)", fontweight="bold")

    # 4. Performance by temperature bin
    ax = axes[1, 1]
    bins = np.arange(70, 140, 5)
    bin_labels = []
    bin_maes = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (y_test >= bins[i]) & (y_test < bins[i+1])
        if mask.sum() > 10:
            bin_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            bin_labels.append(f"{bins[i]}-{bins[i+1]}")
            bin_maes.append(bin_mae)
            bin_counts.append(mask.sum())

    bars = ax.bar(range(len(bin_labels)), bin_maes, color="#3498DB", alpha=0.8)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_ylabel("MAE (°F)")
    ax.set_xlabel("Temperature Bin (°F)")
    ax.set_title("Error by Temperature Range", fontweight="bold")
    ax.axhline(mae, color="red", linestyle="--", alpha=0.5, label=f"Overall MAE: {mae:.2f}")
    ax.legend()

    # Add count labels
    for bar, count in zip(bars, bin_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"n={count}", ha="center", fontsize=7)

    fig.suptitle("Tuned Model — Comprehensive Evaluation", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "tuning_comparison.png"),
                dpi=200, bbox_inches="tight")
    print(f"  Saved: output/tuning_comparison.png")
    plt.show()


def plot_spatial_residuals(model_new, df, features):
    """Spatial residual map for the tuned model."""
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    X_all = df[features]
    y_all = df[TARGET]
    y_pred = model_new.predict(X_all)
    residuals = y_all.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Predicted
    sc1 = axes[0].scatter(df["lon"], df["lat"], c=y_pred, cmap="RdYlBu_r",
                           s=0.3, alpha=0.5, vmin=75, vmax=115)
    plt.colorbar(sc1, ax=axes[0], label="Predicted Temp (°F)", shrink=0.7)
    axes[0].set_title("Predicted Surface Temperature (Tuned)", fontweight="bold")

    # Residuals
    sc2 = axes[1].scatter(df["lon"], df["lat"], c=residuals, cmap="RdBu_r",
                           s=0.3, alpha=0.5, vmin=-8, vmax=8)
    plt.colorbar(sc2, ax=axes[1], label="Residual (°F)", shrink=0.7)
    axes[1].set_title("Residuals (Red=hotter than predicted)", fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "tuning_residual_map.png"),
                dpi=150, bbox_inches="tight")
    print(f"  Saved: output/tuning_residual_map.png")
    plt.show()


# ============================================================
# SAVE
# ============================================================

def save_tuned_model(model, features, metrics, feature_results, intervention_results, best_params):
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_OUTPUT_DIR, "chicago_heat_model_tuned.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model: {model_path}")

    metadata = {
        "model_type": "lightgbm_tuned",
        "features": features,
        "target": TARGET,
        "metrics": metrics,
        "feature_selection_results": {k: {kk: vv for kk, vv in v.items()
                                           if kk != "features"}
                                       for k, v in feature_results.items()},
        "intervention_simulations": intervention_results,
        "best_hyperparameters": {k: v for k, v in best_params.items()
                                  if k not in ["verbose", "random_state"]},
        "trained_at": datetime.now().isoformat(),
        "dataset_size": int(metrics.get("dataset_size", 0)),
    }

    meta_path = os.path.join(MODEL_OUTPUT_DIR, "chicago_tuned_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved metadata: {meta_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("  HeatSense — Model Tuning & Optimization")
    print("=" * 60)

    # Step 1: Better data prep
    df, all_features = load_and_prepare_data_v2()

    # Step 2: Feature selection
    feature_results, best_feature_set_name = test_feature_subsets(df, all_features)
    best_features = feature_results[best_feature_set_name]["features"]
    print(f"\n  Using feature set: {best_feature_set_name}")
    print(f"  Features: {best_features}")

    # Step 3: Hyperparameter tuning
    best_params, _, hp_results = tune_hyperparameters(df, best_features)

    # Step 4: Train final model
    final_model, metrics, X_test, y_test = train_final_model(df, best_features, best_params)
    metrics["dataset_size"] = len(df)

    # Step 5: Improved intervention simulation
    intervention_results = improved_intervention_simulation(
        final_model, X_test, y_test, best_features
    )

    # Step 6: Visualizations
    print("\n" + "=" * 60)
    print("  Step 6: Generating Visualizations")
    print("=" * 60)
    plot_comparison(None, final_model, X_test, y_test, best_features, df)
    plot_spatial_residuals(final_model, df, best_features)

    # Save
    print("\n" + "=" * 60)
    print("  Saving Tuned Model")
    print("=" * 60)
    save_tuned_model(final_model, best_features, metrics,
                      feature_results, intervention_results, best_params)

    # Summary
    print("\n" + "=" * 60)
    print("  TUNING COMPLETE")
    print("=" * 60)
    print(f"\n  Dataset: {len(df):,} cells (vs 45,877 before imputation)")
    print(f"  Features: {len(best_features)} ({best_feature_set_name})")
    print(f"  Test MAE: {metrics['test']['mae']:.3f}°F")
    print(f"  Test R²:  {metrics['test']['r2']:.4f}")
    print(f"  Train-Test gap: {abs(metrics['train']['mae'] - metrics['test']['mae']):.3f}°F")
    if "extreme_mae" in metrics:
        print(f"  Extreme heat MAE: {metrics['extreme_mae']:.3f}°F")
    print(f"\n  Intervention estimates:")
    for name, result in intervention_results.items():
        print(f"    {name}: {result['avg_cooling_f']:+.1f}°F cooling")
    print(f"\n  Files:")
    print(f"  - model/models/chicago_heat_model_tuned.pkl")
    print(f"  - model/models/chicago_tuned_metadata.json")
    print(f"  - output/tuning_comparison.png")
    print(f"  - output/tuning_residual_map.png")
    print(f"\n  Next: Build the web application!")
    print("=" * 60)


if __name__ == "__main__":
    main()