"""
TomorrowLand Heat — Model Service
==================================
Loads the trained model and grid data, provides prediction and
intervention simulation functions for the API layer.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path


class HeatModel:
    """Wrapper around the trained LightGBM model and grid data."""

    def __init__(self, model_dir="../model/models", grid_path="../data/grid/chicago_grid.csv"):
        self.model = None
        self.metadata = None
        self.grid = None
        self.features = None
        self._load(model_dir, grid_path)

    def _load(self, model_dir, grid_path):
        """Load model, metadata, and grid data."""
        model_dir = Path(model_dir)

        # Try tuned model first, fall back to original
        tuned_path = model_dir / "chicago_heat_model_tuned.pkl"
        original_path = model_dir / "chicago_heat_model.pkl"

        if tuned_path.exists():
            model_path = tuned_path
            meta_path = model_dir / "chicago_tuned_metadata.json"
            print(f"  Loading tuned model from {model_path}")
        elif original_path.exists():
            model_path = original_path
            meta_path = model_dir / "chicago_model_metadata.json"
            print(f"  Loading original model from {model_path}")
        else:
            raise FileNotFoundError(f"No model found in {model_dir}")

        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load metadata
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.features = self.metadata.get("features", [])
        else:
            # Fallback feature list
            self.features = [
                "ndvi", "impervious_pct", "lat", "lon",
                "building_count", "building_density", "avg_building_height_m",
                "road_density_km", "distance_to_park_m", "distance_to_water_m",
                "park_area_pct"
            ]

        print(f"  Features: {self.features}")

        # Load grid data
        grid_path = Path(grid_path)
        if grid_path.exists():
            self.grid = pd.read_csv(grid_path)
            self._prepare_grid()
            print(f"  Loaded grid: {len(self.grid):,} cells")
        else:
            raise FileNotFoundError(f"Grid not found at {grid_path}")

    def _prepare_grid(self):
        """Prepare grid data — impute missing values same as training."""
        df = self.grid

        # Same imputation as tune_model.py
        fill_zero = ["building_count", "building_density", "avg_building_height_m",
                      "road_density_km", "park_area_pct"]
        for col in fill_zero:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        if "impervious_pct" in df.columns:
            df["impervious_pct"] = df["impervious_pct"].fillna(df["impervious_pct"].median())

        for col in ["distance_to_park_m", "distance_to_water_m"]:
            if col in df.columns:
                p95 = df[col].quantile(0.95)
                df[col] = df[col].fillna(p95)

        # Add predictions for all cells that have the required features
        valid_mask = df[self.features].notna().all(axis=1) & df["mean_lst_f"].notna()
        self.grid_valid = df[valid_mask].copy()

        # Remove water pixels
        self.grid_valid = self.grid_valid[self.grid_valid["ndvi"] > -0.1].copy()

        # Generate predictions
        self.grid_valid["predicted_lst_f"] = self.model.predict(
            self.grid_valid[self.features]
        )

        print(f"  Valid cells with predictions: {len(self.grid_valid):,}")

    def get_grid_data(self, bbox=None, downsample=None):
        """
        Get grid data, optionally filtered to a bounding box.
        Returns a list of dicts for JSON serialization.

        Args:
            bbox: dict with west, south, east, north (optional)
            downsample: int, return every Nth cell for performance (optional)
        """
        df = self.grid_valid

        if bbox:
            df = df[
                (df["lon"] >= bbox["west"]) &
                (df["lon"] <= bbox["east"]) &
                (df["lat"] >= bbox["south"]) &
                (df["lat"] <= bbox["north"])
            ]

        if downsample and downsample > 1:
            df = df.iloc[::downsample]

        # Select columns to send to frontend
        columns = [
            "cell_id", "lon", "lat",
            "mean_lst_f", "predicted_lst_f", "ndvi", "impervious_pct",
            "building_density", "avg_building_height_m",
            "distance_to_park_m", "distance_to_water_m",
            "road_density_km", "park_area_pct", "heat_risk"
        ]
        columns = [c for c in columns if c in df.columns]

        # Round floats for smaller JSON payload
        result = df[columns].copy()
        for col in result.select_dtypes(include=[np.floating]).columns:
            result[col] = result[col].round(2)

        return result.to_dict(orient="records")

    def get_cell_detail(self, lat, lon):
        """
        Get detailed info for the nearest grid cell to a lat/lon point.
        """
        df = self.grid_valid

        # Find nearest cell using simple distance
        distances = np.sqrt((df["lat"] - lat)**2 + (df["lon"] - lon)**2)
        idx = distances.idxmin()
        cell = df.loc[idx]

        # City-wide stats for comparison
        city_avg_temp = df["mean_lst_f"].mean()
        city_avg_ndvi = df["ndvi"].mean()

        detail = {
            "cell_id": cell.get("cell_id", ""),
            "lat": round(float(cell["lat"]), 5),
            "lon": round(float(cell["lon"]), 5),
            "temperature_f": round(float(cell["mean_lst_f"]), 1),
            "predicted_f": round(float(cell["predicted_lst_f"]), 1),
            "ndvi": round(float(cell["ndvi"]), 3),
            "impervious_pct": round(float(cell.get("impervious_pct", 0)), 1),
            "building_density": round(float(cell.get("building_density", 0)), 3),
            "avg_building_height_m": round(float(cell.get("avg_building_height_m", 0)), 1),
            "distance_to_park_m": round(float(cell.get("distance_to_park_m", 0)), 0),
            "distance_to_water_m": round(float(cell.get("distance_to_water_m", 0)), 0),
            "road_density_km": round(float(cell.get("road_density_km", 0)), 3),
            "park_area_pct": round(float(cell.get("park_area_pct", 0)), 3),
            "heat_risk": cell.get("heat_risk", "unknown"),
            # Comparisons
            "vs_city_avg_f": round(float(cell["mean_lst_f"] - city_avg_temp), 1),
            "city_avg_temp_f": round(float(city_avg_temp), 1),
            "city_avg_ndvi": round(float(city_avg_ndvi), 3),
        }

        # Percentile ranking
        percentile = (df["mean_lst_f"] < cell["mean_lst_f"]).mean() * 100
        detail["temp_percentile"] = round(float(percentile), 1)

        return detail

    def simulate_intervention(self, lat, lon, radius_m=500, intervention_type="moderate"):
        """
        Simulate a green infrastructure intervention around a point.

        Args:
            lat, lon: center of intervention
            radius_m: radius of effect in meters
            intervention_type: "light", "moderate", or "heavy"
        """
        interventions = {
            "light": {
                "name": "Street Trees",
                "description": "Plant street trees along main roads",
                "ndvi_add": 0.12,
                "impervious_reduce": 5,
                "park_area_add": 0.0,
                "park_dist_factor": 1.0,
            },
            "moderate": {
                "name": "Pocket Parks",
                "description": "Convert vacant lots to pocket parks + tree planting",
                "ndvi_add": 0.25,
                "impervious_reduce": 20,
                "park_area_add": 0.05,
                "park_dist_factor": 0.7,
            },
            "heavy": {
                "name": "Green Corridor",
                "description": "Major green infrastructure: parks, green roofs, urban forest",
                "ndvi_add": 0.40,
                "impervious_reduce": 35,
                "park_area_add": 0.15,
                "park_dist_factor": 0.4,
            },
        }

        if intervention_type not in interventions:
            intervention_type = "moderate"

        config = interventions[intervention_type]
        df = self.grid_valid

        # Find cells within radius (approximate using lat/lon degrees)
        # 1 degree lat ≈ 111,000m, 1 degree lon ≈ 82,000m at Chicago's latitude
        lat_range = radius_m / 111000
        lon_range = radius_m / 82000

        mask = (
            (df["lat"] >= lat - lat_range) &
            (df["lat"] <= lat + lat_range) &
            (df["lon"] >= lon - lon_range) &
            (df["lon"] <= lon + lon_range)
        )

        affected = df[mask].copy()

        if len(affected) == 0:
            return {"error": "No grid cells found in this area"}

        # Current state
        before_temp = float(affected["mean_lst_f"].mean())
        before_pred = float(self.model.predict(affected[self.features]).mean())

        # Apply intervention
        X_modified = affected[self.features].copy()

        if "ndvi" in self.features:
            X_modified["ndvi"] = np.clip(X_modified["ndvi"] + config["ndvi_add"], -1, 1)

        if "impervious_pct" in self.features:
            X_modified["impervious_pct"] = np.clip(
                X_modified["impervious_pct"] - config["impervious_reduce"], 0, 100
            )

        if "park_area_pct" in self.features:
            X_modified["park_area_pct"] = np.clip(
                X_modified["park_area_pct"] + config["park_area_add"], 0, 1
            )

        if "distance_to_park_m" in self.features:
            X_modified["distance_to_park_m"] = (
                X_modified["distance_to_park_m"] * config["park_dist_factor"]
            )

        # Predict after intervention
        after_pred = float(self.model.predict(X_modified).mean())
        cooling = before_pred - after_pred

        # Per-cell results for map visualization
        before_preds = self.model.predict(affected[self.features])
        after_preds = self.model.predict(X_modified)
        cell_results = []
        for i, (_, row) in enumerate(affected.iterrows()):
            cell_results.append({
                "lat": round(float(row["lat"]), 5),
                "lon": round(float(row["lon"]), 5),
                "before_f": round(float(before_preds[i]), 1),
                "after_f": round(float(after_preds[i]), 1),
                "cooling_f": round(float(before_preds[i] - after_preds[i]), 1),
            })

        return {
            "intervention": config["name"],
            "description": config["description"],
            "center": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "cells_affected": len(affected),
            "before_avg_temp_f": round(before_pred, 1),
            "after_avg_temp_f": round(after_pred, 1),
            "avg_cooling_f": round(cooling, 1),
            "max_cooling_f": round(float((before_preds - after_preds).max()), 1),
            "cells": cell_results,
        }

    def get_city_stats(self):
        """Get city-wide summary statistics."""
        df = self.grid_valid

        risk_counts = df["heat_risk"].value_counts().to_dict()
        total = len(df)

        return {
            "total_cells": total,
            "avg_temp_f": round(float(df["mean_lst_f"].mean()), 1),
            "min_temp_f": round(float(df["mean_lst_f"].min()), 1),
            "max_temp_f": round(float(df["mean_lst_f"].max()), 1),
            "avg_ndvi": round(float(df["ndvi"].mean()), 3),
            "avg_impervious_pct": round(float(df["impervious_pct"].mean()), 1),
            "heat_risk_distribution": {
                risk: {"count": int(count), "pct": round(count / total * 100, 1)}
                for risk, count in risk_counts.items()
            },
            "model_info": {
                "type": self.metadata.get("model_type", "unknown") if self.metadata else "unknown",
                "features": len(self.features),
                "test_mae_f": self.metadata.get("metrics", {}).get("test", {}).get("mae", None) if self.metadata else None,
            }
        }

    def get_neighborhood_comparison(self, neighborhoods):
        """
        Compare stats across neighborhoods.
        neighborhoods: list of dicts with name, lat, lon, radius_m
        """
        results = []

        for hood in neighborhoods:
            lat, lon = hood["lat"], hood["lon"]
            radius_m = hood.get("radius_m", 1000)

            lat_range = radius_m / 111000
            lon_range = radius_m / 82000

            mask = (
                (self.grid_valid["lat"] >= lat - lat_range) &
                (self.grid_valid["lat"] <= lat + lat_range) &
                (self.grid_valid["lon"] >= lon - lon_range) &
                (self.grid_valid["lon"] <= lon + lon_range)
            )

            cells = self.grid_valid[mask]

            if len(cells) == 0:
                continue

            results.append({
                "name": hood["name"],
                "cells": len(cells),
                "avg_temp_f": round(float(cells["mean_lst_f"].mean()), 1),
                "avg_ndvi": round(float(cells["ndvi"].mean()), 3),
                "avg_impervious_pct": round(float(cells["impervious_pct"].mean()), 1),
                "avg_building_density": round(float(cells["building_density"].mean()), 3),
                "avg_distance_to_park_m": round(float(cells["distance_to_park_m"].mean()), 0),
                "heat_risk_pct": {
                    risk: round(float((cells["heat_risk"] == risk).mean() * 100), 1)
                    for risk in ["extreme", "high", "moderate", "low"]
                }
            })

        return results