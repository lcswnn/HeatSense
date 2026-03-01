"""
TomorrowLand Heat — Model Service
==================================
Loads the trained model and grid data, provides prediction,
intervention simulation, and heat map image generation.
"""

import pickle
import json
import io
import numpy as np
import pandas as pd
from pathlib import Path


def to_python(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


class HeatModel:
    """Wrapper around the trained LightGBM model and grid data."""

    def __init__(self, model_dir="../model/models", grid_path="../data/grid/chicago_grid.csv"):
        self.model = None
        self.metadata = None
        self.grid = None
        self.features = None
        self._load(model_dir, grid_path)

    def _load(self, model_dir, grid_path):
        model_dir = Path(model_dir)

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

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.features = self.metadata.get("features", [])
        else:
            self.features = [
                "ndvi", "impervious_pct", "lat", "lon",
                "building_count", "building_density", "avg_building_height_m",
                "road_density_km", "distance_to_park_m", "distance_to_water_m",
                "park_area_pct"
            ]

        print(f"  Features: {self.features}")

        grid_path = Path(grid_path)
        if grid_path.exists():
            self.grid = pd.read_csv(grid_path)
            self._prepare_grid()
            print(f"  Loaded grid: {len(self.grid):,} cells")
        else:
            raise FileNotFoundError(f"Grid not found at {grid_path}")

    def _prepare_grid(self):
        df = self.grid

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

        valid_mask = df[self.features].notna().all(axis=1) & df["mean_lst_f"].notna()
        self.grid_valid = df[valid_mask].copy()
        self.grid_valid = self.grid_valid[self.grid_valid["ndvi"] > -0.1].copy()

        self.grid_valid["predicted_lst_f"] = self.model.predict(
            self.grid_valid[self.features]
        )

        # Pre-compute grid parameters for image generation
        self._compute_grid_params()

        print(f"  Valid cells with predictions: {len(self.grid_valid):,}")

    def _compute_grid_params(self):
        """Compute the grid dimensions for image generation."""
        df = self.grid_valid
        self.min_lat = float(df["lat"].min())
        self.max_lat = float(df["lat"].max())
        self.min_lon = float(df["lon"].min())
        self.max_lon = float(df["lon"].max())
        self.spacing = 0.0009  # ~100m

        self.grid_cols = int(round((self.max_lon - self.min_lon) / self.spacing)) + 1
        self.grid_rows = int(round((self.max_lat - self.min_lat) / self.spacing)) + 1
        print(f"  Grid image dimensions: {self.grid_cols} x {self.grid_rows}")

    # ============================================================
    # HEAT MAP IMAGE GENERATION
    # ============================================================

    def generate_heatmap_png(self, layer="temperature", opacity=200):
        """
        Generate a PNG heat map image that can be overlaid on a Leaflet map.
        Returns: bytes (PNG image), bounds dict
        """
        from PIL import Image

        img = Image.new("RGBA", (self.grid_cols, self.grid_rows), (0, 0, 0, 0))
        pixels = img.load()

        df = self.grid_valid

        for _, row in df.iterrows():
            col_idx = int(round((row["lon"] - self.min_lon) / self.spacing))
            row_idx = int(round((self.max_lat - row["lat"]) / self.spacing))

            if 0 <= col_idx < self.grid_cols and 0 <= row_idx < self.grid_rows:
                if layer == "temperature":
                    rgb = self._temp_to_rgb(row["mean_lst_f"])
                elif layer == "risk":
                    rgb = self._risk_to_rgb(row.get("heat_risk", ""))
                elif layer == "ndvi":
                    rgb = self._ndvi_to_rgb(row["ndvi"])
                else:
                    rgb = self._temp_to_rgb(row["mean_lst_f"])

                pixels[col_idx, row_idx] = (rgb[0], rgb[1], rgb[2], opacity)

        # Scale up the image so pixels become visible blocks
        scale = 6
        img_scaled = img.resize(
            (self.grid_cols * scale, self.grid_rows * scale),
            Image.NEAREST  # Nearest neighbor = crisp blocks
        )

        buf = io.BytesIO()
        img_scaled.save(buf, format="PNG")
        buf.seek(0)

        bounds = {
            "south": self.min_lat - self.spacing / 2,
            "north": self.max_lat + self.spacing / 2,
            "west": self.min_lon - self.spacing / 2,
            "east": self.max_lon + self.spacing / 2,
        }

        return buf.getvalue(), bounds

    @staticmethod
    def _temp_to_rgb(temp):
        if temp is None or pd.isna(temp): return (80, 80, 80)
        if temp >= 115: return (128, 0, 0)
        if temp >= 110: return (180, 30, 30)
        if temp >= 107: return (210, 50, 40)
        if temp >= 104: return (231, 76, 60)
        if temp >= 101: return (230, 120, 40)
        if temp >= 98:  return (240, 160, 30)
        if temp >= 95:  return (241, 196, 15)
        if temp >= 92:  return (200, 210, 50)
        if temp >= 89:  return (120, 190, 100)
        if temp >= 85:  return (70, 170, 140)
        if temp >= 80:  return (52, 152, 219)
        if temp >= 75:  return (41, 120, 200)
        return (30, 70, 160)

    @staticmethod
    def _risk_to_rgb(risk):
        colors = {
            "extreme": (139, 0, 0),
            "high": (231, 76, 60),
            "moderate": (230, 126, 34),
            "low": (52, 152, 219),
        }
        return colors.get(risk, (60, 60, 60))

    @staticmethod
    def _ndvi_to_rgb(ndvi):
        if ndvi is None or pd.isna(ndvi): return (80, 80, 80)
        if ndvi >= 0.55: return (0, 90, 0)
        if ndvi >= 0.45: return (20, 120, 20)
        if ndvi >= 0.35: return (46, 160, 70)
        if ndvi >= 0.25: return (100, 180, 90)
        if ndvi >= 0.15: return (180, 190, 80)
        if ndvi >= 0.05: return (200, 150, 70)
        return (160, 82, 45)

    # ============================================================
    # GRID DATA (JSON)
    # ============================================================

    def get_grid_data(self, bbox=None, downsample=None):
        df = self.grid_valid

        if bbox:
            df = df[
                (df["lon"] >= bbox["west"]) & (df["lon"] <= bbox["east"]) &
                (df["lat"] >= bbox["south"]) & (df["lat"] <= bbox["north"])
            ]

        if downsample and downsample > 1:
            df = df.iloc[::downsample]

        columns = [
            "cell_id", "lon", "lat",
            "mean_lst_f", "predicted_lst_f", "ndvi", "impervious_pct",
            "building_density", "avg_building_height_m",
            "distance_to_park_m", "distance_to_water_m",
            "road_density_km", "park_area_pct", "heat_risk"
        ]
        columns = [c for c in columns if c in df.columns]

        result = df[columns].copy()
        for col in result.select_dtypes(include=[np.floating]).columns:
            result[col] = result[col].round(2)

        records = result.to_dict(orient="records")
        # Convert numpy types to native Python types
        for rec in records:
            for k, v in rec.items():
                rec[k] = to_python(v)

        return records

    # ============================================================
    # CELL DETAIL
    # ============================================================

    def get_cell_detail(self, lat, lon):
        df = self.grid_valid
        distances = np.sqrt((df["lat"] - lat)**2 + (df["lon"] - lon)**2)
        idx = distances.idxmin()
        cell = df.loc[idx]

        city_avg_temp = float(df["mean_lst_f"].mean())
        city_avg_ndvi = float(df["ndvi"].mean())

        detail = {
            "cell_id": to_python(cell.get("cell_id", "")),
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
            "heat_risk": str(cell.get("heat_risk", "unknown")),
            "vs_city_avg_f": round(float(cell["mean_lst_f"]) - city_avg_temp, 1),
            "city_avg_temp_f": round(city_avg_temp, 1),
            "city_avg_ndvi": round(city_avg_ndvi, 3),
        }

        percentile = float((df["mean_lst_f"] < cell["mean_lst_f"]).mean() * 100)
        detail["temp_percentile"] = round(percentile, 1)

        return detail

    # ============================================================
    # INTERVENTION SIMULATION
    # ============================================================

    def simulate_intervention(self, lat, lon, radius_m=500, intervention_type="moderate"):
        interventions = {
            "light": {
                "name": "Street Trees",
                "description": "Plant street trees along main roads",
                "ndvi_add": 0.12, "impervious_reduce": 5,
                "park_area_add": 0.0, "park_dist_factor": 1.0,
            },
            "moderate": {
                "name": "Pocket Parks",
                "description": "Convert vacant lots to pocket parks + tree planting",
                "ndvi_add": 0.25, "impervious_reduce": 20,
                "park_area_add": 0.05, "park_dist_factor": 0.7,
            },
            "heavy": {
                "name": "Green Corridor",
                "description": "Major green infrastructure: parks, green roofs, urban forest",
                "ndvi_add": 0.40, "impervious_reduce": 35,
                "park_area_add": 0.15, "park_dist_factor": 0.4,
            },
        }

        if intervention_type not in interventions:
            intervention_type = "moderate"
        config = interventions[intervention_type]

        df = self.grid_valid
        lat_range = radius_m / 111000
        lon_range = radius_m / 82000

        mask = (
            (df["lat"] >= lat - lat_range) & (df["lat"] <= lat + lat_range) &
            (df["lon"] >= lon - lon_range) & (df["lon"] <= lon + lon_range)
        )
        affected = df[mask].copy()

        if len(affected) == 0:
            return {"error": "No grid cells found in this area"}

        before_preds = self.model.predict(affected[self.features])
        before_avg = float(before_preds.mean())

        X_mod = affected[self.features].copy()
        if "ndvi" in self.features:
            X_mod["ndvi"] = np.clip(X_mod["ndvi"] + config["ndvi_add"], -1, 1)
        if "impervious_pct" in self.features:
            X_mod["impervious_pct"] = np.clip(X_mod["impervious_pct"] - config["impervious_reduce"], 0, 100)
        if "park_area_pct" in self.features:
            X_mod["park_area_pct"] = np.clip(X_mod["park_area_pct"] + config["park_area_add"], 0, 1)
        if "distance_to_park_m" in self.features:
            X_mod["distance_to_park_m"] = X_mod["distance_to_park_m"] * config["park_dist_factor"]

        after_preds = self.model.predict(X_mod)
        after_avg = float(after_preds.mean())
        cooling = before_avg - after_avg

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
            "before_avg_temp_f": round(before_avg, 1),
            "after_avg_temp_f": round(after_avg, 1),
            "avg_cooling_f": round(cooling, 1),
            "max_cooling_f": round(float((before_preds - after_preds).max()), 1),
            "cells": cell_results,
        }

    # ============================================================
    # CITY STATS
    # ============================================================

    def get_city_stats(self):
        df = self.grid_valid
        risk_counts = df["heat_risk"].value_counts().to_dict()
        total = len(df)

        return {
            "total_cells": int(total),
            "avg_temp_f": round(float(df["mean_lst_f"].mean()), 1),
            "min_temp_f": round(float(df["mean_lst_f"].min()), 1),
            "max_temp_f": round(float(df["mean_lst_f"].max()), 1),
            "avg_ndvi": round(float(df["ndvi"].mean()), 3),
            "avg_impervious_pct": round(float(df["impervious_pct"].mean()), 1),
            "heat_risk_distribution": {
                str(risk): {"count": int(count), "pct": round(int(count) / total * 100, 1)}
                for risk, count in risk_counts.items()
            },
        }

    # ============================================================
    # NEIGHBORHOOD COMPARISON
    # ============================================================

    def get_neighborhood_comparison(self, neighborhoods):
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
                "cells": int(len(cells)),
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

    # ============================================================
    # SIMULATION OVERLAY IMAGE
    # ============================================================

    def generate_simulation_png(self, lat, lon, radius_m, intervention_type, opacity=220):
        """
        Generate a PNG showing before→after cooling for an intervention area.
        Green = cooled cells. Brightness = amount of cooling.
        Returns: bytes (PNG), bounds dict
        """
        from PIL import Image

        config_map = {
            "light":    {"ndvi_add": 0.12, "impervious_reduce": 5,  "park_area_add": 0.0,  "park_dist_factor": 1.0},
            "moderate": {"ndvi_add": 0.25, "impervious_reduce": 20, "park_area_add": 0.05, "park_dist_factor": 0.7},
            "heavy":    {"ndvi_add": 0.40, "impervious_reduce": 35, "park_area_add": 0.15, "park_dist_factor": 0.4},
        }
        config = config_map.get(intervention_type, config_map["moderate"])

        df = self.grid_valid
        lat_range = radius_m / 111000
        lon_range = radius_m / 82000

        mask = (
            (df["lat"] >= lat - lat_range) & (df["lat"] <= lat + lat_range) &
            (df["lon"] >= lon - lon_range) & (df["lon"] <= lon + lon_range)
        )
        affected = df[mask].copy()
        if len(affected) == 0:
            return None, None

        # Predict before/after
        before_preds = self.model.predict(affected[self.features])

        X_mod = affected[self.features].copy()
        if "ndvi" in self.features:
            X_mod["ndvi"] = np.clip(X_mod["ndvi"] + config["ndvi_add"], -1, 1)
        if "impervious_pct" in self.features:
            X_mod["impervious_pct"] = np.clip(X_mod["impervious_pct"] - config["impervious_reduce"], 0, 100)
        if "park_area_pct" in self.features:
            X_mod["park_area_pct"] = np.clip(X_mod["park_area_pct"] + config["park_area_add"], 0, 1)
        if "distance_to_park_m" in self.features:
            X_mod["distance_to_park_m"] = X_mod["distance_to_park_m"] * config["park_dist_factor"]

        after_preds = self.model.predict(X_mod)
        cooling = before_preds - after_preds

        # Build small image for just the affected area
        a_min_lat = float(affected["lat"].min())
        a_max_lat = float(affected["lat"].max())
        a_min_lon = float(affected["lon"].min())
        a_max_lon = float(affected["lon"].max())

        spacing = self.spacing
        cols = int(round((a_max_lon - a_min_lon) / spacing)) + 1
        rows = int(round((a_max_lat - a_min_lat) / spacing)) + 1
        cols = max(cols, 1)
        rows = max(rows, 1)

        img = Image.new("RGBA", (cols, rows), (0, 0, 0, 0))
        pixels = img.load()

        for i, (_, row) in enumerate(affected.iterrows()):
            c = int(round((row["lon"] - a_min_lon) / spacing))
            r = int(round((a_max_lat - row["lat"]) / spacing))
            if 0 <= c < cols and 0 <= r < rows:
                cool = float(cooling[i])
                after_t = float(after_preds[i])
                # Color: green intensity based on cooling amount
                rgb = self._temp_to_rgb(after_t)
                pixels[c, r] = (rgb[0], rgb[1], rgb[2], opacity)

        scale = 6
        img_scaled = img.resize((cols * scale, rows * scale), Image.NEAREST)

        buf = io.BytesIO()
        img_scaled.save(buf, format="PNG")
        buf.seek(0)

        pad = spacing / 2
        bounds = {
            "south": a_min_lat - pad,
            "north": a_max_lat + pad,
            "west": a_min_lon - pad,
            "east": a_max_lon + pad,
        }

        return buf.getvalue(), bounds

    # ============================================================
    # SMART INTERVENTION TARGETING
    # ============================================================

    def find_priority_interventions(self, min_temp_f=100, top_n=15):
        """
        Identify the best locations for green interventions by analyzing
        hot areas and filtering out infeasible locations (airports, rail, highways).

        Scoring logic:
        - Higher temperature → higher priority
        - Lower NDVI (less vegetation) → more room for improvement
        - Higher building density in residential range → more people benefit
        - Closer to existing parks → easier to expand
        - Very high impervious + zero buildings → likely infrastructure (airport/highway), penalize

        Returns list of priority zones with scores and descriptions.
        """
        df = self.grid_valid.copy()

        # Filter to hot cells
        hot = df[df["mean_lst_f"] >= min_temp_f].copy()
        if len(hot) == 0:
            return []

        # ---- Land use classification heuristics ----
        # We don't have explicit zoning data, but we can infer from features:
        # Airport/runway: very high impervious (>90%), near-zero buildings, near-zero NDVI
        # Highway/railroad: very high impervious, zero buildings, high road density
        # Industrial: high impervious, moderate buildings, low NDVI
        # Commercial: high building density, moderate-high impervious
        # Residential: moderate building density, some NDVI
        # Vacant/open: low building density, low NDVI, moderate impervious

        conditions = []
        for _, row in hot.iterrows():
            imp = float(row.get("impervious_pct", 0))
            ndvi = float(row.get("ndvi", 0))
            bd = float(row.get("building_density", 0))
            bc = float(row.get("building_count", 0))
            rd = float(row.get("road_density_km", 0))
            ht = float(row.get("avg_building_height_m", 0))
            park_d = float(row.get("distance_to_park_m", 0))

            # Classify
            if imp > 90 and bd < 0.05 and ndvi < 0.05:
                land_use = "airport_runway"
                feasibility = 0.05  # Can't plant trees on runways
            elif imp > 85 and bd < 0.02 and rd > 0.01:
                land_use = "highway_rail"
                feasibility = 0.1  # Very limited options
            elif imp > 75 and bd > 0.3 and ht > 15:
                land_use = "commercial_highrise"
                feasibility = 0.3  # Green roofs possible
            elif imp > 70 and bd > 0.2 and ht <= 15:
                land_use = "commercial_industrial"
                feasibility = 0.5  # Parking lots, some tree planting
            elif bd > 0.05 and bd <= 0.3:
                land_use = "residential"
                feasibility = 0.9  # Best candidates — street trees, yard programs
            elif bd <= 0.05 and imp < 60:
                land_use = "vacant_open"
                feasibility = 1.0  # Ideal for pocket parks
            else:
                land_use = "mixed_urban"
                feasibility = 0.6

            conditions.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "temp_f": round(float(row["mean_lst_f"]), 1),
                "ndvi": round(ndvi, 3),
                "impervious_pct": round(imp, 1),
                "building_density": round(bd, 3),
                "distance_to_park_m": round(park_d, 0),
                "land_use": land_use,
                "feasibility": feasibility,
            })

        cdf = pd.DataFrame(conditions)

        # ---- Score each cell ----
        # Normalize components to 0-1 range
        temp_range = cdf["temp_f"].max() - cdf["temp_f"].min()
        if temp_range == 0:
            temp_range = 1

        cdf["heat_score"] = (cdf["temp_f"] - cdf["temp_f"].min()) / temp_range
        cdf["veg_need_score"] = 1 - np.clip(cdf["ndvi"] / 0.4, 0, 1)  # Low NDVI = high need
        cdf["park_access_score"] = np.clip(cdf["distance_to_park_m"] / 1500, 0, 1)  # Far from park = high need

        # Combined priority score
        cdf["priority_score"] = (
            cdf["heat_score"] * 0.35 +
            cdf["veg_need_score"] * 0.25 +
            cdf["park_access_score"] * 0.15 +
            cdf["feasibility"] * 0.25
        )

        # ---- Cluster nearby cells into zones ----
        # Group by rounding to ~500m blocks
        cdf["block_lat"] = (cdf["lat"] / 0.005).round() * 0.005
        cdf["block_lon"] = (cdf["lon"] / 0.005).round() * 0.005

        zones = cdf.groupby(["block_lat", "block_lon"]).agg(
            avg_temp_f=("temp_f", "mean"),
            avg_ndvi=("ndvi", "mean"),
            avg_impervious=("impervious_pct", "mean"),
            avg_priority=("priority_score", "mean"),
            avg_feasibility=("feasibility", "mean"),
            cell_count=("temp_f", "count"),
            primary_land_use=("land_use", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
            avg_park_dist=("distance_to_park_m", "mean"),
        ).reset_index()

        # Filter to zones with enough cells to matter
        zones = zones[zones["cell_count"] >= 3]

        # Sort by priority
        zones = zones.sort_values("avg_priority", ascending=False).head(top_n)

        # Format results
        land_use_labels = {
            "airport_runway": "Airport / Runway",
            "highway_rail": "Highway / Railroad",
            "commercial_highrise": "Commercial (High-rise)",
            "commercial_industrial": "Commercial / Industrial",
            "residential": "Residential Neighborhood",
            "vacant_open": "Vacant / Open Land",
            "mixed_urban": "Mixed Urban",
        }

        intervention_recs = {
            "airport_runway": "Not feasible for green intervention",
            "highway_rail": "Limited — consider sound barrier vegetation",
            "commercial_highrise": "Green roofs, rooftop gardens",
            "commercial_industrial": "Parking lot trees, cool pavement, shade structures",
            "residential": "Street trees, yard programs, community gardens",
            "vacant_open": "Pocket parks, urban forest, community gardens",
            "mixed_urban": "Street trees, cool roofs, shade structures",
        }

        results = []
        for _, z in zones.iterrows():
            lu = str(z["primary_land_use"])
            results.append({
                "lat": round(float(z["block_lat"]), 4),
                "lon": round(float(z["block_lon"]), 4),
                "avg_temp_f": round(float(z["avg_temp_f"]), 1),
                "avg_ndvi": round(float(z["avg_ndvi"]), 3),
                "avg_impervious_pct": round(float(z["avg_impervious"]), 1),
                "cell_count": int(z["cell_count"]),
                "priority_score": round(float(z["avg_priority"]), 3),
                "feasibility": round(float(z["avg_feasibility"]), 2),
                "land_use": lu,
                "land_use_label": land_use_labels.get(lu, lu),
                "recommendation": intervention_recs.get(lu, "Evaluate on-site"),
                "distance_to_park_m": round(float(z["avg_park_dist"]), 0),
            })

        return results