"""
TomorrowLand Heat — Step 3: Fetch OpenStreetMap Urban Data
==========================================================
v4 — ALL data fetched via quadrant-based direct Overpass queries.
No OSMnx dependency for data fetching (only used for road graph if available).

Usage:
  python fetch_osm.py

Output:
  - data/osm/chicago_buildings.geojson
  - data/osm/chicago_parks.geojson
  - data/osm/chicago_water.geojson
  - data/osm/chicago_roads.geojson
  - output/chicago_urban_features.png
"""

import os
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon, LineString, MultiPolygon
from datetime import datetime
import threading
import sys
import requests
import time


# ============================================================
# PROGRESS LOGGING
# ============================================================

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"  [{timestamp()}] {msg}")
    sys.stdout.flush()


class ProgressMonitor:
    def __init__(self, task_name, interval=30):
        self.task_name = task_name
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = None

    def _heartbeat(self):
        while self.running:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            log(f"... still working on {self.task_name} ({mins}m {secs}s elapsed)")
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        log(f"Started: {self.task_name}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        log(f"Finished: {self.task_name} (took {mins}m {secs}s)")


# ============================================================
# CONFIGURATION
# ============================================================

def load_config(config_path="config/chicago.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_bbox_into_grid(bbox, rows=3, cols=3):
    """Split a bounding box into a grid of smaller bounding boxes."""
    west, south, east, north = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    lon_step = (east - west) / cols
    lat_step = (north - south) / rows

    quadrants = []
    for r in range(rows):
        for c in range(cols):
            q = {
                "west": west + c * lon_step,
                "south": south + r * lat_step,
                "east": west + (c + 1) * lon_step,
                "north": south + (r + 1) * lat_step,
            }
            quadrants.append(q)
    return quadrants


# ============================================================
# OVERPASS API
# ============================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_query(query, max_retries=3, timeout=200):
    """Execute an Overpass API query with retry logic."""
    for attempt in range(max_retries):
        try:
            log(f"  Overpass request (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=timeout
            )
            if response.status_code == 200:
                size_mb = len(response.content) / 1024 / 1024
                log(f"  Response: {size_mb:.1f} MB")
                return response.json()
            elif response.status_code == 429:
                wait = 30 * (attempt + 1)
                log(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif response.status_code == 504:
                log(f"  Gateway timeout — retrying...")
                time.sleep(15)
            else:
                log(f"  HTTP {response.status_code}")
                time.sleep(10)
        except requests.exceptions.Timeout:
            log(f"  Request timeout, retrying...")
            time.sleep(15)
        except requests.exceptions.ConnectionError:
            log(f"  Connection error, retrying...")
            time.sleep(15)

    log("  ⚠ Failed after all retries")
    return None


def overpass_to_polygons(result):
    """Convert Overpass JSON to GeoDataFrame of polygons."""
    if result is None or "elements" not in result:
        return gpd.GeoDataFrame()

    elements = result["elements"]

    # Build node lookup
    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    # Build polygons from ways
    features = []
    for el in elements:
        if el["type"] == "way" and "nodes" in el:
            coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
            if len(coords) < 3:
                continue
            try:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                geom = Polygon(coords)
                if geom.is_valid and not geom.is_empty:
                    props = el.get("tags", {})
                    props["geometry"] = geom
                    features.append(props)
            except Exception:
                continue

    if not features:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def overpass_to_lines_and_polygons(result):
    """Convert Overpass JSON to GeoDataFrame of lines and polygons (for water/roads)."""
    if result is None or "elements" not in result:
        return gpd.GeoDataFrame()

    elements = result["elements"]

    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    features = []
    for el in elements:
        if el["type"] == "way" and "nodes" in el:
            coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
            if len(coords) < 2:
                continue
            try:
                tags = el.get("tags", {})
                # If first == last and enough points, it's a polygon
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    geom = Polygon(coords)
                else:
                    geom = LineString(coords)

                if geom.is_valid and not geom.is_empty:
                    tags["geometry"] = geom
                    features.append(tags)
            except Exception:
                continue

    if not features:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(features, crs="EPSG:4326")


# ============================================================
# GENERIC QUADRANT FETCHER
# ============================================================

def fetch_by_quadrants(config, query_builder, parser, label, rows=3, cols=3,
                       min_area=None, pause=5):
    """
    Generic function to fetch any OSM data type by quadrants.

    Args:
        config: city config dict
        query_builder: function(s, w, n, e) -> Overpass query string
        parser: function(result) -> GeoDataFrame
        label: display name for logging
        rows, cols: grid dimensions
        min_area: if set, filter out polygons smaller than this (m2)
        pause: seconds to wait between quadrants
    """
    quadrants = split_bbox_into_grid(config["bbox"], rows=rows, cols=cols)
    total = len(quadrants)
    log(f"Splitting into {total} quadrants ({rows}x{cols})")

    all_data = []

    for i, quad in enumerate(quadrants):
        quad_name = f"{label} quadrant {i+1}/{total}"

        monitor = ProgressMonitor(quad_name, interval=30)
        monitor.start()
        try:
            s, w, n, e = quad["south"], quad["west"], quad["north"], quad["east"]
            query = query_builder(s, w, n, e)

            log(f"Fetching {quad_name}...")
            result = overpass_query(query)
            gdf = parser(result)
            log(f"  {quad_name}: {len(gdf):,} features")

            if len(gdf) > 0:
                all_data.append(gdf)
        except Exception as ex:
            log(f"⚠ Error on {quad_name}: {ex}")
        finally:
            monitor.stop()

        if i < total - 1:
            log(f"  Pausing {pause}s...")
            time.sleep(pause)

    if not all_data:
        log(f"⚠ No {label} data fetched")
        return gpd.GeoDataFrame()

    # Merge
    log(f"Merging {label} from all quadrants...")
    merged = pd.concat(all_data, ignore_index=True)

    # Deduplicate
    before = len(merged)
    merged = merged.drop_duplicates(subset=["geometry"])
    dupes = before - len(merged)
    if dupes > 0:
        log(f"  Removed {dupes:,} duplicates")

    # Filter by area if requested
    if min_area is not None and len(merged) > 0:
        merged_proj = merged.to_crs(epsg=32616)
        merged["area_m2"] = merged_proj.geometry.area
        before = len(merged)
        merged = merged[merged["area_m2"] >= min_area].copy()
        log(f"  Filtered out {before - len(merged):,} features < {min_area} m2")

    log(f"✓ Total {label}: {len(merged):,} features")
    return merged


# ============================================================
# BUILDINGS
# ============================================================

def fetch_buildings(config):
    """Fetch building footprints via quadrant Overpass queries."""

    def query_builder(s, w, n, e):
        return f"""
        [out:json][timeout:180];
        (
          way["building"]({s},{w},{n},{e});
        );
        out body;
        >;
        out skel qt;
        """

    buildings = fetch_by_quadrants(
        config, query_builder, overpass_to_polygons,
        label="buildings", rows=3, cols=3, pause=5
    )

    if len(buildings) == 0:
        return buildings

    # Compute area
    log("Computing building areas...")
    buildings_proj = buildings.to_crs(epsg=32616)
    buildings["area_m2"] = buildings_proj.geometry.area
    log(f"  Total footprint: {buildings['area_m2'].sum() / 1e6:.1f} km2")

    # Extract height
    if "building:levels" in buildings.columns:
        buildings["levels"] = buildings["building:levels"].apply(safe_int)
    else:
        buildings["levels"] = np.nan

    if "height" in buildings.columns:
        buildings["height_m"] = buildings["height"].apply(safe_float)
    else:
        buildings["height_m"] = np.nan

    # Estimate height from levels
    mask = buildings["height_m"].isna() & buildings["levels"].notna()
    buildings.loc[mask, "height_m"] = buildings.loc[mask, "levels"] * 3.5

    heights = buildings["height_m"].dropna()
    if len(heights) > 0:
        log(f"  Avg height: {heights.mean():.1f} m ({len(heights):,} with data)")

    # Keep only needed columns
    keep = ["geometry", "area_m2", "height_m", "levels"]
    buildings = buildings[[c for c in keep if c in buildings.columns]].copy()

    return buildings


# ============================================================
# PARKS
# ============================================================

def fetch_parks(config):
    """Fetch parks and green spaces via quadrant Overpass queries."""

    def query_builder(s, w, n, e):
        return f"""
        [out:json][timeout:120];
        (
          way["leisure"="park"]({s},{w},{n},{e});
          way["leisure"="garden"]({s},{w},{n},{e});
          way["leisure"="nature_reserve"]({s},{w},{n},{e});
          way["leisure"="playground"]({s},{w},{n},{e});
          way["landuse"="grass"]({s},{w},{n},{e});
          way["landuse"="forest"]({s},{w},{n},{e});
          way["landuse"="recreation_ground"]({s},{w},{n},{e});
          way["landuse"="meadow"]({s},{w},{n},{e});
          way["natural"="wood"]({s},{w},{n},{e});
          way["natural"="grassland"]({s},{w},{n},{e});
          relation["leisure"="park"]({s},{w},{n},{e});
        );
        out body;
        >;
        out skel qt;
        """

    parks = fetch_by_quadrants(
        config, query_builder, overpass_to_polygons,
        label="parks", rows=3, cols=3, min_area=500, pause=3
    )

    if len(parks) > 0:
        if "area_m2" not in parks.columns:
            parks_proj = parks.to_crs(epsg=32616)
            parks["area_m2"] = parks_proj.geometry.area
        log(f"  Total park area: {parks['area_m2'].sum() / 1e6:.1f} km2")
        parks = parks[["geometry", "area_m2"]].copy()

    return parks


# ============================================================
# WATER
# ============================================================

def fetch_water(config):
    """Fetch water bodies via quadrant Overpass queries."""

    def query_builder(s, w, n, e):
        return f"""
        [out:json][timeout:120];
        (
          way["natural"="water"]({s},{w},{n},{e});
          way["water"]({s},{w},{n},{e});
          way["waterway"="river"]({s},{w},{n},{e});
          way["waterway"="stream"]({s},{w},{n},{e});
          way["waterway"="canal"]({s},{w},{n},{e});
          way["natural"="wetland"]({s},{w},{n},{e});
          relation["natural"="water"]({s},{w},{n},{e});
        );
        out body;
        >;
        out skel qt;
        """

    water = fetch_by_quadrants(
        config, query_builder, overpass_to_lines_and_polygons,
        label="water", rows=3, cols=3, pause=3
    )

    if len(water) > 0:
        water = water[["geometry"]].copy()

    return water


# ============================================================
# ROADS
# ============================================================

def fetch_roads(config):
    """Fetch road network via quadrant Overpass queries."""

    def query_builder(s, w, n, e):
        return f"""
        [out:json][timeout:120];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service)$"]({s},{w},{n},{e});
        );
        out body;
        >;
        out skel qt;
        """

    roads = fetch_by_quadrants(
        config, query_builder, overpass_to_lines_and_polygons,
        label="roads", rows=3, cols=3, pause=3
    )

    if len(roads) > 0:
        # Compute road lengths
        roads_proj = roads.to_crs(epsg=32616)
        roads["length"] = roads_proj.geometry.length

        # Extract highway type
        if "highway" not in roads.columns:
            roads["highway"] = "unknown"

        log(f"  Total road length: {roads['length'].sum() / 1000:.0f} km")
        roads = roads[["geometry", "highway", "length"]].copy()

    return roads


# ============================================================
# UTILITIES
# ============================================================

def safe_int(val):
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return np.nan

def safe_float(val):
    try:
        if isinstance(val, str):
            val = val.replace("m", "").replace("ft", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# SAVE
# ============================================================

def save_geodata(gdf, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if isinstance(gdf, gpd.GeoDataFrame) and len(gdf) > 0:
        log(f"Saving {filepath}...")
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        for col in gdf.columns:
            if col != "geometry":
                try:
                    gdf[col].to_json()
                except (TypeError, ValueError):
                    gdf[col] = gdf[col].astype(str)

        gdf.to_file(filepath, driver="GeoJSON")
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        log(f"✓ Saved {filepath} ({len(gdf):,} features, {file_size_mb:.1f} MB)")
    else:
        log(f"⚠ No data to save for {filepath}")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_urban_features(buildings, parks, water, roads, config,
                             output_path="output/chicago_urban_features.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log("Generating visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    bbox = config["bbox"]
    ax.set_xlim(bbox["west"], bbox["east"])
    ax.set_ylim(bbox["south"], bbox["north"])

    if isinstance(water, gpd.GeoDataFrame) and len(water) > 0:
        log("  Plotting water...")
        water.plot(ax=ax, color="#4A90D9", alpha=0.6, label="Water")

    if isinstance(parks, gpd.GeoDataFrame) and len(parks) > 0:
        log("  Plotting parks...")
        parks.plot(ax=ax, color="#2ECC71", alpha=0.5, label="Parks & Green Space")

    if isinstance(buildings, gpd.GeoDataFrame) and len(buildings) > 0:
        log("  Plotting buildings (may take a moment with 1M+ features)...")
        buildings.plot(ax=ax, color="#E74C3C", alpha=0.3, markersize=0.5, label="Buildings")

    if isinstance(roads, gpd.GeoDataFrame) and len(roads) > 0:
        log("  Plotting roads...")
        roads.plot(ax=ax, color="#7F8C8D", alpha=0.3, linewidth=0.3, label="Roads")

    for hood in config.get("focus_neighborhoods", []):
        lat, lon = hood["center"]
        ax.annotate(
            hood["name"], xy=(lon, lat), fontsize=8, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7)
        )

    ax.set_title(
        f"Urban Features — {config['city']['display_name']}\n"
        f"Buildings (red) | Parks (green) | Water (blue) | Roads (gray)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_facecolor("#F5F5DC")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    log(f"✓ Saved {output_path}")
    plt.show()


def print_summary(buildings, parks, water, roads):
    print()
    print("  ┌─────────────────────────────────────┐")
    print("  │     Urban Feature Summary            │")
    print("  ├─────────────────────────────────────┤")

    if isinstance(buildings, gpd.GeoDataFrame) and len(buildings) > 0:
        total_area = buildings["area_m2"].sum() / 1e6
        print(f"  │ Buildings: {len(buildings):>10,}              │")
        print(f"  │ Total footprint: {total_area:>8.1f} km2        │")
        if "height_m" in buildings.columns:
            avg_h = buildings["height_m"].dropna().mean()
            if not np.isnan(avg_h):
                print(f"  │ Avg height: {avg_h:>8.1f} m             │")

    if isinstance(parks, gpd.GeoDataFrame) and len(parks) > 0:
        park_area = parks["area_m2"].sum() / 1e6 if "area_m2" in parks.columns else 0
        print(f"  │ Parks: {len(parks):>14,}              │")
        print(f"  │ Total park area: {park_area:>8.1f} km2       │")

    if isinstance(water, gpd.GeoDataFrame) and len(water) > 0:
        print(f"  │ Water features: {len(water):>6,}             │")

    if isinstance(roads, gpd.GeoDataFrame) and len(roads) > 0:
        total_km = roads["length"].sum() / 1000
        print(f"  │ Road segments: {len(roads):>7,}             │")
        print(f"  │ Total road length: {total_km:>6.0f} km       │")

    print("  └─────────────────────────────────────┘")


# ============================================================
# MAIN
# ============================================================

def main():
    overall_start = time.time()

    print("=" * 60)
    print("  TomorrowLand Heat — OpenStreetMap Urban Data Pipeline")
    print("  v4 — All quadrant-based Overpass fetching")
    print("=" * 60)
    print()

    config = load_config()
    log(f"City: {config['city']['display_name']}")
    print()

    # ---- Buildings ----
    print("=" * 40)
    print("[1/4] Buildings (3x3 quadrants)")
    print("=" * 40)
    try:
        buildings = fetch_buildings(config)
    except Exception as e:
        log(f"⚠ Building fetch failed: {e}")
        buildings = gpd.GeoDataFrame()

    # ---- Parks ----
    print()
    print("=" * 40)
    print("[2/4] Parks & Green Spaces (3x3 quadrants)")
    print("=" * 40)
    try:
        parks = fetch_parks(config)
    except Exception as e:
        log(f"⚠ Parks fetch failed: {e}")
        parks = gpd.GeoDataFrame()

    # ---- Water ----
    print()
    print("=" * 40)
    print("[3/4] Water Bodies (3x3 quadrants)")
    print("=" * 40)
    try:
        water = fetch_water(config)
    except Exception as e:
        log(f"⚠ Water fetch failed: {e}")
        water = gpd.GeoDataFrame()

    # ---- Roads ----
    print()
    print("=" * 40)
    print("[4/4] Road Network (3x3 quadrants)")
    print("=" * 40)
    try:
        roads = fetch_roads(config)
    except Exception as e:
        log(f"⚠ Roads fetch failed: {e}")
        roads = gpd.GeoDataFrame()

    # ---- Summary ----
    print_summary(buildings, parks, water, roads)

    # ---- Save ----
    print()
    log("Saving all data...")
    save_geodata(buildings, "data/osm/chicago_buildings.geojson")
    save_geodata(parks, "data/osm/chicago_parks.geojson")
    save_geodata(water, "data/osm/chicago_water.geojson")
    save_geodata(roads, "data/osm/chicago_roads.geojson")

    # ---- Visualize ----
    print()
    visualize_urban_features(buildings, parks, water, roads, config)

    # ---- Done ----
    total_elapsed = int(time.time() - overall_start)
    mins, secs = divmod(total_elapsed, 60)

    print()
    print("=" * 60)
    print(f"  Done! Total time: {mins}m {secs}s")
    print()
    print("  Saved to data/osm/:")
    print("  - chicago_buildings.geojson")
    print("  - chicago_parks.geojson")
    print("  - chicago_water.geojson")
    print("  - chicago_roads.geojson")
    print()
    print("  Next: Re-run process_grid.py then train_heat_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()