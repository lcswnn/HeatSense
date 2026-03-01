"""
TomorrowLand Heat — Step 3: Fetch OpenStreetMap Urban Data
==========================================================
This script pulls building footprints, road networks, parks, and water bodies
from OpenStreetMap for Chicago using the OSMnx library.

These features are critical for the heat model because:
  - Building density and height create "urban canyon" effects that trap heat
  - Road density correlates with impervious surface and vehicle heat output
  - Parks and water bodies are natural cooling zones
  - Distance to the nearest park/water is a strong predictor of local temperature

Prerequisites:
  - pip install -r requirements.txt
  - fetch_landsat.py and fetch_ndvi.py already run successfully

Usage:
  python fetch_osm.py

Output:
  - data/osm/chicago_buildings.geojson
  - data/osm/chicago_roads.geojson
  - data/osm/chicago_parks.geojson
  - data/osm/chicago_water.geojson
  - output/chicago_urban_features.png (visualization)
"""

import os
import yaml
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from datetime import datetime
import threading
import sys

# OSMnx for OpenStreetMap data
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    print("⚠ osmnx not installed. Install with: pip install osmnx")

# Alternative: direct Overpass API queries
import requests
import json
import time


# ============================================================
# PROGRESS LOGGING
# ============================================================

def timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    """Print a timestamped log message."""
    print(f"  [{timestamp()}] {msg}")
    sys.stdout.flush()  # Force immediate output


class ProgressMonitor:
    """
    Background thread that prints a heartbeat message every N seconds
    so you know the script isn't stuck.
    """
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


# ============================================================
# OVERPASS API (Fallback if OSMnx has issues)
# ============================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_query(query, max_retries=3):
    """Execute an Overpass API query with retry logic."""
    for attempt in range(max_retries):
        try:
            log(f"Overpass API request (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(
                OVERPASS_URL,
                params={"data": query},
                timeout=180
            )
            if response.status_code == 200:
                log(f"Overpass returned {len(response.content) / 1024 / 1024:.1f} MB")
                return response.json()
            elif response.status_code == 429:
                wait = 30 * (attempt + 1)
                log(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                log(f"Overpass returned status {response.status_code}")
                time.sleep(10)
        except requests.exceptions.Timeout:
            log(f"Timeout on attempt {attempt + 1}, retrying...")
            time.sleep(10)

    log("⚠ Overpass API failed after retries")
    return None


# ============================================================
# BUILDING DATA
# ============================================================

def fetch_buildings_osmnx(config):
    """Fetch building footprints using OSMnx."""
    bbox = config["bbox"]

    monitor = ProgressMonitor("building footprint download", interval=30)
    monitor.start()

    try:
        ox.settings.max_query_area_size = 50_000_000_000_000  # Force 50 trillion right before query
        log("Sending Overpass query for buildings...")
        log(f"Bounding box: W={bbox['west']}, S={bbox['south']}, E={bbox['east']}, N={bbox['north']}")

        # OSMnx expects (north, south, east, west)
        buildings = ox.features_from_bbox(
            bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
            tags={"building": True}
        )
        log(f"Raw response: {len(buildings):,} features received")

        # Keep only polygon geometries (skip nodes)
        log("Filtering to polygon geometries...")
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        log(f"After filtering: {len(buildings):,} building polygons")

        # Extract useful attributes
        log("Extracting building attributes...")
        cols_to_keep = ["geometry"]
        if "building:levels" in buildings.columns:
            buildings["levels"] = buildings["building:levels"].apply(safe_int)
            cols_to_keep.append("levels")
            levels_count = buildings["levels"].notna().sum()
            log(f"  Found {levels_count:,} buildings with level data")
        if "height" in buildings.columns:
            buildings["height_m"] = buildings["height"].apply(safe_float)
            cols_to_keep.append("height_m")
            height_count = buildings["height_m"].notna().sum()
            log(f"  Found {height_count:,} buildings with height data")
        if "building" in buildings.columns:
            buildings["building_type"] = buildings["building"]
            cols_to_keep.append("building_type")

        buildings = buildings[cols_to_keep].copy()

        # Compute area
        log("Computing building areas (projecting to UTM)...")
        buildings_projected = buildings.to_crs(epsg=32616)  # UTM 16N for Chicago
        buildings["area_m2"] = buildings_projected.geometry.area
        log(f"  Total building footprint: {buildings['area_m2'].sum() / 1e6:.1f} km2")

        # Estimate height from levels if actual height not available
        if "height_m" not in buildings.columns:
            buildings["height_m"] = np.nan
        if "levels" in buildings.columns:
            mask = buildings["height_m"].isna() & buildings["levels"].notna()
            buildings.loc[mask, "height_m"] = buildings.loc[mask, "levels"] * 3.5
            log(f"  Estimated heights for {mask.sum():,} buildings from level data")

        log(f"✓ Fetched {len(buildings):,} building footprints")
        return buildings

    finally:
        monitor.stop()


def fetch_buildings_overpass(config):
    """Fetch buildings via Overpass API (fallback)."""
    bbox = config["bbox"]
    query = f"""
    [out:json][timeout:120];
    (
      way["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      relation["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out body;
    >;
    out skel qt;
    """
    log("Fetching buildings via Overpass API (fallback)...")
    result = overpass_query(query)

    if result is None:
        log("⚠ Could not fetch buildings")
        return gpd.GeoDataFrame()

    log(f"✓ Received {len(result.get('elements', []))} elements from Overpass")
    return result


# ============================================================
# PARKS & GREEN SPACES
# ============================================================

def fetch_parks(config):
    """Fetch parks, gardens, and green spaces."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        monitor = ProgressMonitor("parks and green spaces download", interval=20)
        monitor.start()
        try:
            ox.settings.max_query_area_size = 50_000_000_000_000  # Force 50 trillion right before query
            log("Sending Overpass query for parks...")
            parks = ox.features_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                tags={
                    "leisure": ["park", "garden", "nature_reserve", "playground"],
                    "landuse": ["grass", "forest", "recreation_ground", "meadow"],
                    "natural": ["wood", "grassland"],
                }
            )
            log(f"Raw response: {len(parks):,} features received")

            parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
            parks = parks[["geometry"]].copy()
            log(f"After filtering to polygons: {len(parks):,}")

            # Compute area
            log("Computing park areas...")
            parks_projected = parks.to_crs(epsg=32616)
            parks["area_m2"] = parks_projected.geometry.area

            # Filter out tiny polygons (< 500 m2)
            before = len(parks)
            parks = parks[parks["area_m2"] >= 500].copy()
            log(f"Filtered out {before - len(parks):,} tiny polygons (< 500 m2)")

            log(f"✓ Fetched {len(parks):,} parks and green spaces")
            log(f"  Total park area: {parks['area_m2'].sum() / 1e6:.1f} km2")
            return parks

        except Exception as e:
            log(f"⚠ OSMnx parks fetch failed: {e}")
        finally:
            monitor.stop()

    # Fallback: Overpass
    query = f"""
    [out:json][timeout:90];
    (
      way["leisure"="park"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      way["landuse"="grass"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      way["landuse"="forest"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      relation["leisure"="park"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out body;
    >;
    out skel qt;
    """
    log("Fetching parks via Overpass API (fallback)...")
    result = overpass_query(query)
    log(f"✓ Received park data from Overpass")
    return result


# ============================================================
# WATER BODIES
# ============================================================

def fetch_water(config):
    """Fetch water bodies (lakes, rivers, ponds)."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        monitor = ProgressMonitor("water bodies download", interval=20)
        monitor.start()
        try:
            ox.settings.max_query_area_size = 50_000_000_000_000  # Force 50 trillion right before query
            log("Sending Overpass query for water bodies...")
            water = ox.features_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                tags={
                    "natural": ["water", "wetland"],
                    "water": True,
                    "waterway": ["river", "stream", "canal"],
                }
            )
            log(f"Raw response: {len(water):,} features received")

            water = water[water.geometry.type.isin(
                ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
            )].copy()
            water = water[["geometry"]].copy()
            log(f"✓ Fetched {len(water):,} water features")
            return water

        except Exception as e:
            log(f"⚠ OSMnx water fetch failed: {e}")
        finally:
            monitor.stop()

    log("Skipping water bodies (will use NDVI as proxy — water has negative NDVI)")
    return gpd.GeoDataFrame()


# ============================================================
# ROAD NETWORK
# ============================================================

def fetch_roads(config):
    """Fetch road network (drives heat from vehicles + impervious surface)."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        monitor = ProgressMonitor("road network download", interval=30)
        monitor.start()
        try:
            ox.settings.max_query_area_size = 50_000_000_000_000  # Force 50 trillion right before query
            log("Sending Overpass query for road network...")
            # Get drivable road network
            G = ox.graph_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                network_type="drive"
            )
            log(f"Graph received: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

            log("Converting graph to GeoDataFrame...")
            roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
            roads = roads[["geometry", "highway", "length"]].copy()
            log(f"✓ Fetched {len(roads):,} road segments")
            log(f"  Total road length: {roads['length'].sum() / 1000:.0f} km")
            return roads

        except Exception as e:
            log(f"⚠ OSMnx road fetch failed: {e}")
        finally:
            monitor.stop()

    log("Skipping roads (will use impervious surface % as proxy)")
    return gpd.GeoDataFrame()


# ============================================================
# UTILITIES
# ============================================================

def safe_int(val):
    """Safely convert a value to int."""
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return np.nan


def safe_float(val):
    """Safely convert a value to float, handling units like '10 m'."""
    try:
        if isinstance(val, str):
            val = val.replace("m", "").replace("ft", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# SAVE DATA
# ============================================================

def save_geodata(gdf, filepath):
    """Save a GeoDataFrame to GeoJSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if isinstance(gdf, gpd.GeoDataFrame) and len(gdf) > 0:
        log(f"Saving {filepath}...")

        # Ensure CRS is WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        # Drop any non-serializable columns
        for col in gdf.columns:
            if col != "geometry":
                try:
                    gdf[col].to_json()
                except (TypeError, ValueError):
                    gdf[col] = gdf[col].astype(str)

        gdf.to_file(filepath, driver="GeoJSON")
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        log(f"✓ Saved to {filepath} ({len(gdf):,} features, {file_size_mb:.1f} MB)")
    else:
        log(f"⚠ No data to save for {filepath}")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_urban_features(buildings, parks, water, roads, config,
                             output_path="output/chicago_urban_features.png"):
    """Visualize all urban features on a single map."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log("Generating urban features visualization...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    bbox = config["bbox"]
    ax.set_xlim(bbox["west"], bbox["east"])
    ax.set_ylim(bbox["south"], bbox["north"])

    # Plot in order: water (bottom) -> parks -> buildings -> roads (top)
    if isinstance(water, gpd.GeoDataFrame) and len(water) > 0:
        log("  Plotting water...")
        water.plot(ax=ax, color="#4A90D9", alpha=0.6, label="Water")

    if isinstance(parks, gpd.GeoDataFrame) and len(parks) > 0:
        log("  Plotting parks...")
        parks.plot(ax=ax, color="#2ECC71", alpha=0.5, label="Parks & Green Space")

    if isinstance(buildings, gpd.GeoDataFrame) and len(buildings) > 0:
        log("  Plotting buildings (this may take a moment with many features)...")
        buildings.plot(ax=ax, color="#E74C3C", alpha=0.3, markersize=0.5, label="Buildings")

    if isinstance(roads, gpd.GeoDataFrame) and len(roads) > 0:
        log("  Plotting roads...")
        roads.plot(ax=ax, color="#7F8C8D", alpha=0.3, linewidth=0.3, label="Roads")

    # Neighborhood labels
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
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_facecolor("#F5F5DC")  # Light beige background

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    log(f"✓ Urban features map saved to: {output_path}")
    plt.show()


def print_summary(buildings, parks, water, roads):
    """Print a summary of fetched data."""
    print()
    print("  ┌─────────────────────────────────────┐")
    print("  │     Urban Feature Summary            │")
    print("  ├─────────────────────────────────────┤")

    if isinstance(buildings, gpd.GeoDataFrame) and len(buildings) > 0:
        total_area = buildings["area_m2"].sum() / 1e6  # km2
        avg_height = buildings["height_m"].dropna().mean()
        print(f"  │ Buildings: {len(buildings):>10,}              │")
        print(f"  │ Total footprint: {total_area:>8.1f} km2        │")
        if not np.isnan(avg_height):
            print(f"  │ Avg height: {avg_height:>8.1f} m             │")

    if isinstance(parks, gpd.GeoDataFrame) and len(parks) > 0:
        park_area = parks["area_m2"].sum() / 1e6
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
    print("=" * 60)
    print()

    config = load_config()
    log(f"City: {config['city']['display_name']}")
    print()

    if HAS_OSMNX:
        log("Using OSMnx for data fetching")
        # Configure OSMnx
        ox.settings.use_cache = True
        ox.settings.max_query_area_size = 5_000_000_000  # 5 billion
        ox.settings.cache_folder = "data/osm_cache"
        ox.settings.timeout = 300  # 5 minute timeout per request
    else:
        log("Using Overpass API directly (install osmnx for better results)")
    print()

    # ---- Fetch all data ----
    print("=" * 40)
    print("[1/4] Buildings")
    print("=" * 40)
    try:
        buildings = fetch_buildings_osmnx(config) if HAS_OSMNX else gpd.GeoDataFrame()
    except Exception as e:
        log(f"⚠ Building fetch error: {e}")
        log("Trying Overpass API fallback...")
        try:
            buildings = fetch_buildings_overpass(config)
        except Exception as e2:
            log(f"⚠ Fallback also failed: {e2}")
            buildings = gpd.GeoDataFrame()

    print()
    print("=" * 40)
    print("[2/4] Parks & Green Spaces")
    print("=" * 40)
    try:
        parks = fetch_parks(config)
    except Exception as e:
        log(f"⚠ Parks fetch error: {e}")
        parks = gpd.GeoDataFrame()

    print()
    print("=" * 40)
    print("[3/4] Water Bodies")
    print("=" * 40)
    try:
        water = fetch_water(config)
    except Exception as e:
        log(f"⚠ Water fetch error: {e}")
        water = gpd.GeoDataFrame()

    print()
    print("=" * 40)
    print("[4/4] Road Network")
    print("=" * 40)
    try:
        roads = fetch_roads(config)
    except Exception as e:
        log(f"⚠ Roads fetch error: {e}")
        roads = gpd.GeoDataFrame()

    # ---- Summary ----
    print_summary(buildings, parks, water, roads)

    # ---- Save ----
    print()
    log("Saving data...")
    save_geodata(buildings, "data/osm/chicago_buildings.geojson")
    save_geodata(parks, "data/osm/chicago_parks.geojson")
    save_geodata(water, "data/osm/chicago_water.geojson")
    save_geodata(roads, "data/osm/chicago_roads.geojson")

    # ---- Visualize ----
    print()
    visualize_urban_features(buildings, parks, water, roads, config)

    # ---- Final timing ----
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
    print("  Next step: Run process_grid.py to build the analysis grid")
    print("=" * 60)


if __name__ == "__main__":
    main()