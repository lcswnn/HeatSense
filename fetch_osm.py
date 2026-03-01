"""
TomorrowLand Heat — Step 3: Fetch OpenStreetMap Urban Data
==========================================================
This script pulls building footprints, road networks, parks, and water bodies
from OpenStreetMap for Chicago using the OSMnx library.

v3 — Splits Chicago into quadrants for building fetches to avoid timeout.

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
import pandas as pd
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

import requests
import json
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
    """
    Split a bounding box into a grid of smaller bounding boxes.
    This prevents Overpass API timeouts on large areas.
    """
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
# OVERPASS API (Direct queries — more reliable than OSMnx for large areas)
# ============================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def overpass_query(query, max_retries=3, timeout=180):
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
                log(f"  Gateway timeout — query too large or server busy")
                time.sleep(15)
            else:
                log(f"  HTTP {response.status_code}")
                time.sleep(10)
        except requests.exceptions.Timeout:
            log(f"  Request timeout on attempt {attempt + 1}")
            time.sleep(15)
        except requests.exceptions.ConnectionError:
            log(f"  Connection error on attempt {attempt + 1}")
            time.sleep(15)

    log("  ⚠ Overpass API failed after all retries")
    return None


def overpass_to_geodataframe(result, geometry_types=None):
    """
    Convert Overpass JSON result to a GeoDataFrame.
    Handles nodes, ways, and relations.
    """
    from shapely.geometry import Point, Polygon, LineString

    if result is None or "elements" not in result:
        return gpd.GeoDataFrame()

    elements = result["elements"]

    # Build node lookup
    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    # Build geometries from ways
    features = []
    for el in elements:
        if el["type"] == "way" and "nodes" in el:
            coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
            if len(coords) < 3:
                continue
            try:
                # Close the polygon if needed
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

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    return gdf


# ============================================================
# BUILDING DATA — QUADRANT APPROACH
# ============================================================

def fetch_buildings_quadrant(bbox_quad, quad_name):
    """Fetch buildings for a single quadrant via direct Overpass query."""
    s, w, n, e = bbox_quad["south"], bbox_quad["west"], bbox_quad["north"], bbox_quad["east"]

    query = f"""
    [out:json][timeout:180];
    (
      way["building"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """

    log(f"Fetching {quad_name}...")
    result = overpass_query(query, timeout=200)

    if result is None:
        log(f"⚠ {quad_name} failed")
        return gpd.GeoDataFrame()

    gdf = overpass_to_geodataframe(result)
    log(f"  {quad_name}: {len(gdf):,} buildings")
    return gdf


def fetch_buildings_by_quadrants(config, rows=3, cols=3):
    """
    Fetch buildings by splitting Chicago into a grid of smaller queries.
    Each quadrant is fetched separately and then merged.
    """
    quadrants = split_bbox_into_grid(config["bbox"], rows=rows, cols=cols)
    total = len(quadrants)

    log(f"Splitting Chicago into {total} quadrants ({rows}x{cols})")
    log(f"Each quadrant is ~1/{total}th of the city")

    all_buildings = []

    for i, quad in enumerate(quadrants):
        quad_name = f"quadrant {i+1}/{total}"

        monitor = ProgressMonitor(quad_name, interval=30)
        monitor.start()
        try:
            gdf = fetch_buildings_quadrant(quad, quad_name)
            if len(gdf) > 0:
                all_buildings.append(gdf)
        except Exception as e:
            log(f"⚠ Error on {quad_name}: {e}")
        finally:
            monitor.stop()

        # Brief pause between queries to be polite to the API
        if i < total - 1:
            log("  Pausing 5s before next quadrant...")
            time.sleep(5)

    if not all_buildings:
        log("⚠ No buildings fetched from any quadrant")
        return gpd.GeoDataFrame()

    # Merge all quadrants
    log("Merging all quadrants...")
    buildings = pd.concat(all_buildings, ignore_index=True)

    # Remove duplicates (buildings on quadrant borders may appear twice)
    before = len(buildings)
    buildings = buildings.drop_duplicates(subset=["geometry"])
    dupes = before - len(buildings)
    if dupes > 0:
        log(f"  Removed {dupes:,} duplicate buildings on quadrant borders")

    # Compute area
    log("Computing building areas...")
    buildings_projected = buildings.to_crs(epsg=32616)
    buildings["area_m2"] = buildings_projected.geometry.area

    # Extract height from tags
    if "building:levels" in buildings.columns:
        buildings["levels"] = buildings["building:levels"].apply(safe_int)
    else:
        buildings["levels"] = np.nan

    if "height" in buildings.columns:
        buildings["height_m"] = buildings["height"].apply(safe_float)
    else:
        buildings["height_m"] = np.nan

    # Estimate height from levels where height is missing
    mask = buildings["height_m"].isna() & buildings["levels"].notna()
    buildings.loc[mask, "height_m"] = buildings.loc[mask, "levels"] * 3.5

    # Keep only needed columns
    keep_cols = ["geometry", "area_m2", "height_m"]
    if "levels" in buildings.columns:
        keep_cols.append("levels")
    buildings = buildings[[c for c in keep_cols if c in buildings.columns]].copy()

    log(f"✓ Total: {len(buildings):,} buildings fetched")
    log(f"  Total footprint: {buildings['area_m2'].sum() / 1e6:.1f} km2")
    heights = buildings["height_m"].dropna()
    if len(heights) > 0:
        log(f"  Avg height: {heights.mean():.1f} m ({len(heights):,} buildings with height data)")

    return buildings


# ============================================================
# PARKS — OSMnx with fallback
# ============================================================

def fetch_parks(config):
    """Fetch parks, gardens, and green spaces."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        monitor = ProgressMonitor("parks and green spaces", interval=20)
        monitor.start()
        try:
            ox.settings.max_query_area_size = 50_000_000_000_000
            log("Fetching parks via OSMnx...")
            parks = ox.features_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                tags={
                    "leisure": ["park", "garden", "nature_reserve", "playground"],
                    "landuse": ["grass", "forest", "recreation_ground", "meadow"],
                    "natural": ["wood", "grassland"],
                }
            )
            log(f"Raw: {len(parks):,} features")
            parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
            parks = parks[["geometry"]].copy()

            parks_projected = parks.to_crs(epsg=32616)
            parks["area_m2"] = parks_projected.geometry.area
            parks = parks[parks["area_m2"] >= 500].copy()

            log(f"✓ {len(parks):,} parks ({parks['area_m2'].sum() / 1e6:.1f} km2)")
            return parks
        except Exception as e:
            log(f"⚠ OSMnx failed: {e}")
        finally:
            monitor.stop()

    # Fallback: direct Overpass
    log("Trying direct Overpass for parks...")
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    query = f"""
    [out:json][timeout:120];
    (
      way["leisure"="park"]({s},{w},{n},{e});
      way["landuse"="grass"]({s},{w},{n},{e});
      way["landuse"="forest"]({s},{w},{n},{e});
      way["leisure"="garden"]({s},{w},{n},{e});
      relation["leisure"="park"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """
    result = overpass_query(query)
    gdf = overpass_to_geodataframe(result)
    if len(gdf) > 0:
        gdf_proj = gdf.to_crs(epsg=32616)
        gdf["area_m2"] = gdf_proj.geometry.area
        gdf = gdf[gdf["area_m2"] >= 500][["geometry", "area_m2"]].copy()
    log(f"✓ {len(gdf):,} parks from Overpass")
    return gdf


# ============================================================
# WATER BODIES
# ============================================================

def fetch_water(config):
    """Fetch water bodies."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        monitor = ProgressMonitor("water bodies", interval=20)
        monitor.start()
        try:
            ox.settings.max_query_area_size = 50_000_000_000_000
            log("Fetching water via OSMnx...")
            water = ox.features_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                tags={
                    "natural": ["water", "wetland"],
                    "water": True,
                    "waterway": ["river", "stream", "canal"],
                }
            )
            log(f"Raw: {len(water):,} features")
            water = water[water.geometry.type.isin(
                ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
            )].copy()
            water = water[["geometry"]].copy()
            log(f"✓ {len(water):,} water features")
            return water
        except Exception as e:
            log(f"⚠ OSMnx failed: {e}")
        finally:
            monitor.stop()

    # Fallback
    log("Trying direct Overpass for water...")
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    query = f"""
    [out:json][timeout:120];
    (
      way["natural"="water"]({s},{w},{n},{e});
      way["waterway"="river"]({s},{w},{n},{e});
      way["waterway"="canal"]({s},{w},{n},{e});
      relation["natural"="water"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """
    result = overpass_query(query)
    gdf = overpass_to_geodataframe(result)
    gdf = gdf[["geometry"]].copy() if len(gdf) > 0 else gdf
    log(f"✓ {len(gdf):,} water features from Overpass")
    return gdf


# ============================================================
# ROADS
# ============================================================

def fetch_roads(config):
    """Fetch road network."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        monitor = ProgressMonitor("road network", interval=30)
        monitor.start()
        try:
            ox.settings.max_query_area_size = 50_000_000_000_000
            log("Fetching roads via OSMnx...")
            G = ox.graph_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                network_type="drive"
            )
            log(f"Graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
            roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
            roads = roads[["geometry", "highway", "length"]].copy()
            log(f"✓ {len(roads):,} road segments ({roads['length'].sum()/1000:.0f} km)")
            return roads
        except Exception as e:
            log(f"⚠ OSMnx failed: {e}")
        finally:
            monitor.stop()

    log("Skipping roads (will use impervious surface as proxy)")
    return gpd.GeoDataFrame()


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
# SAVE DATA
# ============================================================

def save_geodata(gdf, filepath):
    """Save a GeoDataFrame to GeoJSON."""
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
        log("  Plotting buildings (may take a moment)...")
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
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
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
    print("  v3 — Quadrant-based building fetch")
    print("=" * 60)
    print()

    config = load_config()
    log(f"City: {config['city']['display_name']}")

    if HAS_OSMNX:
        log("OSMnx available (used for parks, water, roads)")
        ox.settings.use_cache = True
        ox.settings.cache_folder = "data/osm_cache"
        ox.settings.timeout = 300
    print()

    # ---- Buildings (quadrant approach — bypasses OSMnx) ----
    print("=" * 40)
    print("[1/4] Buildings (3x3 quadrant fetch)")
    print("=" * 40)
    try:
        buildings = fetch_buildings_by_quadrants(config, rows=3, cols=3)
    except Exception as e:
        log(f"⚠ Building fetch failed: {e}")
        buildings = gpd.GeoDataFrame()

    # ---- Parks ----
    print()
    print("=" * 40)
    print("[2/4] Parks & Green Spaces")
    print("=" * 40)
    try:
        parks = fetch_parks(config)
    except Exception as e:
        log(f"⚠ Parks fetch error: {e}")
        parks = gpd.GeoDataFrame()

    # ---- Water ----
    print()
    print("=" * 40)
    print("[3/4] Water Bodies")
    print("=" * 40)
    try:
        water = fetch_water(config)
    except Exception as e:
        log(f"⚠ Water fetch error: {e}")
        water = gpd.GeoDataFrame()

    # ---- Roads ----
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
    print("  Next step: Re-run process_grid.py to rebuild with OSM features")
    print("=" * 60)


if __name__ == "__main__":
    main()