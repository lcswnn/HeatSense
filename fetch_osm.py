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
            response = requests.get(
                OVERPASS_URL,
                params={"data": query},
                timeout=120
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Overpass returned status {response.status_code}")
                time.sleep(10)
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt + 1}, retrying...")
            time.sleep(10)

    print("  ⚠ Overpass API failed after retries")
    return None


# ============================================================
# BUILDING DATA
# ============================================================

def fetch_buildings_osmnx(config):
    """Fetch building footprints using OSMnx."""
    bbox = config["bbox"]
    print("  Fetching buildings via OSMnx (this may take a few minutes)...")

    # OSMnx expects (north, south, east, west)
    buildings = ox.features_from_bbox(
        bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
        tags={"building": True}
    )

    # Keep only polygon geometries (skip nodes)
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # Extract useful attributes
    cols_to_keep = ["geometry"]
    if "building:levels" in buildings.columns:
        buildings["levels"] = buildings["building:levels"].apply(safe_int)
        cols_to_keep.append("levels")
    if "height" in buildings.columns:
        buildings["height_m"] = buildings["height"].apply(safe_float)
        cols_to_keep.append("height_m")
    if "building" in buildings.columns:
        buildings["building_type"] = buildings["building"]
        cols_to_keep.append("building_type")

    buildings = buildings[cols_to_keep].copy()

    # Compute area
    buildings_projected = buildings.to_crs(epsg=32616)  # UTM 16N for Chicago
    buildings["area_m2"] = buildings_projected.geometry.area

    # Estimate height from levels if actual height not available
    if "height_m" not in buildings.columns:
        buildings["height_m"] = np.nan
    if "levels" in buildings.columns:
        # Average floor height ~3.5m
        mask = buildings["height_m"].isna() & buildings["levels"].notna()
        buildings.loc[mask, "height_m"] = buildings.loc[mask, "levels"] * 3.5

    print(f"  ✓ Fetched {len(buildings):,} building footprints")
    return buildings


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
    print("  Fetching buildings via Overpass API (this may take several minutes)...")
    result = overpass_query(query)

    if result is None:
        print("  ⚠ Could not fetch buildings")
        return gpd.GeoDataFrame()

    # For large datasets, Overpass returns nodes + ways separately
    # We'll use a simplified approach: just count buildings per area in process_grid
    print(f"  ✓ Received {len(result.get('elements', []))} elements from Overpass")
    return result


# ============================================================
# PARKS & GREEN SPACES
# ============================================================

def fetch_parks(config):
    """Fetch parks, gardens, and green spaces."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        print("  Fetching parks and green spaces via OSMnx...")
        try:
            parks = ox.features_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                tags={
                    "leisure": ["park", "garden", "nature_reserve", "playground"],
                    "landuse": ["grass", "forest", "recreation_ground", "meadow"],
                    "natural": ["wood", "grassland"],
                }
            )
            parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
            parks = parks[["geometry"]].copy()

            # Compute area
            parks_projected = parks.to_crs(epsg=32616)
            parks["area_m2"] = parks_projected.geometry.area

            # Filter out tiny polygons (< 500 m²)
            parks = parks[parks["area_m2"] >= 500].copy()

            print(f"  ✓ Fetched {len(parks):,} parks and green spaces")
            return parks

        except Exception as e:
            print(f"  ⚠ OSMnx parks fetch failed: {e}")

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
    print("  Fetching parks via Overpass API...")
    result = overpass_query(query)
    print(f"  ✓ Received park data from Overpass")
    return result


# ============================================================
# WATER BODIES
# ============================================================

def fetch_water(config):
    """Fetch water bodies (lakes, rivers, ponds)."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        print("  Fetching water bodies via OSMnx...")
        try:
            water = ox.features_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                tags={
                    "natural": ["water", "wetland"],
                    "water": True,
                    "waterway": ["river", "stream", "canal"],
                }
            )
            water = water[water.geometry.type.isin(
                ["Polygon", "MultiPolygon", "LineString", "MultiLineString"]
            )].copy()
            water = water[["geometry"]].copy()
            print(f"  ✓ Fetched {len(water):,} water features")
            return water

        except Exception as e:
            print(f"  ⚠ OSMnx water fetch failed: {e}")

    print("  Skipping water bodies (will use NDVI as proxy — water has negative NDVI)")
    return gpd.GeoDataFrame()


# ============================================================
# ROAD NETWORK
# ============================================================

def fetch_roads(config):
    """Fetch road network (drives heat from vehicles + impervious surface)."""
    bbox = config["bbox"]

    if HAS_OSMNX:
        print("  Fetching road network via OSMnx...")
        try:
            # Get drivable road network
            G = ox.graph_from_bbox(
                bbox=(bbox["north"], bbox["south"], bbox["east"], bbox["west"]),
                network_type="drive"
            )
            roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
            roads = roads[["geometry", "highway", "length"]].copy()
            print(f"  ✓ Fetched {len(roads):,} road segments")
            return roads

        except Exception as e:
            print(f"  ⚠ OSMnx road fetch failed: {e}")

    print("  Skipping roads (will use impervious surface % as proxy)")
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
        print(f"  ✓ Saved to {filepath} ({len(gdf):,} features)")
    else:
        print(f"  ⚠ No data to save for {filepath}")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_urban_features(buildings, parks, water, roads, config,
                             output_path="output/chicago_urban_features.png"):
    """Visualize all urban features on a single map."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    bbox = config["bbox"]
    ax.set_xlim(bbox["west"], bbox["east"])
    ax.set_ylim(bbox["south"], bbox["north"])

    # Plot in order: water (bottom) → parks → buildings → roads (top)
    if isinstance(water, gpd.GeoDataFrame) and len(water) > 0:
        water.plot(ax=ax, color="#4A90D9", alpha=0.6, label="Water")

    if isinstance(parks, gpd.GeoDataFrame) and len(parks) > 0:
        parks.plot(ax=ax, color="#2ECC71", alpha=0.5, label="Parks & Green Space")

    if isinstance(buildings, gpd.GeoDataFrame) and len(buildings) > 0:
        buildings.plot(ax=ax, color="#E74C3C", alpha=0.3, markersize=0.5, label="Buildings")

    if isinstance(roads, gpd.GeoDataFrame) and len(roads) > 0:
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
    print(f"  ✓ Urban features map saved to: {output_path}")
    plt.show()


def print_summary(buildings, parks, water, roads):
    """Print a summary of fetched data."""
    print()
    print("  ┌─────────────────────────────────────┐")
    print("  │     Urban Feature Summary            │")
    print("  ├─────────────────────────────────────┤")

    if isinstance(buildings, gpd.GeoDataFrame) and len(buildings) > 0:
        total_area = buildings["area_m2"].sum() / 1e6  # km²
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
    print("=" * 60)
    print("  TomorrowLand Heat — OpenStreetMap Urban Data Pipeline")
    print("=" * 60)
    print()

    config = load_config()
    print(f"City: {config['city']['display_name']}")
    print()

    if HAS_OSMNX:
        print("Using OSMnx for data fetching")
        # Configure OSMnx
        ox.settings.use_cache = True
        ox.settings.max_query_area_size = 50_000_000_000_000  # 50 trillion
        ox.settings.cache_folder = "data/osm_cache"
        ox.settings.timeout = 120
    else:
        print("Using Overpass API directly (install osmnx for better results)")
    print()

    # ---- Fetch all data ----
    print("[1/4] Buildings")
    try:
        buildings = fetch_buildings_osmnx(config) if HAS_OSMNX else gpd.GeoDataFrame()
    except Exception as e:
        print(f"  ⚠ Building fetch error: {e}")
        buildings = gpd.GeoDataFrame()

    print()
    print("[2/4] Parks & Green Spaces")
    try:
        parks = fetch_parks(config)
    except Exception as e:
        print(f"  ⚠ Parks fetch error: {e}")
        parks = gpd.GeoDataFrame()

    print()
    print("[3/4] Water Bodies")
    try:
        water = fetch_water(config)
    except Exception as e:
        print(f"  ⚠ Water fetch error: {e}")
        water = gpd.GeoDataFrame()

    print()
    print("[4/4] Road Network")
    try:
        roads = fetch_roads(config)
    except Exception as e:
        print(f"  ⚠ Roads fetch error: {e}")
        roads = gpd.GeoDataFrame()

    # ---- Summary ----
    print_summary(buildings, parks, water, roads)

    # ---- Save ----
    print()
    print("Saving data...")
    save_geodata(buildings, "data/osm/chicago_buildings.geojson")
    save_geodata(parks, "data/osm/chicago_parks.geojson")
    save_geodata(water, "data/osm/chicago_water.geojson")
    save_geodata(roads, "data/osm/chicago_roads.geojson")

    # ---- Visualize ----
    print()
    print("Creating visualization...")
    visualize_urban_features(buildings, parks, water, roads, config)

    print()
    print("=" * 60)
    print("  Done! Urban feature data collected.")
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