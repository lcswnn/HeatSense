"""
HeatSense — Step 4: Build the Analysis Grid
====================================================
This is the BACKBONE of the project. It takes every data source we've
collected (thermal, NDVI, buildings, parks, water, roads, impervious surface)
and combines them into a unified 100m x 100m grid covering Chicago.

Each grid cell becomes one row in the dataset with all features needed
for the ML model. This is what transforms raw satellite and map data
into a structured, trainable dataset.

Grid cell features:
  - lat, lon (center point)
  - mean_lst_f (mean summer surface temperature, Fahrenheit)
  - mean_lst_c (mean summer surface temperature, Celsius)
  - ndvi (vegetation index, 0-1)
  - impervious_pct (impervious surface percentage, 0-100)
  - building_count (number of buildings in cell)
  - building_density (building footprint area / cell area)
  - avg_building_height_m (average building height)
  - road_density_km (total road length in cell, km)
  - distance_to_park_m (distance to nearest park centroid)
  - distance_to_water_m (distance to nearest water body)
  - park_area_pct (percentage of cell that is park/green space)
  - land_cover_class (dominant land cover type)

Prerequisites:
  - fetch_landsat.py completed (thermal data in Earth Engine)
  - fetch_ndvi.py completed (NDVI data in Earth Engine)
  - fetch_osm.py completed (GeoJSON files in data/osm/)

Usage:
  python process_grid.py

Output:
  - data/grid/chicago_grid.geojson (full grid with all features)
  - data/grid/chicago_grid.csv (tabular version for ML training)
  - output/chicago_grid_preview.png (visualization)
"""

import ee
import geemap
import os
import yaml
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import box, Point
from shapely.ops import nearest_points
from tqdm import tqdm
import tempfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIGURATION
# ============================================================

def load_config(config_path=None, city_slug=None):
    """Load city configuration. Use --city flag or direct path."""
    if config_path is None and city_slug is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        config_path = os.path.join(project_dir, "config", f"{city_slug}.yaml")
    elif config_path is None:
        config_path = "config/chicago.yaml"
    if not os.path.exists(config_path):
        print(f"  Error: Config not found at {config_path}")
        raise SystemExit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def initialize_ee():
    try:
        ee.Initialize(project='quick-composite-462916-q9')  # Replace with your GEE project ID
        print("Earth Engine initialized")
    except:
        ee.Authenticate()
        ee.Initialize(project='quick-composite-462916-q9')


# ============================================================
# STEP 1: CREATE THE GRID
# ============================================================

def create_grid(config):
    """
    Create a grid of square cells covering the study area.
    Each cell is 100m x 100m (configurable in chicago.yaml).
    """
    bbox = config["bbox"]
    cell_size = config["grid"]["cell_size_meters"]
    target_crs = config["grid"]["crs"]  # UTM for meter-based calculations

    print(f"  Cell size: {cell_size}m x {cell_size}m")

    # Create bounding box in WGS84, then project to UTM for meter-based grid
    bbox_geom = box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
    bbox_utm = bbox_gdf.to_crs(target_crs)

    # Get UTM bounds
    minx, miny, maxx, maxy = bbox_utm.total_bounds

    # Generate grid cells
    cells = []
    cell_id = 0
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + cell_size, y + cell_size)
            center = cell.centroid
            cells.append({
                "cell_id": cell_id,
                "geometry": cell,
                "center_x_utm": center.x,
                "center_y_utm": center.y,
            })
            cell_id += 1
            y += cell_size
        x += cell_size

    # Create GeoDataFrame in UTM
    grid = gpd.GeoDataFrame(cells, crs=target_crs)

    # Add lat/lon of center points (WGS84)
    grid_wgs84 = grid.to_crs("EPSG:4326")
    grid["lon"] = grid_wgs84.geometry.centroid.x
    grid["lat"] = grid_wgs84.geometry.centroid.y

    print(f"  Created {len(grid):,} grid cells")
    print(f"  Grid dimensions: ~{int((maxx - minx) / cell_size)} x {int((maxy - miny) / cell_size)}")

    return grid


# ============================================================
# STEP 2: EXTRACT SATELLITE DATA PER GRID CELL
# ============================================================

def fetch_satellite_rasters(config):
    """
    Fetch mean LST and NDVI rasters from Earth Engine and download as numpy arrays.
    Returns arrays + bounds for spatial alignment with the grid.
    """
    from fetch_landsat import fetch_landsat_collection, compute_mean_lst, get_study_area
    from fetch_ndvi import fetch_ndvi_collection, compute_mean_ndvi, fetch_impervious_surface

    study_area = get_study_area(config)

    # Fetch thermal data
    print("  Fetching mean summer LST from Earth Engine...")
    lst_collection = fetch_landsat_collection(config, study_area)
    mean_lst_f, mean_lst_c = compute_mean_lst(lst_collection, study_area)

    # Fetch NDVI
    print("  Fetching mean summer NDVI from Earth Engine...")
    ndvi_collection = fetch_ndvi_collection(config, study_area)
    mean_ndvi = compute_mean_ndvi(ndvi_collection, study_area)

    # Fetch impervious surface
    print("  Fetching impervious surface data...")
    imperv = fetch_impervious_surface(study_area)

    # Download all as numpy arrays at grid resolution
    scale = config["grid"]["cell_size_meters"]  # Match grid cell size
    rasters = {}

    for name, image in [("lst_f", mean_lst_f), ("lst_c", mean_lst_c),
                         ("ndvi", mean_ndvi), ("impervious", imperv)]:
        print(f"  Downloading {name} raster...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"{name}.tif")
                geemap.ee_export_image(
                    image, filename=filepath, scale=scale,
                    region=study_area, file_per_band=False
                )
                import rasterio
                with rasterio.open(filepath) as src:
                    data = src.read(1)
                    transform = src.transform
                    raster_crs = src.crs
                    bounds = src.bounds

                rasters[name] = {
                    "data": data,
                    "transform": transform,
                    "crs": raster_crs,
                    "bounds": bounds
                }
                print(f"    {name}: shape={data.shape}, range=[{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]")
        except Exception as e:
            print(f"    ⚠ Failed to download {name}: {e}")
            rasters[name] = None

    return rasters


def sample_raster_at_points(raster_info, points_gdf):
    """
    Sample raster values at grid cell center points.
    Uses nearest-neighbor sampling.
    """
    if raster_info is None:
        return np.full(len(points_gdf), np.nan)

    import rasterio
    from rasterio.transform import rowcol

    data = raster_info["data"]
    transform = raster_info["transform"]

    # Ensure points are in same CRS as raster
    if points_gdf.crs != raster_info["crs"]:
        points_gdf = points_gdf.to_crs(raster_info["crs"])

    values = []
    for idx, row in points_gdf.iterrows():
        try:
            lon, lat = row.geometry.x, row.geometry.y
            r, c = rowcol(transform, lon, lat)
            r, c = int(r), int(c)
            if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                val = data[r, c]
                values.append(val if val != 0 else np.nan)
            else:
                values.append(np.nan)
        except Exception:
            values.append(np.nan)

    return np.array(values)


def assign_satellite_data(grid, rasters, config):
    """Assign satellite-derived values to each grid cell."""
    # Create center points GeoDataFrame for sampling
    centers = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(grid["lon"], grid["lat"])],
        crs="EPSG:4326"
    )

    print("  Sampling LST at grid centers...")
    grid["mean_lst_f"] = sample_raster_at_points(rasters.get("lst_f"), centers)
    grid["mean_lst_c"] = sample_raster_at_points(rasters.get("lst_c"), centers)

    print("  Sampling NDVI at grid centers...")
    grid["ndvi"] = sample_raster_at_points(rasters.get("ndvi"), centers)

    print("  Sampling impervious surface at grid centers...")
    grid["impervious_pct"] = sample_raster_at_points(rasters.get("impervious"), centers)

    # Quick stats
    valid_lst = grid["mean_lst_f"].dropna()
    valid_ndvi = grid["ndvi"].dropna()
    print(f"  LST: {len(valid_lst):,} cells with data, "
          f"range [{valid_lst.min():.1f}, {valid_lst.max():.1f}] F")
    print(f"  NDVI: {len(valid_ndvi):,} cells with data, "
          f"range [{valid_ndvi.min():.3f}, {valid_ndvi.max():.3f}]")

    return grid


# ============================================================
# STEP 3: ASSIGN OSM URBAN FEATURES
# ============================================================

def load_osm_data(city_slug="chicago"):
    """Load previously saved OSM GeoJSON files."""
    data = {}
    osm_files = {
        "buildings": f"data/{city_slug}/osm/{city_slug}_buildings.geojson",
        "parks": f"data/{city_slug}/osm/{city_slug}_parks.geojson",
        "water": f"data/{city_slug}/osm/{city_slug}_water.geojson",
        "roads": f"data/{city_slug}/osm/{city_slug}_roads.geojson",
    }

    for name, filepath in osm_files.items():
        if os.path.exists(filepath):
            try:
                gdf = gpd.read_file(filepath)
                print(f"  Loaded {name}: {len(gdf):,} features")
                data[name] = gdf
            except Exception as e:
                print(f"  ⚠ Failed to load {name}: {e}")
                data[name] = gpd.GeoDataFrame()
        else:
            print(f"  ⚠ {filepath} not found — skipping {name}")
            data[name] = gpd.GeoDataFrame()

    return data


def assign_building_features(grid, buildings, config):
    """
    For each grid cell, compute:
    - building_count: number of buildings intersecting the cell
    - building_density: fraction of cell area covered by buildings
    - avg_building_height_m: mean height of buildings in the cell
    """
    target_crs = config["grid"]["crs"]
    cell_area = config["grid"]["cell_size_meters"] ** 2  # m²

    if len(buildings) == 0:
        grid["building_count"] = 0
        grid["building_density"] = 0.0
        grid["avg_building_height_m"] = np.nan
        print("  ⚠ No building data — using defaults")
        return grid

    # Project buildings to UTM
    buildings_utm = buildings.to_crs(target_crs)

    # Build spatial index
    print("  Building spatial index for buildings...")
    bldg_sindex = buildings_utm.sindex

    counts = []
    densities = []
    heights = []

    print("  Computing building features per grid cell...")
    for idx, cell in tqdm(grid.iterrows(), total=len(grid), desc="  Buildings",
                          ncols=80, leave=False):
        # Find buildings that might intersect this cell
        possible_idx = list(bldg_sindex.intersection(cell.geometry.bounds))

        if len(possible_idx) == 0:
            counts.append(0)
            densities.append(0.0)
            heights.append(np.nan)
            continue

        # Get actual intersecting buildings
        candidates = buildings_utm.iloc[possible_idx]
        intersecting = candidates[candidates.intersects(cell.geometry)]

        counts.append(len(intersecting))

        if len(intersecting) > 0:
            # Building density: sum of intersection areas / cell area
            intersection_area = intersecting.geometry.intersection(cell.geometry).area.sum()
            densities.append(min(intersection_area / cell_area, 1.0))

            # Average height
            if "height_m" in intersecting.columns:
                h = intersecting["height_m"].dropna()
                heights.append(h.mean() if len(h) > 0 else np.nan)
            else:
                heights.append(np.nan)
        else:
            densities.append(0.0)
            heights.append(np.nan)

    grid["building_count"] = counts
    grid["building_density"] = densities
    grid["avg_building_height_m"] = heights

    print(f"  Building density range: [{grid['building_density'].min():.3f}, "
          f"{grid['building_density'].max():.3f}]")

    return grid


def assign_park_features(grid, parks, config):
    """
    For each grid cell, compute:
    - distance_to_park_m: distance to nearest park
    - park_area_pct: fraction of cell that is park/green space
    """
    target_crs = config["grid"]["crs"]
    cell_area = config["grid"]["cell_size_meters"] ** 2

    if len(parks) == 0:
        grid["distance_to_park_m"] = np.nan
        grid["park_area_pct"] = 0.0
        print("  ⚠ No park data — using defaults")
        return grid

    parks_utm = parks.to_crs(target_crs)

    # Create a union of all parks for distance calculation
    print("  Creating park union geometry for distance calculations...")
    park_union = parks_utm.geometry.union_all()

    park_sindex = parks_utm.sindex

    distances = []
    park_pcts = []

    print("  Computing park features per grid cell...")
    for idx, cell in tqdm(grid.iterrows(), total=len(grid), desc="  Parks",
                          ncols=80, leave=False):
        center = cell.geometry.centroid

        # Distance to nearest park
        dist = center.distance(park_union)
        distances.append(dist)

        # Park area percentage in cell
        possible_idx = list(park_sindex.intersection(cell.geometry.bounds))
        if len(possible_idx) > 0:
            candidates = parks_utm.iloc[possible_idx]
            intersecting = candidates[candidates.intersects(cell.geometry)]
            if len(intersecting) > 0:
                park_area = intersecting.geometry.intersection(cell.geometry).area.sum()
                park_pcts.append(min(park_area / cell_area, 1.0))
            else:
                park_pcts.append(0.0)
        else:
            park_pcts.append(0.0)

    grid["distance_to_park_m"] = distances
    grid["park_area_pct"] = park_pcts

    print(f"  Distance to park range: [{grid['distance_to_park_m'].min():.0f}, "
          f"{grid['distance_to_park_m'].max():.0f}] m")

    return grid


def assign_water_features(grid, water, config):
    """For each grid cell, compute distance to nearest water body."""
    target_crs = config["grid"]["crs"]

    if len(water) == 0:
        grid["distance_to_water_m"] = np.nan
        print("  ⚠ No water data — will be computed from NDVI as fallback")
        return grid

    water_utm = water.to_crs(target_crs)
    water_union = water_utm.geometry.union_all()

    print("  Computing distance to water per grid cell...")
    distances = []
    for idx, cell in tqdm(grid.iterrows(), total=len(grid), desc="  Water",
                          ncols=80, leave=False):
        center = cell.geometry.centroid
        dist = center.distance(water_union)
        distances.append(dist)

    grid["distance_to_water_m"] = distances

    print(f"  Distance to water range: [{grid['distance_to_water_m'].min():.0f}, "
          f"{grid['distance_to_water_m'].max():.0f}] m")

    return grid


def assign_road_features(grid, roads, config):
    """For each grid cell, compute total road length (road density)."""
    target_crs = config["grid"]["crs"]

    if len(roads) == 0:
        grid["road_density_km"] = 0.0
        print("  ⚠ No road data — using defaults")
        return grid

    roads_utm = roads.to_crs(target_crs)
    road_sindex = roads_utm.sindex

    road_lengths = []

    print("  Computing road density per grid cell...")
    for idx, cell in tqdm(grid.iterrows(), total=len(grid), desc="  Roads",
                          ncols=80, leave=False):
        possible_idx = list(road_sindex.intersection(cell.geometry.bounds))
        if len(possible_idx) > 0:
            candidates = roads_utm.iloc[possible_idx]
            intersecting = candidates[candidates.intersects(cell.geometry)]
            if len(intersecting) > 0:
                # Clip roads to cell and sum lengths
                clipped = intersecting.geometry.intersection(cell.geometry)
                total_length = clipped.length.sum() / 1000.0  # Convert to km
                road_lengths.append(total_length)
            else:
                road_lengths.append(0.0)
        else:
            road_lengths.append(0.0)

    grid["road_density_km"] = road_lengths

    print(f"  Road density range: [{grid['road_density_km'].min():.3f}, "
          f"{grid['road_density_km'].max():.3f}] km/cell")

    return grid


# ============================================================
# STEP 4: CLASSIFY HEAT RISK
# ============================================================

def assign_heat_risk(grid, config):
    """
    Assign a heat risk category to each cell based on surface temperature.
    """
    thresholds = config["thresholds"]

    def classify(temp):
        if pd.isna(temp):
            return "no_data"
        elif temp >= thresholds["extreme"]:
            return "extreme"
        elif temp >= thresholds["high"]:
            return "high"
        elif temp >= thresholds["moderate"]:
            return "moderate"
        else:
            return "low"

    grid["heat_risk"] = grid["mean_lst_f"].apply(classify)

    # Print distribution
    risk_counts = grid["heat_risk"].value_counts()
    print("  Heat risk distribution:")
    for risk, count in risk_counts.items():
        pct = count / len(grid) * 100
        print(f"    {risk:>10}: {count:>7,} cells ({pct:.1f}%)")

    return grid


# ============================================================
# STEP 5: SAVE & VISUALIZE
# ============================================================

def save_grid(grid, config, city_slug="chicago"):
    """Save the grid as both GeoJSON and CSV."""
    os.makedirs("data/grid", exist_ok=True)

    # GeoJSON (with geometry)
    grid_wgs84 = grid.to_crs("EPSG:4326")
    os.makedirs(f"data/{city_slug}/grid", exist_ok=True)
    geojson_path = f"data/{city_slug}/grid/{city_slug}_grid.geojson"
    grid_wgs84.to_file(geojson_path, driver="GeoJSON")
    print(f"  Saved GeoJSON: {geojson_path}")

    # CSV (without geometry, for ML training)
    csv_cols = [c for c in grid.columns if c not in ["geometry", "center_x_utm", "center_y_utm"]]
    csv_path = f"data/{city_slug}/grid/{city_slug}_grid.csv"
    grid[csv_cols].to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    print(f"  CSV shape: {grid[csv_cols].shape}")

    # Print column summary
    print()
    print("  Grid columns:")
    for col in csv_cols:
        non_null = grid[col].notna().sum()
        print(f"    {col:<25} {non_null:>7,} non-null values")

    return geojson_path, csv_path


def visualize_grid(grid, config, output_path="output/chicago_grid_preview.png"):
    """Create a multi-panel visualization of the grid features."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    grid_wgs84 = grid.to_crs("EPSG:4326")

    features = [
        ("mean_lst_f", "Surface Temperature (F)", "RdYlBu_r", 80, 115),
        ("ndvi", "Vegetation (NDVI)", "RdYlGn", 0, 0.7),
        ("building_density", "Building Density", "YlOrRd", 0, 0.6),
        ("distance_to_park_m", "Distance to Park (m)", "RdYlGn", 0, 2000),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    for i, (col, title, cmap, vmin, vmax) in enumerate(features):
        ax = axes[i]
        if col in grid_wgs84.columns:
            grid_wgs84.plot(
                column=col, ax=ax, cmap=cmap,
                vmin=vmin, vmax=vmax,
                legend=True, legend_kwds={"shrink": 0.6},
                missing_kwds={"color": "lightgray"},
                markersize=0.5
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Add neighborhood labels
        for hood in config.get("focus_neighborhoods", []):
            lat, lon = hood["center"]
            ax.annotate(
                hood["name"], xy=(lon, lat), fontsize=6,
                color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6)
            )

    fig.suptitle(
        f"Analysis Grid — {config['city']['display_name']}\n"
        f"{len(grid):,} cells at {config['grid']['cell_size_meters']}m resolution",
        fontsize=18, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Grid preview saved to: {output_path}")
    plt.show()


def visualize_heat_risk_map(grid, config, output_path="output/chicago_heat_risk.png"):
    """Create a clean heat risk classification map."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    grid_wgs84 = grid.to_crs("EPSG:4326")

    risk_colors = {
        "low": "#3288BD",
        "moderate": "#FDAE61",
        "high": "#F46D43",
        "extreme": "#A50026",
        "no_data": "#CCCCCC"
    }

    fig, ax = plt.subplots(figsize=(14, 14))

    for risk, color in risk_colors.items():
        subset = grid_wgs84[grid_wgs84["heat_risk"] == risk]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, label=f"{risk.title()} ({len(subset):,})")

    for hood in config.get("focus_neighborhoods", []):
        lat, lon = hood["center"]
        ax.annotate(
            hood["name"], xy=(lon, lat), fontsize=8, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7)
        )

    ax.set_title(
        f"Heat Risk Classification — {config['city']['display_name']}",
        fontsize=16, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=10, title="Risk Level")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"  Heat risk map saved to: {output_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HeatSense — Analysis Grid Builder")
    parser.add_argument("--city", type=str, default="chicago", help="City slug (e.g. chicago, phoenix, dallas)")
    parser.add_argument("--config", type=str, default=None, help="Direct path to config YAML")
    args = parser.parse_args()

    print("=" * 60)
    print("  HeatSense — Analysis Grid Builder")
    print("=" * 60)
    print()

    config = load_config(config_path=args.config, city_slug=args.city)
    city_slug = config.get("city", {}).get("slug", args.city)
    print(f"City: {config['city']['display_name']}")
    print(f"Grid cell size: {config['grid']['cell_size_meters']}m")
    print()

    # Initialize Earth Engine
    initialize_ee()
    print()

    # ---- Step 1: Create grid ----
    print("[Step 1/5] Creating grid...")
    grid = create_grid(config)
    print()

    # ---- Step 2: Satellite data ----
    print("[Step 2/5] Fetching satellite data...")
    rasters = fetch_satellite_rasters(config)
    print()
    print("  Assigning satellite values to grid cells...")
    grid = assign_satellite_data(grid, rasters, config)
    print()

    # ---- Step 3: OSM urban features ----
    print("[Step 3/5] Loading OSM data and computing urban features...")
    osm_data = load_osm_data(city_slug=city_slug)
    print()

    print("  [3a] Building features...")
    grid = assign_building_features(grid, osm_data["buildings"], config)
    print()

    print("  [3b] Park features...")
    grid = assign_park_features(grid, osm_data["parks"], config)
    print()

    print("  [3c] Water features...")
    grid = assign_water_features(grid, osm_data["water"], config)
    print()

    print("  [3d] Road features...")
    grid = assign_road_features(grid, osm_data["roads"], config)
    print()

    # ---- Step 4: Heat risk classification ----
    print("[Step 4/5] Classifying heat risk...")
    grid = assign_heat_risk(grid, config)
    print()

    # ---- Step 5: Save & visualize ----
    print("[Step 5/5] Saving and visualizing...")
    geojson_path, csv_path = save_grid(grid, config, city_slug=city_slug)
    print()

    print("  Creating grid preview...")
    visualize_grid(grid, config, output_path=f"output/{city_slug}_grid_preview.png")
    print()

    print("  Creating heat risk map...")
    visualize_heat_risk_map(grid, config, output_path=f"output/{city_slug}_heat_risk.png")

    # ---- Final summary ----
    print()
    print("=" * 60)
    print("  GRID BUILD COMPLETE")
    print()
    print(f"  Total cells: {len(grid):,}")
    print(f"  Cells with thermal data: {grid['mean_lst_f'].notna().sum():,}")
    print(f"  Cells with NDVI data: {grid['ndvi'].notna().sum():,}")
    print(f"  Cells with building data: {(grid['building_count'] > 0).sum():,}")
    print()
    print("  Output files:")
    print(f"  - {geojson_path} (spatial data)")
    print(f"  - {csv_path} (ML training data)")
    print(f"  - output/{city_slug}_grid_preview.png")
    print(f"  - output/{city_slug}_heat_risk.png")
    print()
    print("  NEXT: Train the ML model with:")
    print(f"  python model/train_heat_model.py --city {city_slug}")
    print("=" * 60)


if __name__ == "__main__":
    main()