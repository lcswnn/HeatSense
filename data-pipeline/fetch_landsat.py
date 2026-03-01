"""
TomorrowLand Heat — Step 1: Fetch Landsat Thermal Data
======================================================
This script pulls Landsat 8/9 thermal infrared data from Google Earth Engine
for Chicago and computes Land Surface Temperature (LST).

Prerequisites:
  1. Google Earth Engine account (registered for non-commercial use ✓)
  2. Run `earthengine authenticate` in your terminal first
  3. pip install -r requirements.txt

Usage:
  python fetch_landsat.py

Output:
  - Saves a GeoTIFF of mean summer surface temperature to data/thermal/
  - Generates a visualization plot to output/
"""

import ee
import geemap
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================
# CONFIGURATION
# ============================================================

def load_config(config_path=None, city_slug=None):
    """Load city configuration. Accepts either a path or a city slug."""
    if config_path is None and city_slug is not None:
        # Find config relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        config_path = os.path.join(project_dir, "config", f"{city_slug}.yaml")
    elif config_path is None:
        config_path = "config/chicago.yaml"
    
    if not os.path.exists(config_path):
        print(f"  Error: Config not found at {config_path}")
        # List available configs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        config_dir = os.path.join(project_dir, "config")
        if os.path.isdir(config_dir):
            print(f"  Available cities:")
            for f in sorted(os.listdir(config_dir)):
                if f.endswith(".yaml") and f != "cities.yaml":
                    print(f"    - {f.replace('.yaml', '')}")
        raise SystemExit(1)
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# EARTH ENGINE HELPERS
# ============================================================

def initialize_ee():
    """Authenticate and initialize Earth Engine."""
    try:
        ee.Initialize(project='quick-composite-462916-q9')  # Replace with your GEE project ID
        print("✓ Earth Engine initialized successfully")
    except Exception as e:
        print("Earth Engine not authenticated. Running authentication flow...")
        ee.Authenticate()
        ee.Initialize(project='quick-composite-462916-q9')  # Replace with your GEE project ID
        print("✓ Earth Engine initialized successfully")


def get_study_area(config):
    """Create an Earth Engine geometry from the config bounding box."""
    bbox = config["bbox"]
    return ee.Geometry.Rectangle([
        bbox["west"], bbox["south"],
        bbox["east"], bbox["north"]
    ])


# ============================================================
# LANDSAT PROCESSING
# ============================================================

def apply_scale_factors(image):
    """
    Apply Landsat Collection 2 scale factors.
    Band 10 (thermal): multiply by 0.00341802 + 149.0 → Kelvin
    Surface reflectance bands: multiply by 0.0000275 + (-0.2)
    """
    thermal_bands = image.select("ST_B10").multiply(0.00341802).add(149.0)
    optical_bands = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return image.addBands(thermal_bands, overwrite=True) \
                .addBands(optical_bands, overwrite=True)


def mask_clouds_landsat(image):
    """
    Mask clouds and cloud shadows using the QA_PIXEL band.
    Bits 3 and 4 are cloud shadow and cloud, respectively.
    """
    qa = image.select("QA_PIXEL")
    cloud_shadow_bit = 1 << 3
    cloud_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0) \
             .And(qa.bitwiseAnd(cloud_bit).eq(0))
    return image.updateMask(mask)


def kelvin_to_fahrenheit(image):
    """Convert thermal band from Kelvin to Fahrenheit."""
    fahrenheit = image.select("ST_B10") \
                      .subtract(273.15) \
                      .multiply(9.0 / 5.0) \
                      .add(32.0) \
                      .rename("LST_F")
    return image.addBands(fahrenheit)


def kelvin_to_celsius(image):
    """Convert thermal band from Kelvin to Celsius."""
    celsius = image.select("ST_B10") \
                   .subtract(273.15) \
                   .rename("LST_C")
    return image.addBands(celsius)


def fetch_landsat_collection(config, study_area):
    """
    Fetch and process Landsat 8/9 imagery for summer months.
    Returns an ImageCollection with cloud-masked, temperature-converted images.
    """
    landsat_config = config["landsat"]
    years = landsat_config["years"]
    start_month = landsat_config["start_month"]
    end_month = landsat_config["end_month"]
    max_cloud = landsat_config["max_cloud_cover"]

    all_images = []

    for year in years:
        start_date = f"{year}-{start_month:02d}-01"
        # End date: last day of the end month
        if end_month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{end_month + 1:02d}-01"

        # Landsat 9
        l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
              .filterBounds(study_area)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt("CLOUD_COVER", max_cloud)))

        # Landsat 8
        l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterBounds(study_area)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt("CLOUD_COVER", max_cloud)))

        all_images.append(l9)
        all_images.append(l8)

    # Merge all collections
    merged = all_images[0]
    for col in all_images[1:]:
        merged = merged.merge(col)

    # Process: scale factors → cloud mask → temperature conversion
    processed = (merged
                 .map(apply_scale_factors)
                 .map(mask_clouds_landsat)
                 .map(kelvin_to_fahrenheit)
                 .map(kelvin_to_celsius))

    count = processed.size().getInfo()
    print(f"✓ Found {count} cloud-free Landsat scenes")

    return processed


def compute_mean_lst(collection, study_area):
    """
    Compute the mean Land Surface Temperature across all summer scenes.
    This gives us the 'typical summer heat' baseline for each pixel.
    """
    mean_lst_f = collection.select("LST_F").mean().clip(study_area)
    mean_lst_c = collection.select("LST_C").mean().clip(study_area)

    return mean_lst_f, mean_lst_c


# ============================================================
# EXPORT & VISUALIZATION
# ============================================================

def export_to_drive(image, description, study_area, scale=100):
    """
    Export an Earth Engine image to Google Drive as a GeoTIFF.
    (Alternative to local download for large areas)
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder="tomorrowland_heat",
        region=study_area,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF"
    )
    task.start()
    print(f"✓ Export task started: {description}")
    print(f"  Check progress at: https://code.earthengine.google.com/tasks")
    return task


def download_as_numpy(image, study_area, scale=100):
    """
    Download a small-ish Earth Engine image as a numpy array.
    Good for visualization; for large areas, use export_to_drive instead.
    """
    print("  Downloading raster data (this may take a minute)...")
    
    # Use geemap to convert to numpy
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "lst.tif")
        geemap.ee_export_image(
            image,
            filename=filepath,
            scale=scale,
            region=study_area,
            file_per_band=False
        )
        
        import rasterio
        with rasterio.open(filepath) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            bounds = src.bounds
    
    # Mask no-data values
    data = np.where(data == 0, np.nan, data)
    
    print(f"  ✓ Downloaded raster: {data.shape} pixels")
    return data, bounds


def visualize_lst(data, bounds, config, output_path="output/chicago_heat_map.png"):
    """
    Create a beautiful visualization of the Land Surface Temperature map.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Custom colormap: cool blues → warm yellows → hot reds
    colors = [
        "#313695",  # Deep blue (coolest)
        "#4575b4",
        "#74add1",
        "#abd9e9",
        "#e0f3f8",
        "#ffffbf",  # Yellow (moderate)
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#d73027",
        "#a50026",  # Deep red (hottest)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("heat_island", colors, N=256)

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Plot the temperature data
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=75,   # Min temp for color scale (°F)
        vmax=115,  # Max temp for color scale (°F)
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        aspect="auto",
        interpolation="nearest"
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Land Surface Temperature (°F)", fontsize=12)

    # Labels
    ax.set_title(
        f"Urban Heat Island Map — {config['city']['display_name']}\n"
        f"Mean Summer Surface Temperature (Landsat 8/9, {config['landsat']['years'][0]}-{config['landsat']['years'][-1]})",
        fontsize=16,
        fontweight="bold",
        pad=20
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Add neighborhood labels
    for hood in config.get("focus_neighborhoods", []):
        lat, lon = hood["center"]
        if bounds.bottom <= lat <= bounds.top and bounds.left <= lon <= bounds.right:
            ax.annotate(
                hood["name"],
                xy=(lon, lat),
                fontsize=8,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ Heat map saved to: {output_path}")
    plt.show()


def create_interactive_map(mean_lst_f, study_area, config):
    """
    Create an interactive Folium map with the heat overlay.
    This is your first look at the data in a web-friendly format!
    """
    center = config["center"]
    m = geemap.Map(center=[center["latitude"], center["longitude"]], zoom=center["zoom"])

    # Add the LST layer
    vis_params = {
        "min": 75,
        "max": 115,
        "palette": [
            "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
            "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"
        ]
    }
    m.addLayer(mean_lst_f, vis_params, "Land Surface Temperature (°F)")

    # Add a colorbar legend
    m.add_colorbar(
        vis_params,
        label="Surface Temperature (°F)",
        layer_name="Land Surface Temperature (°F)"
    )

    # Save to HTML
    output_path = "output/chicago_interactive_map.html"
    os.makedirs("output", exist_ok=True)
    m.to_html(output_path)
    print(f"✓ Interactive map saved to: {output_path}")
    print(f"  Open it in your browser to explore!")

    return m


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HeatSense — Fetch Landsat Thermal Data")
    parser.add_argument("--city", type=str, default="chicago",
                        help="City slug matching a config file (e.g. chicago, phoenix, dallas)")
    parser.add_argument("--config", type=str, default=None,
                        help="Direct path to a city config YAML file")
    parser.add_argument("--west", type=float, default=None)
    parser.add_argument("--south", type=float, default=None)
    parser.add_argument("--east", type=float, default=None)
    parser.add_argument("--north", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start-month", type=int, default=None)
    parser.add_argument("--end-month", type=int, default=None)
    parser.add_argument("--max-cloud", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  HeatSense — Landsat Thermal Data Pipeline")
    print("=" * 60)
    print()

    # Load config — prefer --config path, then --city slug
    if args.config:
        config = load_config(config_path=args.config)
    else:
        config = load_config(city_slug=args.city)

    # Allow CLI overrides for bbox and landsat params
    if args.west is not None:
        config["bbox"]["west"] = args.west
    if args.south is not None:
        config["bbox"]["south"] = args.south
    if args.east is not None:
        config["bbox"]["east"] = args.east
    if args.north is not None:
        config["bbox"]["north"] = args.north
    if args.start_month is not None:
        config["landsat"]["start_month"] = args.start_month
    if args.end_month is not None:
        config["landsat"]["end_month"] = args.end_month
    if args.max_cloud is not None:
        config["landsat"]["max_cloud_cover"] = args.max_cloud

    city_slug = config.get("city", {}).get("slug", args.city)

    print(f"📍 City: {config['city']['display_name']}")
    print(f"📅 Years: {config['landsat']['years']}")
    start_m = config['landsat']['start_month']
    end_m = config['landsat']['end_month']
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    print(f"☀️  Months: {month_names.get(start_m, start_m)} - {month_names.get(end_m, end_m)}")
    print()

    # Initialize Earth Engine
    initialize_ee()

    # Define study area
    study_area = get_study_area(config)
    print(f"✓ Study area defined: {config['bbox']}")
    print()

    # Fetch and process Landsat imagery
    print("Fetching Landsat 8/9 thermal imagery...")
    collection = fetch_landsat_collection(config, study_area)

    # Compute mean summer surface temperature
    print("Computing mean summer Land Surface Temperature...")
    mean_lst_f, mean_lst_c = compute_mean_lst(collection, study_area)
    print("✓ Mean LST computed")
    print()

    # Determine output paths
    output_dir = args.output or f"data/{city_slug}/landsat/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Download and visualize
    print("Downloading data for visualization...")
    try:
        data, bounds = download_as_numpy(mean_lst_f, study_area, scale=100)
        viz_path = f"output/{city_slug}_heat_map.png"
        visualize_lst(data, bounds, config, output_path=viz_path)
    except Exception as e:
        print(f"  ⚠ Local download failed ({e})")
        print(f"  Falling back to Google Drive export...")
        export_to_drive(mean_lst_f, f"{city_slug}_mean_summer_lst_fahrenheit", study_area)
        export_to_drive(mean_lst_c, f"{city_slug}_mean_summer_lst_celsius", study_area)

    # Create interactive map
    print()
    print("Creating interactive map...")
    create_interactive_map(mean_lst_f, study_area, config)

    print()
    print("=" * 60)
    print(f"  🎉 Done! Heat island visualization ready for {config['city']['display_name']}.")
    print()
    print("  Next steps:")
    print(f"  1. Open output/{city_slug}_interactive_map.html in your browser")
    print("  2. Explore the heat patterns")
    print(f"  3. Run fetch_ndvi.py --city {city_slug} to add vegetation data")
    print("=" * 60)


if __name__ == "__main__":
    main()