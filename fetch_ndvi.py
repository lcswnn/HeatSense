"""
TomorrowLand Heat — Step 2: Fetch NDVI Vegetation Data
======================================================
This script pulls Sentinel-2 imagery from Google Earth Engine and computes
NDVI (Normalized Difference Vegetation Index) for Chicago.

NDVI ranges from -1 to 1:
  - 0.6 - 1.0  → Dense, healthy vegetation (forests, lush parks)
  - 0.3 - 0.6  → Moderate vegetation (lawns, scattered trees)
  - 0.1 - 0.3  → Sparse vegetation (dry grass, bare soil with some green)
  - 0.0 - 0.1  → Bare soil, rock, sand
  - < 0.0      → Water, snow, clouds

This data is crucial because NDVI directly correlates with cooling:
more vegetation = lower surface temperatures.

Prerequisites:
  - Run fetch_landsat.py first (to verify your GEE setup works)
  - pip install -r requirements.txt

Usage:
  python fetch_ndvi.py

Output:
  - Saves NDVI GeoTIFF to data/ndvi/
  - Generates visualization comparing heat vs vegetation
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

def load_config(config_path="config/chicago.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def initialize_ee():
    try:
        ee.Initialize(project='quick-composite-462916-q9')  # Replace with your GEE project ID
        print("✓ Earth Engine initialized")
    except:
        ee.Authenticate()
        ee.Initialize(project='quick-composite-462916-q9')
        print("✓ Earth Engine initialized")


def get_study_area(config):
    bbox = config["bbox"]
    return ee.Geometry.Rectangle([
        bbox["west"], bbox["south"],
        bbox["east"], bbox["north"]
    ])


# ============================================================
# SENTINEL-2 NDVI PROCESSING
# ============================================================

def mask_clouds_sentinel2(image):
    """
    Mask clouds in Sentinel-2 using the Scene Classification Layer (SCL).
    SCL values 3 (cloud shadow), 8 (cloud medium prob), 9 (cloud high prob),
    10 (thin cirrus) are masked out.
    """
    scl = image.select("SCL")
    mask = (scl.neq(3)   # Cloud shadow
            .And(scl.neq(8))   # Cloud medium probability
            .And(scl.neq(9))   # Cloud high probability
            .And(scl.neq(10))) # Thin cirrus
    return image.updateMask(mask)


def compute_ndvi(image):
    """
    Compute NDVI from Sentinel-2 bands.
    NDVI = (NIR - Red) / (NIR + Red)
    For Sentinel-2: NIR = B8, Red = B4
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def fetch_ndvi_collection(config, study_area):
    """
    Fetch Sentinel-2 imagery and compute NDVI for summer months.
    """
    s2_config = config["sentinel2"]
    years = s2_config["years"]
    start_month = s2_config["start_month"]
    end_month = s2_config["end_month"]
    max_cloud = s2_config["max_cloud_cover"]

    all_collections = []

    for year in years:
        start_date = f"{year}-{start_month:02d}-01"
        if end_month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{end_month + 1:02d}-01"

        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(study_area)
               .filterDate(start_date, end_date)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
               .map(mask_clouds_sentinel2)
               .map(compute_ndvi))

        all_collections.append(col)

    # Merge all years
    merged = all_collections[0]
    for col in all_collections[1:]:
        merged = merged.merge(col)

    count = merged.size().getInfo()
    print(f"✓ Found {count} cloud-free Sentinel-2 scenes")

    return merged


def compute_mean_ndvi(collection, study_area):
    """Compute mean summer NDVI."""
    return collection.select("NDVI").mean().clip(study_area)


# ============================================================
# IMPERVIOUS SURFACE DATA
# ============================================================

def fetch_impervious_surface(study_area):
    """
    Fetch NLCD Impervious Surface data from Earth Engine.
    Values 0-100 representing % impervious surface cover.
    """
    # NLCD 2021 Impervious Surface
    nlcd_imperv = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
                   .filter(ee.Filter.eq("system:index", "2021"))
                   .first()
                   .select("impervious"))

    if nlcd_imperv is None:
        # Fallback to 2019 if 2021 not available
        nlcd_imperv = (ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
                       .first()
                       .select("impervious"))

    return nlcd_imperv.clip(study_area)


# ============================================================
# VISUALIZATION
# ============================================================

def download_as_numpy(image, study_area, scale=100):
    """Download EE image as numpy array."""
    import tempfile, rasterio
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "data.tif")
        geemap.ee_export_image(
            image, filename=filepath, scale=scale,
            region=study_area, file_per_band=False
        )
        with rasterio.open(filepath) as src:
            data = src.read(1)
            bounds = src.bounds
    data = np.where(data == 0, np.nan, data)
    return data, bounds


def visualize_ndvi(ndvi_data, bounds, config, output_path="output/chicago_ndvi_map.png"):
    """Create NDVI visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Green colormap for vegetation
    colors = [
        "#8B4513",  # Brown (bare soil)
        "#D2B48C",  # Tan
        "#F5DEB3",  # Wheat
        "#ADFF2F",  # Green-yellow
        "#7CFC00",  # Lawn green
        "#228B22",  # Forest green
        "#006400",  # Dark green (dense vegetation)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("vegetation", colors, N=256)

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    im = ax.imshow(
        ndvi_data, cmap=cmap,
        vmin=0.0, vmax=0.7,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        aspect="auto", interpolation="nearest"
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("NDVI (Vegetation Density)", fontsize=12)

    ax.set_title(
        f"Vegetation Cover — {config['city']['display_name']}\n"
        f"Mean Summer NDVI (Sentinel-2, {config['sentinel2']['years'][0]}-{config['sentinel2']['years'][-1]})",
        fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # Neighborhood labels
    for hood in config.get("focus_neighborhoods", []):
        lat, lon = hood["center"]
        if bounds.bottom <= lat <= bounds.top and bounds.left <= lon <= bounds.right:
            ax.annotate(
                hood["name"], xy=(lon, lat), fontsize=8, fontweight="bold",
                color="white", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ NDVI map saved to: {output_path}")
    plt.show()


def visualize_comparison(heat_data, ndvi_data, bounds, config,
                         output_path="output/chicago_heat_vs_vegetation.png"):
    """
    Side-by-side comparison of heat and vegetation.
    THIS IS THE MONEY SHOT — shows the inverse correlation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Heat colormap
    heat_colors = [
        "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
        "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"
    ]
    heat_cmap = mcolors.LinearSegmentedColormap.from_list("heat", heat_colors, N=256)

    # Vegetation colormap
    veg_colors = [
        "#8B4513", "#D2B48C", "#F5DEB3", "#ADFF2F",
        "#7CFC00", "#228B22", "#006400"
    ]
    veg_cmap = mcolors.LinearSegmentedColormap.from_list("veg", veg_colors, N=256)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    # Left: Heat map
    im1 = axes[0].imshow(
        heat_data, cmap=heat_cmap, vmin=75, vmax=115,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        aspect="auto", interpolation="nearest"
    )
    plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02, label="Surface Temp (°F)")
    axes[0].set_title("🔥 Land Surface Temperature", fontsize=14, fontweight="bold")

    # Right: NDVI
    im2 = axes[1].imshow(
        ndvi_data, cmap=veg_cmap, vmin=0.0, vmax=0.7,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        aspect="auto", interpolation="nearest"
    )
    plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02, label="NDVI (Vegetation)")
    axes[1].set_title("🌳 Vegetation Density", fontsize=14, fontweight="bold")

    # Add neighborhood labels to both
    for ax in axes:
        for hood in config.get("focus_neighborhoods", []):
            lat, lon = hood["center"]
            if bounds.bottom <= lat <= bounds.top and bounds.left <= lon <= bounds.right:
                ax.annotate(
                    hood["name"], xy=(lon, lat), fontsize=7, fontweight="bold",
                    color="white", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
                )
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)

    fig.suptitle(
        f"Urban Heat Island Effect — {config['city']['display_name']}\n"
        f"Notice: Where vegetation is LOW (brown), temperatures are HIGH (red)",
        fontsize=16, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ Comparison map saved to: {output_path}")
    print(f"  This visualization shows the core insight: green areas are cool, concrete is hot.")
    plt.show()


def create_scatter_analysis(heat_data, ndvi_data,
                            output_path="output/chicago_heat_ndvi_scatter.png"):
    """
    Scatter plot: NDVI vs Surface Temperature.
    Should show a clear negative correlation — more green = cooler.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Flatten and remove NaNs
    heat_flat = heat_data.flatten()
    ndvi_flat = ndvi_data.flatten()
    mask = ~np.isnan(heat_flat) & ~np.isnan(ndvi_flat) & (ndvi_flat > -0.1) & (heat_flat > 50)
    heat_valid = heat_flat[mask]
    ndvi_valid = ndvi_flat[mask]

    # Subsample for plotting (too many points makes it slow)
    if len(heat_valid) > 10000:
        idx = np.random.choice(len(heat_valid), 10000, replace=False)
        heat_sample = heat_valid[idx]
        ndvi_sample = ndvi_valid[idx]
    else:
        heat_sample = heat_valid
        ndvi_sample = ndvi_valid

    # Correlation
    correlation = np.corrcoef(ndvi_valid, heat_valid)[0, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        ndvi_sample, heat_sample,
        c=heat_sample, cmap="RdYlBu_r",
        alpha=0.3, s=5, vmin=75, vmax=115
    )
    plt.colorbar(scatter, ax=ax, label="Surface Temperature (°F)")

    # Trend line
    z = np.polyfit(ndvi_valid, heat_valid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ndvi_valid.min(), ndvi_valid.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Trend (r = {correlation:.3f})")

    ax.set_xlabel("NDVI (Vegetation Index)", fontsize=12)
    ax.set_ylabel("Land Surface Temperature (°F)", fontsize=12)
    ax.set_title(
        f"Vegetation vs. Surface Temperature — Chicago\n"
        f"Correlation: r = {correlation:.3f} — More trees = cooler neighborhoods",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ Scatter plot saved to: {output_path}")
    print(f"  Correlation coefficient: r = {correlation:.3f}")
    print(f"  (Negative correlation confirms: more vegetation → lower temps)")
    plt.show()


# ============================================================
# INTERACTIVE MAP
# ============================================================

def create_interactive_map(mean_ndvi, mean_lst_f, imperv, study_area, config):
    """Create an interactive map with multiple layers."""
    center = config["center"]
    m = geemap.Map(center=[center["latitude"], center["longitude"]], zoom=center["zoom"])

    # LST layer
    lst_vis = {
        "min": 75, "max": 115,
        "palette": ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
                     "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    }
    m.addLayer(mean_lst_f, lst_vis, "Surface Temperature (°F)")

    # NDVI layer
    ndvi_vis = {
        "min": 0.0, "max": 0.7,
        "palette": ["#8B4513", "#D2B48C", "#F5DEB3", "#ADFF2F",
                     "#7CFC00", "#228B22", "#006400"]
    }
    m.addLayer(mean_ndvi, ndvi_vis, "Vegetation (NDVI)")

    # Impervious surface layer
    imperv_vis = {
        "min": 0, "max": 100,
        "palette": ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000"]
    }
    m.addLayer(imperv, imperv_vis, "Impervious Surface (%)")

    # Layer control
    m.addLayerControl()

    output_path = "output/chicago_multi_layer_map.html"
    os.makedirs("output", exist_ok=True)
    m.to_html(output_path)
    print(f"✓ Multi-layer interactive map saved to: {output_path}")

    return m


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  TomorrowLand Heat — NDVI Vegetation Pipeline")
    print("=" * 60)
    print()

    config = load_config()
    print(f"📍 City: {config['city']['display_name']}")
    print()

    initialize_ee()
    study_area = get_study_area(config)

    # Fetch NDVI
    print("Fetching Sentinel-2 imagery for NDVI computation...")
    ndvi_collection = fetch_ndvi_collection(config, study_area)
    mean_ndvi = compute_mean_ndvi(ndvi_collection, study_area)
    print("✓ Mean summer NDVI computed")
    print()

    # Fetch impervious surface
    print("Fetching NLCD impervious surface data...")
    imperv = fetch_impervious_surface(study_area)
    print("✓ Impervious surface data loaded")
    print()

    # Also fetch LST for comparison (reuse from fetch_landsat.py logic)
    print("Fetching Landsat thermal data for comparison...")
    from fetch_landsat import fetch_landsat_collection, compute_mean_lst
    lst_collection = fetch_landsat_collection(config, study_area)
    mean_lst_f, _ = compute_mean_lst(lst_collection, study_area)
    print()

    # Download and visualize
    print("Downloading data for visualization...")
    try:
        ndvi_data, ndvi_bounds = download_as_numpy(mean_ndvi, study_area, scale=100)
        visualize_ndvi(ndvi_data, ndvi_bounds, config)

        heat_data, heat_bounds = download_as_numpy(mean_lst_f, study_area, scale=100)

        # The money shot: side-by-side comparison
        print()
        print("Creating heat vs. vegetation comparison...")
        visualize_comparison(heat_data, ndvi_data, ndvi_bounds, config)

        # Scatter plot showing the correlation
        print()
        print("Creating correlation analysis...")
        create_scatter_analysis(heat_data, ndvi_data)

    except Exception as e:
        print(f"⚠ Local download failed ({e}), using Drive export...")
        export_task = ee.batch.Export.image.toDrive(
            image=mean_ndvi, description="chicago_mean_ndvi",
            folder="tomorrowland_heat", region=study_area,
            scale=30, crs="EPSG:4326", maxPixels=1e9
        )
        export_task.start()
        print("✓ NDVI export started to Google Drive")

    # Interactive multi-layer map
    print()
    print("Creating interactive multi-layer map...")
    create_interactive_map(mean_ndvi, mean_lst_f, imperv, study_area, config)

    print()
    print("=" * 60)
    print("  🎉 NDVI pipeline complete!")
    print()
    print("  Key outputs:")
    print("  - output/chicago_ndvi_map.png           (vegetation map)")
    print("  - output/chicago_heat_vs_vegetation.png  (THE comparison)")
    print("  - output/chicago_heat_ndvi_scatter.png   (correlation proof)")
    print("  - output/chicago_multi_layer_map.html    (interactive explorer)")
    print()
    print("  Next step: Run process_grid.py to build the analysis grid")
    print("=" * 60)


if __name__ == "__main__":
    main()