#!/usr/bin/env python3
"""
TomorrowLand Heat — City Pipeline Runner
==========================================
Runs the complete data pipeline for a city based on its config file.

Usage:
  python run_city.py --city chicago          # Run full pipeline
  python run_city.py --city phoenix --step fetch_landsat  # Run specific step
  python run_city.py --list                  # List available cities

Steps (in order):
  1. fetch_landsat   — Pull thermal data from Google Earth Engine
  2. fetch_ndvi      — Pull Sentinel-2 vegetation data
  3. fetch_nlcd      — Pull impervious surface data
  4. fetch_osm       — Pull buildings, roads, parks from OpenStreetMap
  5. process_grid    — Build 100m grid, merge all features
  6. train_model     — Train LightGBM heat prediction model
  7. generate_images — Pre-render heat map PNG tiles

Each step reads the city's config/[slug].yaml and outputs to data/[slug]/.
"""

import argparse
import sys
import yaml
from pathlib import Path


def load_config(city_slug):
    """Load a city's configuration file."""
    config_dir = Path(__file__).parent.parent / "config"
    config_path = config_dir / f"{city_slug}.yaml"

    if not config_path.exists():
        print(f"  Error: No config found at {config_path}")
        print(f"  Available configs:")
        for f in sorted(config_dir.glob("*.yaml")):
            if f.name != "cities.yaml":
                print(f"    - {f.stem}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n  City: {config['city']['display_name']}")
    print(f"  Bbox: {config['bbox']}")
    print(f"  CRS:  {config['grid']['crs']}")
    return config


def list_cities():
    """List all available city configurations."""
    config_dir = Path(__file__).parent.parent / "config"
    cities_path = config_dir / "cities.yaml"

    if cities_path.exists():
        with open(cities_path) as f:
            registry = yaml.safe_load(f)
        print("\n  Available cities:\n")
        for city in registry.get("cities", []):
            status_icon = {
                "active": "✅",
                "ready": "🟡",
                "planned": "⬜",
            }.get(city["status"], "?")
            cells = f"{city['cells']:,} cells" if city.get('cells') else "not processed"
            print(f"  {status_icon} {city['slug']:20s} {city['name']:35s} [{city['status']}] {cells}")
            if city.get("key_story"):
                print(f"     └─ {city['key_story']}")
        print()
    else:
        # Fallback: list config files
        for f in sorted(config_dir.glob("*.yaml")):
            if f.name != "cities.yaml":
                print(f"  {f.stem}")


def ensure_dirs(city_slug):
    """Create output directories for a city."""
    base = Path(__file__).parent.parent
    dirs = [
        base / "data" / city_slug / "landsat",
        base / "data" / city_slug / "ndvi",
        base / "data" / city_slug / "nlcd",
        base / "data" / city_slug / "osm",
        base / "data" / city_slug / "grid",
        base / "model" / "models" / city_slug,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  📁 {d}")


def run_step(step, config, city_slug):
    """Run a single pipeline step."""
    print(f"\n{'='*60}")
    print(f"  Running: {step}")
    print(f"{'='*60}\n")

    bbox = config["bbox"]

    if step == "fetch_landsat":
        print("  This step requires Google Earth Engine credentials.")
        print("  Run manually:")
        print(f"    python data-pipeline/fetch_landsat.py \\")
        print(f"      --west {bbox['west']} --south {bbox['south']} \\")
        print(f"      --east {bbox['east']} --north {bbox['north']} \\")
        print(f"      --output data/{city_slug}/landsat/ \\")
        print(f"      --start-month {config['landsat']['start_month']} \\")
        print(f"      --end-month {config['landsat']['end_month']} \\")
        print(f"      --max-cloud {config['landsat']['max_cloud_cover']}")

    elif step == "fetch_ndvi":
        print("  This step requires Google Earth Engine credentials.")
        print("  Run manually:")
        print(f"    python data-pipeline/fetch_ndvi.py \\")
        print(f"      --west {bbox['west']} --south {bbox['south']} \\")
        print(f"      --east {bbox['east']} --north {bbox['north']} \\")
        print(f"      --output data/{city_slug}/ndvi/")

    elif step == "fetch_nlcd":
        print("  Run manually:")
        print(f"    python data-pipeline/fetch_nlcd.py \\")
        print(f"      --west {bbox['west']} --south {bbox['south']} \\")
        print(f"      --east {bbox['east']} --north {bbox['north']} \\")
        print(f"      --output data/{city_slug}/nlcd/")

    elif step == "fetch_osm":
        print("  Fetching OpenStreetMap data (buildings, roads, parks, water)...")
        print(f"    python data-pipeline/fetch_osm.py \\")
        print(f"      --west {bbox['west']} --south {bbox['south']} \\")
        print(f"      --east {bbox['east']} --north {bbox['north']} \\")
        print(f"      --output data/{city_slug}/osm/")

    elif step == "process_grid":
        print(f"  Building {config['grid']['cell_size_meters']}m grid...")
        print(f"    python data-pipeline/process_grid.py \\")
        print(f"      --config config/{city_slug}.yaml \\")
        print(f"      --output data/{city_slug}/grid/")

    elif step == "train_model":
        print(f"  Training LightGBM model...")
        print(f"    python model/train_heat_model.py \\")
        print(f"      --grid data/{city_slug}/grid/{city_slug}_grid.csv \\")
        print(f"      --output model/models/{city_slug}/")

    elif step == "generate_images":
        print(f"  Pre-rendering heat map images...")
        print(f"  (This happens automatically when the API starts)")

    else:
        print(f"  Unknown step: {step}")
        print(f"  Valid steps: fetch_landsat, fetch_ndvi, fetch_nlcd, fetch_osm,")
        print(f"               process_grid, train_model, generate_images")


ALL_STEPS = [
    "fetch_landsat", "fetch_ndvi", "fetch_nlcd", "fetch_osm",
    "process_grid", "train_model", "generate_images"
]


def main():
    parser = argparse.ArgumentParser(
        description="TomorrowLand Heat — City Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_city.py --list                    # List available cities
  python run_city.py --city phoenix            # Show all steps for Phoenix
  python run_city.py --city phoenix --step fetch_osm  # Show specific step
  python run_city.py --city phoenix --init     # Create directories only
        """
    )
    parser.add_argument("--city", type=str, help="City slug (e.g. chicago, phoenix)")
    parser.add_argument("--step", type=str, help="Run specific step")
    parser.add_argument("--list", action="store_true", help="List available cities")
    parser.add_argument("--init", action="store_true", help="Create city directories only")

    args = parser.parse_args()

    if args.list:
        list_cities()
        return

    if not args.city:
        parser.print_help()
        return

    config = load_config(args.city)

    if args.init:
        print(f"\n  Creating directories for {args.city}...")
        ensure_dirs(args.city)
        print(f"\n  Done! Now run pipeline steps.")
        return

    if args.step:
        run_step(args.step, config, args.city)
    else:
        # Show all steps
        print(f"\n  Pipeline steps for {config['city']['display_name']}:")
        print(f"  Grid: {config['grid']['cell_size_meters']}m cells")
        print(f"  Landsat months: {config['landsat']['start_month']}-{config['landsat']['end_month']}")
        print(f"  Cloud max: {config['landsat']['max_cloud_cover']}%")
        print()

        ensure_dirs(args.city)

        for step in ALL_STEPS:
            run_step(step, config, args.city)

        print(f"\n{'='*60}")
        print(f"  Pipeline guide complete for {args.city}")
        print(f"  Run each step above in order.")
        print(f"  Once the grid CSV exists, start the API:")
        print(f"    cd api && uvicorn main:app --reload --port 8000")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
