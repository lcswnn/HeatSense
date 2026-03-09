#!/usr/bin/env python3
"""
HeatSense — City Pipeline Runner
==========================================
Runs the complete data pipeline for a city based on its config file.

Usage:
  python run_city.py --city chicago          # Run full pipeline
  python run_city.py --city phoenix --step fetch_osm  # Run specific step
  python run_city.py --city phoenix --from process_grid  # Start from a step
  python run_city.py --list                  # List available cities

Steps (in order):
  1. fetch_landsat   — Pull thermal data from Google Earth Engine
  2. fetch_ndvi      — Pull Sentinel-2 vegetation + NLCD impervious data
  3. fetch_osm       — Pull buildings, roads, parks from OpenStreetMap
  4. process_grid    — Build 100m grid, merge all features
  5. train_model     — Train LightGBM heat prediction model
  6. tune_model      — Hyperparameter tuning & optimization

Each step reads the city's config/[slug].yaml and outputs to data/[slug]/.
"""

import argparse
import subprocess
import sys
import time
import yaml
from pathlib import Path

# Project root (one level up from data-pipeline/)
PROJECT_ROOT = Path(__file__).parent.parent

ALL_STEPS = [
    "fetch_landsat", "fetch_ndvi", "fetch_osm",
    "process_grid", "train_model", "tune_model"
]


def load_config(city_slug):
    """Load a city's configuration file."""
    config_dir = PROJECT_ROOT / "config"
    config_path = config_dir / f"{city_slug}.yaml"

    if not config_path.exists():
        print(f"\n  Error: No config found at {config_path}")
        print(f"  Available configs:")
        for f in sorted(config_dir.glob("*.yaml")):
            if f.name != "cities.yaml":
                print(f"    - {f.stem}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def list_cities():
    """List all available city configurations."""
    config_dir = PROJECT_ROOT / "config"
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
        print("\n  Available cities:\n")
        for f in sorted(config_dir.glob("*.yaml")):
            if f.name != "cities.yaml":
                print(f"    {f.stem}")
        print()


def ensure_dirs(city_slug):
    """Create output directories for a city."""
    dirs = [
        PROJECT_ROOT / "data" / city_slug / "landsat",
        PROJECT_ROOT / "data" / city_slug / "ndvi",
        PROJECT_ROOT / "data" / city_slug / "osm",
        PROJECT_ROOT / "data" / city_slug / "grid",
        PROJECT_ROOT / "model" / "models",
        PROJECT_ROOT / "output",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def check_output_exists(city_slug, step):
    """Check if a step's output already exists."""
    checks = {
        "fetch_landsat": PROJECT_ROOT / "data" / city_slug / "landsat",
        "fetch_ndvi": PROJECT_ROOT / "data" / city_slug / "ndvi",
        "fetch_osm": PROJECT_ROOT / "data" / city_slug / "osm",
        "process_grid": PROJECT_ROOT / "data" / city_slug / "grid" / f"{city_slug}_grid.csv",
        "train_model": PROJECT_ROOT / "model" / "models" / f"{city_slug}_heat_model.pkl",
        "tune_model": PROJECT_ROOT / "model" / "models" / f"{city_slug}_heat_model_tuned.pkl",
    }

    path = checks.get(step)
    if path is None:
        return False

    if path.is_dir():
        # Check if directory has files (not just empty)
        return any(path.iterdir()) if path.exists() else False
    else:
        return path.exists()


def run_command(cmd, step_name):
    """Run a subprocess command and stream output in real time."""
    print(f"\n  Running: {' '.join(str(c) for c in cmd)}\n")

    start = time.time()
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED: {step_name} exited with code {result.returncode}")
        print(f"  Elapsed: {elapsed:.1f}s")
        return False

    print(f"\n  Completed: {step_name} ({elapsed:.1f}s)")
    return True


def run_step(step, city_slug, config, skip_existing=True):
    """Run a single pipeline step."""
    print(f"\n{'='*60}")
    print(f"  Step: {step}")
    print(f"{'='*60}")

    # Check if output already exists
    if skip_existing and check_output_exists(city_slug, step):
        print(f"  Output already exists — skipping. Use --force to re-run.")
        return True

    pipeline_dir = PROJECT_ROOT / "data-pipeline"
    model_dir = PROJECT_ROOT / "model"

    if step == "fetch_landsat":
        cmd = [
            sys.executable, pipeline_dir / "fetch_landsat.py",
            "--city", city_slug,
        ]
        return run_command(cmd, step)

    elif step == "fetch_ndvi":
        cmd = [
            sys.executable, pipeline_dir / "fetch_ndvi.py",
            "--city", city_slug,
        ]
        return run_command(cmd, step)

    elif step == "fetch_osm":
        cmd = [
            sys.executable, pipeline_dir / "fetch_osm.py",
            "--city", city_slug,
        ]
        return run_command(cmd, step)

    elif step == "process_grid":
        cmd = [
            sys.executable, pipeline_dir / "process_grid.py",
            "--city", city_slug,
        ]
        return run_command(cmd, step)

    elif step == "train_model":
        cmd = [
            sys.executable, model_dir / "train_heat_model.py",
            "--city", city_slug,
        ]
        return run_command(cmd, step)

    elif step == "tune_model":
        cmd = [
            sys.executable, pipeline_dir / "tune_model.py",
            "--city", city_slug,
        ]
        return run_command(cmd, step)

    else:
        print(f"  Unknown step: {step}")
        print(f"  Valid steps: {', '.join(ALL_STEPS)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="HeatSense — City Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_city.py --list                           # List available cities
  python run_city.py --city phoenix                   # Run full pipeline
  python run_city.py --city phoenix --step fetch_osm  # Run one step
  python run_city.py --city phoenix --from process_grid  # Start from a step
  python run_city.py --city phoenix --force           # Re-run even if output exists
        """
    )
    parser.add_argument("--city", type=str, help="City slug (e.g. chicago, phoenix, dallas)")
    parser.add_argument("--step", type=str, choices=ALL_STEPS, help="Run a single step")
    parser.add_argument("--from", dest="from_step", type=str, choices=ALL_STEPS,
                        help="Start from this step (skips earlier steps)")
    parser.add_argument("--list", action="store_true", help="List available cities")
    parser.add_argument("--force", action="store_true",
                        help="Re-run steps even if output already exists")

    args = parser.parse_args()

    if args.list:
        list_cities()
        return

    if not args.city:
        parser.print_help()
        return

    city_slug = args.city.lower().replace(" ", "_")
    config = load_config(city_slug)
    skip_existing = not args.force

    print()
    print("=" * 60)
    print(f"  HeatSense Pipeline — {config['city']['display_name']}")
    print("=" * 60)
    print(f"  Bbox: [{config['bbox']['west']}, {config['bbox']['south']}] "
          f"to [{config['bbox']['east']}, {config['bbox']['north']}]")
    print(f"  Grid: {config['grid']['cell_size_meters']}m cells | CRS: {config['grid']['crs']}")
    print(f"  Landsat: months {config['landsat']['start_month']}-{config['landsat']['end_month']}, "
          f"max cloud {config['landsat']['max_cloud_cover']}%")

    # Create directories
    ensure_dirs(city_slug)

    # Determine which steps to run
    if args.step:
        steps = [args.step]
    elif args.from_step:
        start_idx = ALL_STEPS.index(args.from_step)
        steps = ALL_STEPS[start_idx:]
    else:
        steps = ALL_STEPS

    print(f"\n  Steps to run: {' → '.join(steps)}")

    # Run each step
    total_start = time.time()
    results = {}

    for step in steps:
        success = run_step(step, city_slug, config, skip_existing=skip_existing)
        results[step] = success

        if not success:
            print(f"\n  Pipeline stopped at '{step}'. Fix the error and re-run with:")
            print(f"    python run_city.py --city {city_slug} --from {step}")
            break

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  Pipeline Summary — {config['city']['display_name']}")
    print(f"{'='*60}")
    for step, success in results.items():
        icon = "✅" if success else "❌"
        print(f"  {icon} {step}")
    print(f"\n  Total time: {total_elapsed:.1f}s")

    if all(results.values()):
        print(f"\n  All steps completed! Start the API:")
        print(f"    cd api && uvicorn main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
