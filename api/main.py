"""
TomorrowLand Heat — FastAPI Backend
====================================
Serves heat model predictions, grid data, and heat map images.
Supports multiple cities — auto-discovers cities with grid data + trained models.

Usage:
  cd api
  pip install fastapi uvicorn pillow
  uvicorn main:app --reload --port 8000
"""

import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_model import HeatModel

app = FastAPI(
    title="TomorrowLand Heat API",
    description="Urban heat island prediction and intervention simulation — multi-city",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multi-city storage: { "chicago": HeatModel, "dallas": HeatModel, ... }
models: dict[str, HeatModel] = {}

# Per-city heatmap cache: { "chicago": { "temperature": {png, bounds}, ... }, ... }
heatmap_cache: dict[str, dict] = {}

# Default city (first loaded, or chicago if available)
default_city: str = "chicago"


# ============================================================
# City discovery & startup
# ============================================================

NEIGHBORHOODS = {
    "chicago": [
        {"name": "Pilsen", "lat": 41.8484, "lon": -87.6564},
        {"name": "Englewood", "lat": 41.7798, "lon": -87.6456},
        {"name": "Austin", "lat": 41.8953, "lon": -87.7651},
        {"name": "Back of the Yards", "lat": 41.8097, "lon": -87.6564},
        {"name": "Lincoln Park", "lat": 41.9214, "lon": -87.6513},
        {"name": "Hyde Park", "lat": 41.7943, "lon": -87.5907},
        {"name": "Logan Square", "lat": 41.9236, "lon": -87.7078},
        {"name": "Wicker Park", "lat": 41.9088, "lon": -87.6796},
        {"name": "Loop", "lat": 41.8819, "lon": -87.6278},
        {"name": "Bronzeville", "lat": 41.8231, "lon": -87.6170},
        {"name": "Humboldt Park", "lat": 41.9020, "lon": -87.7234},
        {"name": "Garfield Park", "lat": 41.8803, "lon": -87.7242},
        {"name": "Bridgeport", "lat": 41.8381, "lon": -87.6506},
        {"name": "Rogers Park", "lat": 42.0087, "lon": -87.6723},
        {"name": "South Shore", "lat": 41.7613, "lon": -87.5767},
    ],
    "dallas": [
        {"name": "Downtown", "lat": 32.7810, "lon": -96.7970},
        {"name": "Deep Ellum", "lat": 32.7834, "lon": -96.7836},
        {"name": "Oak Lawn", "lat": 32.8100, "lon": -96.8100},
        {"name": "Bishop Arts", "lat": 32.7460, "lon": -96.8270},
        {"name": "Uptown", "lat": 32.7990, "lon": -96.8020},
        {"name": "Fair Park", "lat": 32.7790, "lon": -96.7620},
        {"name": "South Dallas", "lat": 32.7500, "lon": -96.7700},
        {"name": "Oak Cliff", "lat": 32.7300, "lon": -96.8400},
    ],
    "phoenix": [
        {"name": "Downtown", "lat": 33.4484, "lon": -112.0740},
        {"name": "Tempe", "lat": 33.4255, "lon": -111.9400},
        {"name": "Scottsdale", "lat": 33.4942, "lon": -111.9261},
        {"name": "Mesa", "lat": 33.4152, "lon": -111.8315},
        {"name": "South Mountain", "lat": 33.3500, "lon": -112.0500},
    ],
    "houston": [
        {"name": "Downtown", "lat": 29.7604, "lon": -95.3698},
        {"name": "Midtown", "lat": 29.7430, "lon": -95.3830},
        {"name": "Third Ward", "lat": 29.7220, "lon": -95.3540},
        {"name": "Heights", "lat": 29.7930, "lon": -95.3980},
        {"name": "Montrose", "lat": 29.7460, "lon": -95.3960},
    ],
    "los-angeles": [
        {"name": "Downtown LA", "lat": 34.0407, "lon": -118.2468},
        {"name": "Hollywood", "lat": 34.0928, "lon": -118.3287},
        {"name": "South LA", "lat": 33.9425, "lon": -118.2551},
        {"name": "East LA", "lat": 34.0239, "lon": -118.1720},
        {"name": "Venice", "lat": 33.9850, "lon": -118.4695},
    ],
    "atlanta": [
        {"name": "Downtown", "lat": 33.7490, "lon": -84.3880},
        {"name": "Midtown", "lat": 33.7840, "lon": -84.3834},
        {"name": "Buckhead", "lat": 33.8384, "lon": -84.3797},
        {"name": "West End", "lat": 33.7360, "lon": -84.4120},
    ],
    "miami": [
        {"name": "Downtown", "lat": 25.7743, "lon": -80.1937},
        {"name": "Little Havana", "lat": 25.7650, "lon": -80.2280},
        {"name": "Wynwood", "lat": 25.8010, "lon": -80.1990},
        {"name": "Overtown", "lat": 25.7850, "lon": -80.2050},
    ],
    "nyc": [
        {"name": "Manhattan", "lat": 40.7831, "lon": -73.9712},
        {"name": "South Bronx", "lat": 40.8176, "lon": -73.9209},
        {"name": "Brooklyn", "lat": 40.6782, "lon": -73.9442},
        {"name": "Harlem", "lat": 40.8116, "lon": -73.9465},
        {"name": "Queens", "lat": 40.7282, "lon": -73.7949},
    ],
}


@app.on_event("startup")
def load_models():
    global default_city
    print("\n  Loading TomorrowLand Heat — Multi-City...")
    start = time.time()

    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(api_dir)
    model_dir = os.path.join(project_dir, "model", "models")
    data_dir = os.path.join(project_dir, "data")

    # Auto-discover cities: look for data/{city}/grid/{city}_grid.csv + model
    for entry in sorted(Path(data_dir).iterdir()):
        if not entry.is_dir():
            continue
        city_slug = entry.name
        grid_path = entry / "grid" / f"{city_slug}_grid.csv"
        model_path = Path(model_dir) / f"{city_slug}_heat_model.pkl"

        if not grid_path.exists():
            continue
        if not model_path.exists():
            print(f"  Skipping {city_slug}: grid found but no trained model")
            continue

        try:
            print(f"\n  Loading {city_slug}...")
            city_model = HeatModel(
                model_dir=model_dir,
                grid_path=str(grid_path),
                city_slug=city_slug,
            )
            models[city_slug] = city_model

            # Pre-generate heatmap images
            heatmap_cache[city_slug] = {}
            for layer_name in ["temperature", "risk", "ndvi"]:
                png_bytes, bounds = city_model.generate_heatmap_png(layer=layer_name)
                heatmap_cache[city_slug][layer_name] = {"png": png_bytes, "bounds": bounds}
                print(f"    {layer_name}: {len(png_bytes) / 1024:.0f} KB")

        except Exception as e:
            print(f"  Error loading {city_slug}: {e}")

    if not models:
        print("  WARNING: No cities loaded! Check data/ and model/models/ directories.")
    else:
        if "chicago" in models:
            default_city = "chicago"
        else:
            default_city = next(iter(models))
        print(f"\n  Loaded {len(models)} cities: {list(models.keys())}")
        print(f"  Default city: {default_city}")

    elapsed = time.time() - start
    print(f"  Startup completed in {elapsed:.1f}s")
    print(f"  Ready to serve!\n")


def get_model(city: Optional[str] = None) -> HeatModel:
    """Resolve the model for a city slug."""
    slug = city or default_city
    if slug not in models:
        raise HTTPException(status_code=404, detail=f"City '{slug}' not loaded. Available: {list(models.keys())}")
    return models[slug]


# ============================================================
# Request models
# ============================================================

class SimulationRequest(BaseModel):
    lat: float
    lon: float
    radius_m: float = 500
    intervention_type: str = "moderate"
    city: Optional[str] = None

class Neighborhood(BaseModel):
    name: str
    lat: float
    lon: float
    radius_m: float = 1000

class CompareRequest(BaseModel):
    neighborhoods: list[Neighborhood]
    city: Optional[str] = None


# ============================================================
# Endpoints
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "cities_loaded": list(models.keys()),
        "default_city": default_city,
    }


@app.get("/api/cities")
def list_cities():
    """List all available cities with their data status."""
    return [
        {"slug": slug, "has_model": True, "cells": len(m.grid_valid)}
        for slug, m in models.items()
    ]


@app.get("/api/stats")
def get_city_stats(city: Optional[str] = Query(None)):
    m = get_model(city)
    stats = m.get_city_stats()
    stats["city"] = m.city_slug
    return stats


@app.get("/api/heatmap/{layer}.png")
def get_heatmap_image(layer: str, city: Optional[str] = Query(None)):
    slug = city or default_city
    if slug not in heatmap_cache or layer not in heatmap_cache.get(slug, {}):
        raise HTTPException(status_code=404, detail=f"No heatmap for city='{slug}', layer='{layer}'")

    return Response(
        content=heatmap_cache[slug][layer]["png"],
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/heatmap/{layer}/bounds")
def get_heatmap_bounds(layer: str, city: Optional[str] = Query(None)):
    slug = city or default_city
    if slug not in heatmap_cache or layer not in heatmap_cache.get(slug, {}):
        raise HTTPException(status_code=404, detail=f"No heatmap for city='{slug}', layer='{layer}'")
    return heatmap_cache[slug][layer]["bounds"]


@app.get("/api/grid")
def get_grid(
    city: Optional[str] = Query(None),
    west: Optional[float] = Query(None),
    south: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
    north: Optional[float] = Query(None),
    downsample: Optional[int] = Query(None),
):
    m = get_model(city)
    bbox = None
    if all(v is not None for v in [west, south, east, north]):
        bbox = {"west": west, "south": south, "east": east, "north": north}

    data = m.get_grid_data(bbox=bbox, downsample=downsample)
    return {"count": len(data), "city": m.city_slug, "bbox": bbox, "cells": data}


@app.get("/api/cell")
def get_cell_detail(
    lat: float = Query(...),
    lon: float = Query(...),
    city: Optional[str] = Query(None),
):
    m = get_model(city)
    return m.get_cell_detail(lat, lon)


@app.post("/api/simulate")
def simulate_intervention(request: SimulationRequest):
    m = get_model(request.city)
    return m.simulate_intervention(
        lat=request.lat, lon=request.lon,
        radius_m=request.radius_m,
        intervention_type=request.intervention_type,
    )


@app.post("/api/compare")
def compare_neighborhoods(request: CompareRequest):
    m = get_model(request.city)
    hoods = [{"name": n.name, "lat": n.lat, "lon": n.lon, "radius_m": n.radius_m}
             for n in request.neighborhoods]
    return m.get_neighborhood_comparison(hoods)


@app.get("/api/neighborhoods")
def get_neighborhoods(city: Optional[str] = Query(None)):
    slug = city or default_city
    return NEIGHBORHOODS.get(slug, [])


# ============================================================
# Simulation overlay image
# ============================================================

@app.get("/api/simulate/overlay")
def get_simulation_overlay(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: float = Query(500),
    intervention_type: str = Query("moderate"),
    city: Optional[str] = Query(None),
):
    m = get_model(city)
    png_bytes, bounds = m.generate_simulation_png(
        lat=lat, lon=lon, radius_m=radius_m, intervention_type=intervention_type
    )
    if png_bytes is None:
        return Response(content="No cells in area", status_code=404)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Bounds-South": str(bounds["south"]),
            "X-Bounds-North": str(bounds["north"]),
            "X-Bounds-West": str(bounds["west"]),
            "X-Bounds-East": str(bounds["east"]),
        },
    )


@app.get("/api/simulate/overlay/bounds")
def get_simulation_overlay_bounds(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_m: float = Query(500),
    intervention_type: str = Query("moderate"),
    city: Optional[str] = Query(None),
):
    m = get_model(city)
    _, bounds = m.generate_simulation_png(
        lat=lat, lon=lon, radius_m=radius_m, intervention_type=intervention_type
    )
    return bounds or {"error": "No cells"}


# ============================================================
# Smart intervention targeting
# ============================================================

@app.get("/api/priorities")
def get_priority_interventions(
    min_temp_f: float = Query(100, description="Minimum temperature threshold"),
    top_n: int = Query(15, description="Number of priority zones to return"),
    city: Optional[str] = Query(None),
):
    m = get_model(city)
    return m.find_priority_interventions(min_temp_f=min_temp_f, top_n=top_n)
