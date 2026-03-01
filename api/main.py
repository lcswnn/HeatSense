"""
HeatSense — FastAPI Backend
====================================
Serves heat model predictions, grid data, and heat map images.

Usage:
  cd api
  pip install fastapi uvicorn pillow
  uvicorn main:app --reload --port 8000
"""

import os
import sys
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_model import HeatModel

app = FastAPI(
    title="HeatSense API",
    description="Urban heat island prediction and intervention simulation for Chicago",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: HeatModel = None

# Cache generated heat map images
heatmap_cache = {}


@app.on_event("startup")
def load_model():
    global model
    print("\n  Loading HeatSense model...")
    start = time.time()

    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(api_dir)

    model = HeatModel(
        model_dir=os.path.join(project_dir, "model", "models"),
        grid_path=os.path.join(project_dir, "data", "grid", "chicago_grid.csv"),
    )

    # Pre-generate heat map images for each layer
    print("  Generating heat map images...")
    for layer_name in ["temperature", "risk", "ndvi"]:
        png_bytes, bounds = model.generate_heatmap_png(layer=layer_name)
        heatmap_cache[layer_name] = {"png": png_bytes, "bounds": bounds}
        print(f"    {layer_name}: {len(png_bytes) / 1024:.0f} KB")

    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    print(f"  Ready to serve!\n")


# ============================================================
# Request models
# ============================================================

class SimulationRequest(BaseModel):
    lat: float
    lon: float
    radius_m: float = 500
    intervention_type: str = "moderate"

class Neighborhood(BaseModel):
    name: str
    lat: float
    lon: float
    radius_m: float = 1000

class CompareRequest(BaseModel):
    neighborhoods: list[Neighborhood]


# ============================================================
# Endpoints
# ============================================================

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/api/stats")
def get_city_stats():
    return model.get_city_stats()


@app.get("/api/heatmap/{layer}.png")
def get_heatmap_image(layer: str):
    """
    Serve a pre-rendered heat map PNG image.
    Layers: temperature, risk, ndvi
    """
    if layer not in heatmap_cache:
        return Response(content="Invalid layer", status_code=400)

    return Response(
        content=heatmap_cache[layer]["png"],
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/heatmap/{layer}/bounds")
def get_heatmap_bounds(layer: str):
    """Get the geographic bounds for a heat map image overlay."""
    if layer not in heatmap_cache:
        return {"error": "Invalid layer"}
    return heatmap_cache[layer]["bounds"]


@app.get("/api/grid")
def get_grid(
    west: Optional[float] = Query(None),
    south: Optional[float] = Query(None),
    east: Optional[float] = Query(None),
    north: Optional[float] = Query(None),
    downsample: Optional[int] = Query(None),
):
    bbox = None
    if all(v is not None for v in [west, south, east, north]):
        bbox = {"west": west, "south": south, "east": east, "north": north}

    data = model.get_grid_data(bbox=bbox, downsample=downsample)
    return {"count": len(data), "bbox": bbox, "cells": data}


@app.get("/api/cell")
def get_cell_detail(
    lat: float = Query(...),
    lon: float = Query(...),
):
    return model.get_cell_detail(lat, lon)


@app.post("/api/simulate")
def simulate_intervention(request: SimulationRequest):
    return model.simulate_intervention(
        lat=request.lat, lon=request.lon,
        radius_m=request.radius_m,
        intervention_type=request.intervention_type,
    )


@app.post("/api/compare")
def compare_neighborhoods(request: CompareRequest):
    hoods = [{"name": n.name, "lat": n.lat, "lon": n.lon, "radius_m": n.radius_m}
             for n in request.neighborhoods]
    return model.get_neighborhood_comparison(hoods)


@app.get("/api/neighborhoods")
def get_neighborhoods():
    return [
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
    ]