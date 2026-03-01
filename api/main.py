"""
TomorrowLand Heat — FastAPI Backend
====================================
Serves the heat model predictions and grid data to the React frontend.

Endpoints:
  GET  /api/health              — Health check
  GET  /api/stats               — City-wide summary statistics
  GET  /api/grid                — Grid data (with optional bbox filter)
  GET  /api/cell                — Detailed info for a specific location
  POST /api/simulate            — Run intervention simulation
  POST /api/compare             — Compare two neighborhoods

Usage:
  cd api
  uvicorn main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API documentation.
"""

import os
import sys
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

# Add parent directory to path so we can find model/data directories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat_model import HeatModel


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="TomorrowLand Heat API",
    description="Urban heat island prediction and intervention simulation for Chicago",
    version="1.0.0",
)

# Allow frontend to talk to backend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alternative
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model: HeatModel = None


@app.on_event("startup")
def load_model():
    global model
    print("\n  Loading TomorrowLand Heat model...")
    start = time.time()

    # Resolve paths relative to api/ directory
    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(api_dir)

    model = HeatModel(
        model_dir=os.path.join(project_dir, "model", "models"),
        grid_path=os.path.join(project_dir, "data", "grid", "chicago_grid.csv"),
    )

    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    print(f"  Ready to serve predictions!\n")


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class SimulationRequest(BaseModel):
    lat: float
    lon: float
    radius_m: float = 500
    intervention_type: str = "moderate"  # light, moderate, heavy


class Neighborhood(BaseModel):
    name: str
    lat: float
    lon: float
    radius_m: float = 1000


class CompareRequest(BaseModel):
    neighborhoods: list[Neighborhood]


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "grid_cells": len(model.grid_valid) if model else 0,
    }


@app.get("/api/stats")
def get_city_stats():
    """Get city-wide summary statistics."""
    return model.get_city_stats()


@app.get("/api/grid")
def get_grid(
    west: Optional[float] = Query(None, description="Western bound longitude"),
    south: Optional[float] = Query(None, description="Southern bound latitude"),
    east: Optional[float] = Query(None, description="Eastern bound longitude"),
    north: Optional[float] = Query(None, description="Northern bound latitude"),
    downsample: Optional[int] = Query(None, description="Return every Nth cell"),
):
    """
    Get grid data, optionally filtered to a bounding box.
    Use downsample parameter for zoomed-out views to reduce payload size.

    Example: /api/grid?west=-87.8&south=41.8&east=-87.6&north=41.9
    """
    bbox = None
    if all(v is not None for v in [west, south, east, north]):
        bbox = {"west": west, "south": south, "east": east, "north": north}

    data = model.get_grid_data(bbox=bbox, downsample=downsample)

    return {
        "count": len(data),
        "bbox": bbox,
        "cells": data,
    }


@app.get("/api/cell")
def get_cell_detail(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
):
    """
    Get detailed information for the grid cell nearest to a lat/lon point.

    Example: /api/cell?lat=41.85&lon=-87.75
    """
    return model.get_cell_detail(lat, lon)


@app.post("/api/simulate")
def simulate_intervention(request: SimulationRequest):
    """
    Simulate a green infrastructure intervention.

    Intervention types:
    - "light": Street trees along main roads
    - "moderate": Pocket parks + tree planting
    - "heavy": Major green infrastructure (parks, green roofs, urban forest)

    Example body:
    {
        "lat": 41.85,
        "lon": -87.72,
        "radius_m": 500,
        "intervention_type": "moderate"
    }
    """
    return model.simulate_intervention(
        lat=request.lat,
        lon=request.lon,
        radius_m=request.radius_m,
        intervention_type=request.intervention_type,
    )


@app.post("/api/compare")
def compare_neighborhoods(request: CompareRequest):
    """
    Compare statistics across multiple neighborhoods.

    Example body:
    {
        "neighborhoods": [
            {"name": "Englewood", "lat": 41.7798, "lon": -87.6456, "radius_m": 1000},
            {"name": "Lincoln Park", "lat": 41.9214, "lon": -87.6513, "radius_m": 1000}
        ]
    }
    """
    hoods = [{"name": n.name, "lat": n.lat, "lon": n.lon, "radius_m": n.radius_m}
             for n in request.neighborhoods]
    return model.get_neighborhood_comparison(hoods)


# ============================================================
# PREDEFINED NEIGHBORHOOD DATA
# ============================================================

@app.get("/api/neighborhoods")
def get_neighborhoods():
    """Get list of predefined Chicago neighborhoods with coordinates."""
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