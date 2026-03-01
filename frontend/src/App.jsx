import React, { useState, useEffect, useCallback, useRef } from 'react'
import { MapContainer, TileLayer, ImageOverlay, useMap, useMapEvents } from 'react-leaflet'
import SidePanel from './components/SidePanel'
import SimulatePanel from './components/SimulatePanel'
import ComparePanel from './components/ComparePanel'

const API_BASE = '/api'
const CHICAGO_CENTER = [41.85, -87.72]

// ============================================================
// Map click handler
// ============================================================

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => onMapClick(e.latlng.lat, e.latlng.lng),
  })
  return null
}

// ============================================================
// Legend
// ============================================================

function MapLegend({ layer }) {
  const configs = {
    temperature: {
      title: 'Surface Temperature',
      stops: [
        { color: '#1e46a0', label: '< 75°F' },
        { color: '#3498db', label: '80' },
        { color: '#46aa8c', label: '85' },
        { color: '#c8d232', label: '92' },
        { color: '#f1c40f', label: '95' },
        { color: '#f0a01e', label: '98' },
        { color: '#e67828', label: '101' },
        { color: '#e74c3c', label: '105' },
        { color: '#b41e1e', label: '110' },
        { color: '#800000', label: '115+' },
      ],
    },
    risk: {
      title: 'Heat Risk Level',
      stops: [
        { color: '#3498db', label: 'Low' },
        { color: '#e67e22', label: 'Moderate' },
        { color: '#e74c3c', label: 'High' },
        { color: '#8b0000', label: 'Extreme' },
      ],
    },
    ndvi: {
      title: 'Vegetation Density (NDVI)',
      stops: [
        { color: '#a0522d', label: 'Bare' },
        { color: '#c89646', label: 'Low' },
        { color: '#b4be50', label: 'Med' },
        { color: '#64b45a', label: 'Good' },
        { color: '#147814', label: 'Dense' },
        { color: '#005a00', label: 'Forest' },
      ],
    },
  }

  const cfg = configs[layer] || configs.temperature

  return (
    <div style={{
      position: 'absolute', bottom: 24, right: 16, zIndex: 1000,
      background: 'rgba(18,18,22,0.92)', backdropFilter: 'blur(12px)',
      borderRadius: 8, padding: '10px 14px',
      border: '1px solid rgba(255,255,255,0.08)',
    }}>
      <div style={{ fontSize: 11, color: '#999', marginBottom: 6, fontWeight: 600 }}>
        {cfg.title}
      </div>
      <div style={{
        display: 'flex', height: 14, borderRadius: 3, overflow: 'hidden',
        background: `linear-gradient(to right, ${cfg.stops.map(s => s.color).join(', ')})`,
        width: cfg.stops.length * 36,
      }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
        {cfg.stops.map((stop, i) => (
          <span key={i} style={{ fontSize: 9, color: '#666' }}>{stop.label}</span>
        ))}
      </div>
    </div>
  )
}

// ============================================================
// Main App
// ============================================================

export default function App() {
  const [mode, setMode] = useState('explore')
  const [layer, setLayer] = useState('temperature')
  const [selectedCell, setSelectedCell] = useState(null)
  const [cityStats, setCityStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [opacity, setOpacity] = useState(0.78)

  // Heat map image URLs and bounds per layer
  const [heatmapUrls, setHeatmapUrls] = useState({})
  const [heatmapBounds, setHeatmapBounds] = useState({})

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)

        // Load city stats
        const statsRes = await fetch(`${API_BASE}/stats`)
        if (!statsRes.ok) throw new Error('Failed to load city stats')
        setCityStats(await statsRes.json())

        // Load heat map bounds and image URLs for each layer
        const layers = ['temperature', 'risk', 'ndvi']
        const boundsMap = {}
        const urlsMap = {}

        for (const l of layers) {
          const boundsRes = await fetch(`${API_BASE}/heatmap/${l}/bounds`)
          if (boundsRes.ok) {
            boundsMap[l] = await boundsRes.json()
            // The image URL — browser will fetch this as a regular image
            urlsMap[l] = `${API_BASE}/heatmap/${l}.png`
          }
        }

        setHeatmapBounds(boundsMap)
        setHeatmapUrls(urlsMap)
        setLoading(false)
      } catch (err) {
        console.error('Load error:', err)
        setError(err.message)
        setLoading(false)
      }
    }
    loadData()
  }, [])

  const handleMapClick = useCallback(async (lat, lon) => {
    try {
      const res = await fetch(`${API_BASE}/cell?lat=${lat}&lon=${lon}`)
      if (res.ok) {
        const detail = await res.json()
        setSelectedCell(detail)
        if (mode !== 'simulate') setMode('explore')
      }
    } catch (err) {
      console.error('Cell detail error:', err)
    }
  }, [mode])

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <h1>TomorrowLand Heat</h1>
        <p>Loading Chicago urban heat data...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="loading-overlay">
        <h1>Connection Error</h1>
        <p style={{ color: '#e74c3c', marginBottom: 12 }}>{error}</p>
        <p style={{ color: '#888' }}>Make sure the API server is running:</p>
        <code style={{
          display: 'block', marginTop: 12, padding: 12,
          background: 'rgba(255,255,255,0.05)', borderRadius: 6,
          color: '#ccc', fontSize: 13
        }}>
          cd api && pip install pillow && uvicorn main:app --reload --port 8000
        </code>
      </div>
    )
  }

  // Current layer bounds for the image overlay
  const bounds = heatmapBounds[layer]
  const imageUrl = heatmapUrls[layer]
  const leafletBounds = bounds
    ? [[bounds.south, bounds.west], [bounds.north, bounds.east]]
    : null

  return (
    <div className="app-container">
      {/* Top bar */}
      <div className="top-bar">
        <div className="logo">
          <h1>TomorrowLand Heat</h1>
          <span>Urban Heat Island Mapper — Chicago</span>
        </div>
        <div className="mode-tabs">
          {['explore', 'compare', 'simulate'].map(m => (
            <button
              key={m}
              className={`mode-tab ${mode === m ? 'active' : ''}`}
              onClick={() => setMode(m)}
            >
              {m === 'explore' ? '🔍 Explore' : m === 'compare' ? '⚖️ Compare' : '🌳 Simulate'}
            </button>
          ))}
        </div>
      </div>

      {/* Map */}
      <div className="map-container">
        <MapContainer
          center={CHICAGO_CENTER}
          zoom={11}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
          minZoom={10}
          maxZoom={16}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
            attribution='&copy; OSM &copy; CARTO'
          />

          {/* Heat map image overlay */}
          {imageUrl && leafletBounds && (
            <ImageOverlay
              url={imageUrl}
              bounds={leafletBounds}
              opacity={opacity}
              className="heat-overlay-img"
            />
          )}

          {/* Labels on top of heat overlay */}
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png"
            pane="overlayPane"
          />

          <MapClickHandler onMapClick={handleMapClick} />
        </MapContainer>
      </div>

      {/* Layer controls */}
      <div className="layer-toggles">
        <button
          className={`layer-btn ${layer === 'temperature' ? 'active' : ''}`}
          onClick={() => setLayer('temperature')}
        >
          🌡️ Temperature
        </button>
        <button
          className={`layer-btn ${layer === 'risk' ? 'active' : ''}`}
          onClick={() => setLayer('risk')}
        >
          ⚠️ Heat Risk
        </button>
        <button
          className={`layer-btn ${layer === 'ndvi' ? 'active' : ''}`}
          onClick={() => setLayer('ndvi')}
        >
          🌿 Vegetation
        </button>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, marginLeft: 8,
          padding: '4px 10px', borderRadius: 20,
          background: 'rgba(18,18,22,0.85)', border: '1px solid rgba(255,255,255,0.1)',
        }}>
          <span style={{ fontSize: 11, color: '#888' }}>Opacity</span>
          <input
            type="range" min={0.15} max={1} step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            style={{ width: 70, accentColor: '#e74c3c' }}
          />
        </div>
      </div>

      {/* Legend */}
      <MapLegend layer={layer} />

      {/* Side panels */}
      {mode === 'explore' && (
        <SidePanel cell={selectedCell} cityStats={cityStats} onClose={() => setSelectedCell(null)} />
      )}
      {mode === 'compare' && <ComparePanel />}
      {mode === 'simulate' && <SimulatePanel selectedCell={selectedCell} />}
    </div>
  )
}