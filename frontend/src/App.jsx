import React, { useState, useEffect, useCallback } from 'react'
import { MapContainer, TileLayer, useMap, Rectangle, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import SidePanel from './components/SidePanel'
import SimulatePanel from './components/SimulatePanel'
import ComparePanel from './components/ComparePanel'

const API_BASE = '/api'

// Chicago center
const CHICAGO_CENTER = [41.85, -87.72]
const CHICAGO_BOUNDS = [
  [41.6445, -87.9401],
  [42.023, -87.5241],
]

// Color scale for temperature
function tempToColor(temp, opacity = 0.7) {
  if (temp == null) return `rgba(128,128,128,${opacity})`
  if (temp >= 115) return `rgba(139,0,0,${opacity})`
  if (temp >= 110) return `rgba(192,57,43,${opacity})`
  if (temp >= 105) return `rgba(231,76,60,${opacity})`
  if (temp >= 100) return `rgba(230,126,34,${opacity})`
  if (temp >= 95) return `rgba(241,196,15,${opacity})`
  if (temp >= 90) return `rgba(243,156,18,${opacity})`
  if (temp >= 85) return `rgba(52,152,219,${opacity})`
  return `rgba(41,128,185,${opacity})`
}

function riskToColor(risk, opacity = 0.7) {
  const colors = {
    extreme: `rgba(139,0,0,${opacity})`,
    high: `rgba(231,76,60,${opacity})`,
    moderate: `rgba(230,126,34,${opacity})`,
    low: `rgba(52,152,219,${opacity})`,
    no_data: `rgba(128,128,128,0.2)`,
  }
  return colors[risk] || colors.no_data
}

function ndviToColor(ndvi, opacity = 0.7) {
  if (ndvi == null) return `rgba(128,128,128,${opacity})`
  if (ndvi >= 0.6) return `rgba(0,100,0,${opacity})`
  if (ndvi >= 0.4) return `rgba(34,139,34,${opacity})`
  if (ndvi >= 0.3) return `rgba(46,204,113,${opacity})`
  if (ndvi >= 0.2) return `rgba(144,238,144,${opacity})`
  if (ndvi >= 0.1) return `rgba(222,184,135,${opacity})`
  return `rgba(160,82,45,${opacity})`
}

// Grid layer component that renders heat map cells
function HeatGridLayer({ cells, layer, onCellClick }) {
  const map = useMap()

  useEffect(() => {
    if (!cells || cells.length === 0) return

    const cellSize = 0.001 // ~100m in degrees

    const rectangles = cells.map(cell => {
      let color
      if (layer === 'temperature') {
        color = tempToColor(cell.mean_lst_f)
      } else if (layer === 'risk') {
        color = riskToColor(cell.heat_risk)
      } else if (layer === 'ndvi') {
        color = ndviToColor(cell.ndvi)
      } else {
        color = tempToColor(cell.mean_lst_f)
      }

      const bounds = [
        [cell.lat - cellSize / 2, cell.lon - cellSize / 2],
        [cell.lat + cellSize / 2, cell.lon + cellSize / 2],
      ]

      const rect = L.rectangle(bounds, {
        color: 'transparent',
        fillColor: color,
        fillOpacity: 1,
        weight: 0,
      })

      rect.on('click', () => onCellClick(cell))
      return rect
    })

    const layerGroup = L.layerGroup(rectangles)
    layerGroup.addTo(map)

    return () => {
      map.removeLayer(layerGroup)
    }
  }, [cells, layer, map, onCellClick])

  return null
}

// Click handler component
function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => {
      onMapClick(e.latlng.lat, e.latlng.lng)
    },
  })
  return null
}

export default function App() {
  const [mode, setMode] = useState('explore') // explore, compare, simulate
  const [layer, setLayer] = useState('temperature')
  const [cells, setCells] = useState([])
  const [selectedCell, setSelectedCell] = useState(null)
  const [cityStats, setCityStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Load initial data
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)

        // Fetch city stats
        const statsRes = await fetch(`${API_BASE}/stats`)
        if (!statsRes.ok) throw new Error('Failed to load stats')
        const stats = await statsRes.json()
        setCityStats(stats)

        // Fetch grid data (downsampled for initial load)
        const gridRes = await fetch(`${API_BASE}/grid?downsample=3`)
        if (!gridRes.ok) throw new Error('Failed to load grid')
        const gridData = await gridRes.json()
        setCells(gridData.cells)

        setLoading(false)
      } catch (err) {
        console.error('Load error:', err)
        setError(err.message)
        setLoading(false)
      }
    }
    loadData()
  }, [])

  // Handle cell click
  const handleCellClick = useCallback(async (cell) => {
    try {
      const res = await fetch(`${API_BASE}/cell?lat=${cell.lat}&lon=${cell.lon}`)
      if (res.ok) {
        const detail = await res.json()
        setSelectedCell(detail)
      }
    } catch (err) {
      console.error('Cell detail error:', err)
    }
  }, [])

  // Handle map click (for areas without pre-loaded cells)
  const handleMapClick = useCallback(async (lat, lon) => {
    try {
      const res = await fetch(`${API_BASE}/cell?lat=${lat}&lon=${lon}`)
      if (res.ok) {
        const detail = await res.json()
        setSelectedCell(detail)
      }
    } catch (err) {
      console.error('Map click error:', err)
    }
  }, [])

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <h1>TomorrowLand Heat</h1>
        <p>Loading Chicago heat data...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="loading-overlay">
        <h1>Connection Error</h1>
        <p style={{ color: '#e74c3c', marginBottom: 12 }}>{error}</p>
        <p>Make sure the API server is running:</p>
        <p style={{ color: '#888', fontFamily: 'monospace', marginTop: 8 }}>
          cd api && uvicorn main:app --reload --port 8000
        </p>
      </div>
    )
  }

  return (
    <div className="app-container">
      {/* Top Bar */}
      <div className="top-bar">
        <div className="logo">
          <h1>TomorrowLand Heat</h1>
          <span>Urban Heat Island Mapper — Chicago</span>
        </div>
        <div className="mode-tabs">
          <button
            className={`mode-tab ${mode === 'explore' ? 'active' : ''}`}
            onClick={() => setMode('explore')}
          >
            Explore
          </button>
          <button
            className={`mode-tab ${mode === 'compare' ? 'active' : ''}`}
            onClick={() => setMode('compare')}
          >
            Compare
          </button>
          <button
            className={`mode-tab ${mode === 'simulate' ? 'active' : ''}`}
            onClick={() => setMode('simulate')}
          >
            Simulate
          </button>
        </div>
      </div>

      {/* Map */}
      <div className="map-container">
        <MapContainer
          center={CHICAGO_CENTER}
          zoom={11}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
          maxBounds={CHICAGO_BOUNDS}
          minZoom={10}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>'
          />
          <HeatGridLayer
            cells={cells}
            layer={layer}
            onCellClick={handleCellClick}
          />
          <MapClickHandler onMapClick={handleMapClick} />
        </MapContainer>
      </div>

      {/* Layer toggles */}
      <div className="layer-toggles">
        <button
          className={`layer-btn ${layer === 'temperature' ? 'active' : ''}`}
          onClick={() => setLayer('temperature')}
        >
          Temperature
        </button>
        <button
          className={`layer-btn ${layer === 'risk' ? 'active' : ''}`}
          onClick={() => setLayer('risk')}
        >
          Heat Risk
        </button>
        <button
          className={`layer-btn ${layer === 'ndvi' ? 'active' : ''}`}
          onClick={() => setLayer('ndvi')}
        >
          Vegetation
        </button>
      </div>

      {/* Side Panel — changes based on mode */}
      {mode === 'explore' && (
        <SidePanel
          cell={selectedCell}
          cityStats={cityStats}
          onClose={() => setSelectedCell(null)}
        />
      )}
      {mode === 'compare' && (
        <ComparePanel />
      )}
      {mode === 'simulate' && (
        <SimulatePanel selectedCell={selectedCell} />
      )}
    </div>
  )
}