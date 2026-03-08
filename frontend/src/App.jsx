import React, { useState, useEffect, useCallback } from 'react'
import { MapContainer, TileLayer, ImageOverlay, CircleMarker, Popup, Circle, useMap, useMapEvents } from 'react-leaflet'
import SidePanel from './components/SidePanel'
import SimulatePanel from './components/SimulatePanel'
import ComparePanel from './components/ComparePanel'
import PrioritiesPanel from './components/PrioritiesPanel'

const API_BASE = '/api'
const CHICAGO_CENTER = [41.85, -87.72]

const CITIES = [
  { name: 'Chicago', state: 'Illinois', coords: [41.85, -87.72] },
  { name: 'Phoenix', state: 'Arizona', coords: [33.45, -112.07] },
  { name: 'Houston', state: 'Texas', coords: [29.76, -95.37] },
  { name: 'Los Angeles', state: 'California', coords: [34.05, -118.24] },
  { name: 'Atlanta', state: 'Georgia', coords: [33.75, -84.39] },
  { name: 'Miami', state: 'Florida', coords: [25.76, -80.19] },
  { name: 'New York City', state: 'New York', coords: [40.71, -74.01] },
  { name: 'Dallas', state: 'Texas', coords: [32.78, -96.80] },
]

// ============================================================
// City Search
// ============================================================

function CitySearch({ onSelectCity }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)

  const filtered = query.trim()
    ? CITIES.filter(c =>
        `${c.name} ${c.state}`.toLowerCase().includes(query.toLowerCase())
      )
    : CITIES

  return (
    <div className="city-search">
      <input
        type="text"
        placeholder="Search cities..."
        value={query}
        onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        className="city-search-input"
      />
      {open && filtered.length > 0 && (
        <div className="city-search-dropdown">
          {filtered.map((city) => (
            <button
              key={city.name}
              className="city-search-item"
              onMouseDown={() => {
                onSelectCity(city.coords)
                setQuery(city.name)
                setOpen(false)
              }}
            >
              {city.name}, {city.state}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ============================================================
// Map click handler
// ============================================================

function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => onMapClick(e.latlng.lat, e.latlng.lng),
  })
  return null
}

// Fly to a location
function FlyTo({ center, zoom }) {
  const map = useMap()
  useEffect(() => {
    if (center) {
      map.flyTo(center, zoom || 14, { duration: 0.8 })
    }
  }, [center, zoom, map])
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

  // Heat map images
  const [heatmapUrls, setHeatmapUrls] = useState({})
  const [heatmapBounds, setHeatmapBounds] = useState({})

  // Simulation overlay
  const [simOverlay, setSimOverlay] = useState(null)

  // Priority zone markers
  const [priorityZones, setPriorityZones] = useState([])
  const [flyTarget, setFlyTarget] = useState(null)

  // ---- Load initial data ----
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)

        const statsRes = await fetch(`${API_BASE}/stats`)
        if (!statsRes.ok) throw new Error('Failed to load city stats')
        setCityStats(await statsRes.json())

        const layers = ['temperature', 'risk', 'ndvi']
        const boundsMap = {}
        const urlsMap = {}

        for (const l of layers) {
          const boundsRes = await fetch(`${API_BASE}/heatmap/${l}/bounds`)
          if (boundsRes.ok) {
            boundsMap[l] = await boundsRes.json()
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

  // ---- Load priorities when entering priorities mode ----
  useEffect(() => {
    if (mode === 'priorities') {
      fetch(`${API_BASE}/priorities?min_temp_f=100&top_n=15`)
        .then(res => res.json())
        .then(data => setPriorityZones(data))
        .catch(err => console.error('Priorities error:', err))
    } else {
      setPriorityZones([])
    }
  }, [mode])

  // ---- Handlers ----
  const handleMapClick = useCallback(async (lat, lon) => {
    try {
      const res = await fetch(`${API_BASE}/cell?lat=${lat}&lon=${lon}`)
      if (res.ok) {
        const detail = await res.json()
        setSelectedCell(detail)
        if (mode !== 'simulate' && mode !== 'priorities') setMode('explore')
      }
    } catch (err) {
      console.error('Cell detail error:', err)
    }
  }, [mode])

  const handleSimulationResult = useCallback((result) => {
    if (!result) {
      setSimOverlay(null)
      return
    }
    setSimOverlay({
      imageUrl: result.imageUrl,
      bounds: [[result.bounds.south, result.bounds.west], [result.bounds.north, result.bounds.east]],
    })
  }, [])

  const handleSelectZone = useCallback((zone) => {
    setFlyTarget([zone.lat, zone.lon])
    // Also select that cell for detail
    fetch(`${API_BASE}/cell?lat=${zone.lat}&lon=${zone.lon}`)
      .then(res => res.json())
      .then(detail => setSelectedCell(detail))
      .catch(() => {})
  }, [])

  // Clear simulation overlay when leaving simulate mode
  useEffect(() => {
    if (mode !== 'simulate') {
      setSimOverlay(null)
    }
  }, [mode])

  // ---- Render ----
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
          cd api && uvicorn main:app --reload --port 8000
        </code>
      </div>
    )
  }

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
          {[
            { key: 'explore', label: '🔍 Explore' },
            { key: 'compare', label: '⚖️ Compare' },
            { key: 'simulate', label: '🌳 Simulate' },
            { key: 'priorities', label: '🎯 Priorities' },
          ].map(m => (
            <button
              key={m.key}
              className={`mode-tab ${mode === m.key ? 'active' : ''}`}
              onClick={() => setMode(m.key)}
            >
              {m.label}
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
          minZoom={3}
          maxZoom={16}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
            attribution='&copy; OSM &copy; CARTO'
          />

          {/* Base heat map */}
          {imageUrl && leafletBounds && (
            <ImageOverlay
              url={imageUrl}
              bounds={leafletBounds}
              opacity={opacity}
              className="heat-overlay-img"
            />
          )}

          {/* Simulation overlay (shows post-intervention temps) */}
          {simOverlay && (
            <>
              <ImageOverlay
                url={simOverlay.imageUrl}
                bounds={simOverlay.bounds}
                opacity={0.95}
                className="heat-overlay-img"
              />
              <Circle
                center={[
                  (simOverlay.bounds[0][0] + simOverlay.bounds[1][0]) / 2,
                  (simOverlay.bounds[0][1] + simOverlay.bounds[1][1]) / 2,
                ]}
                radius={
                  Math.max(
                    (simOverlay.bounds[1][0] - simOverlay.bounds[0][0]) * 111000 / 2,
                    200
                  )
                }
                pathOptions={{
                  color: '#2ecc71',
                  weight: 2,
                  dashArray: '6 4',
                  fillOpacity: 0,
                }}
              />
            </>
          )}

          {/* Priority zone markers */}
          {priorityZones.map((zone, i) => {
            const isFeasible = zone.feasibility >= 0.5
            return (
              <CircleMarker
                key={i}
                center={[zone.lat, zone.lon]}
                radius={isFeasible ? 10 : 6}
                pathOptions={{
                  color: isFeasible ? '#2ecc71' : '#e74c3c',
                  weight: 2,
                  fillColor: isFeasible ? '#2ecc71' : '#e74c3c',
                  fillOpacity: 0.6,
                }}
                eventHandlers={{
                  click: () => handleSelectZone(zone),
                }}
              >
                <Popup>
                  <div style={{ fontSize: 12, minWidth: 150 }}>
                    <strong>#{i + 1} {zone.land_use_label}</strong><br />
                    Temp: {zone.avg_temp_f}°F<br />
                    Score: {(zone.priority_score * 100).toFixed(0)}/100<br />
                    <em>{zone.recommendation}</em>
                  </div>
                </Popup>
              </CircleMarker>
            )
          })}

          {/* Labels on top */}
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png"
            pane="overlayPane"
          />

          <MapClickHandler onMapClick={handleMapClick} />
          {flyTarget && <FlyTo center={flyTarget} zoom={14} />}
        </MapContainer>
      </div>

      {/* City search */}
      <CitySearch onSelectCity={(coords) => setFlyTarget(coords)} />

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
      {mode === 'simulate' && (
        <SimulatePanel
          selectedCell={selectedCell}
          onSimulationResult={handleSimulationResult}
        />
      )}
      {mode === 'priorities' && (
        <PrioritiesPanel onSelectZone={handleSelectZone} />
      )}
    </div>
  )
}