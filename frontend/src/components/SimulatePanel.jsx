import React, { useState, useCallback } from 'react'

const API_BASE = '/api'

const INTERVENTION_LABELS = {
  light: { name: 'Street Trees', emoji: '🌳', desc: 'Plant trees along roads' },
  moderate: { name: 'Pocket Parks', emoji: '🏞️', desc: 'Convert lots to parks + trees' },
  heavy: { name: 'Green Corridor', emoji: '🌲', desc: 'Parks, green roofs, urban forest' },
}

export default function SimulatePanel({ selectedCell }) {
  const [interventionType, setInterventionType] = useState('moderate')
  const [radius, setRadius] = useState(500)
  const [result, setResult] = useState(null)
  const [simulating, setSimulating] = useState(false)

  const runSimulation = useCallback(async () => {
    if (!selectedCell) return

    setSimulating(true)
    try {
      const res = await fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: selectedCell.lat,
          lon: selectedCell.lon,
          radius_m: radius,
          intervention_type: interventionType,
        }),
      })

      if (res.ok) {
        const data = await res.json()
        setResult(data)
      }
    } catch (err) {
      console.error('Simulation error:', err)
    }
    setSimulating(false)
  }, [selectedCell, interventionType, radius])

  return (
    <div className="side-panel">
      <h2>Simulate Intervention</h2>
      <p style={{ fontSize: 13, color: '#888', marginBottom: 16 }}>
        {selectedCell
          ? 'Model the impact of adding green infrastructure to this area.'
          : 'Click a location on the map first, then simulate green infrastructure changes.'}
      </p>

      {selectedCell && (
        <>
          {/* Current location info */}
          <div style={{
            padding: 12, background: 'rgba(255,255,255,0.03)',
            borderRadius: 8, marginBottom: 16
          }}>
            <div style={{ fontSize: 13, color: '#888' }}>Selected Location</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#fff', marginTop: 4 }}>
              {selectedCell.temperature_f}°F
            </div>
            <div style={{ fontSize: 12, color: '#666' }}>
              NDVI: {selectedCell.ndvi} · Impervious: {selectedCell.impervious_pct}%
            </div>
          </div>

          {/* Intervention type selector */}
          <h3>Intervention Type</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 4 }}>
            {Object.entries(INTERVENTION_LABELS).map(([key, { name, emoji, desc }]) => (
              <button
                key={key}
                onClick={() => setInterventionType(key)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '10px 12px', borderRadius: 8, border: 'none',
                  background: interventionType === key
                    ? 'rgba(46,204,113,0.15)'
                    : 'rgba(255,255,255,0.03)',
                  borderLeft: interventionType === key
                    ? '3px solid #2ecc71'
                    : '3px solid transparent',
                  cursor: 'pointer', textAlign: 'left', width: '100%',
                  transition: 'all 0.2s',
                }}
              >
                <span style={{ fontSize: 20 }}>{emoji}</span>
                <div>
                  <div style={{
                    fontSize: 13, fontWeight: 600,
                    color: interventionType === key ? '#2ecc71' : '#ccc'
                  }}>
                    {name}
                  </div>
                  <div style={{ fontSize: 11, color: '#666' }}>{desc}</div>
                </div>
              </button>
            ))}
          </div>

          {/* Radius slider */}
          <div className="simulation-controls">
            <div className="sim-slider-group">
              <div className="sim-slider-label">
                <span style={{ color: '#888' }}>Effect Radius</span>
                <span style={{ color: '#fff', fontWeight: 600 }}>{radius}m</span>
              </div>
              <input
                type="range"
                className="sim-slider"
                min={200}
                max={2000}
                step={100}
                value={radius}
                onChange={(e) => setRadius(Number(e.target.value))}
              />
              <div className="bar-label">
                <span>200m (2 blocks)</span>
                <span>2km (neighborhood)</span>
              </div>
            </div>
          </div>

          {/* Run button */}
          <button
            onClick={runSimulation}
            disabled={simulating}
            style={{
              width: '100%', padding: '12px 0', marginTop: 16,
              border: 'none', borderRadius: 8,
              background: simulating ? '#444' : '#2ecc71',
              color: '#fff', fontSize: 14, fontWeight: 600,
              cursor: simulating ? 'wait' : 'pointer',
              transition: 'all 0.2s',
            }}
          >
            {simulating ? 'Running Simulation...' : 'Run Simulation'}
          </button>

          {/* Results */}
          {result && (
            <>
              <div className="result-card" style={{ marginTop: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                  <div>
                    <div className="big-number">-{result.avg_cooling_f}°F</div>
                    <div className="big-label">Average Cooling</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: 18, fontWeight: 600, color: '#fff' }}>
                      {result.after_avg_temp_f}°F
                    </div>
                    <div style={{ fontSize: 11, color: '#888' }}>
                      from {result.before_avg_temp_f}°F
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 12 }}>
                <div className="stat-row">
                  <span className="stat-label">Cells Affected</span>
                  <span className="stat-value">{result.cells_affected}</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Max Cooling</span>
                  <span className="stat-value green">-{result.max_cooling_f}°F</span>
                </div>
                <div className="stat-row">
                  <span className="stat-label">Intervention</span>
                  <span className="stat-value">{result.intervention}</span>
                </div>
              </div>

              <div style={{
                marginTop: 16, padding: 12,
                background: 'rgba(255,255,255,0.03)',
                borderRadius: 8, fontSize: 12, color: '#666', lineHeight: 1.5
              }}>
                {result.description}. This simulation adjusts vegetation, impervious surface,
                and park proximity within {result.radius_m}m of the selected point.
                Estimates based on LightGBM model trained on Landsat satellite data.
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}