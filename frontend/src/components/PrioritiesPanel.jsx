import React, { useState, useEffect, useCallback } from 'react'

const API_BASE = '/api'

const LAND_USE_COLORS = {
  'residential': '#2ecc71',
  'vacant_open': '#27ae60',
  'commercial_industrial': '#e67e22',
  'commercial_highrise': '#f39c12',
  'mixed_urban': '#3498db',
  'highway_rail': '#95a5a6',
  'airport_runway': '#7f8c8d',
}

const FEASIBILITY_LABELS = {
  'airport_runway': '🚫 Not Feasible',
  'highway_rail': '⚠️ Very Limited',
  'commercial_highrise': '🏢 Green Roofs',
  'commercial_industrial': '🅿️ Moderate',
  'residential': '✅ High Potential',
  'vacant_open': '🌟 Ideal',
  'mixed_urban': '🔧 Moderate',
}

export default function PrioritiesPanel({ onSelectZone }) {
  const [priorities, setPriorities] = useState(null)
  const [loading, setLoading] = useState(false)
  const [minTemp, setMinTemp] = useState(100)

  const loadPriorities = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/priorities?min_temp_f=${minTemp}&top_n=15`)
      if (res.ok) {
        const data = await res.json()
        setPriorities(data)
      }
    } catch (err) {
      console.error('Priority load error:', err)
    }
    setLoading(false)
  }, [minTemp])

  useEffect(() => {
    loadPriorities()
  }, [loadPriorities])

  // Summary stats
  const feasibleCount = priorities?.filter(p => p.feasibility >= 0.5).length || 0
  const totalCells = priorities?.reduce((sum, p) => sum + p.cell_count, 0) || 0

  return (
    <div className="side-panel">
      <h2>🎯 Priority Interventions</h2>
      <p style={{ fontSize: 13, color: '#888', marginBottom: 12 }}>
        AI-analyzed hot zones ranked by intervention potential.
        Considers temperature, vegetation need, land use feasibility, and park access.
      </p>

      {/* Temperature threshold */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
          <span style={{ color: '#888' }}>Min Temperature Threshold</span>
          <span style={{ color: '#fff', fontWeight: 600 }}>{minTemp}°F</span>
        </div>
        <input
          type="range"
          className="sim-slider"
          min={90} max={115} step={5}
          value={minTemp}
          onChange={(e) => setMinTemp(Number(e.target.value))}
        />
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: 20, color: '#666' }}>
          Analyzing hot zones...
        </div>
      )}

      {priorities && !loading && (
        <>
          {/* Summary */}
          <div style={{
            display: 'flex', gap: 8, marginBottom: 16,
          }}>
            <div style={{
              flex: 1, padding: 10, borderRadius: 8, textAlign: 'center',
              background: 'rgba(46,204,113,0.1)', border: '1px solid rgba(46,204,113,0.15)',
            }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#2ecc71' }}>
                {feasibleCount}
              </div>
              <div style={{ fontSize: 10, color: '#888' }}>Feasible Zones</div>
            </div>
            <div style={{
              flex: 1, padding: 10, borderRadius: 8, textAlign: 'center',
              background: 'rgba(231,76,60,0.1)', border: '1px solid rgba(231,76,60,0.15)',
            }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#e74c3c' }}>
                {totalCells}
              </div>
              <div style={{ fontSize: 10, color: '#888' }}>Hot Cells Found</div>
            </div>
            <div style={{
              flex: 1, padding: 10, borderRadius: 8, textAlign: 'center',
              background: 'rgba(52,152,219,0.1)', border: '1px solid rgba(52,152,219,0.15)',
            }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#3498db' }}>
                {priorities.length}
              </div>
              <div style={{ fontSize: 10, color: '#888' }}>Zones Analyzed</div>
            </div>
          </div>

          {/* Priority list */}
          {priorities.map((zone, i) => {
            const isFeasible = zone.feasibility >= 0.5
            const color = LAND_USE_COLORS[zone.land_use] || '#888'

            return (
              <div
                key={i}
                onClick={() => onSelectZone?.(zone)}
                style={{
                  padding: 12, marginBottom: 8, borderRadius: 8,
                  background: 'rgba(255,255,255,0.03)',
                  border: `1px solid ${isFeasible ? 'rgba(46,204,113,0.15)' : 'rgba(255,255,255,0.05)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  opacity: isFeasible ? 1 : 0.6,
                }}
              >
                {/* Header row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      display: 'inline-block', width: 20, height: 20, borderRadius: 4,
                      background: color, textAlign: 'center', lineHeight: '20px',
                      fontSize: 11, fontWeight: 700, color: '#fff',
                    }}>
                      {i + 1}
                    </span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#fff' }}>
                      {zone.land_use_label}
                    </span>
                  </div>
                  <span style={{
                    fontSize: 16, fontWeight: 700,
                    color: zone.avg_temp_f >= 110 ? '#e74c3c' : zone.avg_temp_f >= 105 ? '#e67e22' : '#f1c40f',
                  }}>
                    {zone.avg_temp_f}°F
                  </span>
                </div>

                {/* Feasibility badge */}
                <div style={{ fontSize: 11, marginBottom: 6 }}>
                  <span style={{ color: isFeasible ? '#2ecc71' : '#e74c3c' }}>
                    {FEASIBILITY_LABELS[zone.land_use] || '🔧 Moderate'}
                  </span>
                  <span style={{ color: '#555', marginLeft: 8 }}>
                    Score: {(zone.priority_score * 100).toFixed(0)}/100
                  </span>
                </div>

                {/* Stats row */}
                <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#888' }}>
                  <span>NDVI: {zone.avg_ndvi}</span>
                  <span>Imperv: {zone.avg_impervious_pct}%</span>
                  <span>Park: {Math.round(zone.distance_to_park_m)}m</span>
                </div>

                {/* Recommendation */}
                <div style={{
                  marginTop: 6, fontSize: 11, color: '#aaa',
                  fontStyle: 'italic',
                }}>
                  → {zone.recommendation}
                </div>
              </div>
            )
          })}

          <div style={{
            marginTop: 12, padding: 10,
            background: 'rgba(255,255,255,0.02)', borderRadius: 6,
            fontSize: 11, color: '#555', lineHeight: 1.5,
          }}>
            Land use is inferred from building density, impervious surface, road density,
            and vegetation patterns. Click a zone to view it on the map and run a simulation.
          </div>
        </>
      )}
    </div>
  )
}
