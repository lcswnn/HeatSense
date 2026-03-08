import React, { useState, useEffect, useCallback } from 'react'

const API_BASE = '/api'

export default function ComparePanel({ activeCity }) {
  const [neighborhoods, setNeighborhoods] = useState([])
  const [selected, setSelected] = useState([])
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)

  // Load neighborhood list
  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${API_BASE}/neighborhoods?city=${activeCity}`)
        if (res.ok) {
          const data = await res.json()
          setNeighborhoods(data)
        }
      } catch (err) {
        console.error('Failed to load neighborhoods:', err)
      }
    }
    load()
  }, [activeCity])

  // Auto-select first two neighborhoods when data loads
  useEffect(() => {
    if (neighborhoods.length >= 2) {
      setSelected([neighborhoods[0].name, neighborhoods[1].name])
    }
  }, [neighborhoods])

  // Run comparison
  const runComparison = useCallback(async () => {
    if (selected.length < 2) return

    setLoading(true)
    try {
      const hoods = selected.map(name => {
        const hood = neighborhoods.find(n => n.name === name)
        return hood ? { name: hood.name, lat: hood.lat, lon: hood.lon, radius_m: 1000 } : null
      }).filter(Boolean)

      const res = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ neighborhoods: hoods, city: activeCity }),
      })

      if (res.ok) {
        const data = await res.json()
        setComparison(data)
      }
    } catch (err) {
      console.error('Compare error:', err)
    }
    setLoading(false)
  }, [selected, neighborhoods])

  // Auto-run when selection changes and we have neighborhood data
  useEffect(() => {
    if (neighborhoods.length > 0 && selected.length >= 2) {
      runComparison()
    }
  }, [selected, neighborhoods, runComparison])

  const handleSelect = (index, name) => {
    const newSelected = [...selected]
    newSelected[index] = name
    setSelected(newSelected)
  }

  return (
    <div className="side-panel">
      <h2>Compare Neighborhoods</h2>
      <p style={{ fontSize: 13, color: '#888', marginBottom: 16 }}>
        See how heat exposure differs across neighborhoods.
      </p>

      {/* Selectors */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {[0, 1].map(i => (
          <select
            key={i}
            value={selected[i]}
            onChange={(e) => handleSelect(i, e.target.value)}
            style={{
              flex: 1, padding: '8px 10px', borderRadius: 6,
              background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.1)',
              color: '#fff', fontSize: 13,
            }}
          >
            {neighborhoods.map(n => (
              <option key={n.name} value={n.name} style={{ background: '#1a1a1a' }}>
                {n.name}
              </option>
            ))}
          </select>
        ))}
      </div>

      {/* Comparison results */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 20, color: '#666' }}>
          Loading comparison...
        </div>
      )}

      {comparison && comparison.length >= 2 && (
        <>
          {/* Temperature comparison hero */}
          <div style={{
            display: 'flex', gap: 12, marginBottom: 20
          }}>
            {comparison.map((hood, i) => (
              <div key={hood.name} style={{
                flex: 1, padding: 14, borderRadius: 8, textAlign: 'center',
                background: i === 0 ? 'rgba(231,76,60,0.1)' : 'rgba(52,152,219,0.1)',
                border: `1px solid ${i === 0 ? 'rgba(231,76,60,0.2)' : 'rgba(52,152,219,0.2)'}`,
              }}>
                <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>
                  {hood.name}
                </div>
                <div style={{
                  fontSize: 28, fontWeight: 700,
                  color: hood.avg_temp_f >= 100 ? '#e74c3c' : '#3498db'
                }}>
                  {hood.avg_temp_f}°F
                </div>
              </div>
            ))}
          </div>

          {/* Temperature difference */}
          {comparison.length >= 2 && (
            <div style={{
              textAlign: 'center', padding: 10, marginBottom: 16,
              background: 'rgba(255,255,255,0.03)', borderRadius: 8
            }}>
              <span style={{ fontSize: 13, color: '#888' }}>
                {comparison[0].name} is{' '}
              </span>
              <span style={{
                fontSize: 16, fontWeight: 700,
                color: comparison[0].avg_temp_f > comparison[1].avg_temp_f ? '#e74c3c' : '#3498db'
              }}>
                {Math.abs(comparison[0].avg_temp_f - comparison[1].avg_temp_f).toFixed(1)}°F
                {comparison[0].avg_temp_f > comparison[1].avg_temp_f ? ' hotter' : ' cooler'}
              </span>
              <span style={{ fontSize: 13, color: '#888' }}>
                {' '}than {comparison[1].name}
              </span>
            </div>
          )}

          {/* Detailed comparison bars */}
          <CompareMetric
            label="Vegetation (NDVI)"
            values={comparison.map(h => h.avg_ndvi)}
            names={comparison.map(h => h.name)}
            max={0.8}
            format={v => v.toFixed(3)}
            goodDirection="high"
            colorHigh="#2ecc71"
            colorLow="#e74c3c"
          />
          <CompareMetric
            label="Impervious Surface"
            values={comparison.map(h => h.avg_impervious_pct)}
            names={comparison.map(h => h.name)}
            max={100}
            format={v => `${v.toFixed(0)}%`}
            goodDirection="low"
            colorHigh="#e74c3c"
            colorLow="#2ecc71"
          />
          <CompareMetric
            label="Building Density"
            values={comparison.map(h => h.avg_building_density * 100)}
            names={comparison.map(h => h.name)}
            max={60}
            format={v => `${v.toFixed(0)}%`}
            goodDirection="low"
            colorHigh="#e67e22"
            colorLow="#3498db"
          />
          <CompareMetric
            label="Avg Distance to Park"
            values={comparison.map(h => h.avg_distance_to_park_m)}
            names={comparison.map(h => h.name)}
            max={1500}
            format={v => `${Math.round(v)}m`}
            goodDirection="low"
            colorHigh="#e74c3c"
            colorLow="#2ecc71"
          />

          {/* Heat risk breakdown */}
          <h3 style={{ marginTop: 16 }}>Heat Risk Breakdown</h3>
          {comparison.map(hood => (
            <div key={hood.name} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>{hood.name}</div>
              <div style={{ display: 'flex', height: 20, borderRadius: 4, overflow: 'hidden' }}>
                {['extreme', 'high', 'moderate', 'low'].map(risk => {
                  const pct = hood.heat_risk_pct?.[risk] || 0
                  if (pct === 0) return null
                  const colors = {
                    extreme: '#8b0000',
                    high: '#e74c3c',
                    moderate: '#e67e22',
                    low: '#3498db',
                  }
                  return (
                    <div
                      key={risk}
                      style={{
                        width: `${pct}%`,
                        background: colors[risk],
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 9, color: '#fff', fontWeight: 600,
                      }}
                      title={`${risk}: ${pct}%`}
                    >
                      {pct > 10 ? `${pct}%` : ''}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  )
}

// Reusable comparison metric bar
function CompareMetric({ label, values, names, max, format, goodDirection, colorHigh, colorLow }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 6 }}>{label}</div>
      {values.map((val, i) => {
        const pct = Math.min((val / max) * 100, 100)
        const isWorse = goodDirection === 'low' ? val > values[1 - i] : val < values[1 - i]
        const barColor = isWorse ? colorHigh : colorLow

        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 11, color: '#666', width: 80, textAlign: 'right' }}>
              {names[i]}
            </span>
            <div style={{ flex: 1, height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4 }}>
              <div style={{
                width: `${pct}%`, height: '100%', borderRadius: 4,
                background: barColor, transition: 'width 0.5s ease'
              }} />
            </div>
            <span style={{ fontSize: 11, color: '#ccc', width: 50, fontWeight: 600 }}>
              {format(val)}
            </span>
          </div>
        )
      })}
    </div>
  )
}