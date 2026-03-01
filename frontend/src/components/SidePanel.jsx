import React from 'react'

function getRiskClass(risk) {
  if (risk === 'extreme') return 'extreme'
  if (risk === 'high') return 'high'
  if (risk === 'moderate') return 'moderate'
  return 'low'
}

function getTempClass(temp) {
  if (temp >= 105) return 'hot'
  if (temp >= 95) return 'warm'
  return 'cool'
}

export default function SidePanel({ cell, cityStats, onClose }) {
  // Show city overview when no cell selected
  if (!cell) {
    return (
      <div className="side-panel">
        <h2>Chicago Heat Overview</h2>
        <p style={{ fontSize: 13, color: '#888', marginBottom: 16 }}>
          Click anywhere on the map to see detailed heat data for that location.
        </p>

        {cityStats && (
          <>
            <h3>City Statistics</h3>
            <div className="stat-row">
              <span className="stat-label">Avg Surface Temp</span>
              <span className={`stat-value ${getTempClass(cityStats.avg_temp_f)}`}>
                {cityStats.avg_temp_f}°F
              </span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Hottest Cell</span>
              <span className="stat-value hot">{cityStats.max_temp_f}°F</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Coolest Cell</span>
              <span className="stat-value cool">{cityStats.min_temp_f}°F</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Avg Vegetation (NDVI)</span>
              <span className="stat-value green">{cityStats.avg_ndvi}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">Avg Impervious Surface</span>
              <span className="stat-value">{cityStats.avg_impervious_pct}%</span>
            </div>

            <h3>Heat Risk Distribution</h3>
            {cityStats.heat_risk_distribution &&
              ['extreme', 'high', 'moderate', 'low'].map(risk => {
                const data = cityStats.heat_risk_distribution[risk]
                if (!data) return null
                return (
                  <div key={risk} className="comparison-bar" style={{ marginBottom: 10 }}>
                    <div className="stat-row" style={{ padding: '4px 0', borderBottom: 'none' }}>
                      <span className="stat-label" style={{ textTransform: 'capitalize' }}>
                        {risk}
                      </span>
                      <span className={`stat-value`}>{data.pct}%</span>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{
                          width: `${data.pct}%`,
                          background: risk === 'extreme' ? '#8b0000'
                            : risk === 'high' ? '#e74c3c'
                            : risk === 'moderate' ? '#e67e22'
                            : '#3498db'
                        }}
                      />
                    </div>
                  </div>
                )
              })
            }

            <div style={{ marginTop: 20, padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
              <p style={{ fontSize: 12, color: '#999', fontWeight: 600, marginBottom: 4 }}>
                About the data
              </p>
              <p style={{ fontSize: 11, color: '#666', lineHeight: 1.6 }}>
                Temperatures shown are <strong style={{ color: '#aaa' }}>land surface temperatures</strong> measured
                by Landsat 8/9 satellites during summer months (2022–2025). Surface temps are typically
                15–30°F higher than the air temperature you feel, but they drive the urban heat island
                effect — hotter surfaces radiate more heat into surrounding neighborhoods.
              </p>
              <p style={{ fontSize: 11, color: '#666', lineHeight: 1.6, marginTop: 6 }}>
                Vegetation data from Sentinel-2. Building and road data from OpenStreetMap.
              </p>
            </div>
          </>
        )}
      </div>
    )
  }

  // Show cell detail
  return (
    <div className="side-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <h2>Location Detail</h2>
        <button
          onClick={onClose}
          style={{
            background: 'none', border: 'none', color: '#666',
            cursor: 'pointer', fontSize: 18, padding: '0 4px'
          }}
        >
          ✕
        </button>
      </div>

      {/* Temperature hero */}
      <div style={{
        textAlign: 'center', padding: '16px 0', marginBottom: 12,
        background: 'rgba(255,255,255,0.03)', borderRadius: 8
      }}>
        <div style={{ fontSize: 42, fontWeight: 700, color: '#fff' }}>
          {cell.temperature_f}°F
        </div>
        <div style={{ fontSize: 13, color: '#888', marginTop: 4 }}>
          Land Surface Temperature
        </div>
        <div style={{ marginTop: 8 }}>
          <span className={`temp-badge ${getRiskClass(cell.heat_risk)}`}>
            {cell.heat_risk?.toUpperCase()} RISK
          </span>
        </div>
        <div style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
          {cell.vs_city_avg_f > 0 ? '+' : ''}{cell.vs_city_avg_f}°F vs city average
          {' · '}Hotter than {cell.temp_percentile}% of Chicago
        </div>
      </div>

      {/* Surface temp explainer */}
      <div style={{
        padding: 10, marginBottom: 12, borderRadius: 8,
        background: 'rgba(241, 196, 15, 0.06)',
        border: '1px solid rgba(241, 196, 15, 0.12)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <span style={{ fontSize: 12, color: '#999' }}>Est. Air Temperature</span>
          <span style={{ fontSize: 16, fontWeight: 700, color: '#f1c40f' }}>
            ~{Math.round(cell.temperature_f - 20)}–{Math.round(cell.temperature_f - 15)}°F
          </span>
        </div>
        <p style={{ fontSize: 11, color: '#666', lineHeight: 1.5, margin: 0 }}>
          Surface temperature is measured by Landsat satellites looking down at rooftops and
          pavement. Surfaces are typically 15–30°F hotter than the air you feel.
          Areas with higher surface temps radiate more heat, making nearby air temperatures
          warmer — this is the urban heat island effect.
        </p>
      </div>

      {/* Coordinates */}
      <div style={{ fontSize: 11, color: '#555', marginBottom: 12, textAlign: 'center' }}>
        {cell.lat}, {cell.lon}
      </div>

      {/* Stats */}
      <h3>Vegetation & Surface</h3>
      <div className="stat-row">
        <span className="stat-label">Vegetation (NDVI)</span>
        <span className={`stat-value ${cell.ndvi >= 0.3 ? 'green' : cell.ndvi >= 0.15 ? 'warm' : 'hot'}`}>
          {cell.ndvi}
        </span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Impervious Surface</span>
        <span className="stat-value">{cell.impervious_pct}%</span>
      </div>

      <h3>Built Environment</h3>
      <div className="stat-row">
        <span className="stat-label">Building Density</span>
        <span className="stat-value">{(cell.building_density * 100).toFixed(0)}%</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Avg Building Height</span>
        <span className="stat-value">{cell.avg_building_height_m}m</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Road Density</span>
        <span className="stat-value">{cell.road_density_km} km/cell</span>
      </div>

      <h3>Cooling Resources</h3>
      <div className="stat-row">
        <span className="stat-label">Distance to Park</span>
        <span className={`stat-value ${cell.distance_to_park_m > 500 ? 'warm' : 'green'}`}>
          {Math.round(cell.distance_to_park_m)}m
        </span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Distance to Water</span>
        <span className={`stat-value ${cell.distance_to_water_m > 2000 ? 'warm' : 'cool'}`}>
          {Math.round(cell.distance_to_water_m)}m
        </span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Park Coverage</span>
        <span className="stat-value">{(cell.park_area_pct * 100).toFixed(1)}%</span>
      </div>

      {/* Suggestion */}
      <div style={{
        marginTop: 16, padding: 12,
        background: 'rgba(46,204,113,0.08)',
        border: '1px solid rgba(46,204,113,0.15)',
        borderRadius: 8
      }}>
        <p style={{ fontSize: 12, color: '#2ecc71', fontWeight: 600, marginBottom: 4 }}>
          Want to see what could change?
        </p>
        <p style={{ fontSize: 12, color: '#888', lineHeight: 1.5 }}>
          Switch to <strong style={{ color: '#ccc' }}>Simulate</strong> mode
          to model the impact of adding green space to this area.
        </p>
      </div>
    </div>
  )
}