import { useState, useMemo } from "react";

const NUMERIC_FIELDS = [
  { name: "rainfall_mm", label: "Rainfall", unit: "mm", min: 0, sliderMax: 600, step: 1, inputStep: 0.1 },
  { name: "river_level_m", label: "River Level", unit: "m", min: 0, sliderMax: 10, step: 0.1, inputStep: 0.01 },
  { name: "soil_saturation_percent", label: "Soil Saturation", unit: "%", min: 0, max: 100, sliderMax: 100, step: 1, inputStep: 0.1 },
];

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

const YEARS = Array.from({ length: 31 }, (_, i) => 2000 + i);

const INITIAL_VALUES = {
  district: "",
  division: "",
  climate_zone: "",
  drainage_quality: "",
  year: "",
  month: "",
  rainfall_mm: "",
  river_level_m: "",
  soil_saturation_percent: "",
  district_flood_prone: 0,
};

function getContextLevel(value, stats) {
  if (!stats || value === "" || value === null || value === undefined) return null;
  const num = Number(value);
  if (isNaN(num)) return null;
  if (num < stats.p25) return "low";
  if (num > stats.p75) return "high";
  return "avg";
}

const CTX_LABELS = {
  low: "Below avg",
  avg: "Average",
  high: "Above avg",
};

const SCENARIOS = [
  { label: "Monsoon Heavy", rainfall_mm: 450, river_level_m: 7.5, soil_saturation_percent: 85, color: "#dc2626" },
  { label: "Moderate Rain", rainfall_mm: 200, river_level_m: 4.0, soil_saturation_percent: 55, color: "#d97706" },
  { label: "Dry Season", rainfall_mm: 30, river_level_m: 1.2, soil_saturation_percent: 20, color: "#059669" },
  { label: "Post-Flood", rainfall_mm: 120, river_level_m: 5.5, soil_saturation_percent: 75, color: "#7c3aed" },
];

export default function PredictionForm({ metadata, onSubmit, onClear, loading }) {
  const [form, setForm] = useState(INITIAL_VALUES);

  const districtMappings = metadata.district_mappings || {};
  const featureStats = metadata.feature_stats || {};
  const districts = useMemo(
    () => Object.keys(districtMappings).sort(),
    [districtMappings]
  );

  const currentMapping = districtMappings[form.district] || null;

  const handleDistrictChange = (e) => {
    const district = e.target.value;
    const mapping = districtMappings[district];
    if (mapping) {
      setForm((prev) => ({
        ...prev,
        district,
        division: mapping.divisions[0] || "",
        climate_zone: mapping.climate_zone,
        district_flood_prone: mapping.district_flood_prone,
        drainage_quality: mapping.drainage_default || "",
      }));
    } else {
      setForm((prev) => ({
        ...prev,
        district,
        division: "",
        climate_zone: "",
        district_flood_prone: 0,
        drainage_quality: "",
      }));
    }
  };

  const handleYearMonth = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const setFieldValue = (name, value) => {
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleMetricInput = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value === "" ? "" : Number(value) }));
  };

  const handleClear = () => {
    setForm(INITIAL_VALUES);
    onClear();
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* District */}
      <div className="district-selector">
        <label className="district-label" htmlFor="district">Select District</label>
        <select
          id="district"
          name="district"
          value={form.district}
          onChange={handleDistrictChange}
          required
          className="district-select"
        >
          <option value="">Choose a district...</option>
          {districts.map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </div>

      {/* Auto-filled info */}
      {currentMapping && (
        <div className="info-cards">
          <div className="info-card">
            <div className="info-card-label">Division</div>
            <div className="info-card-value">{form.division}</div>
          </div>
          <div className="info-card">
            <div className="info-card-label">Climate</div>
            <div className="info-card-value">{currentMapping.climate_zone}</div>
          </div>
          <div className="info-card">
            <div className="info-card-label">Drainage</div>
            <div className="info-card-value">{form.drainage_quality}</div>
          </div>
          <div className={`info-card ${currentMapping.district_flood_prone ? "info-card-warn" : "info-card-ok"}`}>
            <div className="info-card-label">Flood Prone</div>
            <div className="info-card-value">{currentMapping.district_flood_prone ? "Yes" : "No"}</div>
          </div>
        </div>
      )}

      {/* Period */}
      <div className="period-selector">
        <div className="period-header">
          <span className="period-title">Period</span>
        </div>
        <div className="period-dropdowns">
          <div className="period-dropdown">
            <label htmlFor="month">Month</label>
            <select id="month" name="month" value={form.month} onChange={handleYearMonth} required>
              <option value="">Month...</option>
              {MONTH_NAMES.map((name, i) => (
                <option key={i + 1} value={i + 1}>{name}</option>
              ))}
            </select>
          </div>
          <div className="period-dropdown">
            <label htmlFor="year">Year</label>
            <select id="year" name="year" value={form.year} onChange={handleYearMonth} required>
              <option value="">Year...</option>
              {YEARS.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Scenario Quick-Fill */}
      <div className="scenario-section">
        <div className="scenario-label">Quick Scenarios</div>
        <div className="scenario-buttons">
          {SCENARIOS.map((s) => (
            <button
              key={s.label}
              type="button"
              className="scenario-btn"
              style={{ borderColor: s.color }}
              onClick={() => {
                setForm((prev) => ({
                  ...prev,
                  rainfall_mm: s.rainfall_mm,
                  river_level_m: s.river_level_m,
                  soil_saturation_percent: s.soil_saturation_percent,
                }));
              }}
            >
              <span className="scenario-dot" style={{ background: s.color }} />
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="form-divider"><span>Environmental Measurements</span></div>

      {/* Measurements */}
      <div className="measure-panels">
        {NUMERIC_FIELDS.map((field) => {
          const stats = featureStats[field.name];
          const level = getContextLevel(form[field.name], stats);
          const numVal = form[field.name] === "" ? 0 : Number(form[field.name]);
          const fillPct = Math.min(100, (numVal / field.sliderMax) * 100);

          return (
            <div key={field.name} className="measure-panel">
              <div className="measure-header">
                <span className="measure-label">{field.label}</span>
                <span className="measure-unit">({field.unit})</span>
                {level && (
                  <span className={`input-ctx-badge input-ctx-${level}`}>{CTX_LABELS[level]}</span>
                )}
              </div>

              <div className="measure-input-row">
                <input
                  id={field.name}
                  name={field.name}
                  type="number"
                  className="measure-input"
                  value={form[field.name]}
                  onChange={handleMetricInput}
                  min={field.min}
                  max={field.max}
                  step={field.inputStep}
                  placeholder="0"
                  required
                />
                <span className="measure-input-unit">{field.unit}</span>
              </div>

              <input
                type="range"
                className="measure-slider"
                name={field.name}
                min={field.min}
                max={field.sliderMax}
                step={field.step}
                value={numVal}
                onChange={handleMetricInput}
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${fillPct}%, #e2e8f0 ${fillPct}%, #e2e8f0 100%)`,
                }}
              />

              {stats && (
                <div className="measure-presets">
                  <button
                    type="button"
                    className={`preset-btn ${numVal === stats.p25 ? "preset-active" : ""}`}
                    onClick={() => setFieldValue(field.name, stats.p25)}
                  >
                    <span className="preset-label">Low</span>
                    <span className="preset-value">{stats.p25}</span>
                  </button>
                  <button
                    type="button"
                    className={`preset-btn ${numVal === stats.mean ? "preset-active" : ""}`}
                    onClick={() => setFieldValue(field.name, stats.mean)}
                  >
                    <span className="preset-label">Avg</span>
                    <span className="preset-value">{stats.mean}</span>
                  </button>
                  <button
                    type="button"
                    className={`preset-btn ${numVal === stats.p75 ? "preset-active" : ""}`}
                    onClick={() => setFieldValue(field.name, stats.p75)}
                  >
                    <span className="preset-label">High</span>
                    <span className="preset-value">{stats.p75}</span>
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Actions */}
      <div className="form-actions">
        <button type="submit" disabled={loading || !form.district} className="predict-btn">
          {loading ? (
            <><span className="btn-spinner" /> Analyzing...</>
          ) : (
            <>
              Predict Flood Risk
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14" /><path d="m12 5 7 7-7 7" />
              </svg>
            </>
          )}
        </button>
        <button type="button" onClick={handleClear} disabled={loading} className="clear-btn">
          Clear
        </button>
      </div>
    </form>
  );
}
