const FIELDS = [
  { key: "rainfall_mm", label: "Rainfall", unit: "mm" },
  { key: "river_level_m", label: "River Level", unit: "m" },
  { key: "soil_saturation_percent", label: "Soil Saturation", unit: "%" },
];

const LEVEL_CONFIG = {
  Low: { className: "ctx-low", dotClass: "ctx-dot-low", label: "Below Avg" },
  Average: { className: "ctx-avg", dotClass: "ctx-dot-avg", label: "Average" },
  High: { className: "ctx-high", dotClass: "ctx-dot-high", label: "Above Avg" },
};

export default function InputContextPanel({ inputContext }) {
  if (!inputContext) return null;

  return (
    <div className="context-section">
      <h4 className="section-title">Your Input Values</h4>
      <div className="context-cards">
        {FIELDS.map(({ key, label, unit }) => {
          const ctx = inputContext[key];
          if (!ctx) return null;
          const cfg = LEVEL_CONFIG[ctx.level] || LEVEL_CONFIG.Average;

          const maxRange = Math.max(ctx.value, ctx.p75) * 1.4 || 1;
          const valuePos = Math.max(3, Math.min(97, (ctx.value / maxRange) * 100));
          const p25Pos = (ctx.p25 / maxRange) * 100;
          const p75Pos = (ctx.p75 / maxRange) * 100;

          return (
            <div key={key} className="context-card">
              <div className="context-info">
                <div className="context-top-row">
                  <div className="context-label">{label}</div>
                  <span className={`context-badge ${cfg.className}`}>{cfg.label}</span>
                </div>
                <div className="context-value">
                  {ctx.value} {unit}
                </div>
                <div className="ctx-range">
                  <div className="ctx-range-track">
                    <div
                      className="ctx-range-zone"
                      style={{ left: `${p25Pos}%`, width: `${p75Pos - p25Pos}%` }}
                    />
                    <div
                      className={`ctx-range-dot ${cfg.dotClass}`}
                      style={{ left: `${valuePos}%` }}
                    />
                  </div>
                  <div className="ctx-range-labels">
                    <span>P25</span>
                    <span>P75</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
