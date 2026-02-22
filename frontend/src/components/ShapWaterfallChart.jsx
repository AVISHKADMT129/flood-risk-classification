import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
  ReferenceLine,
} from "recharts";

export default function ShapWaterfallChart({ shapValues }) {
  if (!shapValues || shapValues.length === 0) return null;

  const data = [...shapValues]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 7)
    .reverse();

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.shap_value)));
  const domain = [-maxAbs * 1.3, maxAbs * 1.3];

  return (
    <div className="chart-section">
      <h4 className="section-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 6, verticalAlign: -2 }}>
          <path d="M12 20V10" /><path d="M18 20V4" /><path d="M6 20v-4" />
        </svg>
        SHAP Feature Impact
      </h4>
      <p className="shap-subtitle">
        How each feature pushed the prediction toward or away from flood
      </p>
      <ResponsiveContainer width="100%" height={data.length * 38 + 16}>
        <BarChart
          layout="vertical"
          data={data}
          margin={{ left: 0, right: 50, top: 2, bottom: 2 }}
        >
          <XAxis type="number" domain={domain} hide />
          <YAxis
            type="category"
            dataKey="display_name"
            width={100}
            tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 500 }}
            axisLine={false}
            tickLine={false}
          />
          <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
          <Tooltip
            formatter={(val) => [
              `${val > 0 ? "+" : ""}${val.toFixed(4)}`,
              val > 0 ? "Increases flood risk" : "Decreases flood risk",
            ]}
            labelStyle={{ fontWeight: 600 }}
            contentStyle={{
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.1)",
              fontSize: 12,
              background: "rgba(15,23,42,0.9)",
              color: "#f1f5f9",
              boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
            }}
          />
          <Bar dataKey="shap_value" radius={[4, 4, 4, 4]} barSize={16}>
            <LabelList
              dataKey="shap_value"
              position="right"
              formatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(3)}`}
              style={{ fontSize: 9, fontWeight: 600, fill: "#94a3b8" }}
            />
            {data.map((entry) => (
              <Cell
                key={entry.feature}
                fill={entry.shap_value > 0 ? "#dc2626" : "#2563eb"}
                opacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="shap-legend">
        <span className="shap-legend-item">
          <span className="shap-dot shap-dot-red" />
          Increases flood risk
        </span>
        <span className="shap-legend-item">
          <span className="shap-dot shap-dot-blue" />
          Decreases flood risk
        </span>
      </div>
    </div>
  );
}
