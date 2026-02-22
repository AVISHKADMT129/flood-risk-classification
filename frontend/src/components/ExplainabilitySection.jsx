import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
  LineChart,
  Line,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

const IMPORTANCE_COLORS = [
  "#1e40af", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd",
  "#bfdbfe", "#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed",
  "#6d28d9", "#5b21b6",
];

const PDP_COLORS = ["#2563eb", "#059669", "#d97706"];

const TABS = [
  { id: "importance", label: "Feature Importance", icon: "M3 3v18h18" },
  { id: "pdp", label: "PDP", icon: "M22 12h-4l-3 9L9 3l-3 9H2" },
  { id: "shap", label: "SHAP", icon: "M12 20V10M18 20V4M6 20v-4" },
  { id: "lime", label: "LIME", icon: "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" },
];

function TabIcon({ d }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d={d} />
    </svg>
  );
}

function ImportanceTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{d.display_name}</div>
      <div className="chart-tooltip-row">
        <span>Importance:</span>
        <strong>{(d.importance * 100).toFixed(2)}%</strong>
      </div>
    </div>
  );
}

function PdpTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{label} {unit}</div>
      <div className="chart-tooltip-row">
        <span>Avg. Prediction:</span>
        <strong>{(payload[0].value * 100).toFixed(1)}%</strong>
      </div>
    </div>
  );
}

function ShapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{d.display_name}</div>
      <div className="chart-tooltip-row">
        <span>{d.shap_value > 0 ? "Increases" : "Decreases"} risk:</span>
        <strong>{d.shap_value > 0 ? "+" : ""}{d.shap_value.toFixed(4)}</strong>
      </div>
    </div>
  );
}

function LimeTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{d.display_name}</div>
      <div className="chart-tooltip-row">
        <span>{d.lime_weight > 0 ? "Supports" : "Opposes"} flood:</span>
        <strong>{d.lime_weight > 0 ? "+" : ""}{d.lime_weight.toFixed(4)}</strong>
      </div>
    </div>
  );
}

// ── Feature Importance Tab ──
function FeatureImportanceTab({ data }) {
  if (!data || data.length === 0) {
    return <p className="xai-empty">No feature importance data available.</p>;
  }

  const chartData = [...data].reverse();

  return (
    <div className="xai-tab-inner">
      <p className="xai-description">
        Global feature importance shows which features the model relies on most across all predictions in the training data.
        Higher values mean the feature has a stronger overall influence on every prediction the model makes.
      </p>
      <ResponsiveContainer width="100%" height={data.length * 36 + 24}>
        <BarChart layout="vertical" data={chartData} margin={{ left: 4, right: 60, top: 4, bottom: 4 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category" dataKey="display_name" width={140}
            tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 500 }}
            axisLine={false} tickLine={false}
          />
          <Tooltip content={<ImportanceTooltip />} />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={18}>
            <LabelList
              dataKey="importance" position="right"
              formatter={(v) => `${(v * 100).toFixed(1)}%`}
              style={{ fontSize: 10, fontWeight: 600, fill: "#94a3b8" }}
            />
            {chartData.map((_, i) => (
              <Cell key={i} fill={IMPORTANCE_COLORS[chartData.length - 1 - i] || "#3b82f6"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── PDP Tab ──
function PdpTab({ data }) {
  if (!data || Object.keys(data).length === 0) {
    return <p className="xai-empty">No PDP data available.</p>;
  }

  const features = Object.entries(data);

  return (
    <div className="xai-tab-inner">
      <p className="xai-description">
        Partial Dependence Plots show how changing a single feature affects the average flood prediction,
        while keeping all other features constant. The y-axis represents the average predicted flood probability.
      </p>
      <div className="pdp-grid">
        {features.map(([key, feat], idx) => {
          const chartData = feat.values.map((v, i) => ({
            x: v,
            y: feat.predictions[i],
          }));

          return (
            <div key={key} className="pdp-chart-card">
              <h5 className="pdp-chart-title">{feat.label} ({feat.unit})</h5>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis
                    dataKey="x" type="number"
                    tick={{ fontSize: 10, fill: "#94a3b8" }}
                    tickLine={false} axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#94a3b8" }}
                    tickLine={false} axisLine={false}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  />
                  <Tooltip content={(props) => <PdpTooltip {...props} unit={feat.unit} />} />
                  <Line
                    type="monotone" dataKey="y"
                    stroke={PDP_COLORS[idx % PDP_COLORS.length]}
                    strokeWidth={2.5} dot={false}
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── SHAP Tab ──
function ShapTab({ shapValues }) {
  if (!shapValues || shapValues.length === 0) {
    return (
      <div className="xai-tab-inner">
        <p className="xai-description">
          SHAP (SHapley Additive exPlanations) uses game theory to explain how each feature
          contributes to pushing the prediction toward or away from a flood outcome.
        </p>
        <p className="xai-empty">Make a prediction to see per-instance SHAP values.</p>
      </div>
    );
  }

  const data = [...shapValues]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 7)
    .reverse();

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.shap_value)));
  const domain = [-maxAbs * 1.3, maxAbs * 1.3];

  return (
    <div className="xai-tab-inner">
      <p className="xai-description">
        SHAP values for your current prediction. Each bar shows how one feature pushed the prediction
        toward (red) or away from (blue) a flood outcome compared to the baseline.
      </p>
      <ResponsiveContainer width="100%" height={data.length * 40 + 20}>
        <BarChart layout="vertical" data={data} margin={{ left: 4, right: 56, top: 4, bottom: 4 }}>
          <XAxis type="number" domain={domain} hide />
          <YAxis
            type="category" dataKey="display_name" width={120}
            tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 500 }}
            axisLine={false} tickLine={false}
          />
          <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
          <Tooltip content={<ShapTooltip />} />
          <Bar dataKey="shap_value" radius={[4, 4, 4, 4]} barSize={18}>
            <LabelList
              dataKey="shap_value" position="right"
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
      <div className="xai-legend">
        <span className="xai-legend-item">
          <span className="xai-legend-dot" style={{ background: "#dc2626" }} />
          Increases flood risk
        </span>
        <span className="xai-legend-item">
          <span className="xai-legend-dot" style={{ background: "#2563eb" }} />
          Decreases flood risk
        </span>
      </div>
    </div>
  );
}

// ── LIME Tab ──
function LimeTab({ limeValues }) {
  if (!limeValues || limeValues.length === 0) {
    return (
      <div className="xai-tab-inner">
        <p className="xai-description">
          LIME (Local Interpretable Model-agnostic Explanations) creates a simple, interpretable
          model around each prediction to explain which features were most influential locally.
        </p>
        <p className="xai-empty">Make a prediction to see per-instance LIME explanations.</p>
      </div>
    );
  }

  const data = [...limeValues]
    .sort((a, b) => Math.abs(b.lime_weight) - Math.abs(a.lime_weight))
    .slice(0, 7)
    .reverse();

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.lime_weight)));
  const domain = [-maxAbs * 1.3, maxAbs * 1.3];

  return (
    <div className="xai-tab-inner">
      <p className="xai-description">
        LIME explanation for your current prediction. Each bar shows how strongly a feature supports
        (green) or opposes (orange) the flood prediction in this specific case.
      </p>
      <ResponsiveContainer width="100%" height={data.length * 40 + 20}>
        <BarChart layout="vertical" data={data} margin={{ left: 4, right: 56, top: 4, bottom: 4 }}>
          <XAxis type="number" domain={domain} hide />
          <YAxis
            type="category" dataKey="display_name" width={120}
            tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 500 }}
            axisLine={false} tickLine={false}
          />
          <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
          <Tooltip content={<LimeTooltip />} />
          <Bar dataKey="lime_weight" radius={[4, 4, 4, 4]} barSize={18}>
            <LabelList
              dataKey="lime_weight" position="right"
              formatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(3)}`}
              style={{ fontSize: 9, fontWeight: 600, fill: "#94a3b8" }}
            />
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.lime_weight > 0 ? "#059669" : "#d97706"}
                opacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="xai-legend">
        <span className="xai-legend-item">
          <span className="xai-legend-dot" style={{ background: "#059669" }} />
          Supports flood prediction
        </span>
        <span className="xai-legend-item">
          <span className="xai-legend-dot" style={{ background: "#d97706" }} />
          Opposes flood prediction
        </span>
      </div>
    </div>
  );
}

// ── Main Component ──
export default function ExplainabilitySection({ globalData, predictionResult }) {
  const [activeTab, setActiveTab] = useState("importance");

  const shapValues = predictionResult?.shap_values || [];
  const limeValues = predictionResult?.lime_values || [];

  return (
    <div className="xai-card">
      <div className="xai-header">
        <div className="xai-header-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4M12 8h.01" />
          </svg>
        </div>
        <div>
          <h3 className="xai-title">Model Explainability</h3>
          <p className="xai-subtitle">Understand how the model makes predictions using multiple XAI methods</p>
        </div>
      </div>

      <div className="xai-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`xai-tab ${activeTab === tab.id ? "xai-tab-active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <TabIcon d={tab.icon} />
            {tab.label}
          </button>
        ))}
      </div>

      <div className="xai-content">
        {activeTab === "importance" && (
          <FeatureImportanceTab data={globalData?.feature_importance} />
        )}
        {activeTab === "pdp" && (
          <PdpTab data={globalData?.pdp_data} />
        )}
        {activeTab === "shap" && (
          <ShapTab shapValues={shapValues} />
        )}
        {activeTab === "lime" && (
          <LimeTab limeValues={limeValues} />
        )}
      </div>
    </div>
  );
}
