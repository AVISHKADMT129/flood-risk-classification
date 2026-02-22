import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from "recharts";

const MODEL_COLORS = {
  LogisticRegression: "#94a3b8",
  RandomForest: "#3b82f6",
  GradientBoosting: "#8b5cf6",
};

const MODEL_LABELS = {
  LogisticRegression: "Logistic Reg.",
  RandomForest: "Random Forest",
  GradientBoosting: "Gradient Boost",
};

const METRIC_KEYS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"];
const METRIC_LABELS = {
  test_accuracy: "Accuracy",
  test_precision: "Precision",
  test_recall: "Recall",
  test_f1: "F1 Score",
  test_roc_auc: "ROC AUC",
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="chart-tooltip-row">
          <span className="chart-tooltip-dot" style={{ background: p.color }} />
          <span>{p.name}:</span>
          <strong>{(p.value * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
}

function RadarTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const item = payload[0];
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-label">{item.payload.metric}</div>
      <div className="chart-tooltip-row">
        <strong>{(item.value * 100).toFixed(1)}%</strong>
      </div>
    </div>
  );
}

export default function ModelInfoCard({ metrics }) {
  if (!metrics) return null;

  const comparison = metrics.model_comparison || {};
  const cm = metrics.confusion_matrix;
  const total = cm
    ? cm.true_positives + cm.true_negatives + cm.false_positives + cm.false_negatives
    : 0;

  // Radar data for selected model
  const radarData = METRIC_KEYS.map((key) => ({
    metric: METRIC_LABELS[key],
    value: comparison[metrics.model_name]?.[key] ?? 0,
    fullMark: 1,
  }));

  // Grouped bar comparison data
  const barData = METRIC_KEYS.map((key) => {
    const row = { metric: METRIC_LABELS[key] };
    Object.entries(comparison).forEach(([model, data]) => {
      row[model] = data[key];
    });
    return row;
  });

  // Confusion matrix donut data
  const cmDonutData = cm
    ? [
        { name: "True Neg", value: cm.true_negatives, fill: "#3b82f6" },
        { name: "True Pos", value: cm.true_positives, fill: "#059669" },
        { name: "False Pos", value: cm.false_positives, fill: "#dc2626" },
        { name: "False Neg", value: cm.false_negatives, fill: "#f59e0b" },
      ]
    : [];

  return (
    <div className="model-info-card">
      {/* Header */}
      <div className="model-info-header">
        <div className="model-info-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 20V10" /><path d="M18 20V4" /><path d="M6 20v-4" />
          </svg>
        </div>
        <div>
          <h3 className="model-info-title">Model Performance</h3>
          <p className="model-info-sub">
            {metrics.model_name} classifier &middot; {total.toLocaleString()} test samples
          </p>
        </div>
      </div>

      {/* Top: Radar + Confusion Matrix side by side */}
      <div className="model-charts-row">
        {/* Radar Chart */}
        <div className="model-chart-card">
          <h4 className="section-title">{metrics.model_name} Metrics</h4>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="rgba(255,255,255,0.1)" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 500 }}
              />
              <PolarRadiusAxis
                domain={[0, 1]}
                tickCount={5}
                tick={{ fontSize: 9, fill: "#94a3b8" }}
                axisLine={false}
              />
              <Radar
                name={metrics.model_name}
                dataKey="value"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.2}
                strokeWidth={2}
                dot={{ r: 4, fill: "#3b82f6", strokeWidth: 0 }}
              />
              <Tooltip content={<RadarTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Confusion Matrix */}
        {cm && (
          <div className="model-chart-card">
            <h4 className="section-title">Confusion Matrix</h4>
            <div className="cm-chart-layout">
              <div className="cm-grid">
                <div className="cm-corner" />
                <div className="cm-header">Pred. No</div>
                <div className="cm-header">Pred. Yes</div>
                <div className="cm-row-label">Actual No</div>
                <div className="cm-cell cm-tn">
                  <span className="cm-val">{cm.true_negatives}</span>
                  <span className="cm-pct">{((cm.true_negatives / total) * 100).toFixed(1)}%</span>
                </div>
                <div className="cm-cell cm-fp">
                  <span className="cm-val">{cm.false_positives}</span>
                  <span className="cm-pct">{((cm.false_positives / total) * 100).toFixed(1)}%</span>
                </div>
                <div className="cm-row-label">Actual Yes</div>
                <div className="cm-cell cm-fn">
                  <span className="cm-val">{cm.false_negatives}</span>
                  <span className="cm-pct">{((cm.false_negatives / total) * 100).toFixed(1)}%</span>
                </div>
                <div className="cm-cell cm-tp">
                  <span className="cm-val">{cm.true_positives}</span>
                  <span className="cm-pct">{((cm.true_positives / total) * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="cm-donut-wrap">
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie
                      data={cmDonutData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={65}
                      paddingAngle={2}
                      dataKey="value"
                      strokeWidth={0}
                    >
                      {cmDonutData.map((entry) => (
                        <Cell key={entry.name} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(val, name) => [`${val} (${((val / total) * 100).toFixed(1)}%)`, name]}
                      contentStyle={{ borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", fontSize: 12, background: "rgba(15,23,42,0.9)", color: "#f1f5f9" }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="cm-donut-legend">
                  {cmDonutData.map((d) => (
                    <span key={d.name} className="cm-donut-legend-item">
                      <span className="cm-donut-dot" style={{ background: d.fill }} />
                      {d.name}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom: Full-width model comparison bar chart */}
      {barData.length > 0 && (
        <div className="model-chart-card model-chart-full">
          <h4 className="section-title">Model Comparison</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <XAxis
                dataKey="metric"
                tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 500 }}
                axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 1]}
                tickCount={6}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                formatter={(value) => MODEL_LABELS[value] || value}
              />
              {Object.keys(comparison).map((model) => (
                <Bar
                  key={model}
                  dataKey={model}
                  name={model}
                  fill={MODEL_COLORS[model] || "#94a3b8"}
                  radius={[4, 4, 0, 0]}
                  barSize={28}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
