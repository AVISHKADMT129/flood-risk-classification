import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const METRIC_LABELS = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1_score: "F1 Score",
  roc_auc: "ROC AUC",
};

const MODEL_COLORS = {
  LogisticRegression: "#94a3b8",
  RandomForest: "#3b82f6",
  GradientBoosting: "#8b5cf6",
};

export default function ModelInfoCard({ metrics }) {
  if (!metrics) return null;

  const mainStats = [
    { label: "Accuracy", value: metrics.accuracy },
    { label: "F1 Score", value: metrics.f1_score },
    { label: "ROC AUC", value: metrics.roc_auc },
    { label: "Precision", value: metrics.precision },
    { label: "Recall", value: metrics.recall },
  ];

  // Build comparison chart data
  const comparison = metrics.model_comparison || {};
  const compData = Object.entries(comparison).map(([name, data]) => ({
    name: name.replace(/([a-z])([A-Z])/g, "$1 $2"),
    f1: data.test_f1,
    roc_auc: data.test_roc_auc,
    accuracy: data.test_accuracy,
    isSelected: name === metrics.model_name,
  }));

  // Confusion matrix
  const cm = metrics.confusion_matrix;
  const total = cm
    ? cm.true_positives + cm.true_negatives + cm.false_positives + cm.false_negatives
    : 0;

  return (
    <div className="model-info-card">
      <div className="model-info-header">
        <div className="model-info-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 20V10" /><path d="M18 20V4" /><path d="M6 20v-4" />
          </svg>
        </div>
        <div>
          <h3 className="model-info-title">Model Performance</h3>
          <p className="model-info-sub">{metrics.model_name} classifier with {total.toLocaleString()} test samples</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="model-stats-row">
        {mainStats.map((s) => (
          <div key={s.label} className="model-stat">
            <div className="model-stat-value">{(s.value * 100).toFixed(1)}%</div>
            <div className="model-stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="model-info-grid">
        {/* Model Comparison */}
        {compData.length > 0 && (
          <div className="model-comparison-section">
            <h4 className="section-title">Model Comparison (F1 Score)</h4>
            <ResponsiveContainer width="100%" height={compData.length * 48 + 16}>
              <BarChart
                layout="vertical"
                data={compData}
                margin={{ left: 8, right: 40, top: 4, bottom: 4 }}
              >
                <XAxis type="number" domain={[0, 1]} hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={120}
                  tick={{ fontSize: 11, fill: "#64748b", fontWeight: 500 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  formatter={(val) => `${(val * 100).toFixed(1)}%`}
                  contentStyle={{
                    borderRadius: 8,
                    border: "1px solid #e2e8f0",
                    fontSize: 12,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                  }}
                />
                <Bar dataKey="f1" radius={[0, 4, 4, 0]} barSize={20}>
                  {compData.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={MODEL_COLORS[entry.name.replace(/\s/g, "")] || "#94a3b8"}
                      opacity={entry.isSelected ? 1 : 0.5}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Confusion Matrix */}
        {cm && (
          <div className="confusion-matrix-section">
            <h4 className="section-title">Confusion Matrix</h4>
            <div className="cm-grid">
              <div className="cm-corner" />
              <div className="cm-header">Predicted No</div>
              <div className="cm-header">Predicted Yes</div>
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
          </div>
        )}
      </div>
    </div>
  );
}
