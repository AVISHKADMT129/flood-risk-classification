import { useState } from "react";
import FeatureImportanceChart from "./FeatureImportanceChart";
import InputComparisonChart from "./InputComparisonChart";
import ShapWaterfallChart from "./ShapWaterfallChart";

const R = 50;
const CIRC = 2 * Math.PI * R;

const RECOMMENDATIONS = {
  High: [
    { title: "Evacuate if advised", desc: "Follow local authority evacuation orders immediately" },
    { title: "Move valuables upstairs", desc: "Protect important documents and electronics from water damage" },
    { title: "Avoid flood water", desc: "Do not walk, swim, or drive through flood waters" },
    { title: "Emergency kit ready", desc: "Keep water, food, flashlight, and first aid supplies accessible" },
  ],
  Medium: [
    { title: "Stay informed", desc: "Monitor weather forecasts and government alerts closely" },
    { title: "Clear drains nearby", desc: "Remove debris from drains and gutters to prevent blockage" },
    { title: "Prepare supplies", desc: "Stock essentials in case conditions worsen" },
    { title: "Plan escape routes", desc: "Know your evacuation routes and meeting points" },
  ],
  Low: [
    { title: "General awareness", desc: "Stay informed about seasonal weather patterns" },
    { title: "Maintain drainage", desc: "Ensure drainage systems are clear and functional" },
    { title: "Review insurance", desc: "Check your flood insurance coverage is up to date" },
  ],
};

function CircularGauge({ pct, riskClass }) {
  const offset = CIRC - (parseFloat(pct) / 100) * CIRC;
  return (
    <div className="gauge-wrap" style={{ position: "relative" }}>
      <svg className="gauge-svg" viewBox="0 0 120 120">
        <circle className="gauge-track" cx="60" cy="60" r={R} />
        <circle
          className={`gauge-fill gauge-fill-${riskClass}`}
          cx="60" cy="60" r={R}
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="gauge-text">
        <span className={`gauge-pct gauge-pct-${riskClass}`}>{pct}%</span>
        <span className="gauge-sub">Probability</span>
      </div>
    </div>
  );
}

const TABS = [
  { id: "features", label: "Feature Analysis" },
  { id: "context", label: "Input Context" },
  { id: "safety", label: "Safety Tips" },
];

export default function PredictionResult({ result, loading }) {
  const [activeTab, setActiveTab] = useState("features");
  if (loading) {
    return (
      <div className="result-empty result-loading-state">
        <div className="result-empty-spinner">
          <div className="spinner" style={{ width: 28, height: 28 }} />
        </div>
        <h3>Analyzing...</h3>
        <p>Running ML model prediction</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-empty">
        <div className="result-empty-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
          </svg>
        </div>
        <h3>Prediction Results</h3>
        <p>Select a district and fill in measurements, then click predict</p>
      </div>
    );
  }

  const isFlood = result.prediction === 1;
  const pct = (result.probability * 100).toFixed(1);
  const safePct = (100 - parseFloat(pct)).toFixed(1);
  const risk = result.risk_level;
  const riskClass = risk === "Low" ? "low" : risk === "Medium" ? "med" : "high";
  const contributions = result.feature_contributions || [];
  const shapValues = result.shap_values || [];
  const tips = RECOMMENDATIONS[risk] || RECOMMENDATIONS.Low;
  const probNum = parseFloat(pct);

  return (
    <div className={`result-card result-${riskClass}`}>
      <div className={`result-accent accent-${riskClass}`} />

      <div className="result-body">
        {/* Gauge + Verdict centered */}
        <div className="result-gauge-area">
          <CircularGauge pct={pct} riskClass={riskClass} />
          <h2 className={`result-verdict result-verdict-${riskClass}`}>
            {isFlood ? "Flood Likely" : "No Flood Expected"}
          </h2>
          <span className={`result-risk-label risk-label-${riskClass}`}>
            {risk} Risk
          </span>
          <div className="result-prob-summary">
            <span>{pct}% flood</span>
            <span className="prob-dot" />
            <span>{safePct}% safe</span>
          </div>
        </div>

        {/* Risk Scale */}
        <div className="risk-scale">
          <div className="risk-scale-bar">
            <div className="risk-scale-zone risk-zone-low" />
            <div className="risk-scale-zone risk-zone-med" />
            <div className="risk-scale-zone risk-zone-high" />
            <div
              className={`risk-scale-marker marker-${riskClass}`}
              style={{ left: `${Math.min(98, Math.max(2, probNum))}%` }}
            />
          </div>
          <div className="risk-scale-labels">
            <span>Low (0-35%)</span>
            <span>Medium (35-65%)</span>
            <span>High (65-100%)</span>
          </div>
        </div>

        {/* Explanation */}
        {result.explanation && (
          <p className={`result-explanation result-explanation-${riskClass}`}>
            {result.explanation}
          </p>
        )}

        {/* Tabs */}
        <div className="result-tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`result-tab ${activeTab === tab.id ? "result-tab-active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="result-tab-content">
          {activeTab === "features" && (
            <>
              <FeatureImportanceChart contributions={contributions} />
              <div className="result-divider" />
              <ShapWaterfallChart shapValues={shapValues} />
            </>
          )}

          {activeTab === "context" && (
            <InputComparisonChart inputContext={result.input_context} />
          )}

          {activeTab === "safety" && (
            <div className="recommendations-section">
              <h4 className="section-title">Safety Recommendations</h4>
              <div className="recommendations-list">
                {tips.map((tip, i) => (
                  <div key={i} className={`rec-item rec-item-${riskClass}`}>
                    <div className="rec-number">{i + 1}</div>
                    <div className="rec-content">
                      <div className="rec-title">{tip.title}</div>
                      <div className="rec-desc">{tip.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
