import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";

const FIELDS = [
  { key: "rainfall_mm", label: "Rainfall", max: 600 },
  { key: "river_level_m", label: "River Level", max: 10 },
  { key: "soil_saturation_percent", label: "Soil Sat.", max: 100 },
];

export default function InputComparisonChart({ inputContext }) {
  if (!inputContext) return null;

  const data = FIELDS.map(({ key, label, max }) => {
    const ctx = inputContext[key];
    if (!ctx) return null;
    return {
      metric: label,
      "Your Value": Math.round(Math.min(100, (ctx.value / max) * 100)),
      Average: Math.round(Math.min(100, (ctx.mean / max) * 100)),
    };
  }).filter(Boolean);

  if (data.length === 0) return null;

  return (
    <div className="chart-section">
      <h4 className="section-title">Your Input vs Dataset Average</h4>
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={data} outerRadius={80}>
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis
            dataKey="metric"
            tick={{ fontSize: 11, fill: "#64748b", fontWeight: 500 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fontSize: 9, fill: "#94a3b8" }}
            tickCount={4}
          />
          <Radar
            name="Average"
            dataKey="Average"
            stroke="#94a3b8"
            fill="#94a3b8"
            fillOpacity={0.12}
            strokeWidth={1.5}
          />
          <Radar
            name="Your Value"
            dataKey="Your Value"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.2}
            strokeWidth={2}
          />
          <Tooltip
            formatter={(val) => `${val}%`}
            contentStyle={{
              borderRadius: 8,
              border: "1px solid #e2e8f0",
              fontSize: 12,
              boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, fontWeight: 500 }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
