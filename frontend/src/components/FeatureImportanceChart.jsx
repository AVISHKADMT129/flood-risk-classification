import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";

const COLORS = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"];

export default function FeatureImportanceChart({ contributions }) {
  if (!contributions || contributions.length === 0) return null;

  const data = [...contributions].reverse();

  return (
    <div className="chart-section">
      <h4 className="section-title">Feature Importance</h4>
      <ResponsiveContainer width="100%" height={contributions.length * 40 + 8}>
        <BarChart
          layout="vertical"
          data={data}
          margin={{ left: 0, right: 36, top: 2, bottom: 2 }}
        >
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey="display_name"
            width={100}
            tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 500 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(val) => `${(val * 100).toFixed(1)}%`}
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
          <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={18}>
            <LabelList
              dataKey="importance"
              position="right"
              formatter={(v) => `${(v * 100).toFixed(1)}%`}
              style={{ fontSize: 10, fontWeight: 600, fill: "#94a3b8" }}
            />
            {data.map((entry, index) => (
              <Cell
                key={entry.feature}
                fill={COLORS[data.length - 1 - index] || COLORS[0]}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
