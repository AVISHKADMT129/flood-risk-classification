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
            tick={{ fontSize: 11, fill: "#64748b", fontWeight: 500 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(val) => `${(val * 100).toFixed(1)}%`}
            labelStyle={{ fontWeight: 600 }}
            contentStyle={{
              borderRadius: 8,
              border: "1px solid #e2e8f0",
              fontSize: 12,
              boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
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
