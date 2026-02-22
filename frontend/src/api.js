const API_BASE = "http://localhost:8003";

export async function fetchMetadata() {
  const res = await fetch(`${API_BASE}/metadata`);
  if (!res.ok) throw new Error("Failed to fetch metadata");
  return res.json();
}

export async function fetchExplainability() {
  const res = await fetch(`${API_BASE}/explainability`);
  if (!res.ok) throw new Error("Failed to fetch explainability data");
  return res.json();
}

export async function fetchPrediction(inputData) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inputData),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}
