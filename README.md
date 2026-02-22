# Flood Risk Classification for Sri Lanka

A full-stack ML application that predicts flood risk in Sri Lankan districts using a Random Forest classifier. Built with **FastAPI** (backend), **React + Vite** (frontend), and **scikit-learn** (ML pipeline).

---

## Prerequisites

- Python 3.9+
- Node.js 18+

---

## Getting Started

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m src.train
```

This generates the trained model at `models/flood_model.pkl` along with evaluation outputs in the `outputs/` folder.

### 3. Run the Backend

```bash
uvicorn api.main:app --reload --port 8003
```

The API will be available at `http://localhost:8003`.
Interactive API docs (Swagger UI) at `http://localhost:8003/docs`.

**API Endpoints:**

| Method | Endpoint          | Description                                      |
| ------ | ----------------- | ------------------------------------------------ |
| GET    | `/health`         | Health check                                     |
| GET    | `/metadata`       | Available districts, categories, and statistics   |
| GET    | `/explainability` | Feature importance and partial dependence data    |
| POST   | `/predict`        | Submit features, get prediction + risk level      |

### 4. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
├── api/                  # FastAPI backend
│   ├── main.py
│   ├── schemas.py
│   └── model_loader.py
├── frontend/             # React frontend
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/
├── src/                  # ML training pipeline
│   ├── config.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── inference.py
├── data/                 # Dataset
│   └── sri_lanka_flood_risk.csv
├── models/               # Trained model
│   └── flood_model.pkl
├── outputs/              # Metrics, plots, and visualizations
├── tests/                # Test suite
└── requirements.txt
```

---

## Tech Stack

| Layer    | Technology                       |
| -------- | -------------------------------- |
| ML       | scikit-learn, pandas, numpy      |
| Backend  | FastAPI, Uvicorn, Pydantic       |
| Frontend | React 19, Vite, Recharts         |
| Testing  | pytest, httpx                    |
