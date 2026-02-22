# FINAL MASTER SPECIFICATION (UPDATED FOR FULL MARKS)
# A Machine Learning-Based Flood Risk Classification Model for Sri Lanka
## Complete Academic-Grade Implementation Guide for GitHub Copilot (Agent Mode)
**Traditional ML only (NO deep learning / NO image processing).**

---

## 0) Purpose of This File
This markdown is a **single source of truth** for Copilot Agent Mode to implement the full solution **end-to-end**:
- Dataset → preprocessing → model training (with validation strategy) → tuning → evaluation → explainability → packaging
- FastAPI backend
- React frontend (Vite)
- Documentation (report-ready)
- Reproducibility + tests

This spec is designed to satisfy:
- **Model Training & Evaluation (20 marks)** requirements (split strategy, hyperparameters, metrics, plots, interpretation)
- **Explainability & Interpretation (20 marks)** requirements (at least one method: Feature Importance + PDP; optional SHAP/LIME)

---

## 1) Inputs (Local Files)
- Dataset CSV: `data/sri_lanka_flood_risk_10000.csv`
- Target column: `flood_occurred` (0/1)

### Feature Columns
Categorical:
- `district`
- `division` *(optional: can be dropped for simpler UI; if dropped, document why)*
- `climate_zone`
- `drainage_quality`

Numerical:
- `year`
- `month`
- `rainfall_mm`
- `river_level_m`
- `soil_saturation_percent`
- `district_flood_prone`

---

## 2) Hard Constraints (Non‑Negotiables)
- ✅ Sri Lankan context framing (district-based flood risk).
- ✅ Traditional ML only:
  - Allowed: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes.
  - Not allowed: neural networks, deep learning, image processing.
- ✅ Must include preprocessing, evaluation, interpretation.
- ✅ Must not contain any personal/sensitive data.
- ✅ Must publish dataset and code.
- ✅ Must include frontend + FastAPI (bonus-ready).

---

## 3) Required Project Structure
```
.
├── data/
│   └── sri_lanka_flood_risk_10000.csv
├── models/
│   └── flood_model.pkl
├── outputs/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── calibration_curve.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   ├── pdp_rainfall_mm.png
│   ├── pdp_river_level_m.png
│   └── pdp_soil_saturation_percent.png
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── inference.py
│   └── utils.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── model_loader.py
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/
├── tests/
│   ├── test_preprocess.py
│   ├── test_training_pipeline.py
│   └── test_api.py
├── requirements.txt
├── README.md
└── docker-compose.yml (optional)
```

---

## 4) Model Training & Evaluation (20 Marks) — MUST IMPLEMENT ALL

### 4.1 Split Strategy (Train/Validation/Test)
Implement one of these **valid** strategies (preferred order):

**Option A (Recommended):**
- Split into **Train+Validation (80%)** and **Test (20%)** using stratification.
- Within Train+Validation, use **StratifiedKFold (k=5)** for validation during hyperparameter tuning.

**Implementation requirement:**
- Use `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` to create the held-out test set.
- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` inside GridSearchCV.

This satisfies “train/validation/test split” because:
- Test is held out
- Validation is cross-validation folds on train

### 4.2 Preprocessing (Mandatory)
Use a **Pipeline** + **ColumnTransformer**:
- Categorical → OneHotEncoder(handle_unknown="ignore")
- Numerical → StandardScaler()

Missing values:
- Numeric → median imputation
- Categorical → mode imputation

### 4.3 Models (Train Minimum 3)
Train and compare at least 3 classical models:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting **or** SVM (RBF)

### 4.4 Hyperparameter Tuning (Must Explain Choices)
Use GridSearchCV (small, academic-friendly grids). Save the grid in the report.

**Logistic Regression grid**
- `C`: [0.1, 1, 10]
- `penalty`: ["l2"]
- `solver`: ["lbfgs"]
- `class_weight`: [None, "balanced"]

**Random Forest grid**
- `n_estimators`: [200, 400]
- `max_depth`: [None, 8, 16]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]
- `class_weight`: [None, "balanced"]

**Gradient Boosting grid**
- `n_estimators`: [150, 300]
- `learning_rate`: [0.05, 0.1]
- `max_depth`: [2, 3]

**SVM (if used) grid**
- `C`: [0.5, 1, 5]
- `gamma`: ["scale", 0.1]
- `class_weight`: [None, "balanced"]

### 4.5 Metrics (Must Report & Justify)
Compute and report:
- Accuracy
- Precision
- Recall
- F1-score (primary)
- ROC-AUC
- PR-AUC (recommended for imbalance)
- Confusion Matrix
- Optional: Calibration curve

### 4.6 Results & Interpretation (Mandatory)
Produce:
- A model comparison table: **model, best_params, accuracy, precision, recall, f1, roc_auc, pr_auc**
- Interpretation of results:
  - best model and why
  - tradeoffs (false alarms vs missed floods)
  - why final model was selected

### 4.7 Required Plots (Graphs/Tables)
Generate and save:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/feature_importance.png`
- PDP plots (Section 5)

---

## 5) Explainability & Interpretation (20 Marks) — MUST IMPLEMENT ALL

### 5.1 Feature Importance Analysis (Required)
Outputs:
- `outputs/feature_importance.csv` (top features + score)
- `outputs/feature_importance.png` (bar chart top 20)

### 5.2 Partial Dependence Plots (PDP) (Required)
Generate PDP for:
- rainfall_mm
- river_level_m
- soil_saturation_percent

Save:
- `outputs/pdp_rainfall_mm.png`
- `outputs/pdp_river_level_m.png`
- `outputs/pdp_soil_saturation_percent.png`

### 5.3 Interpretation Narrative (Mandatory)
Explain:
- what model learned
- most influential features
- alignment with Sri Lankan flood domain knowledge
- limitations and ethical considerations

(Optional): SHAP/LIME if allowed; PDP + feature importance is sufficient.

---

## 6) Model Packaging (Must)
- Save full pipeline to `models/flood_model.pkl` using `joblib.dump`
- Provide CLI:
  - `python -m src.train`
  - `python -m src.inference --json '{...}'`

---

## 7) FastAPI Backend (Must)
Endpoints:
- GET `/health`
- GET `/metadata`
- POST `/predict`

Return risk_level:
- p < 0.35 → Low
- 0.35–0.65 → Medium
- p > 0.65 → High

Requirements:
- load model once at startup
- validate inputs (422 on invalid)
- enable CORS

---

## 8) Frontend (React + Vite) (Must)
- fetch `/metadata`
- form inputs (dropdowns + numeric)
- call `/predict`
- display prediction, probability, risk_level
- loading + error states

---

## 9) Documentation (Must)
README must include:
- problem statement
- dataset description
- split strategy (train/val/test)
- hyperparameter grids
- metrics table + plots
- explainability (feature importance + PDP) + interpretation
- ethical considerations
- how to run training/api/frontend

---

## 10) Reproducibility (Must)
- random_state=42
- save best params + metrics to outputs/metrics.json
- deterministic training script

---

## 11) Testing (Must Minimal)
- preprocessing pipeline test
- API health/predict tests

---

## 12) requirements.txt (Must)
Include only:
- pandas, numpy, scikit-learn, joblib, matplotlib
- fastapi, uvicorn, pydantic
- pytest

---

## 13) Commands (Must Work)
```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
uvicorn api.main:app --reload --port 8000
cd frontend
npm install
npm run dev
```

---

## 14) Acceptance Criteria
Complete when:
- model saved: `models/flood_model.pkl`
- outputs generated (metrics, comparison table, plots)
- API predicts successfully
- frontend shows prediction
- README documents training/evaluation/explainability clearly
- no deep learning used

---

**END OF UPDATED MASTER SPECIFICATION**
