# Flood Risk Classification for Sri Lanka

A Machine Learning-based Flood Risk Classification System for Sri Lankan districts using traditional ML techniques.

---

## 1. Problem Statement

Sri Lanka is highly vulnerable to monsoon-driven flooding, with events affecting agriculture, infrastructure, and livelihoods. This project builds a **binary classification model** to predict whether a flood is likely to occur (`flood_occurred = 0 or 1`), given environmental and district-level indicators. The goal is to support early warning systems and disaster preparedness planning at the district level.

**Key constraints:**
- Traditional ML only (no deep learning or image processing)
- Sri Lankan geographic and climatic context
- Interpretable, explainable predictions

---

## 2. Dataset Description

- **File:** `data/sri_lanka_flood_risk.csv`
- **Size:** 9,986 usable records (after validation)
- **Target Variable:** `flood_occurred` (0 = No Flood, 1 = Flood)
- **Class Distribution:** 78.02% No Flood (7,791) / 21.98% Flood (2,195) — imbalanced

### Features

| Feature | Type | Description |
|---|---|---|
| `district` | Categorical | Administrative district (e.g., Colombo, Galle) |
| `division` | Categorical | Divisional secretariat within the district |
| `climate_zone` | Categorical | Wet / Dry / Intermediate |
| `drainage_quality` | Categorical | Good / Moderate / Poor |
| `year` | Numerical | Year of observation (2000-2023) |
| `month` | Numerical | Month of observation (1-12) |
| `rainfall_mm` | Numerical | Rainfall in millimeters |
| `river_level_m` | Numerical | River water level in meters |
| `soil_saturation_percent` | Numerical | Soil moisture saturation (0-100%) |
| `district_flood_prone` | Numerical | Whether the district is historically flood-prone (0/1) |

The dataset contains **no personal or sensitive data**.

---

## 3. Model Training and Evaluation

### 3.1 Train/Validation/Test Split Strategy

We use a two-level splitting approach that satisfies train/validation/test requirements:

1. **Train+Validation (80%) / Test (20%):** Using `train_test_split(test_size=0.2, stratify=y, random_state=42)` to create a held-out test set that is never seen during training or hyperparameter tuning.

2. **5-Fold Stratified Cross-Validation on Train Set:** Using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` inside `GridSearchCV` for hyperparameter tuning. Each fold preserves the class ratio, and shuffling ensures randomness with reproducibility.

This means:
- **Test set:** 1,998 samples (held out, used only for final evaluation)
- **Training/Validation:** 7,988 samples (split into 5 folds during CV)
- Stratification ensures the 78/22 class ratio is maintained in every split

### 3.2 Preprocessing Pipeline

Implemented using `sklearn.compose.ColumnTransformer` within a `Pipeline` to prevent data leakage:

- **Categorical features** (`district`, `division`, `climate_zone`, `drainage_quality`):
  - Missing value imputation with most frequent strategy
  - `OneHotEncoder(handle_unknown="ignore")` to handle unseen categories at prediction time
- **Numerical features** (`year`, `month`, `rainfall_mm`, `river_level_m`, `soil_saturation_percent`, `district_flood_prone`):
  - Missing value imputation with median strategy
  - `StandardScaler()` for zero mean and unit variance

The entire preprocessing + model is saved as a single `Pipeline` object, ensuring consistent transforms at inference time.

### 3.3 Models Trained (3 Classical ML Models)

#### Logistic Regression
A linear model serving as a baseline. Uses L2 regularization to prevent overfitting.

| Hyperparameter | Grid Values | Justification |
|---|---|---|
| `C` | [0.1, 1, 10] | Controls regularization strength; lower = more regularization |
| `penalty` | ["l2"] | Standard regularization for binary classification |
| `solver` | ["lbfgs"] | Efficient for small-to-medium datasets with L2 |
| `class_weight` | [None, "balanced"] | Tests whether reweighting minority class helps |

#### Random Forest
An ensemble of decision trees that captures non-linear relationships and feature interactions.

| Hyperparameter | Grid Values | Justification |
|---|---|---|
| `n_estimators` | [200, 400] | More trees reduce variance; 200-400 balances accuracy and speed |
| `max_depth` | [None, 8, 16] | Controls tree complexity; None = fully grown |
| `min_samples_split` | [2, 5] | Minimum samples to split a node; higher = more regularization |
| `min_samples_leaf` | [1, 2] | Minimum samples in leaf; prevents overly specific leaves |
| `class_weight` | [None, "balanced"] | Adjusts weights inversely proportional to class frequency |

#### Gradient Boosting
Sequential ensemble that builds trees to correct prior errors. Generally strong on tabular data.

| Hyperparameter | Grid Values | Justification |
|---|---|---|
| `n_estimators` | [150, 300] | Number of boosting rounds |
| `learning_rate` | [0.05, 0.1] | Step size shrinkage; lower = more robust but slower |
| `max_depth` | [2, 3] | Shallow trees (stumps) work best for boosting |

### 3.4 Performance Metrics

We evaluate using multiple metrics because this is an **imbalanced binary classification** problem (only 22% positive class):

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness, but can be misleading with imbalance |
| **Precision** | Of predicted floods, how many were correct (false alarm rate) |
| **Recall** | Of actual floods, how many were detected (miss rate) |
| **F1-Score** (primary) | Harmonic mean of precision and recall; balances both |
| **ROC-AUC** | Discrimination ability across all thresholds |
| **PR-AUC** | More informative than ROC-AUC for imbalanced data |

**F1-Score** is the primary metric for model selection because it directly balances the cost of false alarms (precision) against missed floods (recall), which is critical for disaster warning systems.

### 3.5 Results

#### Model Comparison Table

| Model | Best Params | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | C=0.1, class_weight=balanced | 0.7838 | 0.5051 | **0.7950** | 0.6177 | 0.8603 | 0.6880 |
| **Random Forest** | max_depth=16, n_estimators=200, balanced | **0.8093** | 0.5497 | 0.7312 | **0.6276** | 0.8559 | 0.6677 |
| Gradient Boosting | learning_rate=0.05, max_depth=3, n_estimators=150 | 0.8478 | **0.6991** | 0.5399 | 0.6093 | **0.8598** | **0.6830** |

#### Best Model: Random Forest (CV F1 = 0.6226, Test F1 = 0.6276)

**Confusion Matrix (Test Set, n=1998):**

|  | Predicted No Flood | Predicted Flood |
|---|---|---|
| **Actual No Flood** | 1,296 (TN) | 263 (FP) |
| **Actual Flood** | 118 (FN) | 321 (TP) |

### 3.6 Interpretation of Results

**Why Random Forest was selected:**
- Achieved the **highest cross-validated F1-score** (0.6226), indicating the best balance between precision and recall during training
- Test F1 (0.6276) closely matches CV F1, suggesting **no overfitting**
- Detects 73.1% of actual floods (recall) while maintaining reasonable precision (55%)

**Trade-off analysis:**
- **Logistic Regression** had the highest recall (79.5%) -- it catches the most floods but at the cost of many false alarms (precision only 50.5%). This is a valid choice if the priority is never missing a flood
- **Gradient Boosting** had the highest precision (69.9%) and accuracy (84.8%) -- fewer false alarms but misses 46% of actual floods. This would be preferred if false evacuations are costly
- **Random Forest** provides the best middle ground between these extremes, which is why it was selected as the final model

**Why F1 is not higher:**
- The 22% class imbalance means the model must learn to identify a minority pattern
- Environmental flood drivers are complex and non-linear -- a single threshold on rainfall or river level is not sufficient
- The `class_weight="balanced"` parameter helps but does not fully resolve the imbalance

### 3.7 Plots and Visualizations

| Plot | Path | Description |
|---|---|---|
| Confusion Matrix | `outputs/confusion_matrix.png` | Shows TP, FP, TN, FN distribution |
| ROC Curve | `outputs/roc_curve.png` | AUC = 0.8559, well above random baseline (0.5) |
| Precision-Recall Curve | `outputs/pr_curve.png` | PR-AUC = 0.6677, accounts for class imbalance |
| Calibration Curve | `outputs/calibration_curve.png` | Shows how well predicted probabilities match actual outcomes |
| Model Comparison | `outputs/model_comparison.csv` | Side-by-side metrics for all 3 models |

---

## 4. Explainability and Interpretation

### 4.1 Feature Importance Analysis

We extract feature importances from the Random Forest model using the built-in `feature_importances_` attribute, which measures the mean decrease in impurity (Gini importance) contributed by each feature across all trees.

**Top 10 Most Important Features:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `river_level_m` | 0.1866 |
| 2 | `soil_saturation_percent` | 0.1708 |
| 3 | `rainfall_mm` | 0.1511 |
| 4 | `climate_zone_Wet` | 0.0971 |
| 5 | `climate_zone_Dry` | 0.0663 |
| 6 | `district_flood_prone` | 0.0527 |
| 7 | `year` | 0.0381 |
| 8 | `month` | 0.0362 |
| 9 | `drainage_quality_Good` | 0.0110 |
| 10 | `drainage_quality_Poor` | 0.0096 |

**Outputs:**
- `outputs/feature_importance.csv` -- full feature importance table
- `outputs/feature_importance.png` -- bar chart of top 20 features

### 4.2 Partial Dependence Plots (PDP)

PDPs show the marginal effect of a single feature on the predicted flood probability, averaged over all other features. This reveals the relationship the model has learned between each feature and the target.

**Generated PDP plots:**
- `outputs/pdp_rainfall_mm.png`
- `outputs/pdp_river_level_m.png`
- `outputs/pdp_soil_saturation_percent.png`

### 4.3 What the Model Has Learned

The model's behavior reveals intuitive, domain-consistent patterns:

1. **River level is the strongest predictor** (importance: 0.187). Higher river water levels directly increase flood probability. The PDP for river_level_m shows a steep increase in flood probability as water levels rise, which aligns with the physical reality that overflowing rivers are the primary flood mechanism.

2. **Soil saturation is the second strongest predictor** (importance: 0.171). Saturated soil cannot absorb additional rainfall, leading to surface runoff. The PDP shows increasing flood probability as soil saturation rises, with a sharper increase above approximately 60-70%.

3. **Rainfall is the third strongest predictor** (importance: 0.151). Heavy rainfall is the triggering event for most floods. The PDP shows a non-linear relationship -- moderate rainfall has little effect, but flood probability increases rapidly beyond certain thresholds.

4. **Climate zone matters significantly.** Wet zone districts (importance: 0.097) have inherently higher flood risk due to higher annual rainfall, while dry zone districts (importance: 0.066) face different flood patterns.

5. **Historical flood-prone status** (importance: 0.053) captures district-level vulnerability factors like topography and drainage infrastructure that are not directly measured.

### 4.4 Alignment with Domain Knowledge

The model's learned patterns align well with Sri Lankan flood science:

- **Monsoon-driven flooding:** Sri Lanka experiences two monsoon seasons (May-September and October-January). The model captures seasonal patterns via the `month` feature.
- **Wet zone vulnerability:** Districts in the wet zone (southwestern Sri Lanka) receive 2,500+ mm rainfall annually and are correctly identified as higher risk.
- **Infrastructure effects:** The `drainage_quality` feature correctly shows that poor drainage increases flood risk, consistent with urban flooding patterns in Colombo and other cities.
- **Multi-factor causation:** The model correctly identifies that floods are not caused by a single factor but by the combination of high rainfall, saturated soil, and elevated river levels.

### 4.5 Limitations and Ethical Considerations

**Model limitations:**
- The dataset is synthetically generated and may not capture all real-world complexities
- Temporal dependencies (e.g., cumulative rainfall over multiple days) are not modeled
- The model predicts at the district level; sub-district variations are not captured
- Feature interactions between rainfall intensity and drainage capacity may be underrepresented

**Ethical considerations:**
- Predictions should supplement, not replace, professional meteorological assessments
- Equal attention must be given to all districts regardless of socioeconomic status
- False negatives (missed floods) are more dangerous than false positives (false alarms)
- The model should be periodically retrained with updated real-world data
- Overreliance on automated predictions could reduce investment in physical flood defenses

---

## 5. Risk Classification Thresholds

The model outputs a flood probability which is mapped to risk levels:

| Probability | Risk Level |
|---|---|
| p < 0.35 | Low |
| 0.35 - 0.65 | Medium |
| p > 0.65 | High |

---

## 6. How to Run

### Prerequisites

- Python 3.9+
- Node.js 18+

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python -m src.train
```

This generates:
- `models/flood_model.pkl` -- trained pipeline
- `outputs/metrics.json` -- evaluation metrics
- `outputs/model_comparison.csv` -- side-by-side model comparison
- `outputs/confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `calibration_curve.png`
- `outputs/feature_importance.csv`, `feature_importance.png`
- `outputs/pdp_rainfall_mm.png`, `pdp_river_level_m.png`, `pdp_soil_saturation_percent.png`

### Step 3: Run Inference (CLI)

```bash
python -m src.inference --json '{"district":"Colombo","division":"Colombo","climate_zone":"Wet","drainage_quality":"Moderate","year":2023,"month":5,"rainfall_mm":300,"river_level_m":4.0,"soil_saturation_percent":75,"district_flood_prone":1}'
```

### Step 4: Run the API

```bash
uvicorn api.main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Endpoints:**
- `GET /health` -- health check
- `GET /metadata` -- available districts, categories, and district mappings
- `POST /predict` -- submit features, receive prediction + probability + risk level

### Step 5: Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at `http://localhost:5173`.

### Step 6: Run Tests

```bash
pytest tests/ -v
```

18 tests covering preprocessing, API endpoints, and training pipeline validation.

---

## 7. Reproducibility

- **Random seed:** `42` used across all splits, models, and cross-validation
- **Deterministic splitting:** `train_test_split(stratify=y, random_state=42)`
- **Deterministic CV:** `StratifiedKFold(shuffle=True, random_state=42)`
- **All hyperparameters** documented in `src/config.py` and `src/train.py`
- **Saved artifacts:** Pipeline in `models/flood_model.pkl`, metrics in `outputs/metrics.json`

---

## 8. Project Structure

```
.
├── data/
│   └── sri_lanka_flood_risk.csv
├── models/
│   └── flood_model.pkl
├── outputs/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── calibration_curve.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
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
│       ├── App.css
│       ├── api.js
│       └── components/
│           ├── PredictionForm.jsx
│           └── PredictionResult.jsx
├── tests/
│   ├── test_preprocess.py
│   ├── test_training_pipeline.py
│   └── test_api.py
├── requirements.txt
└── README.md
```
