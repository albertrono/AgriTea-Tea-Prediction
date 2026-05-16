# 🍃 AgriTea — Tea Yield Prediction System

A Streamlit web application that predicts tea yield (in KG per acre) using trained machine learning models. Farmers enter current field conditions and receive an instant yield estimate, contextual feedback, and agronomic recommendations — all without needing a database or internet connection after setup.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Models](#models)
- [Input Features](#input-features)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [App Pages](#app-pages)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Recommendations Guide](#recommendations-guide)
- [Dependencies](#dependencies)
- [Notes & Limitations](#notes--limitations)

---

## Overview

AgriTea is built on three machine learning models trained in `Tea_Project.ipynb` using a dataset of historical tea farm records. The models learn from five agronomic features — rainfall, lagged rainfall, temperature, and soil pH — to predict the expected yield in kilograms per acre for a given month.

The app is designed to be practical for agricultural extension officers and farm managers. It requires no database, stores no previous predictions, and runs entirely locally once the model files are in place.

---

## Features

- **Multi-model prediction** — choose from four trained models per session; the recommended model is pre-selected
- **XGBoost (Tuned) recommended** — hyperparameter-optimised via GridSearchCV for best generalisation
- **Yield gauge chart** — visual speedometer showing predicted yield against a 200 KG baseline
- **Contextual feedback** — low / moderate / good yield interpretation shown after every prediction
- **Input summary expander** — review the exact values sent to the model
- **Recommendations tab** — four categories of evidence-based agronomic guidance (Soil, Water, Harvesting, Agronomy)
- **Model Performance tab** — comparison table, XGBoost feature importance bar chart, and metric explainers
- **Sidebar model metrics** — live R² and MAE shown for the currently selected model
- **No database required** — stateless, prediction-only application

---

## Project Structure

```
agritea/
│
├── app.py                        # Main Streamlit application
├── README.md                     # This file
│
├── best_xgboost_model.joblib     # ⭐ Tuned XGBoost (recommended)
├── xgboost_model.joblib          # Base XGBoost model
├── random_forest_model.joblib    # Random Forest model
├── linear_regression_model.joblib# Linear Regression model
│
└── Tea_Project.ipynb             # Training notebook (reference only)
```

All four `.joblib` files must live in the **same directory as `app.py`**. The app loads them with `os.path.dirname(__file__)` so the path is always relative to the script, regardless of where you launch Streamlit from.

---

## Models

Four models are available in the sidebar dropdown. The app defaults to **XGBoost (Tuned) ⭐**.

| Model | Description |
|---|---|
| **XGBoost (Tuned) ⭐** | XGBoost trained with best hyperparameters found by GridSearchCV (`n_estimators=100`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`). Recommended for production use. |
| **XGBoost** | Base XGBoost with manually chosen hyperparameters (`n_estimators=300`, `learning_rate=0.07`, `max_depth=4`). Slightly higher R² on the test split, but the tuned model generalises better. |
| **Random Forest** | `RandomForestRegressor(random_state=42)` with default sklearn settings. |
| **Linear Regression** | Baseline `LinearRegression()` from scikit-learn. Lowest R², included for comparison. |

All models expect the same five input features in the same order and were trained without a feature scaler — **raw values are passed directly**.

---

## Input Features

| Feature | Label in App | Unit | Description |
|---|---|---|---|
| `Rainfall_mm` | Current month rainfall | mm | Total rainfall recorded this month |
| `Rainfall_Lag1_mm` | Last month rainfall (Lag 1) | mm | Rainfall from 1 month ago — captures delayed soil moisture effects |
| `Rainfall_Lag2_mm` | Two months ago rainfall (Lag 2) | mm | Rainfall from 2 months ago |
| `Avg_Temp_C` | Average temperature | °C | Mean ambient temperature this month |
| `Soil_pH` | Soil pH | 0–14 scale | Single most important predictor (86.6% XGBoost importance). Optimal range for tea: 4.5–6.0 |

> **Why lag features?** Tea root systems absorb moisture over several weeks. Rainfall from the previous month continues to affect leaf growth in the current month. Including 1-month and 2-month lagged rainfall significantly improves model accuracy.

---

## Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/agritea.git
cd agritea
```

Or simply place all files in one folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install manually:

```bash
pip install streamlit pandas numpy plotly joblib scikit-learn xgboost
```

> **scikit-learn version note:** The `.joblib` files were saved with scikit-learn `1.6.1`. Loading them with a different version will show an `InconsistentVersionWarning` but will not break functionality. To silence the warning, match the version: `pip install scikit-learn==1.6.1`.

---

## Running the App

```bash
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser automatically. If it doesn't, navigate there manually.

To run on a specific port:

```bash
streamlit run app.py --server.port 8080
```

---

## App Pages

### Predict Yield

The main prediction interface. Fill in the five fields and click **Predict Yield**.

**Rainfall Conditions (left column)**
- Current month rainfall (mm) — range 0–600
- Rainfall Lag 1: last month's rainfall (mm)
- Rainfall Lag 2: two months ago rainfall (mm)

**Soil & Climate (right column)**
- Average temperature (°C) — range 5–45
- Soil pH slider — range 3.0–9.0 in 0.1 steps

After prediction the app shows:
- A large result card with the predicted KG value and active model name
- A Plotly gauge chart with colour zones (red < 150 KG, amber 150–280 KG, green > 280 KG) and delta vs. 200 KG baseline
- An interpretation banner (low / moderate / good)
- An expandable input summary table

---

### Recommendations

Four tabbed sections of agronomic guidance, each colour-coded by type:

| Colour | Meaning |
|---|---|
| 🟢 Green border | Best practice / positive action |
| 🟡 Amber border | Warning / corrective action needed |
| 🔵 Blue border | Informational tip |

**Soil Management** — pH targets, liming, sulphur application, soil testing schedule, organic matter

**Water & Irrigation** — rainfall monitoring, dry-spell thresholds, drip irrigation, rain gauge setup

**Harvesting** — two-leaves-and-a-bud standard, plucking frequency, wet-leaf avoidance

**General Agronomy** — fertiliser type and timing, split-application schedule, disease watch, shade trees, temperature range

---

### Model Performance

A reference page showing how the four models compare on the held-out test set.

- **Metrics table** — MAE, RMSE, and R² for each model; the recommended model row is highlighted in green
- **Feature importance bar chart** — horizontal bar chart of XGBoost gain-based importances for all five features
- **Metric explainers** — plain-language descriptions of MAE, RMSE, and R² for non-technical users

---

## Model Performance

Results recorded from `Tea_Project.ipynb` on the test split (60% test, 40% train, `random_state=42`):

| Model | MAE (KG) | RMSE (KG) | R² |
|---|---|---|---|
| **XGBoost (Tuned) ⭐** | **7.746** | **10.164** | **0.723** |
| XGBoost | 7.470 | 10.131 | 0.725 |
| Random Forest | — | — | — |
| Linear Regression | 8.238 | 10.988 | 0.676 |

The base XGBoost has marginally better test-set metrics, but the tuned model was optimised via 3-fold cross-validation across 243 hyperparameter combinations, making it more reliable on unseen data from new farms.

Random Forest test metrics were not captured in the notebook output cells; the model file is valid and loads correctly.

---

## Feature Importance

XGBoost feature importances (gain-based, from `xgb.feature_importances_`):

| Feature | Importance |
|---|---|
| `Soil_pH` | **86.6%** |
| `Rainfall_Lag1_mm` | 4.8% |
| `Avg_Temp_C` | 4.3% |
| `Rainfall_Lag2_mm` | 2.9% |
| `Rainfall_mm` | 1.4% |

Soil pH is by far the dominant predictor. A farm with pH outside the 4.5–6.0 range will see very different yield estimates even if all other conditions are ideal. This aligns with agronomy literature — tea is an acid-loving crop and is highly sensitive to pH deviation.

---

## Recommendations Guide

The Recommendations page is designed to complement prediction results. Suggested workflow:

1. Run a prediction on the **Predict Yield** page.
2. If yield is low or moderate, switch to **Recommendations**.
3. Check the **Soil Management** tab first — pH corrections typically have the highest impact.
4. Cross-reference **Water & Irrigation** if rainfall lag values were below 80mm.
5. Review **General Agronomy** before the next plucking round.

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
joblib>=1.3.0
scikit-learn==1.6.1
xgboost>=2.0.0
```

Generate a `requirements.txt` from your virtual environment after installing:

```bash
pip freeze > requirements.txt
```

---

## Notes & Limitations

**No historical data storage.** The app is stateless — each prediction is independent and nothing is saved between sessions. If you want to track predictions over time, export them manually from the input summary expander or integrate a CSV logging step into `app.py`.

**No feature scaling.** The models were trained on raw feature values without StandardScaler or MinMaxScaler. Do not add a scaler — it will break predictions.

**Training split.** The dataset used a 40/60 train/test split, which is unconventional (most projects use 80/20). This means the models had less training data than usual, which may limit accuracy on farms with very different conditions from the training set.

**Soil pH sensitivity.** Because pH accounts for 86.6% of model importance, predictions are highly sensitive to this value. Ensure pH readings come from a calibrated meter or a lab test — a 0.5-point error in pH can shift the predicted yield substantially.

**Model versioning.** The `.joblib` files were created with scikit-learn `1.6.1`. If you retrain the models with a newer version, replace the files and update the `MODEL_METRICS` dictionary in `app.py` with the new performance numbers.

**Streamlit caching.** Models are loaded once per session using `@st.cache_resource`. If you replace a `.joblib` file while the app is running, restart the Streamlit process to reload the new model.

---

*Built with [Streamlit](https://streamlit.io) · Trained in [Jupyter](https://jupyter.org) · Models: XGBoost, Random Forest, Linear Regression*
