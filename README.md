# ⚽ Football Intelligence Scouting Platform

> An end-to-end **football analytics & machine learning** platform that uncovers undervalued players, forecasts career trajectories, predicts transfer success, and builds data-driven recruitment shortlists — wrapped in a polished, dark-mode, one-page scouting dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49%2B-e11d3a)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple)
![License](https://img.shields.io/badge/License-MIT-green)

> **No FIFA / EA branding.** This project uses a generic football-scouting theme with **synthetic, non-copyrighted** player, club and league names. No official logos, team crests or trademarked assets are used.

---

## Quickstart

```bash
git clone https://github.com/<your-username>/football-intelligence-scouting-platform.git
cd football-intelligence-scouting-platform
pip install -r requirements.txt
streamlit run app.py            # opens http://localhost:8501
```

No API keys, no manual data download — the app self-bootstraps a synthetic sample dataset and trains its models on first run (~10–15s).

## Table of contents

- [Overview](#-overview) · [Business problem](#-business-problem) · [Features](#-features)
- [Tech stack](#-tech-stack) · [Screenshots](#-screenshots) · [Dataset](#-dataset-information)
- [Architecture](#-project-architecture) · [Run locally](#-how-to-run-locally) · [Deploy](#-how-to-deploy)
- [Methodology](#-methodology) · [Models](#-ml-models-used) · [Metrics](#-evaluation-metrics)
- [Insights](#-key-insights) · [Limitations](#️-limitations) · [Future work](#-future-improvements)
- [Resume bullets](#-resume-bullets) · [LinkedIn description](#-linkedin-project-description)

---

## Overview

Professional clubs spend hundreds of millions on transfers, yet many decisions still rely on gut feel. This platform demonstrates how a data team can support recruitment with **repeatable, explainable analytics**: identifying value in the market, quantifying risk, and forecasting growth — all from a single interactive dashboard.

It is built as a **recruiter-ready Data Analyst / Data Science / AI-ML portfolio project**, showcasing the full lifecycle: data cleaning → EDA → feature engineering → predictive modeling → scoring/recommendation systems → an interactive product → documentation → deployment.

## Business problem

> *"Given a pool of players, which ones are undervalued, which transfers are most likely to succeed, how will players' values evolve, and who should a club shortlist within its budget and needs?"*

The platform answers this with four decision tools, each surfaced as a dashboard section.

## Features

| # | Module | What it does |
|---|--------|--------------|
| 1 | **Hidden Talent Discovery** | Ranks undervalued players via a transparent, weighted **Undervalue Score** (predicted growth + potential + youth + performance-for-price + wage efficiency). |
| 2 | **Transfer Success Predictor** | ML classifier estimates transfer-success probability (Low/Med/High risk), then adjusts for target-club level & playing style. |
| 3 | **Career Trajectory Prediction** | Regression model projects future market value and assigns a trajectory: *Rising Star / Stable Performer / Declining Asset / High-Risk Prospect*. |
| 4 | **Club Recruitment Engine** | Returns a ranked shortlist for a club's position, budget, age, potential and style needs, with a 0–100 **Fit Score** and reasons. |
| + | **Player Comparison** | Head-to-head profile cards, an overlaid attribute radar, and an automatic verdict on value, upside and quality. |
| + | **Model Lab** | Live evaluation metrics, permutation-importance charts, and a data-quality report. |
| + | **Insights & Story** | Plain-English business narrative auto-derived from the data. |

Dashboard UX: dark mode + red accents, a sticky **quick-jump section nav**, KPI tiles, glass cards, player cards, scouting **score rings**, transfer-risk **badges**, searchable/filterable tables, radar comparisons, and interactive Plotly charts.

## Tech stack

- **Language:** Python 3.10+
- **App / UI:** Streamlit (single-page dashboard, custom CSS theme)
- **Data:** pandas, NumPy
- **ML:** scikit-learn (RandomForest regression & classification, `TransformedTargetRegressor`, permutation importance, train/test split, scaling pipeline)
- **Viz:** Plotly (express + graph_objects)
- **Persistence:** joblib (models), pyarrow (processed parquet)

> **Why Streamlit (not Next.js + FastAPI)?** For a portfolio data-science project, Streamlit delivers a polished, interactive, *deployable* app in pure Python — no separate frontend/backend, build step, or hosting glue. That means faster completion, one-command local run, and free one-click cloud hosting, while keeping all focus on the data and models. A React/FastAPI split would add significant surface area for little analytical benefit here.

## Dataset information

The app loads data with a **3-tier fallback** (first match wins):

1. **Real data** — any `.csv` you place in [`data/raw/`](data/raw/). Common FIFA / SoFIFA / market-value column names are auto-mapped (see [`src/preprocessing.py`](src/preprocessing.py) `COLUMN_ALIASES`).
2. **Bundled sample** — [`data/sample/players_sample.csv`](data/sample/) (generated on first run).
3. **Synthetic generation** — a realistic, correlated dataset of 600 players created by [`src/data_loader.py`](src/data_loader.py).

**The default experience uses synthetic *sample* data** so the project runs out-of-the-box with zero setup or API keys. It is clearly labelled as sample data in the UI. To use real data:

```bash
# Example: a Kaggle FIFA players CSV
# 1. download players_22.csv (or similar)
# 2. drop it in:
data/raw/players_22.csv
# 3. rerun — the loader auto-detects and the alias map standardizes columns
```

Optional API keys (Kaggle, football-data.org, API-FOOTBALL) live in [`.env.example`](.env.example); none are required.

**Expected columns** (others are imputed/derived): `name, age, nationality, club, league, position, overall_rating, potential, market_value_eur, wage_eur` + optional attributes (`pace, shooting, passing, dribbling, defending, physic`) and performance (`appearances, goals, assists, minutes_played`). Full schema in [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Project architecture

```
football-intelligence-scouting-platform/
├── app.py                      # Streamlit one-page dashboard (UI + section renderers)
├── requirements.txt
├── README.md  /  LICENSE
├── .gitignore  /  .env.example
├── .streamlit/config.toml      # dark + red theme
│
├── data/
│   ├── raw/                    # drop a real CSV here (git-ignored)
│   ├── processed/              # cached processed parquet (generated)
│   └── sample/                 # synthetic sample CSV (generated)
│
├── notebooks/
│   └── 01_eda_modeling.ipynb   # EDA + modeling walkthrough
│
├── src/
│   ├── data_loader.py          # 3-tier load + synthetic generator
│   ├── preprocessing.py        # cleaning, column standardization, imputation
│   ├── feature_engineering.py  # derived features + proxy targets
│   ├── modeling.py             # train/evaluate regression + classification
│   ├── recommendation.py       # undervalue score, recruitment, similarity, trajectory
│   ├── visualization.py        # themed Plotly figures
│   └── utils.py                # paths, constants, theme, helpers
│
├── models/                     # saved model bundle (generated, git-ignored)
├── assets/styles.css           # dashboard styling
└── docs/                       # project_summary, methodology, data_dictionary, model_card, deployment_guide
```

**Data flow:** `load_raw_data → clean_data → engineer_features → train_all_models → compute_undervalue_score → add_trajectory_category → dashboard`. The whole pipeline is cached with `st.cache_resource`, so models train once per session.

## How to run locally

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the dashboard
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501). On first run the app generates the sample dataset and trains the models (~10–15s), then everything is cached.

Optional — regenerate the sample dataset from the CLI:

```bash
python -m src.data_loader
```

## Methodology

Full detail in [`docs/methodology.md`](docs/methodology.md). Summary:

- **Cleaning:** column-name standardization + real-dataset alias mapping, robust money parsing (`€12.5M` → `12500000`), primary-position extraction, group-median imputation, dedup, type coercion.
- **Feature engineering:** potential gap, years-to-peak, per-90 output, role-weighted performance index, wage/value efficiency ratios.
- **Proxy targets (documented, because real labels don't ship with the data):**
  - `future_value_eur` — a transparent age-curve × potential × form projection + lognormal noise; the regression model *learns* this valuation function from features.
  - `transfer_success` — a binary label from a latent youth/potential/quality/value/reputation score + noise, thresholded at the median (~balanced classes).
- **Undervalue Score** — explainable weighted composite (not a black box): `0.30·upside + 0.20·potential + 0.15·age + 0.20·quality-per-cost + 0.15·wage-efficiency`, scaled 0–100.

## ML models used

| Task | Model | Notes |
|------|-------|-------|
| Future value (regression) | `RandomForestRegressor` inside `TransformedTargetRegressor` (log1p/expm1 target) | wide value range stabilized via log target |
| Transfer success (classification) | `Pipeline(StandardScaler → RandomForestClassifier(class_weight="balanced"))` | balanced proxy label |
| Explainability | `permutation_importance` | surfaced in the Model Lab |
| Recommendation / similarity | Weighted composite scores + Euclidean nearest neighbours | transparent, no training needed |

## Evaluation metrics

Held-out 20% test split. Indicative results on the synthetic sample data (will vary with a real dataset):

- **Future value regression:** R² ≈ **0.92**, MAE ≈ **€0.46M**, RMSE ≈ **€0.93M**.
- **Transfer success classification:** ROC-AUC ≈ **0.85**, Accuracy ≈ **0.78**, Precision ≈ **0.80**, Recall ≈ **0.73**, F1 ≈ **0.77**.

> Current market value is (correctly) the strongest predictor of future value, so high R² is expected — the model's job is to learn the *growth adjustment* on top of it.

## Key insights

- The strongest value signals cluster around **young players with a wide potential–rating gap** and modest wages.
- **Position groups differ** in average value efficiency — a useful starting point for budget-conscious recruitment (the dashboard surfaces the leader dynamically).
- Combining **upside (growth)** with **price** separates genuine bargains from merely cheap players.

## Limitations

- The bundled dataset is **synthetic sample data**; insights are illustrative until a real dataset is supplied.
- `future_value` and `transfer_success` are **documented proxy targets**, not observed outcomes — the modeling *workflow* is the deliverable, not ground-truth transfer results.
- No injury, contract-clause, tactical-system, or temporal/season data; club-fit is a transparent heuristic layered on the base model.

## Future improvements

- Plug in a real, time-stamped dataset and validate against **actual** future values / transfer outcomes.
- Add SHAP explanations, hyperparameter tuning, and model comparison (XGBoost/LightGBM, gradient boosting).
- Time-series career curves per player; injury & contract risk features.
- SQL analytics layer (DuckDB/SQLite) and a saved-shortlist export.
- User accounts + a scouting watchlist; A/B-tested ranking weights.

---

## Resume bullets

- Built an interactive **football scouting analytics platform** using **Python, Streamlit, pandas, scikit-learn, and Plotly** to identify undervalued players, predict career growth, and generate recruitment shortlists, packaged as a polished one-page dashboard.
- Engineered **player valuation and performance features** and trained **regression and classification models** (R² ≈ 0.92; ROC-AUC ≈ 0.85) to compute undervalue scores, forecast future market value, and estimate transfer-success probability.
- Designed an **explainable Undervalue Score and a Club Recruitment Engine** with weighted fit scoring, enabling data-driven, budget-aware recruitment decisions and transparent, stakeholder-friendly recommendations.
- Owned the **full project lifecycle** — data cleaning, EDA, feature engineering, modeling, evaluation, dashboard UX, documentation, and deployment configuration — producing a reproducible, GitHub- and Streamlit-Cloud–ready portfolio project.

## LinkedIn project description

> **Football Intelligence Scouting Platform** — an end-to-end data science project that brings analytics to football recruitment. I built an interactive, dark-mode Streamlit dashboard that discovers undervalued players, predicts transfer success and career trajectories, and recommends recruitment shortlists. The stack: Python, pandas/NumPy, scikit-learn (Random Forest regression & classification with permutation-importance explainability), and Plotly. I engineered valuation and performance features, designed a transparent Undervalue Score and a club Fit Score, evaluated models with MAE/RMSE/R² and Accuracy/Precision/Recall/F1/ROC-AUC, and made the whole project reproducible and deployable. Built with a generic football theme using synthetic, non-branded data — easily swapped for a real dataset. #DataScience #MachineLearning #Python #Streamlit #SportsAnalytics #Portfolio

