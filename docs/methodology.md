# Methodology

This document explains how raw player data becomes the scores and predictions shown in the dashboard. Every step lives in `src/` and is unit-exercised by the pipeline smoke test.

## 1. Data loading (`src/data_loader.py`)
A 3-tier fallback (first match wins):
1. Real CSV in `data/raw/`.
2. Bundled sample CSV in `data/sample/`.
3. Freshly generated synthetic dataset.

**Synthetic generator.** 600 players with intentionally *correlated* fields so downstream analysis behaves realistically:
- Overall rating ~ N(70, 7), nudged up in prime age (24–29).
- Potential = overall + age-dependent headroom (younger ⇒ more).
- Position-appropriate attributes (defenders score on defending, forwards on shooting, etc.).
- Market value = exponential in rating × youth premium × potential factor × position premium × reputation × lognormal noise.
- Wage tracks value; performance (apps/goals/assists/minutes) scales with role and rating.
- Names/clubs/leagues are fictional to avoid any trademark/branding issues.

## 2. Cleaning & standardization (`src/preprocessing.py`)
- **Column standardization:** lowercase + underscores, then a `COLUMN_ALIASES` map translates common real-world names (`value_eur`, `short_name`, `nationality_name`, `player_positions`, `physicality`, …) to canonical names.
- **Money parsing:** strings like `€12.5M`, `500K`, `1.2B` → floats.
- **Position handling:** multi-position strings (`"ST, CF"`) reduced to a primary position; unknown positions default to `CM`; a `position_group` is mapped.
- **Missing values:** numeric columns imputed by **position-group median**, then global median, then 0. Categorical defaults applied. `potential` floored at `overall_rating`.
- **Integrity:** duplicates dropped on `player_id`; key fields coerced to int.
- A before/after `preprocessing_report` is surfaced in the Model Lab.

## 3. Feature engineering (`src/feature_engineering.py`)
Derived analytical features:
- `potential_gap = potential − overall_rating` (upside headroom).
- `years_to_peak = 27 − age`.
- `value_per_rating`, `rating_per_wage` (wage efficiency), `value_to_wage_ratio`.
- `goals_per_90`, `assists_per_90`, `goal_contributions_per_90` (guard against 0 minutes).
- `performance_index` — role-weighted blend of the six attributes (0–100) + a capped output bonus.

### Proxy targets (clearly documented)
Real data does not ship with "future value" or "transfer success" labels, so two **proxy** targets are constructed to demonstrate the supervised workflow:

- **`future_value_eur`** (regression target): a transparent 2-season projection
  `current_value × age_curve × age_decline × potential_factor × form_factor × lognormal_noise`.
  Value rises toward peak age and declines after; unrealized potential and form lift it. The model then *learns* this valuation function from features — noise keeps it from being perfectly recoverable.

- **`transfer_success`** (classification target): a latent score
  `0.22·youth + 0.24·potential_gap + 0.20·quality + 0.16·performance + 0.12·value_efficiency + 0.06·reputation + noise`, thresholded at the median for ~balanced classes.

> These assumptions are repeated in `model_card.md` and shown to users in the UI. They make the project honest: the *process* (cleaning, feature design, training, evaluation, explainability, productization) is the portfolio value, not literal transfer outcomes.

## 4. Modeling (`src/modeling.py`)
- **Regression:** `RandomForestRegressor` wrapped in `TransformedTargetRegressor(log1p/expm1)` — stabilizes the wide value range. 80/20 split. Metrics: MAE, RMSE, R².
- **Classification:** `Pipeline(StandardScaler → RandomForestClassifier(class_weight="balanced"))`. Stratified 80/20 split. Metrics: accuracy, precision, recall, F1, ROC-AUC.
- **Explainability:** `permutation_importance` on the test set for both models.
- Predictions (`predicted_future_value_eur`, `predicted_growth_pct`, `transfer_success_proba`) are attached to the full dataset for the UI.
- Models are bundled and persisted with joblib (`models/model_bundle.joblib`).

## 5. Scoring & recommendation (`src/recommendation.py`)
- **Undervalue Score (0–100):** explainable weighted composite —
  `0.30·upside + 0.20·potential + 0.15·age_advantage + 0.20·quality_per_cost + 0.15·wage_efficiency`.
  Each component is min-max scaled across the cohort; the top two contributors generate a human-readable reason.
- **Club Recruitment Engine:** hard filters (position, budget, age, potential, rating) narrow the pool; a soft **Fit Score (0–100)** ranks it —
  `0.30·quality + 0.25·style_fit + 0.25·value_for_money + 0.20·budget_fit`, with a generated reason string.
- **Career trajectory:** rule-based bucketing from predicted growth + age + potential gap into *Rising Star / Stable Performer / Declining Asset / High-Risk Prospect*.
- **Similarity:** Euclidean distance in a min-max-scaled attribute space for player-comparison radars.

## 6. Visualization (`src/visualization.py`)
A shared dark/red Plotly theme powers all figures: value-vs-predicted scatter with break-even line, value-by-position box, attribute radar, permutation-importance bars, age-vs-value scatter, trajectory bar, growth gauge, correlation heatmap.

## Reproducibility
- All paths derive from the project root (`src/utils.py`) — no hardcoded machine paths.
- Random seeds fixed (data gen, feature targets, model training).
- The full pipeline is cached (`st.cache_resource`) and re-runnable via `streamlit run app.py` or `python -m src.data_loader`.
