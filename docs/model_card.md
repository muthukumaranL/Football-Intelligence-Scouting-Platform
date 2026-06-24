# Model Card

Two models ship with the platform, plus transparent composite scores. This card
documents intended use, data, metrics, assumptions, and limitations honestly.

## Intended use
- **Purpose:** demonstrate a data-driven scouting workflow (portfolio project).
- **Users:** recruiters/hiring managers evaluating the author's skills; students.
- **Out of scope:** real recruitment or transfer decisions. The shipped data is
  synthetic and the targets are proxies (see below).

## Models

### 1. Future-value regressor
- **Algorithm:** `RandomForestRegressor` (300 trees, max_depth 14, min_samples_leaf 3)
  wrapped in `TransformedTargetRegressor` with `log1p`/`expm1` target transform.
- **Target:** `future_value_eur` — a **proxy** 2-season value projection.
- **Features:** age, overall, potential, potential_gap, years_to_peak, market value,
  wage, reputation, skill moves, weak foot, the six attributes, performance_index,
  goal_contributions_per_90, contract_years_left.
- **Split:** 80/20 random.
- **Indicative metrics (sample data):** R² ≈ 0.92, MAE ≈ €0.46M, RMSE ≈ €0.93M.
- **Top drivers (permutation importance):** market value, potential_gap, potential.

### 2. Transfer-success classifier
- **Algorithm:** `Pipeline(StandardScaler → RandomForestClassifier(300 trees,
  max_depth 10, min_samples_leaf 4, class_weight="balanced"))`.
- **Target:** `transfer_success` — a **proxy** binary label.
- **Features:** age, overall, potential, potential_gap, years_to_peak, value, wage,
  rating_per_wage, reputation, performance_index, goal_contributions_per_90,
  contract_years_left.
- **Split:** stratified 80/20.
- **Indicative metrics (sample data):** ROC-AUC ≈ 0.85, Accuracy ≈ 0.78,
  Precision ≈ 0.80, Recall ≈ 0.73, F1 ≈ 0.77.
- **Top drivers:** potential, potential_gap, age.
- **UI layer:** the dashboard adds a documented heuristic **club-fit adjustment**
  (target club level + playing style) on top of the base probability.

### 3. Composite scores (not ML)
- **Undervalue Score** and **Recruitment Fit Score** are transparent weighted
  formulas (weights in `recommendation.py`), chosen for explainability over
  black-box optimization.

## Training data
- **Default:** 600 synthetic players generated with correlated, realistic
  relationships. Fixed random seeds for reproducibility.
- **Real data:** any CSV dropped in `data/raw/` (column aliases auto-applied).

## ⚠️ Key assumptions & proxy-target rationale
Real player datasets do **not** include observed "future value" or "transfer
success" outcomes. To demonstrate the supervised-learning workflow, two proxy
targets are constructed:

1. **`future_value_eur`** — built from an explicit age curve, potential headroom,
   and form, plus lognormal noise. The model learns this valuation function.
   Because current value is part of the inputs, high R² is expected and correct;
   the model's value-add is the *growth adjustment*.
2. **`transfer_success`** — built from a latent youth/potential/quality/value/
   reputation score + noise, thresholded at the median.

Both are clearly labelled in the README, methodology, and the app UI.

## Limitations
- Proxy targets ≠ real outcomes; metrics measure recovery of a synthetic signal.
- No temporal/season history, injuries, contract clauses, tactical fit, or
  league-strength normalization.
- Synthetic names/clubs/leagues; no real-world validity until real data is added.
- Random Forests can't extrapolate beyond the training value range.

## Ethical & legal notes
- **No FIFA/EA branding, club crests, player likenesses, or trademarks** are used.
- Intended for educational/portfolio purposes; not for betting, scouting, or
  commercial valuation.

## How to improve credibility
Swap in a real, time-stamped dataset; validate against actual future values and
transfer outcomes; add SHAP, hyperparameter tuning, and gradient-boosted baselines.
