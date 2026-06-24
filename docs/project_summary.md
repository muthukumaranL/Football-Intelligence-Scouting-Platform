# Project Summary

## Football Intelligence Scouting Platform

A one-page, dark-mode football scouting dashboard that turns player data into four recruitment decision tools. Built as a Data Analyst / Data Science / AI-ML portfolio project demonstrating the full lifecycle from raw data to a deployed interactive product.

### Problem
Clubs make high-stakes, high-cost recruitment decisions, often without a repeatable analytical process. This platform shows how a data team can support those decisions with explainable analytics.

### What it delivers
1. **Hidden Talent Discovery** — undervalued players ranked by a transparent Undervalue Score.
2. **Transfer Success Predictor** — ML probability + risk level + club-fit adjustment.
3. **Career Trajectory Prediction** — future value forecast + trajectory category.
4. **Club Recruitment Engine** — ranked, budget-aware shortlists with fit scores and reasons.

### Approach (at a glance)
`load → clean → engineer features → train models → score → recommend → visualize`, all cached in a single Streamlit app.

### Models
- Future-value **regression**: RandomForest on a log-transformed target (R² ≈ 0.92).
- Transfer-success **classification**: scaled RandomForest on a documented proxy label (ROC-AUC ≈ 0.85).
- Explainability via permutation importance.
- Transparent composite scores for undervalue, recruitment fit, and similarity.

### Data
Runs out-of-the-box on **synthetic, non-branded sample data** (600 players, realistic correlations). A real CSV can be dropped into `data/raw/` at any time — common FIFA/market-value column names are auto-mapped.

### Tech stack
Python · Streamlit · pandas · NumPy · scikit-learn · Plotly · joblib.

### Status
Complete, tested end-to-end (pipeline smoke test + Streamlit AppTest pass with zero exceptions), documented, and deployment-ready (Streamlit Cloud / Render / Hugging Face Spaces).

### Honest limitations
Sample data + documented **proxy** targets mean results are illustrative; the modeling *workflow* and product are the deliverable. See `model_card.md` for assumptions.
