"""Football Intelligence Scouting Platform — Streamlit dashboard.

A one-page, dark-mode scouting dashboard that turns player data into four
decision tools: Hidden Talent Discovery, Transfer Success Prediction, Career
Trajectory Forecasting and a Club Recruitment Engine.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_raw_data
from src.feature_engineering import engineer_features
from src.modeling import train_all_models
from src.preprocessing import clean_data, preprocessing_report
from src.recommendation import (
    STYLE_TO_ATTRIBUTE,
    add_trajectory_category,
    compute_undervalue_score,
    find_similar_players,
    recommend_players,
)
from src.utils import (
    ASSETS_DIR,
    POSITION_GROUP_ORDER,
    PROCESSED_DATA_PATH,
    THEME,
    clamp,
    format_currency,
)
from src.visualization import (
    age_value_scatter,
    attribute_radar,
    correlation_heatmap,
    feature_importance_bar,
    growth_gauge,
    score_by_category_bar,
    trajectory_bar,
    value_by_position_box,
    value_vs_predicted_scatter,
)

# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Football Intelligence Scouting Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


# Section anchors used by the quick-jump nav (label -> header anchor id).
NAV_SECTIONS = [
    ("Explore", "dataset-explorer"),
    ("Discover", "hidden-talent"),
    ("Compare", "player-comparison"),
    ("Predict", "transfer-predictor"),
    ("Forecast", "trajectory"),
    ("Recruit", "recruitment"),
    ("Model Lab", "model-lab"),
    ("Insights", "insights"),
]


def render_nav() -> None:
    """Sticky quick-jump navigation bar linking to each section anchor."""
    links = "".join(
        f"<a class='nav-link' href='#{anchor}'>{label}</a>"
        for label, anchor in NAV_SECTIONS
    )
    st.markdown(
        f"<div class='nav-bar'><span class='nav-brand'>⚽ FISP</span>"
        f"<div class='nav-links'>{links}</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Cached data + model pipeline (runs once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Building scouting intelligence (loading data + training models)…")
def build_platform():
    """Full pipeline: load → clean → engineer → train → score. Cached."""
    raw, source = load_raw_data()
    clean = clean_data(raw)
    feats = engineer_features(clean)
    report = preprocessing_report(raw, clean)

    bundle = train_all_models(feats)
    data = compute_undervalue_score(bundle.predictions)
    data = add_trajectory_category(data)

    # Best-effort artifact persistence (non-fatal if the FS is read-only).
    try:
        data.to_parquet(PROCESSED_DATA_PATH, index=False)
    except Exception:  # noqa: BLE001 - artifact caching is optional
        pass

    return data, bundle, source, report


# --------------------------------------------------------------------------- #
# Small HTML component helpers
# --------------------------------------------------------------------------- #
def kpi_tile(value: str, label: str, foot: str = "") -> str:
    foot_html = f"<div class='kpi-foot'>{foot}</div>" if foot else ""
    return (f"<div class='kpi-tile'><div class='kpi-value'>{value}</div>"
            f"<div class='kpi-label'>{label}</div>{foot_html}</div>")


def score_ring(score: float, label: str = "", size: int = 92) -> str:
    """Inline SVG circular progress ring colored along the red family."""
    score = float(clamp(score, 0, 100))
    r = size / 2 - 8
    circ = 2 * np.pi * r
    dash = circ * score / 100
    color = THEME["green"] if score >= 66 else THEME["amber"] if score >= 40 else THEME["red_bright"]
    label_html = f"<div class='ring-label'>{label}</div>" if label else ""
    return f"""
    <div class='ring-wrap'>
      <svg width='{size}' height='{size}' viewBox='0 0 {size} {size}'>
        <circle cx='{size/2}' cy='{size/2}' r='{r}' fill='none'
                stroke='rgba(255,255,255,0.07)' stroke-width='8'/>
        <circle cx='{size/2}' cy='{size/2}' r='{r}' fill='none'
                stroke='{color}' stroke-width='8' stroke-linecap='round'
                stroke-dasharray='{dash} {circ}'
                transform='rotate(-90 {size/2} {size/2})'/>
        <text x='50%' y='50%' text-anchor='middle' dy='.35em'
              fill='#fff' font-size='{size*0.26}' font-family='Rajdhani' font-weight='700'>
          {score:.0f}</text>
      </svg>{label_html}
    </div>"""


def risk_badge(prob: float) -> tuple[str, str]:
    """Map a success probability to (risk_level, badge_html)."""
    if prob >= 0.66:
        return "Low", "<span class='badge badge-low'>● Low risk</span>"
    if prob >= 0.40:
        return "Medium", "<span class='badge badge-med'>● Medium risk</span>"
    return "High", "<span class='badge badge-high'>● High risk</span>"


def section_header(tag: str, title: str, anchor: str) -> None:
    st.markdown(f"<div class='section-tag'>{tag}</div>", unsafe_allow_html=True)
    st.header(title, anchor=anchor)


def player_card(row: pd.Series, extra_rows: list[tuple[str, str]], reason: str = "") -> str:
    rows_html = "".join(
        f"<div class='pc-row'><span>{k}</span><b>{v}</b></div>" for k, v in extra_rows
    )
    reason_html = f"<div class='pc-reason'>{reason}</div>" if reason else ""
    return (
        f"<div class='player-card'>"
        f"<div class='pc-name'>{row['name']}</div>"
        f"<div class='pc-meta'><span class='badge badge-pos'>{row['position']}</span> "
        f"· {row['club']} · Age {int(row['age'])}</div>"
        f"{rows_html}{reason_html}</div>"
    )


# =========================================================================== #
# Sections
# =========================================================================== #
def render_hero(data: pd.DataFrame, source: str) -> None:
    src_label = ("Real dataset" if source.startswith("real")
                 else "Sample (synthetic) dataset")
    st.markdown(
        f"""
        <div class="hero">
          <span class="hero-kicker">⚽ Data-Driven Scouting · {src_label}</span>
          <div class="hero-title">Football Intelligence<br>Scouting Platform</div>
          <p class="hero-sub">
            An end-to-end analytics platform that uncovers <b>undervalued players</b>,
            forecasts <b>career trajectories</b>, predicts <b>transfer success</b>, and
            builds <b>data-driven recruitment shortlists</b> - turning raw player data
            into scouting decisions.
          </p>
          <div class="pill-row">
            <span class="pill red">Python</span>
            <span class="pill red">Streamlit</span>
            <span class="pill red">pandas / NumPy</span>
            <span class="pill red">scikit-learn</span>
            <span class="pill red">Plotly</span>
            <span class="pill red">Random Forest</span>
            <span class="pill red">Permutation Importance</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(data: pd.DataFrame, bundle) -> None:
    hidden_talents = int((data["undervalue_score"] >= 65).sum())
    r2 = bundle.regression_metrics.get("r2", 0)
    auc = bundle.classification_metrics.get("roc_auc", 0)
    cols = st.columns(5)
    tiles = [
        kpi_tile(f"{len(data):,}", "Players analyzed", "full dataset"),
        kpi_tile(format_currency(data["market_value_eur"].mean()), "Avg market value",
                 f"median {format_currency(data['market_value_eur'].median())}"),
        kpi_tile(f"{data['position'].nunique()}", "Distinct positions",
                 f"{data['position_group'].nunique()} position groups"),
        kpi_tile(f"{hidden_talents:,}", "Hidden talents", "undervalue score ≥ 65"),
        kpi_tile(f"{r2:.2f} / {auc:.2f}", "Model score (R² / AUC)", "value reg / transfer clf"),
    ]
    for col, html in zip(cols, tiles):
        col.markdown(html, unsafe_allow_html=True)


def sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## 🔍 Global filters")
    st.sidebar.caption("Filter the dataset explorer and talent pool.")

    search = st.sidebar.text_input("Search player / club", "")

    groups = st.sidebar.multiselect(
        "Position group", POSITION_GROUP_ORDER, default=POSITION_GROUP_ORDER
    )
    leagues = st.sidebar.multiselect(
        "League", sorted(data["league"].unique()),
        default=sorted(data["league"].unique()),
    )
    nationalities = st.sidebar.multiselect(
        "Nationality (optional)", sorted(data["nationality"].unique()), default=[]
    )

    age = st.sidebar.slider("Age range", int(data["age"].min()), int(data["age"].max()),
                            (int(data["age"].min()), int(data["age"].max())))
    rating = st.sidebar.slider("Overall rating", 40, 99,
                               (int(data["overall_rating"].min()), 99))
    potential = st.sidebar.slider("Potential", 40, 99,
                                  (int(data["potential"].min()), 99))

    vmax = float(data["market_value_eur"].max())
    value = st.sidebar.slider("Max market value (€M)", 0.0, round(vmax / 1e6, 1),
                              round(vmax / 1e6, 1))
    wmax = float(data["wage_eur"].max())
    wage = st.sidebar.slider("Max weekly wage (€K)", 0.0, round(wmax / 1e3, 1),
                             round(wmax / 1e3, 1))

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Drop a real CSV into `data/raw/` and restart to replace the sample data. "
        "See the README for the expected columns."
    )

    mask = (
        data["position_group"].isin(groups)
        & data["league"].isin(leagues)
        & data["age"].between(*age)
        & data["overall_rating"].between(*rating)
        & data["potential"].between(*potential)
        & (data["market_value_eur"] <= value * 1e6)
        & (data["wage_eur"] <= wage * 1e3)
    )
    if nationalities:
        mask &= data["nationality"].isin(nationalities)
    if search.strip():
        s = search.strip().lower()
        mask &= (data["name"].str.lower().str.contains(s)
                 | data["club"].str.lower().str.contains(s))

    return data[mask].copy()


def render_dataset_explorer(fdf: pd.DataFrame, total: int) -> None:
    section_header("01 · Explore", "Dataset Explorer", "dataset-explorer")
    st.caption(f"Showing **{len(fdf):,}** of {total:,} players after filters.")

    if fdf.empty:
        st.warning("No players match the current filters. Loosen them in the sidebar.")
        return

    display_cols = [
        "name", "age", "position", "position_group", "club", "league", "nationality",
        "overall_rating", "potential", "market_value_eur", "wage_eur",
        "predicted_future_value_eur", "undervalue_score", "transfer_success_proba",
    ]
    table = fdf[display_cols].sort_values("market_value_eur", ascending=False)
    st.dataframe(
        table,
        width="stretch", hide_index=True, height=430,
        column_config={
            "market_value_eur": st.column_config.NumberColumn("Value (€)", format="€%d"),
            "wage_eur": st.column_config.NumberColumn("Wage (€/wk)", format="€%d"),
            "predicted_future_value_eur": st.column_config.NumberColumn("Pred. value (€)", format="€%d"),
            "undervalue_score": st.column_config.ProgressColumn(
                "Undervalue", min_value=0, max_value=100, format="%.0f"),
            "transfer_success_proba": st.column_config.ProgressColumn(
                "Transfer success p", min_value=0, max_value=1, format="%.2f"),
            "overall_rating": st.column_config.NumberColumn("OVR"),
            "potential": st.column_config.NumberColumn("POT"),
        },
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(value_by_position_box(fdf), width="stretch")
    with c2:
        st.plotly_chart(age_value_scatter(fdf), width="stretch")


def render_hidden_talent(data: pd.DataFrame) -> None:
    section_header("02 · Discover", "Hidden Talent Discovery", "hidden-talent")
    st.markdown(
        "Players ranked by a transparent **Undervalue Score** - a weighted blend of "
        "predicted value growth, unrealized potential, youth, performance-for-price "
        "and wage efficiency."
    )

    c1, c2, c3 = st.columns([1.2, 1, 1])
    pos_group = c1.selectbox("Position group", ["Any"] + POSITION_GROUP_ORDER, key="ht_pos")
    budget = c2.slider("Max budget (€M)", 0.5, round(float(data["market_value_eur"].max()) / 1e6, 1),
                       min(50.0, round(float(data["market_value_eur"].max()) / 1e6, 1)), key="ht_budget")
    top_n = c3.slider("How many", 3, 12, 6, key="ht_topn")

    pool = data[data["market_value_eur"] <= budget * 1e6]
    if pos_group != "Any":
        pool = pool[pool["position_group"] == pos_group]
    top = pool.sort_values("undervalue_score", ascending=False).head(top_n)

    if top.empty:
        st.warning("No players in this budget / position. Increase the budget.")
        return

    st.plotly_chart(value_vs_predicted_scatter(pool), width="stretch")
    st.caption("Points above the dashed break-even line are predicted to gain value. "
               "Brighter = higher undervalue score.")

    cards = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        rows = [
            ("Undervalue", f"{row['undervalue_score']:.0f}/100"),
            ("Current value", format_currency(row["market_value_eur"])),
            ("Predicted value", format_currency(row["predicted_future_value_eur"])),
            ("Growth", f"{row['predicted_growth_pct']:+.0f}%"),
            ("OVR / POT", f"{int(row['overall_rating'])} / {int(row['potential'])}"),
        ]
        cards[i % 3].markdown(
            player_card(row, rows, row["undervalue_reason"]), unsafe_allow_html=True
        )


def _player_options(data: pd.DataFrame) -> dict:
    """Return {player_id: label} for selectboxes (label includes context)."""
    return {
        int(r.player_id): f"{r.name} · {r.position} · OVR {int(r.overall_rating)} · {r.club}"
        for r in data.itertuples()
    }


def render_comparison(data: pd.DataFrame) -> None:
    section_header("03 · Compare", "Player Comparison", "player-comparison")
    st.markdown(
        "Put two players head-to-head: profile cards, an overlaid attribute radar, "
        "and an automatic verdict on value, upside and quality."
    )

    opts = _player_options(data)
    ids = list(opts.keys())
    c1, c2 = st.columns(2)
    pid_a = c1.selectbox("Player A", ids, index=0, format_func=lambda k: opts[k], key="cmp_a")
    pid_b = c2.selectbox("Player B", ids, index=min(1, len(ids) - 1),
                         format_func=lambda k: opts[k], key="cmp_b")

    pa = data[data["player_id"] == pid_a].iloc[0]
    pb = data[data["player_id"] == pid_b].iloc[0]

    def _compare_rows(p: pd.Series) -> list[tuple[str, str]]:
        return [
            ("Overall / Potential", f"{int(p['overall_rating'])} / {int(p['potential'])}"),
            ("Market value", format_currency(p["market_value_eur"])),
            ("Predicted value", format_currency(p["predicted_future_value_eur"])),
            ("Growth", f"{p['predicted_growth_pct']:+.0f}%"),
            ("Undervalue score", f"{p['undervalue_score']:.0f}/100"),
            ("Trajectory", str(p["trajectory_category"])),
        ]

    cards = st.columns([1, 1.25, 1])
    cards[0].markdown(player_card(pa, _compare_rows(pa)), unsafe_allow_html=True)
    with cards[1]:
        st.plotly_chart(attribute_radar(pd.concat([pa.to_frame().T, pb.to_frame().T])),
                        width="stretch")
    cards[2].markdown(player_card(pb, _compare_rows(pb)), unsafe_allow_html=True)

    # Automatic verdict on the metrics that matter most.
    better_value = pa["name"] if pa["undervalue_score"] >= pb["undervalue_score"] else pb["name"]
    better_growth = pa["name"] if pa["predicted_growth_pct"] >= pb["predicted_growth_pct"] else pb["name"]
    better_quality = pa["name"] if pa["overall_rating"] >= pb["overall_rating"] else pb["name"]
    st.markdown(
        f"<div class='glass'><b>Verdict:</b> "
        f"<span class='badge badge-high'>Best value</span> {better_value} &nbsp; "
        f"<span class='badge badge-low'>Most upside</span> {better_growth} &nbsp; "
        f"<span class='badge badge-pos'>Higher quality now</span> {better_quality}</div>",
        unsafe_allow_html=True,
    )


def render_transfer_predictor(data: pd.DataFrame, bundle) -> None:
    section_header("04 · Predict", "Transfer Success Predictor", "transfer-predictor")
    st.markdown(
        "Estimates the probability a transfer **works out**, then adjusts for the "
        "target club's level and style. The base model is trained on a documented "
        "**proxy label** (see the Model Lab section)."
    )

    opts = _player_options(data)
    c1, c2, c3 = st.columns([1.6, 1, 1])
    pid = c1.selectbox("Select a player", list(opts.keys()),
                       format_func=lambda k: opts[k], key="tp_player")
    tier = c2.selectbox("Target club level",
                        ["Elite (CL)", "Established", "Mid-table", "Developing"], index=1)
    style = c3.selectbox("Club playing style", list(STYLE_TO_ATTRIBUTE.keys()), key="tp_style")

    player = data[data["player_id"] == pid].iloc[0]
    base = float(player["transfer_success_proba"])

    # --- Heuristic club-fit adjustment (documented, on top of base model) -- #
    tier_overall = {"Elite (CL)": 84, "Established": 77, "Mid-table": 70, "Developing": 63}[tier]
    gap = player["overall_rating"] - tier_overall
    if gap < -6:
        fit_adj, fit_note = -0.18, "may struggle for minutes at this level"
    elif gap < 0:
        fit_adj, fit_note = -0.05, "slightly below the typical level here"
    elif gap <= 8:
        fit_adj, fit_note = 0.10, "well matched to the squad level"
    else:
        fit_adj, fit_note = 0.04, "likely an instant starter (limited stretch)"

    if player["age"] <= 23 and tier in ("Developing", "Mid-table"):
        fit_adj += 0.05  # young player + project club synergy

    style_attr = STYLE_TO_ATTRIBUTE.get(style)
    style_match = False
    if style_attr is not None:
        top_attr = player[["pace", "shooting", "passing", "dribbling", "defending", "physic"]].idxmax()
        if top_attr == style_attr:
            fit_adj += 0.05
            style_match = True

    final = float(np.clip(base + fit_adj, 0.02, 0.98))
    _, badge = risk_badge(final)

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown(f"<div class='glass'>{score_ring(final * 100, 'Success probability', 120)}"
                    f"<div style='margin-top:10px'>{badge}</div></div>", unsafe_allow_html=True)
        st.metric("Base model probability", f"{base:.0%}",
                  delta=f"{fit_adj:+.0%} club fit")
    with right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(f"**Fit summary - {player['name']} → {tier} club**")
        st.write(
            f"- Player level **OVR {int(player['overall_rating'])} / POT {int(player['potential'])}** "
            f"vs club benchmark **{tier_overall}** → {fit_note}.\n"
            f"- Age **{int(player['age'])}**, value **{format_currency(player['market_value_eur'])}**, "
            f"predicted growth **{player['predicted_growth_pct']:+.0f}%**.\n"
            f"- Style **{style}** → "
            + ("**matches** the player's strongest attribute." if style_match
               else "not the player's standout strength.")
        )
        st.markdown(f"Overall risk level: {badge}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Key factors the model weighs most** (permutation importance):")
    st.plotly_chart(
        feature_importance_bar(bundle.classification_importance, ""),
        width="stretch",
    )


def render_trajectory(data: pd.DataFrame, bundle) -> None:
    section_header("05 · Forecast", "Career Trajectory Prediction", "trajectory")
    st.markdown(
        "Projects a player's **future market value** (2-season horizon) and assigns a "
        "career-trajectory category from predicted growth and age."
    )

    opts = _player_options(data)
    pid = st.selectbox("Select a player", list(opts.keys()),
                       format_func=lambda k: opts[k], key="traj_player")
    player = data[data["player_id"] == pid].iloc[0]

    cat_colors = {"Rising Star": "badge-low", "Stable Performer": "badge-pos",
                  "Declining Asset": "badge-high", "High-Risk Prospect": "badge-med"}
    cat = player["trajectory_category"]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_tile(format_currency(player["market_value_eur"]), "Current value"), unsafe_allow_html=True)
    c2.markdown(kpi_tile(format_currency(player["predicted_future_value_eur"]), "Predicted value"), unsafe_allow_html=True)
    c3.markdown(kpi_tile(f"{player['predicted_growth_pct']:+.0f}%", "Expected growth"), unsafe_allow_html=True)
    c4.markdown(
        f"<div class='kpi-tile'><div class='kpi-value'><span class='badge {cat_colors[cat]}'>{cat}</span></div>"
        f"<div class='kpi-label'>Trajectory category</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(growth_gauge(float(player["predicted_growth_pct"])),
                        width="stretch")
    with right:
        similar = find_similar_players(data, pid, top_n=4)
        radar_players = pd.concat([player.to_frame().T, similar]).head(3)
        st.plotly_chart(attribute_radar(radar_players), width="stretch")
        st.caption("Attribute profile vs the most similar players in the dataset.")

    st.markdown("**How the cohort breaks down by trajectory:**")
    st.plotly_chart(trajectory_bar(data), width="stretch")


def render_recruitment(data: pd.DataFrame) -> None:
    section_header("06 · Recruit", "Club Recruitment Engine", "recruitment")
    st.markdown(
        "Define a club's needs and get a ranked shortlist scored on quality, style "
        "fit, value-for-money and budget headroom."
    )

    c1, c2, c3 = st.columns(3)
    pos_group = c1.selectbox("Target position group", ["Any"] + POSITION_GROUP_ORDER, key="rec_pos")
    style = c2.selectbox("Playing style priority", list(STYLE_TO_ATTRIBUTE.keys()), key="rec_style")
    top_n = c3.slider("Shortlist size", 3, 15, 8, key="rec_n")

    c4, c5, c6 = st.columns(3)
    budget = c4.slider("Max budget (€M)", 0.5,
                       round(float(data["market_value_eur"].max()) / 1e6, 1), 40.0, key="rec_budget")
    age_range = c5.slider("Age range", int(data["age"].min()), int(data["age"].max()), (18, 27), key="rec_age")
    min_pot = c6.slider("Minimum potential", 50, 95, 75, key="rec_pot")

    shortlist = recommend_players(
        data, position_group=pos_group, max_budget=budget * 1e6,
        age_range=age_range, min_potential=min_pot, style=style, top_n=top_n,
    )

    if shortlist.empty:
        st.warning("No players match these criteria. Raise the budget or relax the filters.")
        return

    st.success(f"Top {len(shortlist)} recommended targets within "
               f"{format_currency(budget * 1e6)} budget.")

    show = shortlist[[
        "name", "position", "club", "age", "market_value_eur",
        "predicted_future_value_eur", "undervalue_score", "fit_score",
        "recommendation_reason",
    ]]
    st.dataframe(
        show, width="stretch", hide_index=True,
        column_config={
            "market_value_eur": st.column_config.NumberColumn("Value (€)", format="€%d"),
            "predicted_future_value_eur": st.column_config.NumberColumn("Pred. value (€)", format="€%d"),
            "undervalue_score": st.column_config.ProgressColumn("Undervalue", min_value=0, max_value=100, format="%.0f"),
            "fit_score": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%.0f"),
            "recommendation_reason": st.column_config.TextColumn("Why", width="large"),
        },
    )

    st.markdown("##### Shortlist cards")
    cards = st.columns(3)
    for i, (_, row) in enumerate(shortlist.head(6).iterrows()):
        rows = [
            ("Fit score", f"{row['fit_score']:.0f}/100"),
            ("Value", format_currency(row["market_value_eur"])),
            ("Pred. value", format_currency(row["predicted_future_value_eur"])),
            ("Age / POT", f"{int(row['age'])} / {int(row['potential'])}"),
        ]
        cards[i % 3].markdown(
            player_card(row, rows, row["recommendation_reason"]), unsafe_allow_html=True
        )


def render_model_lab(bundle, report: dict) -> None:
    section_header("07 · Under the hood", "Model Lab & Data Quality", "model-lab")
    rm, cm = bundle.regression_metrics, bundle.classification_metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_tile(f"{rm['r2']:.2f}", "Value model R²", f"on {rm['n_test']} test players"), unsafe_allow_html=True)
    c2.markdown(kpi_tile(format_currency(rm["mae"]), "Value model MAE", "avg abs error"), unsafe_allow_html=True)
    c3.markdown(kpi_tile(f"{cm['roc_auc']:.2f}", "Transfer model AUC", f"F1 {cm['f1']:.2f}"), unsafe_allow_html=True)
    c4.markdown(kpi_tile(f"{cm['accuracy']:.0%}", "Transfer accuracy", f"recall {cm['recall']:.0%}"), unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Value model importance", "Transfer model importance", "Data quality"])
    with t1:
        st.plotly_chart(feature_importance_bar(bundle.regression_importance,
                        "What drives predicted value"), width="stretch")
    with t2:
        st.plotly_chart(feature_importance_bar(bundle.classification_importance,
                        "What drives transfer success"), width="stretch")
    with t3:
        rc1, rc2 = st.columns([1, 1.3])
        with rc1:
            st.write("**Cleaning summary**")
            st.json(report)
        with rc2:
            st.plotly_chart(correlation_heatmap(bundle.predictions),
                            width="stretch")


def render_insights(data: pd.DataFrame) -> None:
    section_header("08 · Story", "Insights & Business Value", "insights")

    # Data-derived insight: which position group offers best value-for-money.
    by_group = (data.groupby("position_group")["undervalue_score"].mean()
                .sort_values(ascending=False))
    best_group = by_group.index[0]
    young_risers = int(((data["age"] <= 23) & (data["predicted_growth_pct"] >= 15)).sum())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class='glass story'>
            <h4>📌 What's undervalued</h4>
            <p>The strongest value signals cluster around <span class='accent'>younger players
            with a wide gap between potential and current rating</span>, especially when wages
            stay modest. <b>{young_risers}</b> players under 23 are projected to grow their value
            by 15%+.</p>
            <h4>🎯 Best value by position</h4>
            <p><span class='accent'>{best_group}s</span> show the highest average undervalue score
            in this dataset - a useful starting point for budget-conscious recruitment.</p>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(
            """<div class='glass story'>
            <h4>🏟️ How clubs use this</h4>
            <p>Scouts shortlist with the <span class='accent'>Recruitment Engine</span>, sanity-check
            upside with <span class='accent'>Trajectory</span> forecasts, and de-risk deals with the
            <span class='accent'>Transfer Success</span> model - replacing gut feel with a repeatable,
            explainable process.</p>
            <h4>⚠️ Limitations</h4>
            <p>Targets here are <span class='accent'>documented proxies</span>, not observed outcomes,
            and the bundled data is synthetic sample data. Swap in a real dataset (see README) to
            move from demonstration to production.</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("##### Average undervalue score by position group")
    st.plotly_chart(score_by_category_bar(by_group, "Avg undervalue score"),
                    width="stretch")


def render_footer() -> None:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer">
          <b>Football Intelligence Scouting Platform</b> - A Data Analytics project.<br>
          Built with Python · Streamlit · pandas · NumPy · scikit-learn · Plotly.
          Synthetic, non-branded data.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================== #
# Main
# =========================================================================== #
def main() -> None:
    inject_css()
    try:
        data, bundle, source, report = build_platform()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Failed to build the platform. This usually means the dataset could not "
            "be loaded.\n\n"
            f"**Details:** {exc}\n\n"
            "Fix: ensure dependencies are installed (`pip install -r requirements.txt`). "
            "To use your own data, place a CSV in `data/raw/` (see README for columns); "
            "otherwise the app generates synthetic sample data automatically."
        )
        st.stop()


    render_nav()
    render_hero(data, source)
    render_kpis(data, bundle)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)

    fdf = sidebar_filters(data)

    render_dataset_explorer(fdf, total=len(data))
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    render_hidden_talent(data)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    render_comparison(data)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    render_transfer_predictor(data, bundle)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    render_trajectory(data, bundle)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    render_recruitment(data)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    render_model_lab(bundle, report)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    render_insights(data)
    render_footer()


if __name__ == "__main__":
    main()
