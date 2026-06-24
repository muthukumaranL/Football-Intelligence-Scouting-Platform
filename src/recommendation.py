"""Scoring & recommendation logic.

Three building blocks used by the dashboard:

  * ``compute_undervalue_score`` — a transparent, weighted composite index that
    flags players whose upside (predicted growth, potential, youth, output and
    wage efficiency) outstrips their current price tag.
  * ``recommend_players`` — the Club Recruitment Engine: hard filters on a
    club's needs + a soft fit score for ranking a shortlist.
  * ``find_similar_players`` — attribute-space nearest neighbours for the
    player-comparison views.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import ATTRIBUTE_COLS, safe_minmax

# --------------------------------------------------------------------------- #
# Undervalue score
# --------------------------------------------------------------------------- #
# Documented, explainable weights (sum to 1.0). See docs/methodology.md.
UNDERVALUE_WEIGHTS = {
    "upside": 0.30,          # predicted value growth
    "potential": 0.20,       # unrealized rating headroom
    "age_advantage": 0.15,   # years to peak
    "quality_per_cost": 0.20,  # performance relative to price
    "wage_efficiency": 0.15,   # rating delivered per wage
}

_COMPONENT_LABELS = {
    "upside": "high predicted value growth",
    "potential": "large unrealized potential",
    "age_advantage": "youth / room to develop",
    "quality_per_cost": "strong performance for the price",
    "wage_efficiency": "excellent wage efficiency",
}


def compute_undervalue_score(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``undervalue_score`` (0-100), its components, and a reason string.

    Requires ``predicted_growth_pct`` / ``predicted_future_value_eur`` from the
    model bundle; falls back to ``value_growth_pct`` if predictions are absent.
    """
    df = df.copy()

    growth = df.get("predicted_growth_pct", df.get("value_growth_pct", 0))
    # Clip extreme growth so a few outliers don't dominate the scale.
    upside = safe_minmax(np.clip(growth, -50, 150))
    potential = safe_minmax(df["potential_gap"])
    age_adv = safe_minmax(df["years_to_peak"].clip(lower=0))
    quality_per_cost = safe_minmax(
        df["performance_index"] / np.log1p(df["market_value_eur"])
    )
    wage_eff = safe_minmax(df["rating_per_wage"])

    components = {
        "upside": upside,
        "potential": potential,
        "age_advantage": age_adv,
        "quality_per_cost": quality_per_cost,
        "wage_efficiency": wage_eff,
    }
    for name, series in components.items():
        df[f"uv_{name}"] = (series * 100).round(1)

    score = sum(components[name] * w for name, w in UNDERVALUE_WEIGHTS.items())
    df["undervalue_score"] = (score * 100).round(1)

    # Build a short, human reason from the two strongest weighted components.
    comp_df = pd.DataFrame(
        {name: components[name] * w for name, w in UNDERVALUE_WEIGHTS.items()}
    )
    top2 = comp_df.apply(lambda r: r.nlargest(2).index.tolist(), axis=1)
    df["undervalue_reason"] = top2.apply(
        lambda names: "Flagged for " + " and ".join(_COMPONENT_LABELS[n] for n in names) + "."
    )
    return df


# --------------------------------------------------------------------------- #
# Club Recruitment Engine
# --------------------------------------------------------------------------- #
# Playing-style priorities map to the attribute they emphasize.
STYLE_TO_ATTRIBUTE = {
    "Balanced (overall quality)": None,
    "Pace & transitions": "pace",
    "Clinical finishing": "shooting",
    "Creative playmaking": "passing",
    "Dribbling & flair": "dribbling",
    "Defensive solidity": "defending",
    "Physical duels": "physic",
}

RECRUITMENT_WEIGHTS = {
    "quality": 0.30,
    "style": 0.25,
    "value_for_money": 0.25,
    "budget_fit": 0.20,
}


def recommend_players(
    df: pd.DataFrame,
    position_group: str | None = None,
    positions: list[str] | None = None,
    max_budget: float | None = None,
    age_range: tuple[int, int] = (16, 40),
    min_potential: int = 0,
    min_rating: int = 0,
    style: str = "Balanced (overall quality)",
    top_n: int = 10,
) -> pd.DataFrame:
    """Return a ranked shortlist of players matching a club's needs.

    Hard filters narrow the pool; a soft 0-100 ``fit_score`` ranks what's left.
    Requires ``undervalue_score`` to already be present (run
    :func:`compute_undervalue_score` first).
    """
    pool = df.copy()

    # --- Hard filters (club constraints) ---------------------------------- #
    if position_group and position_group != "Any":
        pool = pool[pool["position_group"] == position_group]
    if positions:
        pool = pool[pool["position"].isin(positions)]
    if max_budget is not None:
        pool = pool[pool["market_value_eur"] <= max_budget]
    pool = pool[pool["age"].between(age_range[0], age_range[1])]
    pool = pool[pool["potential"] >= min_potential]
    pool = pool[pool["overall_rating"] >= min_rating]

    if pool.empty:
        return pool.assign(fit_score=[], recommendation_reason=[])

    # --- Soft fit score ---------------------------------------------------- #
    quality = safe_minmax(0.6 * pool["overall_rating"] + 0.4 * pool["potential"])

    style_attr = STYLE_TO_ATTRIBUTE.get(style)
    style_score = (
        safe_minmax(pool["performance_index"]) if style_attr is None
        else safe_minmax(pool[style_attr])
    )

    value_for_money = safe_minmax(pool["undervalue_score"])

    if max_budget:
        # Spending less of the budget scores higher (leaves room elsewhere).
        budget_fit = (1 - (pool["market_value_eur"] / max_budget)).clip(0, 1)
    else:
        budget_fit = safe_minmax(-pool["market_value_eur"])

    fit = (
        RECRUITMENT_WEIGHTS["quality"] * quality
        + RECRUITMENT_WEIGHTS["style"] * style_score
        + RECRUITMENT_WEIGHTS["value_for_money"] * value_for_money
        + RECRUITMENT_WEIGHTS["budget_fit"] * budget_fit
    )
    pool = pool.assign(fit_score=(fit * 100).round(1))

    def _reason(row: pd.Series) -> str:
        bits = []
        if row["potential_gap"] >= 4:
            bits.append(f"+{int(row['potential_gap'])} potential headroom")
        if style_attr:
            bits.append(f"{style.split('(')[0].strip().lower()} fit ({int(row[style_attr])})")
        if row["undervalue_score"] >= 60:
            bits.append("strong value")
        if max_budget and row["market_value_eur"] <= 0.6 * max_budget:
            bits.append("well within budget")
        if not bits:
            bits.append(f"solid all-round profile (OVR {int(row['overall_rating'])})")
        return "; ".join(bits[:3]).capitalize()

    pool["recommendation_reason"] = pool.apply(_reason, axis=1)
    return pool.sort_values("fit_score", ascending=False).head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Player similarity (for comparison views)
# --------------------------------------------------------------------------- #
def find_similar_players(df: pd.DataFrame, player_id: int, top_n: int = 5) -> pd.DataFrame:
    """Find the most similar players in attribute space (Euclidean distance)."""
    feature_cols = ATTRIBUTE_COLS + ["overall_rating", "potential", "age"]
    feats = df[feature_cols].apply(safe_minmax)

    if player_id not in df["player_id"].values:
        return df.head(0)

    target_idx = df.index[df["player_id"] == player_id][0]
    target_vec = feats.loc[target_idx].to_numpy()
    distances = np.sqrt(((feats.to_numpy() - target_vec) ** 2).sum(axis=1))

    out = df.copy()
    out["similarity"] = (1 / (1 + distances)).round(3)
    out = out[out["player_id"] != player_id]
    return out.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Career trajectory categorization
# --------------------------------------------------------------------------- #
def classify_trajectory(row: pd.Series) -> str:
    """Bucket a player into a career-trajectory category.

    Combines predicted growth with age to separate genuine risers from
    declining or speculative profiles.
    """
    growth = row.get("predicted_growth_pct", row.get("value_growth_pct", 0))
    age = row["age"]
    potential_gap = row.get("potential_gap", 0)
    overall = row.get("overall_rating", 70)

    # Very young, big unrealized gap, still raw: high upside but speculative.
    if age <= 20 and potential_gap >= 11 and overall < 68:
        return "High-Risk Prospect"
    # Clear, near-term value growth while still young.
    if growth >= 14 and age <= 24:
        return "Rising Star"
    # Losing value, or older with no growth left.
    if growth <= -5 or (age >= 32 and growth < 2):
        return "Declining Asset"
    return "Stable Performer"


def add_trajectory_category(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized helper to add the ``trajectory_category`` column."""
    df = df.copy()
    df["trajectory_category"] = df.apply(classify_trajectory, axis=1)
    return df
