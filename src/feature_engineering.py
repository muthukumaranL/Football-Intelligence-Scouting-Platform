"""Feature engineering and target construction.

This module derives the analytical features used by the models and the
dashboard, and constructs two **proxy targets** because the underlying data
(real FIFA-style exports included) does not ship with ground-truth labels for
"future value" or "transfer success":

  * ``future_value_eur`` — an expected 2-season market value built from a
    transparent age/potential domain curve plus noise. Models then *learn* this
    valuation function from player features. (See docs/methodology.md.)

  * ``transfer_success`` — a binary proxy label derived from a latent
    "fit/upside" score plus noise, used to demonstrate the classification
    workflow. It is NOT observed transfer outcomes.

Both assumptions are documented in docs/model_card.md and surfaced in the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_minmax

# Peak age used for the value/age curve.
PEAK_AGE = 27

# Role weights for the composite performance index (sum to 1 per role).
_ROLE_WEIGHTS = {
    "Goalkeeper": {"defending": 0.30, "physic": 0.25, "passing": 0.20,
                   "pace": 0.10, "dribbling": 0.10, "shooting": 0.05},
    "Defender": {"defending": 0.40, "physic": 0.20, "pace": 0.15,
                 "passing": 0.15, "dribbling": 0.07, "shooting": 0.03},
    "Midfielder": {"passing": 0.28, "dribbling": 0.22, "defending": 0.18,
                   "physic": 0.12, "pace": 0.10, "shooting": 0.10},
    "Forward": {"shooting": 0.32, "dribbling": 0.26, "pace": 0.22,
                "passing": 0.10, "physic": 0.06, "defending": 0.04},
}


def _per90(count: pd.Series, minutes: pd.Series) -> pd.Series:
    """Per-90-minutes rate, guarding against divide-by-zero."""
    minutes = minutes.replace(0, np.nan)
    return (count / minutes * 90).fillna(0)


def _performance_index(df: pd.DataFrame) -> pd.Series:
    """Role-weighted attribute score (0-100) with an output bonus."""
    idx = pd.Series(0.0, index=df.index)
    for group, weights in _ROLE_WEIGHTS.items():
        mask = df["position_group"] == group
        if not mask.any():
            continue
        score = sum(df.loc[mask, attr] * w for attr, w in weights.items())
        idx.loc[mask] = score

    # Output bonus from goal contributions per 90 (small, capped).
    contrib = _per90(df["goals"] + df["assists"], df["minutes_played"])
    idx = idx + np.clip(contrib * 4, 0, 12)
    return idx.clip(0, 100)


def _expected_future_value(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Transparent 2-season value projection used as the regression target.

    Multiplicative growth from three intuitive drivers:
      * Age curve  — value rises toward the peak age, declines after.
      * Potential  — unrealized headroom (potential - overall) lifts value.
      * Form       — strong performers retain/grow value.
    A lognormal noise term keeps the relationship learnable, not perfect.
    """
    age = df["age"].to_numpy()
    # Smooth bump peaking at PEAK_AGE; younger players have most upside.
    age_curve = 1.0 + 0.22 * np.exp(-((age - (PEAK_AGE - 3)) ** 2) / 30.0)
    age_decline = np.where(age > PEAK_AGE, 1.0 - (age - PEAK_AGE) * 0.045, 1.0)

    potential_gap = (df["potential"] - df["overall_rating"]).to_numpy()
    potential_factor = 1.0 + potential_gap * 0.045

    form = safe_minmax(df["performance_index"]).to_numpy()
    form_factor = 0.92 + 0.18 * form

    growth = age_curve * age_decline * potential_factor * form_factor
    noise = rng.lognormal(0, 0.14, size=len(df))
    future = df["market_value_eur"].to_numpy() * growth * noise
    return pd.Series(np.clip(future, 20_000, 350_000_000), index=df.index)


def _transfer_success_label(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Binary proxy for 'would a transfer of this player tend to succeed'.

    Built from a latent score combining youth, unrealized potential, current
    quality, value-for-money and reputation — then thresholded at the median so
    classes are roughly balanced. Noise prevents a trivially separable target.
    """
    youth = safe_minmax(-df["age"])
    potential_gap = safe_minmax(df["potential"] - df["overall_rating"])
    quality = safe_minmax(df["overall_rating"])
    performance = safe_minmax(df["performance_index"])
    # Value efficiency: quality per euro (cheaper, good players = better bet).
    value_eff = safe_minmax(df["overall_rating"] / np.log1p(df["market_value_eur"]))
    reputation = safe_minmax(df["international_reputation"])

    latent = (
        0.22 * youth + 0.24 * potential_gap + 0.20 * quality
        + 0.16 * performance + 0.12 * value_eff + 0.06 * reputation
    )
    latent = latent + rng.normal(0, 0.09, size=len(df))
    threshold = np.median(latent)
    return (latent > threshold).astype(int)


def engineer_features(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Add derived features and proxy targets. Returns a new DataFrame."""
    rng = np.random.default_rng(seed)
    df = df.copy()

    # --- Efficiency & gap features ---------------------------------------- #
    df["potential_gap"] = (df["potential"] - df["overall_rating"]).clip(lower=0)
    df["years_to_peak"] = (PEAK_AGE - df["age"]).clip(lower=-12)
    df["value_per_rating"] = df["market_value_eur"] / df["overall_rating"].clip(lower=1)
    # Wage efficiency: rating delivered per €1k of weekly wage.
    df["rating_per_wage"] = df["overall_rating"] / (df["wage_eur"].clip(lower=1) / 1000)
    df["value_to_wage_ratio"] = df["market_value_eur"] / df["wage_eur"].clip(lower=1)

    # --- Performance features --------------------------------------------- #
    df["goals_per_90"] = _per90(df["goals"], df["minutes_played"]).round(3)
    df["assists_per_90"] = _per90(df["assists"], df["minutes_played"]).round(3)
    df["goal_contributions_per_90"] = (df["goals_per_90"] + df["assists_per_90"]).round(3)
    df["performance_index"] = _performance_index(df).round(2)

    # --- Proxy targets ----------------------------------------------------- #
    df["future_value_eur"] = _expected_future_value(df, rng).round(-3)
    df["value_growth_pct"] = (
        (df["future_value_eur"] - df["market_value_eur"])
        / df["market_value_eur"].clip(lower=1) * 100
    ).round(1)
    df["transfer_success"] = _transfer_success_label(df, rng)

    return df


# Feature lists consumed by the models (kept here so they stay in sync).
REGRESSION_FEATURES = [
    "age", "overall_rating", "potential", "potential_gap", "years_to_peak",
    "market_value_eur", "wage_eur", "international_reputation", "skill_moves",
    "weak_foot", "pace", "shooting", "passing", "dribbling", "defending",
    "physic", "performance_index", "goal_contributions_per_90",
    "contract_years_left",
]

CLASSIFICATION_FEATURES = [
    "age", "overall_rating", "potential", "potential_gap", "years_to_peak",
    "market_value_eur", "wage_eur", "rating_per_wage", "international_reputation",
    "performance_index", "goal_contributions_per_90", "contract_years_left",
]
