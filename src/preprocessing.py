"""Cleaning and standardization.

Turns a raw DataFrame (either our synthetic sample or a real third-party CSV)
into a clean, consistently-named frame the rest of the pipeline expects.

The column-alias map lets common real-world FIFA / market-value datasets
(e.g. SoFIFA / EA-FC style exports) flow straight in without code changes.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .utils import (
    ATTRIBUTE_COLS,
    PERFORMANCE_COLS,
    POSITION_GROUPS,
)

# Map of {raw_name_lower: canonical_name}. Keys are matched case-insensitively
# after light normalization (spaces/dots -> underscores).
COLUMN_ALIASES: dict[str, str] = {
    "short_name": "name",
    "long_name": "name",
    "player_name": "name",
    "full_name": "name",
    "club_name": "club",
    "team": "club",
    "league_name": "league",
    "league": "league",
    "nationality_name": "nationality",
    "nation": "nationality",
    "player_positions": "position",
    "best_position": "position",
    "overall": "overall_rating",
    "rating": "overall_rating",
    "value_eur": "market_value_eur",
    "value": "market_value_eur",
    "market_value": "market_value_eur",
    "wage": "wage_eur",
    "height": "height_cm",
    "weight": "weight_kg",
    "preferred_foot": "preferred_foot",
    "international_reputation": "international_reputation",
    "skill_moves": "skill_moves",
    "weak_foot": "weak_foot",
    # Attribute aliases
    "pace": "pace",
    "shooting": "shooting",
    "passing": "passing",
    "dribbling": "dribbling",
    "defending": "defending",
    "physic": "physic",
    "physicality": "physic",
}

REQUIRED_NUMERIC = [
    "age", "overall_rating", "potential", "market_value_eur", "wage_eur",
]


def _normalize_name(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[\s\.\-/]+", "_", col)
    col = re.sub(r"[^0-9a-z_]", "", col)
    return col


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/underscore column names and apply the alias map."""
    df = df.copy()
    df.columns = [_normalize_name(c) for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    # If aliasing produced duplicate names, keep the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _coerce_market_value(series: pd.Series) -> pd.Series:
    """Parse values that may arrive as strings like '€12.5M', '500K', '1.2'."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def parse(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().replace("€", "").replace(",", "").upper()
        mult = 1.0
        if s.endswith("M"):
            mult, s = 1_000_000, s[:-1]
        elif s.endswith("K"):
            mult, s = 1_000, s[:-1]
        elif s.endswith("B"):
            mult, s = 1_000_000_000, s[:-1]
        try:
            return float(s) * mult
        except ValueError:
            return np.nan

    return series.map(parse)


def _primary_position(series: pd.Series) -> pd.Series:
    """Real datasets list multiple positions ('ST, CF'); keep the first."""
    return (
        series.astype(str)
        .str.split(r"[,/|]").str[0]
        .str.strip().str.upper()
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: standardize, coerce, impute, deduplicate."""
    df = standardize_columns(df)

    # --- Identity & categorical fields ------------------------------------ #
    if "name" not in df.columns:
        df["name"] = [f"Player {i}" for i in range(len(df))]
    if "player_id" not in df.columns:
        df["player_id"] = np.arange(100000, 100000 + len(df))

    for col, default in [("club", "Unknown Club"), ("league", "Unknown League"),
                         ("nationality", "Unknown"), ("preferred_foot", "Right")]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str).str.strip()

    # --- Position & position group ---------------------------------------- #
    if "position" not in df.columns:
        df["position"] = "CM"
    df["position"] = _primary_position(df["position"])
    df.loc[~df["position"].isin(POSITION_GROUPS), "position"] = "CM"
    df["position_group"] = df["position"].map(POSITION_GROUPS)

    # --- Numeric coercion -------------------------------------------------- #
    if "market_value_eur" in df.columns:
        df["market_value_eur"] = _coerce_market_value(df["market_value_eur"])
    if "wage_eur" in df.columns:
        df["wage_eur"] = _coerce_market_value(df["wage_eur"])

    numeric_like = (
        REQUIRED_NUMERIC + ATTRIBUTE_COLS + PERFORMANCE_COLS
        + ["height_cm", "weight_kg", "contract_years_left",
           "international_reputation", "skill_moves", "weak_foot"]
    )
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Ensure essential numeric columns exist with sane defaults -------- #
    defaults = {
        "age": 25, "overall_rating": 65, "potential": 70,
        "market_value_eur": 500_000, "wage_eur": 10_000,
        "height_cm": 180, "weight_kg": 75, "contract_years_left": 2,
        "international_reputation": 1, "skill_moves": 2, "weak_foot": 3,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Attributes / performance: create if missing, then impute by group median.
    for col in ATTRIBUTE_COLS + PERFORMANCE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # --- Missing-value imputation ----------------------------------------- #
    # Numeric: fill by position-group median, then global median as backstop.
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df.groupby("position_group")[num_cols].transform(
        lambda s: s.fillna(s.median())
    )
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[num_cols] = df[num_cols].fillna(0)

    # Potential should never be below current rating.
    df["potential"] = np.maximum(df["potential"], df["overall_rating"])

    # --- Deduplicate & finalize ------------------------------------------- #
    df = df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    df["age"] = df["age"].round().astype(int)
    df["overall_rating"] = df["overall_rating"].round().astype(int)
    df["potential"] = df["potential"].round().astype(int)

    return df


def preprocessing_report(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Lightweight before/after summary for documentation / the UI."""
    return {
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "raw_cols": raw.shape[1],
        "clean_cols": clean.shape[1],
        "missing_before": int(raw.isna().sum().sum()),
        "missing_after": int(clean.isna().sum().sum()),
        "duplicates_removed": len(raw) - len(clean),
    }
