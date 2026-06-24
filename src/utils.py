"""Shared constants, paths and small helper functions.

Keeping configuration in one place avoids hardcoding machine-specific paths
throughout the project. All paths are resolved relative to the project root,
so the project is portable across machines and works on Streamlit Cloud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the project root, never hardcoded)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

SAMPLE_DATA_PATH = SAMPLE_DIR / "players_sample.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "players_processed.parquet"

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
# Detailed positions mapped to high-level groups used across the app.
POSITION_GROUPS: dict[str, str] = {
    "GK": "Goalkeeper",
    "CB": "Defender",
    "LB": "Defender",
    "RB": "Defender",
    "LWB": "Defender",
    "RWB": "Defender",
    "CDM": "Midfielder",
    "CM": "Midfielder",
    "CAM": "Midfielder",
    "LM": "Midfielder",
    "RM": "Midfielder",
    "LW": "Forward",
    "RW": "Forward",
    "ST": "Forward",
    "CF": "Forward",
}

DETAILED_POSITIONS = list(POSITION_GROUPS.keys())
POSITION_GROUP_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

# FIFA-style attribute columns (0-99 scale).
ATTRIBUTE_COLS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

# Performance columns (per-season).
PERFORMANCE_COLS = ["appearances", "goals", "assists", "minutes_played"]

# Career-trajectory buckets used by the trajectory module.
TRAJECTORY_CATEGORIES = [
    "Rising Star",
    "Stable Performer",
    "Declining Asset",
    "High-Risk Prospect",
]

# --------------------------------------------------------------------------- #
# Theme palette (dark mode, shades of red) — referenced by visualization.py
# --------------------------------------------------------------------------- #
THEME = {
    "bg": "#0b0d12",
    "surface": "#13161d",
    "surface_alt": "#1b1f29",
    "text": "#e8eaf0",
    "muted": "#8b93a7",
    "grid": "#262b36",
    # Red accent family
    "red": "#e11d3a",
    "red_bright": "#ff2e4d",
    "red_soft": "#ff6b81",
    "red_deep": "#8c0f23",
    # Supporting accents for multi-series charts
    "amber": "#f5a524",
    "green": "#21c17a",
    "blue": "#3b82f6",
    "purple": "#a855f7",
}

# Continuous color scale for value/heat charts (dark -> bright red).
RED_SCALE = [
    [0.0, "#2a0a12"],
    [0.4, "#7a1224"],
    [0.7, "#d11e3c"],
    [1.0, "#ff5a72"],
]


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def format_currency(value: float, symbol: str = "€") -> str:
    """Human-readable money formatting, e.g. 12_500_000 -> '€12.5M'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    value = float(value)
    abs_v = abs(value)
    if abs_v >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{symbol}{value / 1_000:.0f}K"
    return f"{symbol}{value:,.0f}"


def safe_minmax(series: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1] without dividing by zero on constant columns."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def clamp(value: float, low: float, high: float) -> float:
    """Constrain a value to the inclusive [low, high] range."""
    return max(low, min(high, value))


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert '#rrggbb' to an 'rgba(r, g, b, a)' string (for chart fills)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"
