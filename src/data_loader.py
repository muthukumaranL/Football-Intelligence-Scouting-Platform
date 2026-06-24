"""Data loading layer.

Loading priority (first match wins):
    1. Any CSV the user drops into ``data/raw/`` (a real dataset).
    2. The bundled sample CSV at ``data/sample/players_sample.csv``.
    3. A freshly generated synthetic dataset (also written to the sample path).

This makes the project run out-of-the-box with realistic *sample* data while
letting anyone swap in a real Kaggle FIFA / market-value CSV later without
touching the code — see README "Dataset information".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    DETAILED_POSITIONS,
    POSITION_GROUPS,
    RAW_DIR,
    SAMPLE_DATA_PATH,
)

# Pools used to build synthetic, non-copyrighted player identities.
_FIRST_NAMES = [
    "Liam", "Mateo", "Noah", "Diego", "Luca", "Hugo", "Adam", "Omar", "Kai",
    "Eli", "Theo", "Aaron", "Ivan", "Marco", "Felix", "Samir", "Bruno", "Jonas",
    "Yusuf", "Ravi", "Tariq", "Andre", "Nikola", "Pavel", "Carlos", "Daichi",
    "Mamadou", "Sven", "Emre", "Nicolas", "Joel", "Rafael", "Tom", "Leon",
    "Sergio", "Kenji", "Dani", "Victor", "Aleksandr",
]
_LAST_NAMES = [
    "Hart", "Vega", "Moretti", "Bauer", "Larsson", "Costa", "Novak", "Diallo",
    "Tanaka", "Petrov", "Almeida", "Keller", "Romano", "Haaland", "Mensah",
    "Okafor", "Silva", "Marchetti", "Dubois", "Fischer", "Sato", "Nielsen",
    "Kovac", "Reyes", "Andersen", "Volkov", "Bianchi", "Schmidt", "Rossi",
    "Ferreira", "Lindqvist", "Adeyemi", "Toure", "Stankovic", "Marin", "Weber",
    "Ibrahim", "Castro", "Janssen", "Park",
]
_NATIONALITIES = [
    "Brazil", "Argentina", "France", "Spain", "Germany", "Italy", "England",
    "Portugal", "Netherlands", "Belgium", "Japan", "Senegal", "Nigeria",
    "Croatia", "Serbia", "Sweden", "Norway", "Denmark", "Mexico", "Uruguay",
    "Morocco", "Ghana", "South Korea", "USA", "Colombia",
]
# Fictional league/club names so we avoid any trademarked branding.
_LEAGUES = {
    "Crown League": ["Northgate United", "Riverside FC", "Kings Park", "Ironside AFC", "Crystal Rovers"],
    "Continental A": ["Olympia Verde", "Real Montaña", "Atlético Sol", "Costa Azul", "Sporting Lumen"],
    "Bundesklasse": ["Eisern Berlin", "Rhein Adler", "Schwarzwald SV", "Hansa Nord", "Bayern Stein"],
    "Serie Oro": ["Inter Lazio", "Roma Vera", "Milano Nero", "Napoli Sud", "Torino Granata"],
    "Ligue Première": ["Paris Étoile", "Lyon Sud", "Marseille Bleu", "Lille Or", "Monaco Rocher"],
    "Eredivisie Pro": ["Amsterdam Lance", "Rotterdam Haven", "Eindhoven Volt", "Utrecht Dom", "Groningen Noord"],
}

_FEET = ["Right", "Left"]


def _assign_attributes(rng: np.random.Generator, group: str, overall: int) -> dict[str, int]:
    """Generate position-appropriate FIFA-style attributes around a base level.

    Attributes are anchored to the player's overall rating and biased by role
    (e.g. defenders score higher on defending, forwards on shooting).
    """
    base = overall
    noise = lambda spread: int(rng.normal(0, spread))  # noqa: E731

    if group == "Goalkeeper":
        profile = {
            "pace": base - 25 + noise(5),
            "shooting": base - 35 + noise(5),
            "passing": base - 18 + noise(6),
            "dribbling": base - 22 + noise(5),
            "defending": base - 20 + noise(6),
            "physic": base - 5 + noise(6),
        }
    elif group == "Defender":
        profile = {
            "pace": base - 6 + noise(7),
            "shooting": base - 22 + noise(7),
            "passing": base - 8 + noise(7),
            "dribbling": base - 12 + noise(7),
            "defending": base + 4 + noise(5),
            "physic": base + 2 + noise(6),
        }
    elif group == "Midfielder":
        profile = {
            "pace": base - 4 + noise(7),
            "shooting": base - 8 + noise(8),
            "passing": base + 4 + noise(5),
            "dribbling": base + 2 + noise(6),
            "defending": base - 6 + noise(8),
            "physic": base - 3 + noise(7),
        }
    else:  # Forward
        profile = {
            "pace": base + 4 + noise(6),
            "shooting": base + 5 + noise(5),
            "passing": base - 5 + noise(7),
            "dribbling": base + 4 + noise(6),
            "defending": base - 22 + noise(8),
            "physic": base - 4 + noise(7),
        }
    return {k: int(np.clip(v, 25, 99)) for k, v in profile.items()}


def generate_synthetic_players(n_players: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic player dataset.

    Relationships are intentionally correlated (value rises with rating and
    potential, falls with age; wages track value; forwards carry a price
    premium) so the downstream EDA and models behave like the real thing.

    NOTE: This is *sample* data for demonstration. It is not real and should
    not be used for actual scouting decisions.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_players):
        position = rng.choice(DETAILED_POSITIONS)
        group = POSITION_GROUPS[position]

        age = int(np.clip(rng.normal(25, 4.2), 16, 39))

        # Overall rating: broad bell curve, lightly nudged by "prime age".
        prime_bonus = 3 if 24 <= age <= 29 else 0
        overall = int(np.clip(rng.normal(70 + prime_bonus, 7), 47, 94))

        # Potential is overall plus an age-dependent headroom (young -> more).
        headroom = max(0, rng.normal((30 - age) * 0.8, 3))
        potential = int(np.clip(overall + headroom, overall, 96))

        attrs = _assign_attributes(rng, group, overall)

        # --- Market value model (EUR) -------------------------------------- #
        # Exponential in rating, premium for youth & potential headroom,
        # position premium for attackers, multiplicative noise.
        rating_factor = np.exp((overall - 50) / 11.0)
        youth_factor = np.clip(1.8 - (age - 18) * 0.05, 0.35, 1.8)
        potential_factor = 1.0 + (potential - overall) * 0.06
        position_premium = {
            "Forward": 1.30, "Midfielder": 1.10,
            "Defender": 0.95, "Goalkeeper": 0.75,
        }[group]
        rep_factor = 1.0 + 0.12 * (rng.integers(1, 6) - 1)
        base_value = (
            120_000 * rating_factor * youth_factor * potential_factor
            * position_premium * rep_factor
        )
        market_value = float(np.clip(base_value * rng.lognormal(0, 0.22), 30_000, 220_000_000))

        # Wage roughly tracks value with noise (weekly EUR).
        wage = float(np.clip(market_value * rng.uniform(0.0006, 0.0016), 1_000, 700_000))

        league = rng.choice(list(_LEAGUES.keys()))
        club = rng.choice(_LEAGUES[league])

        # Performance: minutes scale with rating; goals/assists by role.
        appearances = int(np.clip(rng.normal(24, 8), 0, 38))
        minutes = int(np.clip(appearances * rng.uniform(45, 90), 0, 3420))
        attack_rate = {"Forward": 0.55, "Midfielder": 0.28, "Defender": 0.07, "Goalkeeper": 0.0}[group]
        goals = int(np.clip(rng.poisson(attack_rate * appearances * (overall / 75)), 0, 40))
        assist_rate = {"Forward": 0.22, "Midfielder": 0.30, "Defender": 0.10, "Goalkeeper": 0.01}[group]
        assists = int(np.clip(rng.poisson(assist_rate * appearances * (overall / 78)), 0, 30))

        rows.append({
            "player_id": 100000 + i,
            "name": f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
            "age": age,
            "nationality": rng.choice(_NATIONALITIES),
            "club": club,
            "league": league,
            "position": position,
            "position_group": group,
            "overall_rating": overall,
            "potential": potential,
            "market_value_eur": round(market_value, -3),
            "wage_eur": round(wage, -2),
            "height_cm": int(np.clip(rng.normal(182 if group != "Goalkeeper" else 189, 6), 162, 205)),
            "weight_kg": int(np.clip(rng.normal(76, 7), 58, 100)),
            "preferred_foot": rng.choice(_FEET, p=[0.76, 0.24]),
            "contract_years_left": int(rng.integers(1, 6)),
            "international_reputation": int(np.clip(round(rng.normal(1.6, 0.9)), 1, 5)),
            "skill_moves": int(np.clip(round(rng.normal(2.8, 0.9)), 1, 5)),
            "weak_foot": int(np.clip(round(rng.normal(3.0, 0.8)), 1, 5)),
            **attrs,
            "appearances": appearances,
            "goals": goals,
            "assists": assists,
            "minutes_played": minutes,
        })

    return pd.DataFrame(rows)


def _find_raw_csv() -> Path | None:
    """Return the first CSV found in data/raw/, if any."""
    candidates = sorted(RAW_DIR.glob("*.csv"))
    return candidates[0] if candidates else None


def load_raw_data(force_regenerate: bool = False) -> tuple[pd.DataFrame, str]:
    """Load the best available dataset.

    Returns
    -------
    (DataFrame, source_label)
        ``source_label`` is one of ``"real"`` or ``"sample"`` so the UI can
        clearly tell the user which dataset is in use.
    """
    raw_csv = _find_raw_csv()
    if raw_csv is not None and not force_regenerate:
        df = pd.read_csv(raw_csv)
        return df, f"real:{raw_csv.name}"

    if SAMPLE_DATA_PATH.exists() and not force_regenerate:
        return pd.read_csv(SAMPLE_DATA_PATH), "sample"

    # Generate, persist, and return synthetic sample data.
    df = generate_synthetic_players()
    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return df, "sample"


if __name__ == "__main__":
    # Allow `python -m src.data_loader` to (re)build the sample dataset.
    frame, source = load_raw_data(force_regenerate=True)
    print(f"Generated {len(frame)} players -> {SAMPLE_DATA_PATH} (source={source})")
    print(frame.head())
