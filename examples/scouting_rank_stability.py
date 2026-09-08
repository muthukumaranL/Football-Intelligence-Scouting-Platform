"""Estimate how stable a scouting shortlist is under score uncertainty.

Input CSV columns:
    player,score,score_std

Run with built-in demo data:
    python examples/scouting_rank_stability.py

Or:
    python examples/scouting_rank_stability.py --csv shortlist.csv --top-k 3

The simulation samples plausible player scores from each player's uncertainty
band and reports average rank plus probability of finishing inside the top-k.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"player", "score", "score_std"}


def simulate_rank_stability(
    players: pd.DataFrame,
    top_k: int = 3,
    simulations: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    missing = REQUIRED - set(players.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if top_k < 1 or top_k > len(players):
        raise ValueError("top_k must be between 1 and the number of players")
    if (players["score_std"] < 0).any():
        raise ValueError("score_std cannot be negative")

    rng = np.random.default_rng(seed)
    means = players["score"].to_numpy(dtype=float)
    stds = players["score_std"].to_numpy(dtype=float)
    sampled = rng.normal(means, stds, size=(simulations, len(players)))

    order = np.argsort(-sampled, axis=1)
    ranks = np.empty_like(order)
    row_ids = np.arange(simulations)[:, None]
    ranks[row_ids, order] = np.arange(1, len(players) + 1)

    result = players[["player", "score", "score_std"]].copy()
    result["mean_rank"] = ranks.mean(axis=0)
    result[f"top_{top_k}_probability"] = (ranks <= top_k).mean(axis=0)
    return result.sort_values([f"top_{top_k}_probability", "mean_rank"], ascending=[False, True])


def demo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": ["A", "B", "C", "D", "E", "F"],
            "score": [88.0, 86.5, 85.8, 84.2, 82.5, 80.0],
            "score_std": [1.5, 2.8, 1.0, 3.5, 1.8, 2.0],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv) if args.csv else demo()
    report = simulate_rank_stability(frame, top_k=args.top_k, simulations=args.simulations)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
