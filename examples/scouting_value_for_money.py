"""Identify scouting targets that deliver the most performance per transfer cost."""

from __future__ import annotations

import pandas as pd


def value_for_money_frontier(
    players: pd.DataFrame,
    *,
    score_col: str = "performance_score",
    cost_col: str = "market_value_m",
) -> pd.DataFrame:
    """Return players not dominated by a cheaper, equally strong alternative."""
    required = {score_col, cost_col}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if (players[cost_col] <= 0).any():
        raise ValueError("market values must be positive")

    ranked = players.copy()
    ranked["value_score"] = ranked[score_col] / ranked[cost_col]
    ranked = ranked.sort_values([cost_col, score_col], ascending=[True, False]).reset_index(drop=True)

    best_score = float("-inf")
    efficient = []
    for _, row in ranked.iterrows():
        is_efficient = float(row[score_col]) > best_score
        efficient.append(is_efficient)
        best_score = max(best_score, float(row[score_col]))
    ranked["cost_performance_frontier"] = efficient
    return ranked.sort_values("value_score", ascending=False)


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "player": ["A", "B", "C", "D"],
            "performance_score": [82, 88, 84, 91],
            "market_value_m": [12, 30, 18, 48],
        }
    )
    print(value_for_money_frontier(sample).to_string(index=False))
