"""Rank scouting targets while penalizing uncertain performance estimates."""

from __future__ import annotations

import pandas as pd


def uncertainty_adjusted_ranking(
    players: pd.DataFrame,
    *,
    score_col: str = "performance_score",
    uncertainty_col: str = "score_std",
    risk_aversion: float = 1.0,
) -> pd.DataFrame:
    """Return conservative rankings using score - risk_aversion * uncertainty."""
    required = {score_col, uncertainty_col}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if risk_aversion < 0 or (players[uncertainty_col] < 0).any():
        raise ValueError("risk_aversion and uncertainty must be non-negative")

    ranked = players.copy()
    ranked["risk_adjusted_score"] = (
        ranked[score_col] - risk_aversion * ranked[uncertainty_col]
    )
    ranked["raw_rank"] = ranked[score_col].rank(method="min", ascending=False).astype(int)
    ranked["risk_adjusted_rank"] = ranked["risk_adjusted_score"].rank(
        method="min", ascending=False
    ).astype(int)
    ranked["rank_change"] = ranked["raw_rank"] - ranked["risk_adjusted_rank"]
    return ranked.sort_values(["risk_adjusted_rank", score_col])


if __name__ == "__main__":
    sample = pd.DataFrame({
        "player": ["A", "B", "C", "D"],
        "performance_score": [88, 86, 84, 82],
        "score_std": [7, 2, 1, 3],
    })
    print(uncertainty_adjusted_ranking(sample).to_string(index=False))
