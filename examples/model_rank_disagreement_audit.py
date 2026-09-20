"""Measure disagreement between independent player-ranking models."""

from __future__ import annotations

import numpy as np


def rank_disagreement(
    player_ids: list[str],
    score_matrix: np.ndarray,
) -> list[dict[str, float | str]]:
    """Return each player's mean rank and disagreement across scoring models."""
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(player_ids) or scores.shape[0] < 2:
        raise ValueError("score_matrix must be models x players with at least two models")
    if len(set(player_ids)) != len(player_ids):
        raise ValueError("player_ids must be unique")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")

    ranks = np.empty_like(scores, dtype=float)
    for model in range(scores.shape[0]):
        order = np.argsort(-scores[model], kind="stable")
        ranks[model, order] = np.arange(1, scores.shape[1] + 1, dtype=float)

    rows: list[dict[str, float | str]] = []
    for idx, player in enumerate(player_ids):
        player_ranks = ranks[:, idx]
        rows.append({
            "player_id": player,
            "mean_rank": float(np.mean(player_ranks)),
            "rank_std": float(np.std(player_ranks, ddof=1)),
            "rank_range": float(np.max(player_ranks) - np.min(player_ranks)),
            "best_rank": float(np.min(player_ranks)),
            "worst_rank": float(np.max(player_ranks)),
        })
    return sorted(rows, key=lambda item: float(item["rank_std"]), reverse=True)


if __name__ == "__main__":
    scores = np.array([[0.91, 0.82, 0.70, 0.65], [0.74, 0.89, 0.78, 0.61], [0.88, 0.75, 0.81, 0.63]])
    print(rank_disagreement(["A", "B", "C", "D"], scores))