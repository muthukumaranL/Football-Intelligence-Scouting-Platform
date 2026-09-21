"""Summarize scouting rank uncertainty across bootstrap or resampling runs."""

from __future__ import annotations

import numpy as np


def bootstrap_rank_confidence(
    player_ids: list[str],
    score_runs: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    """Estimate rank intervals and top-k inclusion probability for each player."""
    scores = np.asarray(score_runs, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(player_ids) or scores.shape[0] < 2:
        raise ValueError("score_runs must be resamples x players with at least two runs")
    if len(set(player_ids)) != len(player_ids) or not np.all(np.isfinite(scores)):
        raise ValueError("player_ids must be unique and scores must be finite")
    if not 1 <= top_k <= len(player_ids):
        raise ValueError("top_k must be between 1 and the number of players")

    ranks = np.empty_like(scores, dtype=float)
    for run in range(scores.shape[0]):
        order = np.argsort(-scores[run], kind="stable")
        ranks[run, order] = np.arange(1, scores.shape[1] + 1, dtype=float)

    rows: list[dict[str, float | str]] = []
    for idx, player in enumerate(player_ids):
        player_ranks = ranks[:, idx]
        rows.append({
            "player_id": player,
            "median_rank": float(np.median(player_ranks)),
            "rank_p05": float(np.quantile(player_ranks, 0.05)),
            "rank_p95": float(np.quantile(player_ranks, 0.95)),
            "top_k_probability": float(np.mean(player_ranks <= top_k)),
            "mean_score": float(np.mean(scores[:, idx])),
        })
    return sorted(rows, key=lambda row: (-float(row["top_k_probability"]), float(row["median_rank"])))


if __name__ == "__main__":
    runs = np.array([[0.91, 0.86, 0.72], [0.88, 0.90, 0.70], [0.93, 0.84, 0.76], [0.89, 0.87, 0.78]])
    print(bootstrap_rank_confidence(["A", "B", "C"], runs, top_k=2))
