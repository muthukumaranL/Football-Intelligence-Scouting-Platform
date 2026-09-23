"""Audit how sensitive player rankings are to scouting score weights."""

from __future__ import annotations

import numpy as np


def score_weight_sensitivity(
    player_names: list[str],
    feature_matrix: np.ndarray,
    weight_scenarios: np.ndarray,
    top_k: int = 5,
) -> dict[str, object]:
    """Measure top-k inclusion and rank variability across plausible weight scenarios."""
    x = np.asarray(feature_matrix, dtype=float)
    weights = np.asarray(weight_scenarios, dtype=float)
    if x.ndim != 2 or weights.ndim != 2 or x.shape[0] != len(player_names):
        raise ValueError("feature_matrix must align with player_names and weights must be 2D")
    if x.shape[1] != weights.shape[1] or x.shape[0] == 0 or weights.shape[0] < 2:
        raise ValueError("feature and weight dimensions must match with at least two scenarios")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("features and weights must be finite; weights must be non-negative")
    if np.any(np.sum(weights, axis=1) <= 0) or not 1 <= top_k <= len(player_names):
        raise ValueError("each scenario needs positive total weight and top_k must be valid")

    normalized = weights / np.sum(weights, axis=1, keepdims=True)
    scores = x @ normalized.T
    ranks = np.empty_like(scores, dtype=int)
    for scenario in range(scores.shape[1]):
        order = np.argsort(-scores[:, scenario], kind="stable")
        ranks[order, scenario] = np.arange(1, len(player_names) + 1)

    rows = []
    for idx, name in enumerate(player_names):
        player_ranks = ranks[idx]
        rows.append({
            "player": name,
            "mean_rank": float(np.mean(player_ranks)),
            "rank_std": float(np.std(player_ranks)),
            "best_rank": int(np.min(player_ranks)),
            "worst_rank": int(np.max(player_ranks)),
            "top_k_inclusion_rate": float(np.mean(player_ranks <= top_k)),
        })
    rows.sort(key=lambda row: (row["mean_rank"], row["rank_std"]))
    return {"scenarios": int(weights.shape[0]), "top_k": top_k, "players": rows}


if __name__ == "__main__":
    names = ["A", "B", "C", "D"]
    features = np.array([[0.9, 0.6, 0.7], [0.7, 0.9, 0.6], [0.8, 0.7, 0.8], [0.6, 0.8, 0.9]])
    scenarios = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.3, 0.2, 0.5]])
    print(score_weight_sensitivity(names, features, scenarios, top_k=2))
