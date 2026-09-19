"""Measure how stable scouting ranks are across model or data perturbations."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def rank_stability_audit(rankings: list[list[str]], *, top_k: int = 10) -> dict[str, object]:
    """Summarize player rank variance and top-k retention across repeated runs."""
    if len(rankings) < 2 or top_k < 1:
        raise ValueError("provide at least two rankings and a positive top_k")
    if any(len(set(run)) != len(run) for run in rankings):
        raise ValueError("each ranking must contain unique player identifiers")

    positions: dict[str, list[int]] = defaultdict(list)
    for run in rankings:
        for position, player in enumerate(run, start=1):
            positions[player].append(position)

    common = set(rankings[0][:top_k])
    for run in rankings[1:]:
        common &= set(run[:top_k])

    player_stats = {}
    for player, ranks in positions.items():
        if len(ranks) == len(rankings):
            player_stats[player] = {
                "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks)),
                "rank_range": float(max(ranks) - min(ranks)),
            }
    return {
        "runs": len(rankings),
        "top_k": top_k,
        "top_k_retention": float(len(common) / min(top_k, len(rankings[0]))),
        "always_top_k": sorted(common),
        "player_rank_stability": player_stats,
    }


if __name__ == "__main__":
    runs = [["A", "B", "C", "D"], ["B", "A", "D", "C"], ["A", "C", "B", "D"]]
    print(rank_stability_audit(runs, top_k=3))
