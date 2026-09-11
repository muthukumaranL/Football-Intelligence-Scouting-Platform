"""Build position-relative percentile profiles for scouting attributes.

Raw player metrics are difficult to compare across roles. This helper ranks a
player only against peers in the same position and returns percentile scores
for selected numeric features, producing a role-aware scouting profile.
"""
from __future__ import annotations

import numpy as np


def _percentile_rank(values: np.ndarray, target: float) -> float:
    below = np.sum(values < target)
    equal = np.sum(values == target)
    return float((below + 0.5 * equal) / len(values) * 100.0)


def position_percentiles(names, positions, features, feature_names, target_name: str) -> dict:
    names = np.asarray(names)
    positions = np.asarray(positions)
    x = np.asarray(features, dtype=float)
    feature_names = list(feature_names)

    if x.ndim != 2 or len(names) != len(positions) or len(names) != len(x):
        raise ValueError("names, positions, and feature rows must align")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names must match feature columns")
    matches = np.where(names == target_name)[0]
    if len(matches) != 1:
        raise ValueError("target_name must identify exactly one player")

    target_idx = int(matches[0])
    position = positions[target_idx]
    peer_mask = positions == position
    peers = x[peer_mask]
    if len(peers) < 2:
        raise ValueError("at least two players are required in the target position")

    profile = {
        feature_names[j]: _percentile_rank(peers[:, j], x[target_idx, j])
        for j in range(x.shape[1])
    }
    return {
        "player": str(target_name),
        "position": str(position),
        "peer_count": int(len(peers)),
        "percentiles": profile,
    }


if __name__ == "__main__":
    players = ["A", "B", "C", "D", "E", "F"]
    positions = ["CM", "CM", "CM", "FW", "FW", "FW"]
    metrics = [
        [84, 78, 65], [76, 82, 70], [88, 74, 68],
        [72, 66, 89], [75, 70, 84], [69, 64, 92],
    ]
    print(position_percentiles(players, positions, metrics, ["passing", "vision", "finishing"], "A"))
