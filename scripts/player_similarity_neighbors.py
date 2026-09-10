"""Find statistically similar players from numeric scouting attributes.

The helper standardizes features before Euclidean distance so attributes with
large numeric scales (for example market value) do not dominate the result.
"""
from __future__ import annotations

import numpy as np


def nearest_players(names, features, target_name: str, k: int = 5):
    names = np.asarray(names)
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or len(names) != len(x):
        raise ValueError("names and feature rows must align")
    matches = np.where(names == target_name)[0]
    if len(matches) != 1:
        raise ValueError("target_name must identify exactly one player")
    if k < 1:
        raise ValueError("k must be positive")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std
    target_idx = int(matches[0])
    distance = np.linalg.norm(z - z[target_idx], axis=1)
    order = np.argsort(distance)
    order = order[order != target_idx][:k]
    return [(str(names[i]), float(distance[i])) for i in order]


if __name__ == "__main__":
    players = ["A", "B", "C", "D", "E"]
    # pace, passing, finishing, defensive contribution
    matrix = [[82, 76, 71, 42], [80, 78, 69, 45], [65, 88, 61, 55],
              [90, 68, 84, 30], [79, 75, 73, 44]]
    for name, distance in nearest_players(players, matrix, "A", k=3):
        print(f"{name}: distance={distance:.3f}")
