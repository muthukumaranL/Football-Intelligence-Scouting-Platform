"""Measure concentration risk in a football scouting shortlist.

A high-scoring shortlist can still be fragile when most candidates come from
one club, league, or age band. This utility reports HHI concentration and the
effective number of represented groups for any categorical shortlist field.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable


def concentration_metrics(values: Iterable[str]) -> dict[str, float]:
    cleaned = [str(value).strip() or "unknown" for value in values]
    if not cleaned:
        raise ValueError("values cannot be empty")

    counts = Counter(cleaned)
    total = len(cleaned)
    shares = [count / total for count in counts.values()]
    hhi = sum(share * share for share in shares)
    effective_groups = 1.0 / hhi
    largest_share = max(shares)

    return {
        "items": float(total),
        "unique_groups": float(len(counts)),
        "hhi": float(hhi),
        "effective_groups": float(effective_groups),
        "largest_group_share": float(largest_share),
    }


def audit_shortlist(players: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    if not players:
        raise ValueError("players cannot be empty")

    age_bands = []
    for player in players:
        age = int(player.get("age", 0))
        if age < 21:
            age_bands.append("u21")
        elif age <= 25:
            age_bands.append("21-25")
        elif age <= 29:
            age_bands.append("26-29")
        else:
            age_bands.append("30+")

    return {
        "club": concentration_metrics(player.get("club", "unknown") for player in players),
        "league": concentration_metrics(player.get("league", "unknown") for player in players),
        "age_band": concentration_metrics(age_bands),
    }


if __name__ == "__main__":
    shortlist = [
        {"name": "A", "club": "North FC", "league": "Premier", "age": 22},
        {"name": "B", "club": "North FC", "league": "Premier", "age": 24},
        {"name": "C", "club": "City SC", "league": "Premier", "age": 20},
        {"name": "D", "club": "Rovers", "league": "Championship", "age": 27},
        {"name": "E", "club": "Union", "league": "Eredivisie", "age": 23},
    ]

    for dimension, metrics in audit_shortlist(shortlist).items():
        print(f"\n{dimension}")
        for metric, value in metrics.items():
            print(f"{metric:>22}: {value:.3f}")
