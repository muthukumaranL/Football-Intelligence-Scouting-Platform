"""Audit whether repeated scouting recommendations over-expose a small player set."""

from __future__ import annotations

from collections import Counter


def recommendation_exposure_audit(
    ranked_lists: list[list[str]],
    *,
    top_k: int = 5,
) -> dict[str, object]:
    """Measure player exposure concentration across repeated shortlist runs."""
    if not ranked_lists or top_k < 1:
        raise ValueError("ranked_lists must be non-empty and top_k positive")

    exposure = Counter()
    total_slots = 0
    for ranking in ranked_lists:
        selected = ranking[:top_k]
        exposure.update(selected)
        total_slots += len(selected)
    if total_slots == 0:
        raise ValueError("ranked lists must contain at least one player")

    shares = {player: count / total_slots for player, count in exposure.items()}
    hhi = sum(share * share for share in shares.values())
    return {
        "total_recommendation_slots": total_slots,
        "unique_players_exposed": len(exposure),
        "largest_player_share": max(shares.values()),
        "exposure_hhi": hhi,
        "effective_player_count": 1.0 / hhi if hhi else 0.0,
        "exposure_share": dict(sorted(shares.items(), key=lambda x: x[1], reverse=True)),
    }


if __name__ == "__main__":
    runs = [
        ["A", "B", "C", "D"],
        ["A", "C", "E", "B"],
        ["A", "B", "F", "C"],
    ]
    print(recommendation_exposure_audit(runs, top_k=3))
