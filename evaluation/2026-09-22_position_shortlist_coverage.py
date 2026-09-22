"""Audit whether a scouting shortlist covers required squad positions."""

from __future__ import annotations

from collections import Counter


def position_coverage_audit(
    shortlisted_positions: list[str],
    required_slots: dict[str, int],
) -> dict[str, object]:
    """Compare shortlist representation with required positional depth."""
    if not shortlisted_positions:
        raise ValueError("shortlisted_positions must not be empty")
    if not required_slots or any(v < 1 for v in required_slots.values()):
        raise ValueError("required_slots must contain positive requirements")

    counts = Counter(shortlisted_positions)
    rows: list[dict[str, int | float | str]] = []
    total_required = sum(required_slots.values())
    total_covered = 0
    for position, required in required_slots.items():
        available = counts.get(position, 0)
        covered = min(available, required)
        total_covered += covered
        rows.append({
            "position": position,
            "required": required,
            "available": available,
            "shortfall": max(0, required - available),
            "coverage": covered / required,
        })

    uncovered = [row["position"] for row in rows if row["shortfall"] > 0]
    return {
        "slot_coverage": total_covered / total_required,
        "positions_with_shortfall": uncovered,
        "position_details": rows,
    }


if __name__ == "__main__":
    shortlist = ["CB", "CB", "CM", "RW", "ST", "ST", "GK", "LB"]
    needs = {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 2, "RW": 1, "ST": 1}
    print(position_coverage_audit(shortlist, needs))