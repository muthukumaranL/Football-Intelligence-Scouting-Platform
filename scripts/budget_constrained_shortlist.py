"""Optimize a scouting shortlist under a transfer budget.

A ranking alone does not answer the recruitment question: which combination of
players maximizes expected scouting value within a fixed budget? This example
uses 0/1 dynamic programming with integer budget units.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Player:
    name: str
    cost_units: int
    score: float


def optimize_shortlist(players: list[Player], budget_units: int) -> dict[str, object]:
    if budget_units < 0:
        raise ValueError("budget_units must be non-negative")
    if any(player.cost_units <= 0 for player in players):
        raise ValueError("player costs must be positive integer units")

    # dp[b] = (best_score, selected_indices)
    dp: list[tuple[float, tuple[int, ...]]] = [(0.0, ()) for _ in range(budget_units + 1)]

    for index, player in enumerate(players):
        for budget in range(budget_units, player.cost_units - 1, -1):
            previous_score, previous_indices = dp[budget - player.cost_units]
            candidate_score = previous_score + player.score
            if candidate_score > dp[budget][0]:
                dp[budget] = (candidate_score, previous_indices + (index,))

    best_budget = max(range(budget_units + 1), key=lambda b: dp[b][0])
    best_score, selected_indices = dp[best_budget]
    selected = [players[index] for index in selected_indices]
    total_cost = sum(player.cost_units for player in selected)

    return {
        "selected_players": [player.name for player in selected],
        "total_score": float(best_score),
        "total_cost_units": int(total_cost),
        "unused_budget_units": int(budget_units - total_cost),
        "players_selected": len(selected),
    }


if __name__ == "__main__":
    candidates = [
        Player("Forward A", 28, 84.0),
        Player("Midfielder B", 22, 78.0),
        Player("Defender C", 16, 66.0),
        Player("Winger D", 20, 73.0),
        Player("Keeper E", 12, 55.0),
    ]

    result = optimize_shortlist(candidates, budget_units=60)
    print("Budget-constrained scouting shortlist")
    for key, value in result.items():
        print(f"{key:>22}: {value}")
