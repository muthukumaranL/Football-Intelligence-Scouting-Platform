"""Find non-dominated scouting options across value, performance and risk.

Run:
    python examples/scouting_pareto_frontier.py

A player is Pareto-efficient when no other player is simultaneously cheaper,
better-performing and lower-risk. This is useful for shortlist decisions where
a single weighted score can hide meaningful trade-offs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Player:
    name: str
    market_value_m: float
    performance_score: float
    risk_score: float


def dominates(a: Player, b: Player) -> bool:
    no_worse = (
        a.market_value_m <= b.market_value_m
        and a.performance_score >= b.performance_score
        and a.risk_score <= b.risk_score
    )
    strictly_better = (
        a.market_value_m < b.market_value_m
        or a.performance_score > b.performance_score
        or a.risk_score < b.risk_score
    )
    return no_worse and strictly_better


def pareto_frontier(players: list[Player]) -> list[Player]:
    return [p for p in players if not any(dominates(other, p) for other in players if other != p)]


if __name__ == "__main__":
    shortlist = [
        Player("Player A", 18.0, 82.0, 0.24),
        Player("Player B", 12.0, 78.0, 0.18),
        Player("Player C", 21.0, 86.0, 0.21),
        Player("Player D", 15.0, 76.0, 0.29),
        Player("Player E", 10.0, 73.0, 0.16),
    ]
    print("Pareto-efficient shortlist:")
    for player in pareto_frontier(shortlist):
        print(f"- {player.name}: €{player.market_value_m:.1f}m, perf={player.performance_score:.1f}, risk={player.risk_score:.2f}")
