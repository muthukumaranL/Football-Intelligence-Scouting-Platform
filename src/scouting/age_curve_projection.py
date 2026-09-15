"""Simple, transparent age-curve projection for scouting scenarios.

This is a scenario tool rather than a learned causal model. It makes the age
assumption explicit so scouts can stress-test a shortlist instead of treating
current performance as permanent.
"""
from __future__ import annotations

import pandas as pd


def project_age_curve(players: pd.DataFrame, peak_age: int = 27, annual_growth: float = 0.025,
                      annual_decline: float = 0.035, horizon_years: int = 3) -> pd.DataFrame:
    required = {"player", "age", "performance_score"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if horizon_years < 0:
        raise ValueError("horizon_years must be non-negative")

    out = players.copy()
    projected = []
    for row in out.itertuples(index=False):
        score = float(row.performance_score)
        age = int(row.age)
        for year in range(horizon_years):
            next_age = age + year + 1
            rate = annual_growth if next_age <= peak_age else -annual_decline
            score *= 1 + rate
        projected.append(score)

    out["projected_age"] = out["age"] + horizon_years
    out["projected_score"] = projected
    out["projected_change_pct"] = 100 * (out["projected_score"] / out["performance_score"] - 1)
    return out.sort_values("projected_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    sample = pd.DataFrame({
        "player": ["Prospect", "Prime", "Veteran"],
        "age": [21, 26, 30],
        "performance_score": [72.0, 80.0, 84.0],
    })
    print(project_age_curve(sample).round(2).to_string(index=False))
