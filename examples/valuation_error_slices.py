"""Audit football player valuation errors across interpretable slices.

Overall R-squared can hide systematic misses for specific player groups. This
standalone example calculates MAE and signed bias by age and market-value band
so scouting users can see where a valuation model tends to over- or under-price.

Run:
    pip install numpy pandas scikit-learn
    python examples/valuation_error_slices.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def summarize_slice_errors(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return sample size, MAE and signed prediction bias for each slice."""
    required = {group_col, "actual_value", "predicted_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for label, group in frame.groupby(group_col, observed=True):
        error = group["predicted_value"] - group["actual_value"]
        rows.append(
            {
                group_col: label,
                "players": len(group),
                "mae": mean_absolute_error(group["actual_value"], group["predicted_value"]),
                "bias": float(error.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(11)
    n_players = 900
    age = rng.integers(17, 36, size=n_players)
    rating = rng.normal(72, 8, size=n_players).clip(45, 94)
    potential = np.maximum(rating, rating + rng.normal(4, 4, size=n_players)).clip(45, 96)
    minutes = rng.integers(250, 3300, size=n_players)
    goals_assists = rng.poisson(np.maximum((rating - 55) / 5, 0.5), size=n_players)

    actual_value = (
        0.55 * rating**2
        + 0.40 * potential**2
        + 2.0 * goals_assists
        + 0.004 * minutes
        - 18 * np.maximum(age - 27, 0)
        + rng.normal(0, 350, size=n_players)
    ).clip(100, None)

    X = pd.DataFrame(
        {
            "age": age,
            "rating": rating,
            "potential": potential,
            "minutes": minutes,
            "goals_assists": goals_assists,
        }
    )
    y = pd.Series(actual_value, name="actual_value")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11)

    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=11, n_jobs=-1)
    model.fit(X_train, y_train)
    audit = X_test.copy()
    audit["actual_value"] = y_test.to_numpy()
    audit["predicted_value"] = model.predict(X_test)
    audit["age_band"] = pd.cut(audit["age"], [16, 21, 25, 29, 36], labels=["17-21", "22-25", "26-29", "30+"])
    audit["value_band"] = pd.qcut(audit["actual_value"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    print(f"Overall MAE: {mean_absolute_error(audit['actual_value'], audit['predicted_value']):.1f}\n")
    print("Error by age band")
    print(summarize_slice_errors(audit, "age_band").to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\nError by market-value quartile")
    print(summarize_slice_errors(audit, "value_band").to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
