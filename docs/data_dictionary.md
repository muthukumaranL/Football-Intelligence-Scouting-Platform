# Data Dictionary

Columns after `clean_data` + `engineer_features`. Types are post-cleaning.
Money is in EUR; attributes use the 0–99 scale.

## Identity & profile
| Column | Type | Description |
|--------|------|-------------|
| `player_id` | int | Unique player identifier. |
| `name` | str | Player display name (synthetic in sample data). |
| `age` | int | Age in years. |
| `nationality` | str | Nationality. |
| `club` | str | Current club (fictional in sample data). |
| `league` | str | League (fictional in sample data). |
| `position` | str | Primary position code (GK, CB, CM, ST, …). |
| `position_group` | str | Goalkeeper / Defender / Midfielder / Forward. |
| `preferred_foot` | str | Right / Left. |
| `height_cm` | int | Height in cm. |
| `weight_kg` | int | Weight in kg. |
| `contract_years_left` | int | Years remaining on contract. |

## Ratings & reputation
| Column | Type | Description |
|--------|------|-------------|
| `overall_rating` | int | Current ability (≈40–94). |
| `potential` | int | Peak projected ability (≥ overall). |
| `international_reputation` | int | 1–5. |
| `skill_moves` | int | 1–5. |
| `weak_foot` | int | 1–5. |

## Attributes (0–99)
| Column | Description |
|--------|-------------|
| `pace` | Speed/acceleration. |
| `shooting` | Finishing/shot power. |
| `passing` | Passing/vision. |
| `dribbling` | Ball control/dribbling. |
| `defending` | Defensive ability. |
| `physic` | Physicality/strength. |

## Market
| Column | Type | Description |
|--------|------|-------------|
| `market_value_eur` | float | Current market value (EUR). |
| `wage_eur` | float | Weekly wage (EUR). |

## Performance (per season)
| Column | Type | Description |
|--------|------|-------------|
| `appearances` | int | Matches played. |
| `goals` | int | Goals scored. |
| `assists` | int | Assists. |
| `minutes_played` | int | Total minutes. |

## Engineered features
| Column | Description |
|--------|-------------|
| `potential_gap` | `potential − overall_rating` (upside). |
| `years_to_peak` | `27 − age`. |
| `value_per_rating` | Value per rating point. |
| `rating_per_wage` | Rating per €1k weekly wage (wage efficiency). |
| `value_to_wage_ratio` | Value ÷ wage. |
| `goals_per_90`, `assists_per_90`, `goal_contributions_per_90` | Per-90 output. |
| `performance_index` | Role-weighted attribute score (0–100) + output bonus. |

## Targets & predictions
| Column | Description |
|--------|-------------|
| `future_value_eur` | **Proxy** regression target — projected 2-season value. |
| `value_growth_pct` | Implied growth from the proxy target. |
| `transfer_success` | **Proxy** binary label (0/1) for transfer success. |
| `predicted_future_value_eur` | Model-predicted future value. |
| `predicted_growth_pct` | Model-predicted growth %. |
| `transfer_success_proba` | Model probability of transfer success. |

## Scores (computed in the app)
| Column | Description |
|--------|-------------|
| `undervalue_score` | 0–100 composite undervalue index. |
| `uv_upside`, `uv_potential`, `uv_age_advantage`, `uv_quality_per_cost`, `uv_wage_efficiency` | Undervalue components (0–100). |
| `undervalue_reason` | Human-readable explanation. |
| `trajectory_category` | Rising Star / Stable Performer / Declining Asset / High-Risk Prospect. |
| `fit_score` | 0–100 recruitment fit (Recruitment Engine output). |
| `recommendation_reason` | Why a player was shortlisted. |
| `similarity` | 0–1 similarity to a selected player. |

## Using a real dataset
Place a CSV in `data/raw/`. The alias map in `src/preprocessing.py` handles common names, e.g.:
`value_eur → market_value_eur`, `short_name/long_name → name`, `club_name → club`,
`league_name → league`, `nationality_name → nationality`, `player_positions/best_position → position`,
`overall → overall_rating`, `physicality → physic`. Missing columns are imputed or derived.
