"""Reusable Plotly figures with a consistent dark / red theme.

Every builder returns a ready-to-render ``plotly.graph_objects.Figure`` styled
via :func:`apply_theme`, so the dashboard stays visually consistent.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import ATTRIBUTE_COLS, RED_SCALE, THEME, hex_to_rgba


def apply_theme(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the shared dark theme + red accent styling to a figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text"], family="Inter, Segoe UI, sans-serif", size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=[THEME["red"], THEME["amber"], THEME["blue"],
                  THEME["green"], THEME["purple"], THEME["red_soft"]],
    )
    fig.update_xaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])
    fig.update_yaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])
    if height:
        fig.update_layout(height=height)
    return fig


def value_vs_predicted_scatter(df: pd.DataFrame) -> go.Figure:
    """Current vs predicted value, with a y=x reference line and undervalue heat."""
    fig = px.scatter(
        df,
        x="market_value_eur",
        y="predicted_future_value_eur",
        color="undervalue_score",
        size="potential",
        hover_name="name",
        hover_data={"position": True, "age": True, "overall_rating": True,
                    "market_value_eur": ":,.0f", "predicted_future_value_eur": ":,.0f",
                    "undervalue_score": ":.0f", "potential": False},
        color_continuous_scale=RED_SCALE,
        labels={"market_value_eur": "Current value (€)",
                "predicted_future_value_eur": "Predicted future value (€)",
                "undervalue_score": "Undervalue"},
    )
    lo = float(min(df["market_value_eur"].min(), df["predicted_future_value_eur"].min()))
    hi = float(max(df["market_value_eur"].max(), df["predicted_future_value_eur"].max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Break-even",
        line=dict(color=THEME["muted"], dash="dash", width=1.5),
        hoverinfo="skip",
    ))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return apply_theme(fig, height=460)


def value_by_position_box(df: pd.DataFrame) -> go.Figure:
    """Distribution of market value by position group."""
    fig = px.box(
        df, x="position_group", y="market_value_eur", color="position_group",
        points="outliers",
        labels={"position_group": "Position group", "market_value_eur": "Market value (€)"},
    )
    fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, height=380)


def attribute_radar(players: pd.DataFrame) -> go.Figure:
    """Radar chart comparing FIFA-style attributes for up to ~4 players."""
    categories = [c.capitalize() for c in ATTRIBUTE_COLS]
    fig = go.Figure()
    palette = [THEME["red_bright"], THEME["blue"], THEME["amber"], THEME["green"]]
    for i, (_, row) in enumerate(players.iterrows()):
        values = [float(row[c]) for c in ATTRIBUTE_COLS]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself", name=str(row["name"]),
            line=dict(color=color, width=2),
            fillcolor=hex_to_rgba(color, 0.18),
            opacity=0.9,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, range=[0, 99], gridcolor=THEME["grid"]),
            angularaxis=dict(gridcolor=THEME["grid"]),
        ),
    )
    return apply_theme(fig, height=420)


def feature_importance_bar(importance: pd.DataFrame, title: str = "") -> go.Figure:
    """Horizontal bar chart of permutation importances."""
    data = importance.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=data["importance"], y=data["feature"], orientation="h",
        marker=dict(color=data["importance"], colorscale=RED_SCALE),
    ))
    fig.update_layout(title=title, xaxis_title="Permutation importance",
                      yaxis_title="")
    return apply_theme(fig, height=380)


def age_value_scatter(df: pd.DataFrame) -> go.Figure:
    """Age vs market value, colored by predicted growth."""
    fig = px.scatter(
        df, x="age", y="market_value_eur", color="predicted_growth_pct",
        size="overall_rating", hover_name="name",
        hover_data={"position": True, "potential": True,
                    "market_value_eur": ":,.0f", "predicted_growth_pct": ":.1f"},
        color_continuous_scale=RED_SCALE,
        labels={"age": "Age", "market_value_eur": "Market value (€)",
                "predicted_growth_pct": "Growth %"},
    )
    fig.update_yaxes(type="log")
    return apply_theme(fig, height=420)


def trajectory_bar(df: pd.DataFrame) -> go.Figure:
    """Count of players per career-trajectory category."""
    order = ["Rising Star", "Stable Performer", "Declining Asset", "High-Risk Prospect"]
    counts = df["trajectory_category"].value_counts().reindex(order).fillna(0)
    colors = {"Rising Star": THEME["green"], "Stable Performer": THEME["blue"],
              "Declining Asset": THEME["red"], "High-Risk Prospect": THEME["amber"]}
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[colors[c] for c in counts.index],
        text=counts.values.astype(int), textposition="outside",
    ))
    fig.update_layout(yaxis_title="Players", xaxis_title="")
    return apply_theme(fig, height=340)


def growth_gauge(growth_pct: float) -> go.Figure:
    """Gauge showing a single player's predicted growth percentage."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=growth_pct,
        number={"suffix": "%", "font": {"color": THEME["text"]}},
        gauge={
            "axis": {"range": [-40, 80], "tickcolor": THEME["muted"]},
            "bar": {"color": THEME["red_bright"]},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [-40, 0], "color": "rgba(225,29,58,0.15)"},
                {"range": [0, 25], "color": "rgba(245,165,36,0.15)"},
                {"range": [25, 80], "color": "rgba(33,193,122,0.18)"},
            ],
        },
    ))
    return apply_theme(fig, height=260)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap across key numeric features."""
    cols = ["age", "overall_rating", "potential", "market_value_eur", "wage_eur",
            "performance_index", "future_value_eur"] + ATTRIBUTE_COLS
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=RED_SCALE, zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r"),
    ))
    return apply_theme(fig, height=520)


def score_by_category_bar(series: pd.Series, x_title: str = "") -> go.Figure:
    """Themed horizontal bar of a score indexed by category (e.g. by position)."""
    data = series.sort_values()
    fig = go.Figure(go.Bar(
        x=data.values, y=data.index, orientation="h",
        marker=dict(color=data.values, colorscale=RED_SCALE,
                    line=dict(color=hex_to_rgba(THEME["red_deep"], 0.6), width=1)),
        text=[f"{v:.1f}" for v in data.values], textposition="outside",
        textfont=dict(color=THEME["text"]),
    ))
    fig.update_layout(xaxis_title=x_title, yaxis_title="")
    return apply_theme(fig, height=320)
