"""
Plotly-based court visualization and chart helpers.

All coordinates are in the NBA Stats API system:
  LOC_X: tenths of feet from basket center, left = negative
  LOC_Y: tenths of feet from basket toward mid-court (positive)
  Baseline sits at Y ≈ -47.5; mid-court at Y ≈ 422.5.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── court geometry constants ──────────────────────────────────────────────────

_BASELINE_Y = -47.5
_PAINT_X = 80          # 8 ft each side of center
_PAINT_TOP_Y = 142.5   # 19 ft from baseline to FT line
_FT_CIRCLE_R = 60      # 6 ft radius
_RA_RADIUS = 40        # restricted area 4 ft
_HOOP_R = 7.5          # hoop radius
_CORNER_3_X = 220      # corner 3 is 22 ft from center
_THREE_PT_R = 237.5    # 3-pt arc radius 23.75 ft

# Y where the corner-3 straight line meets the arc
_CORNER_ARC_Y = float(np.sqrt(_THREE_PT_R**2 - _CORNER_3_X**2))


# ── court drawing helpers ─────────────────────────────────────────────────────

def _arc_xy(cx, cy, r, theta_start_deg, theta_end_deg, n=200):
    theta = np.linspace(np.radians(theta_start_deg), np.radians(theta_end_deg), n)
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _court_traces(line_color: str = "#222222") -> list[go.BaseTraceType]:
    """Return a list of Scatter traces that draw an NBA half-court."""
    kw = dict(mode="lines", hoverinfo="skip", showlegend=False,
               line=dict(color=line_color, width=1.8))
    traces = []

    def add(x, y):
        traces.append(go.Scatter(x=list(x), y=list(y), **kw))

    # Half-court boundary
    add([-250, 250, 250, -250, -250],
        [_BASELINE_Y, _BASELINE_Y, 422.5, 422.5, _BASELINE_Y])

    # Paint (lane)
    add([-_PAINT_X, -_PAINT_X, _PAINT_X, _PAINT_X],
        [_BASELINE_Y, _PAINT_TOP_Y, _PAINT_TOP_Y, _BASELINE_Y])

    # Free-throw line
    add([-_PAINT_X, _PAINT_X], [_PAINT_TOP_Y, _PAINT_TOP_Y])

    # Free-throw circle (full)
    fx, fy = _arc_xy(0, _PAINT_TOP_Y, _FT_CIRCLE_R, 0, 360)
    add(fx, fy)

    # Basket / hoop
    hx, hy = _arc_xy(0, 0, _HOOP_R, 0, 360)
    add(hx, hy)

    # Backboard
    add([-30, 30], [-7.5, -7.5])

    # Restricted area (upper semi-circle)
    rx, ry = _arc_xy(0, 0, _RA_RADIUS, 0, 180)
    add(rx, ry)

    # Corner 3 straight lines
    add([-_CORNER_3_X, -_CORNER_3_X], [_BASELINE_Y, _CORNER_ARC_Y])
    add([_CORNER_3_X, _CORNER_3_X], [_BASELINE_Y, _CORNER_ARC_Y])

    # 3-point arc
    t_start = float(np.degrees(np.arctan2(_CORNER_ARC_Y, _CORNER_3_X)))
    t_end = 180.0 - t_start
    ax, ay = _arc_xy(0, 0, _THREE_PT_R, t_start, t_end)
    add(ax, ay)

    return traces


def _court_layout_updates() -> dict:
    return dict(
        xaxis=dict(
            range=[-260, 260],
            showgrid=False, zeroline=False, showticklabels=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[-60, 440],
            showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=1,
            fixedrange=True,
        ),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )


# ── public chart functions ────────────────────────────────────────────────────

def shot_chart_makes_misses(df: pd.DataFrame, title: str = "Shot Chart") -> go.Figure:
    """
    Scatter plot of shot locations coloured green (make) / red (miss).
    """
    fig = go.Figure(_court_traces())

    if not df.empty:
        made = df[df["SHOT_MADE_FLAG"] == 1]
        missed = df[df["SHOT_MADE_FLAG"] == 0]

        _hover = "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} ft<extra></extra>"

        fig.add_trace(go.Scatter(
            x=made["LOC_X"], y=made["LOC_Y"],
            mode="markers",
            name=f"Made ({len(made):,})",
            marker=dict(color="#2ecc71", size=5, opacity=0.65, symbol="circle"),
            customdata=made[["PLAYER_NAME", "ACTION_TYPE", "SHOT_DISTANCE"]].values,
            hovertemplate=_hover,
        ))
        fig.add_trace(go.Scatter(
            x=missed["LOC_X"], y=missed["LOC_Y"],
            mode="markers",
            name=f"Missed ({len(missed):,})",
            marker=dict(color="#e74c3c", size=4, opacity=0.45, symbol="x"),
            customdata=missed[["PLAYER_NAME", "ACTION_TYPE", "SHOT_DISTANCE"]].values,
            hovertemplate=_hover,
        ))

    fig.update_layout(
        title=title, height=530,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        **_court_layout_updates(),
    )
    return fig


def shot_chart_heatmap(
    df: pd.DataFrame,
    value_col: str = "EXPECTED_POINTS",
    title: str = "Expected Points per Shot — Court Heatmap",
) -> go.Figure:
    """
    Hexbin-style density heatmap of a value column over shot locations.
    """
    fig = go.Figure(_court_traces())

    if not df.empty:
        fig.add_trace(go.Histogram2dContour(
            x=df["LOC_X"],
            y=df["LOC_Y"],
            z=df[value_col],
            histfunc="avg",
            colorscale="RdYlGn",
            ncontours=20,
            showscale=True,
            colorbar=dict(
                title=dict(text=value_col.replace("_", " ").title(), side="right"),
                thickness=14,
            ),
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=title, height=530,
        **_court_layout_updates(),
    )
    return fig


def svoe_distribution(df: pd.DataFrame) -> go.Figure:
    """League-wide SVOE-per-shot histogram."""
    fig = px.histogram(
        df,
        x="SVOE",
        nbins=80,
        color_discrete_sequence=["#3498db"],
        title="League-Wide SVOE Distribution (per shot)",
        labels={"SVOE": "SVOE (Actual − Expected Points)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Zero")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0", title="Shots"),
        showlegend=False,
        height=350,
    )
    return fig


def leaderboard_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: str | None = None,
    n: int = 20,
) -> go.Figure:
    """Horizontal bar chart for leaderboard views."""
    top = df.nlargest(n, x_col) if x_col in df.columns else df.head(n)
    top = top.sort_values(x_col)

    color_seq = top[color_col] if color_col and color_col in top.columns else "#3498db"

    fig = go.Figure(
        go.Bar(
            x=top[x_col],
            y=top[y_col],
            orientation="h",
            marker_color=color_seq if isinstance(color_seq, str) else None,
            text=top[x_col].round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        height=max(400, n * 26),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=160, r=60, t=50, b=40),
    )
    return fig


def zone_profile_bar(zone_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Bar chart for shot profile by zone."""
    fig = px.bar(
        zone_df.sort_values(metric, ascending=False),
        x="SHOT_ZONE_BASIC",
        y=metric,
        color=metric,
        color_continuous_scale="RdYlGn",
        title=title,
        labels={metric: metric.replace("_", " ").title()},
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        coloraxis_showscale=False,
        height=380,
    )
    return fig


def sustainability_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    xlab: str,
    ylab: str,
) -> go.Figure:
    """Scatter with regression line for sustainability analysis."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        text=label_col,
        trendline="ols",
        title=title,
        labels={x_col: xlab, y_col: ylab},
        color_discrete_sequence=["#2196F3"],
    )
    fig.update_traces(textposition="top center", marker_size=8)
    fig.add_hline(y=0, line_dash="dash", line_color="#aaa")
    fig.add_vline(x=0, line_dash="dash", line_color="#aaa")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
        height=460,
    )
    return fig


def comparison_radar(
    player1_stats: dict,
    player2_stats: dict,
    p1_name: str,
    p2_name: str,
) -> go.Figure:
    """Radar chart comparing two players across key metrics."""
    categories = [
        "SVOE/100", "EP/Shot", "AP/Shot",
        "3PT%", "2PT%", "Attempts/Game",
    ]
    # Normalise each metric to [0,1] across the two players for radar
    def _norm(a, b):
        rng = max(abs(a), abs(b), 1e-9)
        return [(a + rng) / (2 * rng), (b + rng) / (2 * rng)]

    vals = {}
    for k in ["SVOE_PER_100", "EP_PER_SHOT", "AP_PER_SHOT"]:
        vals[k] = _norm(
            player1_stats.get(k, 0), player2_stats.get(k, 0)
        )

    def _pct(d, zone):
        rows = [r for r in (d.get("zones") or []) if zone in r.get("SHOT_TYPE", "")]
        if not rows:
            return 0
        made = sum(r["ACTUAL_POINTS"] / r.get("SHOT_VALUE", 2) for r in rows)
        att = sum(r["ATTEMPTS"] for r in rows)
        return made / att if att else 0

    p1_vals = [
        player1_stats.get("SVOE_PER_100", 0),
        player1_stats.get("EP_PER_SHOT", 0),
        player1_stats.get("AP_PER_SHOT", 0),
        player1_stats.get("FG3_PCT", 0),
        player1_stats.get("FG2_PCT", 0),
        player1_stats.get("ATTEMPTS", 0) / max(player1_stats.get("GAMES", 1), 1),
    ]
    p2_vals = [
        player2_stats.get("SVOE_PER_100", 0),
        player2_stats.get("EP_PER_SHOT", 0),
        player2_stats.get("AP_PER_SHOT", 0),
        player2_stats.get("FG3_PCT", 0),
        player2_stats.get("FG2_PCT", 0),
        player2_stats.get("ATTEMPTS", 0) / max(player2_stats.get("GAMES", 1), 1),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p1_vals + [p1_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=p1_name,
        line_color="#2196F3",
        opacity=0.7,
    ))
    fig.add_trace(go.Scatterpolar(
        r=p2_vals + [p2_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=p2_name,
        line_color="#E91E63",
        opacity=0.7,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"{p1_name} vs {p2_name}",
        height=420,
    )
    return fig
