"""
Shot Value Over Expected (SVOE) — Streamlit Dashboard
======================================================
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import statsmodels  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVOE — Shot Value Over Expected",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── paths ────────────────────────────────────────────────────────────────────
SVOE_PATH = Path("data/processed/shots_with_svoe.parquet")
METRICS_PATH = Path("models/metrics.json")

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 18px 24px;
        text-align: center;
        border-left: 4px solid #2196F3;
    }
    .metric-card .label {font-size:0.82rem; color:#555; font-weight:600; letter-spacing:.04em;}
    .metric-card .value {font-size:1.6rem; font-weight:700; color:#1a1a2e; margin-top:4px;}
    .section-note {
        background:#EFF8FF; border-left:4px solid #2196F3;
        padding:10px 14px; border-radius:6px;
        font-size:0.88rem; color:#333; margin-bottom:12px;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 12px;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading shot data …")
def load_data() -> pd.DataFrame | None:
    if not SVOE_PATH.exists():
        return None

    # Only load columns the dashboard actually needs — every dropped column
    # saves RAM across all the per-tab filter slices.
    USE_COLS = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_NAME",
        "SHOT_MADE_FLAG", "SHOT_TYPE", "ACTION_TYPE",
        "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
        "SHOT_DISTANCE", "LOC_X", "LOC_Y",
        "SHOT_VALUE", "SEASON", "SEASON_TYPE", "GAME_HALF",
        "EXPECTED_POINTS", "ACTUAL_POINTS", "SVOE",
    ]
    # Read only the columns that actually exist in the file (schema peek — no data loaded)
    import pyarrow.parquet as pq
    schema_cols = set(pq.read_schema(SVOE_PATH).names)
    load_cols = [c for c in USE_COLS if c in schema_cols]
    df = pd.read_parquet(SVOE_PATH, columns=load_cols)

    # ── downcast to smallest valid dtype ─────────────────────────────────────
    float32_cols = ["LOC_X", "LOC_Y", "SHOT_DISTANCE",
                    "EXPECTED_POINTS", "ACTUAL_POINTS", "SVOE"]
    int8_cols    = ["SHOT_MADE_FLAG", "SHOT_VALUE"]
    cat_cols     = ["PLAYER_NAME", "TEAM_NAME", "SHOT_TYPE", "ACTION_TYPE",
                    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
                    "SEASON", "SEASON_TYPE", "GAME_HALF"]

    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in int8_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def _card(col, label: str, value: str) -> None:
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _note(text: str) -> None:
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


# ── aggregation helpers ───────────────────────────────────────────────────────

def agg_svoe(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            ATTEMPTS=("SHOT_MADE_FLAG", "count"),
            EXPECTED_POINTS=("EXPECTED_POINTS", "sum"),
            ACTUAL_POINTS=("ACTUAL_POINTS", "sum"),
            TOTAL_SVOE=("SVOE", "sum"),
        )
        .reset_index()
    )
    agg["SVOE_PER_100"] = (agg["TOTAL_SVOE"] / agg["ATTEMPTS"] * 100).round(2)
    agg["EP_PER_SHOT"] = (agg["EXPECTED_POINTS"] / agg["ATTEMPTS"]).round(4)
    agg["AP_PER_SHOT"] = (agg["ACTUAL_POINTS"] / agg["ATTEMPTS"]).round(4)
    agg["TOTAL_SVOE"] = agg["TOTAL_SVOE"].round(2)
    agg["EXPECTED_POINTS"] = agg["EXPECTED_POINTS"].round(2)
    agg["ACTUAL_POINTS"] = agg["ACTUAL_POINTS"].round(2)
    return agg


# ── court drawing (inline for self-contained app) ─────────────────────────────

_BASELINE_Y = -47.5
_PAINT_X = 80
_PAINT_TOP_Y = 142.5
_FT_R = 60
_RA_R = 40
_HOOP_R = 7.5
_C3_X = 220
_3PT_R = 237.5
_C3_Y = float(np.sqrt(_3PT_R**2 - _C3_X**2))


def _arc(cx, cy, r, t0, t1, n=180):
    t = np.linspace(np.radians(t0), np.radians(t1), n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _court_traces():
    kw = dict(mode="lines", hoverinfo="skip", showlegend=False,
               line=dict(color="#333", width=1.6))

    def seg(x, y):
        return go.Scatter(x=list(x), y=list(y), **kw)

    traces = [
        seg([-250, 250, 250, -250, -250], [_BASELINE_Y, _BASELINE_Y, 422.5, 422.5, _BASELINE_Y]),
        seg([-_PAINT_X, -_PAINT_X, _PAINT_X, _PAINT_X], [_BASELINE_Y, _PAINT_TOP_Y, _PAINT_TOP_Y, _BASELINE_Y]),
        seg([-_PAINT_X, _PAINT_X], [_PAINT_TOP_Y, _PAINT_TOP_Y]),
        seg(*_arc(0, _PAINT_TOP_Y, _FT_R, 0, 360)),
        seg(*_arc(0, 0, _HOOP_R, 0, 360)),
        seg([-30, 30], [-7.5, -7.5]),
        seg(*_arc(0, 0, _RA_R, 0, 180)),
        seg([-_C3_X, -_C3_X], [_BASELINE_Y, _C3_Y]),
        seg([_C3_X, _C3_X], [_BASELINE_Y, _C3_Y]),
    ]
    t0 = float(np.degrees(np.arctan2(_C3_Y, _C3_X)))
    traces.append(seg(*_arc(0, 0, _3PT_R, t0, 180 - t0)))
    return traces


_COURT_LAYOUT = dict(
    xaxis=dict(range=[-260, 260], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
    yaxis=dict(range=[-60, 440], showgrid=False, zeroline=False, showticklabels=False,
               scaleanchor="x", scaleratio=1, fixedrange=True),
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="white",
    margin=dict(l=0, r=0, t=40, b=0),
    height=530,
)


def shot_chart_fig(df, title="Shot Chart") -> go.Figure:
    fig = go.Figure(_court_traces())
    if not df.empty:
        made = df[df["SHOT_MADE_FLAG"] == 1]
        miss = df[df["SHOT_MADE_FLAG"] == 0]
        ht = "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} ft<extra></extra>"
        fig.add_trace(go.Scatter(x=made["LOC_X"], y=made["LOC_Y"], mode="markers",
            name=f"Made ({len(made):,})", hovertemplate=ht,
            customdata=made[["PLAYER_NAME", "ACTION_TYPE", "SHOT_DISTANCE"]].values,
            marker=dict(color="#2ecc71", size=5, opacity=0.65, symbol="circle")))
        fig.add_trace(go.Scatter(x=miss["LOC_X"], y=miss["LOC_Y"], mode="markers",
            name=f"Missed ({len(miss):,})", hovertemplate=ht,
            customdata=miss[["PLAYER_NAME", "ACTION_TYPE", "SHOT_DISTANCE"]].values,
            marker=dict(color="#e74c3c", size=4, opacity=0.45, symbol="x")))
    fig.update_layout(title=title, legend=dict(y=0.99, x=0.01), **_COURT_LAYOUT)
    return fig


def heatmap_fig(df, val_col="EXPECTED_POINTS", title="Expected Points Heatmap") -> go.Figure:
    fig = go.Figure(_court_traces())
    if not df.empty:
        fig.add_trace(go.Histogram2dContour(
            x=df["LOC_X"], y=df["LOC_Y"], z=df[val_col],
            histfunc="avg", colorscale="RdYlGn", ncontours=20,
            showscale=True,
            colorbar=dict(title=dict(text=val_col.replace("_", " ").title(), side="right"), thickness=14),
            hoverinfo="skip",
        ))
    fig.update_layout(title=title, **_COURT_LAYOUT)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

df_full = load_data()
metrics = load_metrics()

# ── not-ready guard ───────────────────────────────────────────────────────────
if df_full is None:
    st.title("🏀 Shot Value Over Expected (SVOE)")
    st.error("No processed data found.")
    st.markdown(
        """
        **To get started:**
        ```bash
        pip install -r requirements.txt
        python train.py
        streamlit run app.py
        ```
        `train.py` will fetch NBA shot data, train models, and save results.
        This takes ~5 min on first run (API calls) and ~10 min to train.
        """
    )
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 SVOE Analytics")
    st.markdown("---")

    all_seasons = sorted(df_full["SEASON"].unique())
    sel_seasons = st.multiselect("Season", all_seasons, default=all_seasons, key="sb_seasons")

    all_stypes = sorted(df_full["SEASON_TYPE"].unique()) if "SEASON_TYPE" in df_full.columns else ["Regular Season"]
    sel_stypes = st.multiselect("Season Type", all_stypes, default=all_stypes, key="sb_stypes")

    st.markdown("---")
    if metrics:
        best = metrics.get("best_model", "?")
        st.markdown(f"**Model:** {best.replace('_', ' ').title()}")
        bm = metrics.get(best, {})
        if bm:
            st.markdown(
                f"AUC `{bm.get('roc_auc', '?')}` · "
                f"Brier `{bm.get('brier_score', '?')}`"
            )

    st.markdown("---")
    st.caption("Data via nba_api · ShotChartDetail")

# ── apply global season/type filter (no copy — just a boolean slice) ─────────
_mask = pd.Series(True, index=df_full.index)
if sel_seasons:
    _mask &= df_full["SEASON"].isin(sel_seasons)
if sel_stypes and "SEASON_TYPE" in df_full.columns:
    _mask &= df_full["SEASON_TYPE"].isin(sel_stypes)
df = df_full[_mask]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ── tabs ──────────────────────────────────────────────────────────────────────
(
    tab_overview,
    tab_player,
    tab_team,
    tab_map,
    tab_profile,
    tab_sustain,
    tab_compare,
) = st.tabs([
    "📋 Overview",
    "👤 Player Leaderboard",
    "🏢 Team Leaderboard",
    "🗺️ Shot Map",
    "📊 Shot Profile",
    "📈 Sustainability",
    "⚔️ Player Comparison",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## Shot Value Over Expected (SVOE)")
    st.markdown(
        """
        **SVOE** measures whether a player or team converts *more* points than
        a machine-learning model would predict for the exact same set of shots.

        | Term | Formula |
        |------|---------|
        | **Shot Value** | 2 for a 2-pointer · 3 for a 3-pointer |
        | **Expected Points** | Predicted Make Probability × Shot Value |
        | **Actual Points** | Shot Made Flag × Shot Value |
        | **SVOE** | Actual Points − Expected Points |

        A positive SVOE means the player/team shot **better than expected** given
        their shot selection and context. SVOE separates *shot-making ability*
        from *shot-selection quality*.
        """
    )

    st.markdown("### Model")
    best_name = metrics.get("best_model", "?").replace("_", " ").title()
    feat_list = (
        "LOC_X · LOC_Y · Shot Distance · Shot Type · Action Type · "
        "Shot Zone (Basic / Area / Range) · Period · Time Remaining · "
        "Home/Away · Team · Opponent"
    )
    st.markdown(
        f"Predictions use **{best_name}**, calibrated with isotonic regression. "
        f"Features: {feat_list}. "
        "Player identity is **not** a feature — SVOE purely reflects shot-making "
        "above expectation, not player talent absorbed by the model."
    )

    st.warning(
        "⚠️ **Important limitation: this model does NOT know where the defender is.** "
        "It cannot see whether a shot is wide open, lightly contested, or tightly guarded. "
        "Two shots from the exact same spot in the exact same situation — one wide open, "
        "one with a hand in the face — receive identical Expected Points. "
        "SVOE measures shot-making above a *location and context* baseline, "
        "not above a fully contest-adjusted baseline. Keep this in mind when interpreting results."
    )

    st.markdown("---")
    st.markdown("### League-Wide Snapshot")

    total = len(df)
    avg_ep = df["EXPECTED_POINTS"].mean()
    avg_ap = df["ACTUAL_POINTS"].mean()
    total_svoe = df["SVOE"].sum()
    make_rate = df["SHOT_MADE_FLAG"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    _card(c1, "Total Shots", f"{total:,}")
    _card(c2, "League FG%", f"{make_rate:.1%}")
    _card(c3, "Avg Expected Pts/Shot", f"{avg_ep:.3f}")
    _card(c4, "Avg Actual Pts/Shot", f"{avg_ap:.3f}")
    _card(c5, "Total SVOE", f"{total_svoe:+,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_hist, col_meta = st.columns([3, 2])

    with col_hist:
        fig_dist = px.histogram(
            df, x="SVOE", nbins=80,
            color_discrete_sequence=["#3498db"],
            title="League-Wide SVOE Distribution (per shot)",
            labels={"SVOE": "SVOE (Actual − Expected Points)"},
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="black",
                           annotation_text="Zero")
        fig_dist.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=350,
            showlegend=False, yaxis_title="Shots",
            xaxis=dict(gridcolor="#E0E0E0"), yaxis=dict(gridcolor="#E0E0E0"),
        )
        st.plotly_chart(fig_dist, width='stretch')
        _note(
            "Each bar is the count of shots with that SVOE value. "
            "The distribution is centred at zero by construction — a make adds "
            "+2 or +3 SVOE while a miss adds −(predicted probability × shot value)."
        )

    with col_meta:
        st.markdown("#### Model Performance")
        if metrics:
            rows = []
            for name in ["logistic_regression", "random_forest", "xgboost"]:
                if name in metrics:
                    m = metrics[name]
                    rows.append({
                        "Model": name.replace("_", " ").title(),
                        "Log Loss": f"{m['log_loss']:.4f}",
                        "Brier Score": f"{m['brier_score']:.4f}",
                        "ROC-AUC": f"{m['roc_auc']:.4f}",
                    })
            if rows:
                tbl = pd.DataFrame(rows)
                best_row = tbl[tbl["Model"].str.lower().str.replace(" ", "_") == metrics.get("best_model", "")]
                st.dataframe(tbl, hide_index=True, width='stretch')
                st.markdown(
                    f"**Best model:** {best_name}  \n"
                    "Selected by lowest Brier score (measures calibration quality)."
                )
        else:
            st.info("Run `train.py` to generate model metrics.")

        st.markdown("#### Shot Breakdown")
        two_pt = (df["SHOT_VALUE"] == 2).sum()
        three_pt = (df["SHOT_VALUE"] == 3).sum()
        br = pd.DataFrame({
            "Type": ["2PT", "3PT"],
            "Attempts": [two_pt, three_pt],
            "FG%": [
                df[df["SHOT_VALUE"] == 2]["SHOT_MADE_FLAG"].mean(),
                df[df["SHOT_VALUE"] == 3]["SHOT_MADE_FLAG"].mean(),
            ],
            "EP/Shot": [
                df[df["SHOT_VALUE"] == 2]["EXPECTED_POINTS"].mean(),
                df[df["SHOT_VALUE"] == 3]["EXPECTED_POINTS"].mean(),
            ],
            "AP/Shot": [
                df[df["SHOT_VALUE"] == 2]["ACTUAL_POINTS"].mean(),
                df[df["SHOT_VALUE"] == 3]["ACTUAL_POINTS"].mean(),
            ],
        })
        br["FG%"] = br["FG%"].map("{:.1%}".format)
        br["EP/Shot"] = br["EP/Shot"].map("{:.3f}".format)
        br["AP/Shot"] = br["AP/Shot"].map("{:.3f}".format)
        st.dataframe(br, hide_index=True, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PLAYER LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_player:
    st.markdown("## Player Leaderboard")
    _note(
        "SVOE/100 = total SVOE divided by attempts × 100. "
        "Positive means the player shoots better than the model expects "
        "given the exact locations and contexts of their shots."
    )

    # ── filters ──────────────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        team_opts = sorted(df["TEAM_NAME"].dropna().unique())
        sel_team_p = fc1.multiselect("Team", team_opts, key="p_team")
        zone_opts = sorted(df["SHOT_ZONE_BASIC"].dropna().unique())
        sel_zone_p = fc2.multiselect("Shot Zone", zone_opts, key="p_zone")
        stype_opts = sorted(df["SHOT_TYPE"].dropna().unique())
        sel_stype_p = fc3.multiselect("Shot Type", stype_opts, key="p_stype")
        min_att = fc4.slider("Min Attempts", 10, 500, 100, step=10, key="p_minatts")

    _pm = pd.Series(True, index=df.index)
    if sel_team_p:
        _pm &= df["TEAM_NAME"].isin(sel_team_p)
    if sel_zone_p:
        _pm &= df["SHOT_ZONE_BASIC"].isin(sel_zone_p)
    if sel_stype_p:
        _pm &= df["SHOT_TYPE"].isin(sel_stype_p)
    dff = df[_pm]

    pld = agg_svoe(dff, ["PLAYER_ID", "PLAYER_NAME", "TEAM_NAME", "SEASON"])
    pld = pld[pld["ATTEMPTS"] >= min_att].sort_values("SVOE_PER_100", ascending=False)

    if pld.empty:
        st.info("No players meet the filter criteria.")
    else:
        st.markdown(f"**{len(pld):,} players** | sorted by SVOE/100 shots")

        col_tbl, col_bar = st.columns([3, 2])

        with col_tbl:
            display_cols = [
                "PLAYER_NAME", "TEAM_NAME", "SEASON",
                "ATTEMPTS", "EP_PER_SHOT", "AP_PER_SHOT",
                "TOTAL_SVOE", "SVOE_PER_100",
            ]
            disp = pld[display_cols].rename(columns={
                "PLAYER_NAME": "Player", "TEAM_NAME": "Team",
                "SEASON": "Season", "ATTEMPTS": "FGA",
                "EP_PER_SHOT": "EP/Shot", "AP_PER_SHOT": "AP/Shot",
                "TOTAL_SVOE": "Total SVOE", "SVOE_PER_100": "SVOE/100",
            })
            st.dataframe(
                disp.style
                    .format({"EP/Shot": "{:.3f}", "AP/Shot": "{:.3f}",
                             "Total SVOE": "{:+.2f}", "SVOE/100": "{:+.2f}"})
                    .background_gradient(subset=["SVOE/100"], cmap="RdYlGn", vmin=-3, vmax=3),
                width='stretch',
                height=480,
                hide_index=True,
            )

        with col_bar:
            # Aggregate across seasons/teams → one row per player for the chart.
            # This prevents plotly from stacking bars when the same player name
            # appears multiple times (different seasons or mid-season trades).
            chart_agg = agg_svoe(dff, ["PLAYER_ID", "PLAYER_NAME"])
            chart_agg = chart_agg[chart_agg["ATTEMPTS"] >= min_att]
            top20 = chart_agg.nlargest(20, "SVOE_PER_100").sort_values("SVOE_PER_100")

            fig_bar = go.Figure()
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top20["SVOE_PER_100"]]
            fig_bar.add_trace(go.Bar(
                x=top20["SVOE_PER_100"],
                y=top20["PLAYER_NAME"],
                orientation="h",
                marker_color=colors,
                text=top20["SVOE_PER_100"].map("{:+.2f}".format),
                textposition="outside",
                customdata=top20["ATTEMPTS"].values,
                hovertemplate="<b>%{y}</b><br>SVOE/100: %{x:+.2f}<br>FGA: %{customdata:,}<extra></extra>",
            ))
            fig_bar.update_layout(
                title="Top 20 Players — SVOE/100 Shots (all selected seasons combined)",
                height=560,
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#E0E0E0", title="SVOE/100"),
                yaxis=dict(title=""),
                margin=dict(l=140, r=70, t=50, b=40),
                showlegend=False,
            )
            fig_bar.add_vline(x=0, line_dash="dash", line_color="#aaa")
            st.plotly_chart(fig_bar, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — TEAM LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_team:
    st.markdown("## Team Leaderboard")
    _note(
        "Team SVOE separates two things: **shot quality** (EP/Shot — do they create "
        "good looks?) and **shot making** (AP/Shot − EP/Shot — do they convert above "
        "expectation?). The best teams score high on both."
    )

    t_grp_col = st.radio(
        "Group by", ["Season", "Career (all seasons combined)"],
        horizontal=True, key="t_grp"
    )
    if t_grp_col == "Season":
        td = agg_svoe(df, ["TEAM_NAME", "SEASON"])
    else:
        td = agg_svoe(df, ["TEAM_NAME"])

    td = td.sort_values("SVOE_PER_100", ascending=False)

    col_t1, col_t2 = st.columns([5, 3])

    with col_t1:
        t_display = [c for c in ["TEAM_NAME", "SEASON", "ATTEMPTS", "EP_PER_SHOT",
                                   "AP_PER_SHOT", "TOTAL_SVOE", "SVOE_PER_100"] if c in td.columns]
        t_disp = td[t_display].rename(columns={
            "TEAM_NAME": "Team", "SEASON": "Season", "ATTEMPTS": "FGA",
            "EP_PER_SHOT": "EP/Shot", "AP_PER_SHOT": "AP/Shot",
            "TOTAL_SVOE": "Total SVOE", "SVOE_PER_100": "SVOE/100",
        })
        st.dataframe(
            t_disp.style
                .format({"EP/Shot": "{:.3f}", "AP/Shot": "{:.3f}",
                         "Total SVOE": "{:+.2f}", "SVOE/100": "{:+.2f}"})
                .background_gradient(subset=["SVOE/100"], cmap="RdYlGn", vmin=-3, vmax=3),
            width='stretch',
            height=520,
            hide_index=True,
        )

    with col_t2:
        # Scatter: shot quality (EP/Shot) vs shot making (SVOE/100)
        fig_sq = px.scatter(
            td,
            x="EP_PER_SHOT",
            y="SVOE_PER_100",
            text="TEAM_NAME",
            color="SVOE_PER_100",
            color_continuous_scale="RdYlGn",
            title="Shot Quality vs Shot Making",
            labels={
                "EP_PER_SHOT": "EP/Shot (Shot Quality →)",
                "SVOE_PER_100": "SVOE/100 (Shot Making ↑)",
            },
        )
        fig_sq.update_traces(textposition="top center", marker_size=10, showlegend=False)
        fig_sq.add_hline(y=0, line_dash="dash", line_color="#aaa")
        fig_sq.add_vline(x=td["EP_PER_SHOT"].mean(), line_dash="dash", line_color="#aaa")
        fig_sq.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#E0E0E0"), yaxis=dict(gridcolor="#E0E0E0"),
            coloraxis_showscale=False, height=520,
        )
        st.plotly_chart(fig_sq, width='stretch')
        _note(
            "Top-right = good shot selection AND good conversion. "
            "Bottom-left = poor on both. Left-top = bad shots but making them anyway."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SHOT MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("## Shot Map")

    with st.expander("Filters", expanded=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        player_opts = sorted(df["PLAYER_NAME"].dropna().unique())
        sel_player_m = m1.selectbox("Player (blank = all)", ["All"] + player_opts, key="m_player")
        team_opts_m = sorted(df["TEAM_NAME"].dropna().unique())
        sel_team_m = m2.selectbox("Team (blank = all)", ["All"] + team_opts_m, key="m_team")
        season_opts_m = sorted(df["SEASON"].unique())
        sel_season_m = m3.selectbox("Season", ["All"] + season_opts_m, key="m_season")
        zone_opts_m = sorted(df["SHOT_ZONE_BASIC"].dropna().unique())
        sel_zone_m = m4.multiselect("Shot Zone", zone_opts_m, key="m_zone")
        stype_opts_m = sorted(df["SHOT_TYPE"].dropna().unique())
        sel_stype_m = m5.multiselect("Shot Type", stype_opts_m, key="m_stype")

    _mm = pd.Series(True, index=df.index)
    if sel_player_m != "All":
        _mm &= df["PLAYER_NAME"] == sel_player_m
    if sel_team_m != "All":
        _mm &= df["TEAM_NAME"] == sel_team_m
    if sel_season_m != "All":
        _mm &= df["SEASON"] == sel_season_m
    if sel_zone_m:
        _mm &= df["SHOT_ZONE_BASIC"].isin(sel_zone_m)
    if sel_stype_m:
        _mm &= df["SHOT_TYPE"].isin(sel_stype_m)
    dff_m = df[_mm]

    MAX_SCATTER = 5_000
    if len(dff_m) > MAX_SCATTER:
        dff_m_sample = dff_m.sample(MAX_SCATTER, random_state=42)
        st.info(f"Showing a random sample of {MAX_SCATTER:,} from {len(dff_m):,} shots.")
    else:
        dff_m_sample = dff_m

    map_mode = st.radio(
        "View", ["Makes vs Misses", "Expected Points Heatmap", "SVOE Heatmap"],
        horizontal=True, key="m_mode"
    )

    title_suffix = f" — {sel_player_m}" if sel_player_m != "All" else (
        f" — {sel_team_m}" if sel_team_m != "All" else " — League"
    )

    if map_mode == "Makes vs Misses":
        fig_m = shot_chart_fig(dff_m_sample, title=f"Shot Chart{title_suffix}")
    elif map_mode == "Expected Points Heatmap":
        fig_m = heatmap_fig(dff_m, val_col="EXPECTED_POINTS",
                            title=f"Expected Points/Shot{title_suffix}")
    else:
        fig_m = heatmap_fig(dff_m, val_col="SVOE",
                            title=f"SVOE per Shot{title_suffix}")

    st.plotly_chart(fig_m, width='stretch')

    if map_mode == "Makes vs Misses":
        _note(
            "Green circles = makes, red × = misses. "
            "Hover over any shot to see player name, action type, and distance."
        )
    elif map_mode == "Expected Points Heatmap":
        _note(
            "Warmer colours = higher expected points per shot. "
            "The model assigns the most value to rim shots and open corner threes."
        )
    else:
        _note(
            "Warmer colours = player/team is over-performing expectation from that zone. "
            "Cold spots show where shots are falling short of what the model predicts."
        )

    # Stats summary for filtered shots
    if not dff_m.empty:
        st.markdown("#### Summary for filtered shots")
        s_agg = agg_svoe(dff_m, ["SHOT_ZONE_BASIC"])
        s_disp = s_agg.rename(columns={
            "SHOT_ZONE_BASIC": "Zone", "ATTEMPTS": "FGA",
            "EP_PER_SHOT": "EP/Shot", "AP_PER_SHOT": "AP/Shot",
            "TOTAL_SVOE": "Total SVOE", "SVOE_PER_100": "SVOE/100",
        }).sort_values("FGA", ascending=False)
        keep_cols = [c for c in ["Zone", "FGA", "EP/Shot", "AP/Shot", "Total SVOE", "SVOE/100"]
                     if c in s_disp.columns]
        st.dataframe(
            s_disp[keep_cols].style
                .format({"EP/Shot": "{:.3f}", "AP/Shot": "{:.3f}",
                         "Total SVOE": "{:+.2f}", "SVOE/100": "{:+.2f}"})
                .background_gradient(subset=["SVOE/100"], cmap="RdYlGn", vmin=-5, vmax=5),
            hide_index=True, width='stretch',
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — SHOT PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown("## Shot Profile by Zone & Action Type")
    _note(
        "A shot profile breaks down where shots come from and how well each zone "
        "performs relative to expectation. High EP/Shot means good shot quality "
        "(smart location). High SVOE/100 means great conversion above expectation."
    )

    pr1, pr2 = st.columns(2)
    profile_scope = pr1.radio("Scope", ["League", "Team", "Player"], horizontal=True, key="pr_scope")
    profile_group = pr2.radio("Group by", ["Shot Zone", "Action Type"], horizontal=True, key="pr_grp")

    pr_title_suffix = ""
    dff_pr = df  # start as a reference, narrow below only if needed

    if profile_scope == "Team":
        t_sel = st.selectbox("Select Team", sorted(df["TEAM_NAME"].dropna().unique()), key="pr_team")
        dff_pr = df[df["TEAM_NAME"] == t_sel]
        pr_title_suffix = f" — {t_sel}"
    elif profile_scope == "Player":
        p_sel = st.selectbox("Select Player",
                              sorted(df["PLAYER_NAME"].dropna().unique()), key="pr_player")
        dff_pr = df[df["PLAYER_NAME"] == p_sel]
        pr_title_suffix = f" — {p_sel}"

    group_col = "SHOT_ZONE_BASIC" if profile_group == "Shot Zone" else "ACTION_TYPE"
    pr_agg = agg_svoe(dff_pr, [group_col]).sort_values("ATTEMPTS", ascending=False)

    # Filter to top 15 action types if needed
    if profile_group == "Action Type":
        pr_agg = pr_agg.head(15)

    col_pr1, col_pr2 = st.columns(2)

    with col_pr1:
        st.markdown("#### Attempts & Expected Points per Shot")
        pr_disp = pr_agg.rename(columns={
            group_col: profile_group, "ATTEMPTS": "FGA",
            "EP_PER_SHOT": "EP/Shot", "AP_PER_SHOT": "AP/Shot",
            "TOTAL_SVOE": "Total SVOE", "SVOE_PER_100": "SVOE/100",
        })
        keep = [profile_group, "FGA", "EP/Shot", "AP/Shot", "Total SVOE", "SVOE/100"]
        keep = [c for c in keep if c in pr_disp.columns]
        st.dataframe(
            pr_disp[keep].style
                .format({"EP/Shot": "{:.3f}", "AP/Shot": "{:.3f}",
                         "Total SVOE": "{:+.2f}", "SVOE/100": "{:+.2f}",
                         "FGA": "{:,}"})
                .background_gradient(subset=["SVOE/100"], cmap="RdYlGn", vmin=-5, vmax=5),
            width='stretch', hide_index=True,
        )

    with col_pr2:
        # EP vs AP chart
        fig_ep_ap = go.Figure()
        fig_ep_ap.add_trace(go.Bar(
            name="Expected Pts/Shot",
            x=pr_agg[group_col],
            y=pr_agg["EP_PER_SHOT"],
            marker_color="#3498db",
            opacity=0.85,
        ))
        fig_ep_ap.add_trace(go.Bar(
            name="Actual Pts/Shot",
            x=pr_agg[group_col],
            y=pr_agg["AP_PER_SHOT"],
            marker_color="#2ecc71",
            opacity=0.85,
        ))
        fig_ep_ap.update_layout(
            title=f"Expected vs Actual Points per Shot{pr_title_suffix}",
            barmode="group",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#E0E0E0", title=""),
            yaxis=dict(gridcolor="#E0E0E0", title="Points per Shot"),
            height=380,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        st.plotly_chart(fig_ep_ap, width='stretch')

    # SVOE/100 bar
    fig_svoe_zone = px.bar(
        pr_agg.sort_values("SVOE_PER_100", ascending=False),
        x=group_col,
        y="SVOE_PER_100",
        color="SVOE_PER_100",
        color_continuous_scale="RdYlGn",
        title=f"SVOE/100 Shots by {profile_group}{pr_title_suffix}",
        labels={group_col: profile_group, "SVOE_PER_100": "SVOE/100"},
        text="SVOE_PER_100",
    )
    fig_svoe_zone.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
    fig_svoe_zone.add_hline(y=0, line_dash="dash", line_color="black")
    fig_svoe_zone.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0", title="SVOE/100 Shots"),
        coloraxis_showscale=False, height=380,
    )
    st.plotly_chart(fig_svoe_zone, width='stretch')
    _note(
        "Bars above zero = over-performing expectation from that zone. "
        "A team might take great shots (high EP/Shot) but still have negative SVOE "
        "if they miss them — or take average shots and still post positive SVOE "
        "through elite finishing."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — SUSTAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sustain:
    st.markdown("## SVOE Sustainability Analysis")
    _note(
        "Shot-making luck reverts to the mean. Players with a very high H1 SVOE "
        "often regress in H2 — and vice versa. The H1 vs H2 chart reveals who is "
        "likely sustainable and who benefited from temporary variance."
    )

    sus_scope = st.radio("View", ["Players", "Teams"], horizontal=True, key="sus_scope")
    min_att_sus = st.slider("Min FGA per half", 50, 300, 100, 25, key="sus_minatts")

    name_col = "PLAYER_NAME" if sus_scope == "Players" else "TEAM_NAME"
    id_col = "PLAYER_ID" if sus_scope == "Players" else "TEAM_NAME"

    if "GAME_HALF" not in df.columns:
        st.warning("GAME_HALF column not found — re-run train.py to generate it.")
    else:
        h1 = df[df["GAME_HALF"] == "H1"]
        h2 = df[df["GAME_HALF"] == "H2"]

        h1_agg = agg_svoe(h1, [name_col, "SEASON"])
        h2_agg = agg_svoe(h2, [name_col, "SEASON"])

        merged = h1_agg.merge(
            h2_agg, on=[name_col, "SEASON"], suffixes=("_H1", "_H2")
        )
        merged = merged[
            (merged["ATTEMPTS_H1"] >= min_att_sus) &
            (merged["ATTEMPTS_H2"] >= min_att_sus)
        ]

        if merged.empty:
            st.info("Not enough data — lower the minimum FGA threshold.")
        else:
            fig_sus = px.scatter(
                merged,
                x="SVOE_PER_100_H1",
                y="SVOE_PER_100_H2",
                text=name_col,
                trendline="ols" if HAS_STATSMODELS else None,
                color="SVOE_PER_100_H2",
                color_continuous_scale="RdYlGn",
                title=f"{sus_scope} — First Half SVOE/100 vs Second Half SVOE/100",
                labels={
                    "SVOE_PER_100_H1": "H1 SVOE/100",
                    "SVOE_PER_100_H2": "H2 SVOE/100",
                },
                hover_data={"ATTEMPTS_H1": True, "ATTEMPTS_H2": True},
            )
            fig_sus.update_traces(
                textposition="top center", marker_size=8, showlegend=False
            )
            fig_sus.add_hline(y=0, line_dash="dash", line_color="#aaa")
            fig_sus.add_vline(x=0, line_dash="dash", line_color="#aaa")
            fig_sus.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#E0E0E0"),
                yaxis=dict(gridcolor="#E0E0E0"),
                coloraxis_showscale=False, height=520,
            )
            st.plotly_chart(fig_sus, width='stretch')
            _note(
                "Points near the diagonal = consistent performance. "
                "Points far above the diagonal = shot-making improved in H2. "
                "Far below = SVOE regressed. Top-left quadrant = hot H1 that cooled off."
            )

            # Biggest divergers table
            merged["H1_to_H2_CHANGE"] = merged["SVOE_PER_100_H2"] - merged["SVOE_PER_100_H1"]
            st.markdown("#### Biggest H1 → H2 Changes")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Biggest Improvements (H2 > H1)**")
                improvers = merged.nlargest(10, "H1_to_H2_CHANGE")[[
                    name_col, "SEASON", "SVOE_PER_100_H1", "SVOE_PER_100_H2", "H1_to_H2_CHANGE"
                ]].rename(columns={
                    name_col: sus_scope[:-1],
                    "SVOE_PER_100_H1": "H1 SVOE/100",
                    "SVOE_PER_100_H2": "H2 SVOE/100",
                    "H1_to_H2_CHANGE": "Change",
                })
                st.dataframe(
                    improvers.style.format({
                        "H1 SVOE/100": "{:+.2f}", "H2 SVOE/100": "{:+.2f}", "Change": "{:+.2f}"
                    }),
                    hide_index=True, width='stretch',
                )
            with cc2:
                st.markdown("**Biggest Regressions (H1 > H2)**")
                regressors = merged.nsmallest(10, "H1_to_H2_CHANGE")[[
                    name_col, "SEASON", "SVOE_PER_100_H1", "SVOE_PER_100_H2", "H1_to_H2_CHANGE"
                ]].rename(columns={
                    name_col: sus_scope[:-1],
                    "SVOE_PER_100_H1": "H1 SVOE/100",
                    "SVOE_PER_100_H2": "H2 SVOE/100",
                    "H1_to_H2_CHANGE": "Change",
                })
                st.dataframe(
                    regressors.style.format({
                        "H1 SVOE/100": "{:+.2f}", "H2 SVOE/100": "{:+.2f}", "Change": "{:+.2f}"
                    }),
                    hide_index=True, width='stretch',
                )

    # Year-over-year stability (if multiple seasons)
    if len(df["SEASON"].unique()) >= 2:
        st.markdown("---")
        st.markdown("### Year-over-Year SVOE Stability")
        _note(
            "If SVOE were purely random, the year-over-year correlation would be ~0. "
            "A positive correlation means true shot-making skill is being measured. "
            "Players persistently above zero are genuine elite shot-makers."
        )

        seasons_sorted = sorted(df["SEASON"].unique())
        yoy_frames = []
        for i in range(len(seasons_sorted) - 1):
            s1, s2 = seasons_sorted[i], seasons_sorted[i + 1]
            a1 = agg_svoe(df[df["SEASON"] == s1], [name_col])
            a1 = a1.rename(columns={"SVOE_PER_100": f"SVOE_{s1}", "ATTEMPTS": f"ATT_{s1}"})
            a2 = agg_svoe(df[df["SEASON"] == s2], [name_col])
            a2 = a2.rename(columns={"SVOE_PER_100": f"SVOE_{s2}", "ATTEMPTS": f"ATT_{s2}"})
            yoy = a1.merge(a2, on=name_col).dropna()
            yoy = yoy[(yoy[f"ATT_{s1}"] >= 100) & (yoy[f"ATT_{s2}"] >= 100)]
            if not yoy.empty:
                corr = yoy[f"SVOE_{s1}"].corr(yoy[f"SVOE_{s2}"])
                fig_yoy = px.scatter(
                    yoy,
                    x=f"SVOE_{s1}",
                    y=f"SVOE_{s2}",
                    text=name_col,
                    trendline="ols" if HAS_STATSMODELS else None,
                    title=f"YoY SVOE/100 Stability: {s1} → {s2}  (r = {corr:.2f})",
                    labels={
                        f"SVOE_{s1}": f"{s1} SVOE/100",
                        f"SVOE_{s2}": f"{s2} SVOE/100",
                    },
                    color_discrete_sequence=["#3498db"],
                )
                fig_yoy.update_traces(textposition="top center", marker_size=7, showlegend=False)
                fig_yoy.add_hline(y=0, line_dash="dash", line_color="#aaa")
                fig_yoy.add_vline(x=0, line_dash="dash", line_color="#aaa")
                fig_yoy.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#E0E0E0"), yaxis=dict(gridcolor="#E0E0E0"),
                    height=480,
                )
                st.plotly_chart(fig_yoy, width='stretch')
                st.caption(
                    f"YoY correlation r = {corr:.2f}  — "
                    + ("strong signal of true skill." if corr > 0.4 else
                       "moderate skill signal." if corr > 0.2 else
                       "weak signal — large variance component.")
                )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — PLAYER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## Head-to-Head Player Comparison")
    _note(
        "Select two players to compare shot maps, efficiency, and SVOE across "
        "every dimension. Shot maps show a random sample (≤ 2,000 shots each)."
    )

    all_players = sorted(df["PLAYER_NAME"].dropna().unique())
    default_p1 = all_players[0] if all_players else ""
    default_p2 = all_players[1] if len(all_players) > 1 else ""

    cmp1, cmp2 = st.columns(2)
    p1_name = cmp1.selectbox("Player 1", all_players, index=0, key="cmp_p1")
    p2_name = cmp2.selectbox("Player 2", all_players,
                               index=min(1, len(all_players) - 1), key="cmp_p2")

    df_p1 = df[df["PLAYER_NAME"] == p1_name]
    df_p2 = df[df["PLAYER_NAME"] == p2_name]

    if df_p1.empty or df_p2.empty:
        st.warning("One or both players have no shots in the selected season/type filters.")
    else:
        # ── KPIs side-by-side ─────────────────────────────────────────────────
        def _player_kpis(d):
            return {
                "FGA": len(d),
                "FG%": f"{d['SHOT_MADE_FLAG'].mean():.1%}",
                "EP/Shot": f"{d['EXPECTED_POINTS'].mean():.3f}",
                "AP/Shot": f"{d['ACTUAL_POINTS'].mean():.3f}",
                "Total SVOE": f"{d['SVOE'].sum():+.1f}",
                "SVOE/100": f"{d['SVOE'].mean() * 100:+.2f}",
            }

        k1 = _player_kpis(df_p1)
        k2 = _player_kpis(df_p2)

        kpi_cols = st.columns(len(k1))
        for i, (key, v1) in enumerate(k1.items()):
            v2 = k2[key]
            with kpi_cols[i]:
                st.markdown(f"**{key}**")
                delta_val = None
                try:
                    raw1 = float(v1.replace("+", "").replace(",", "").replace("%", ""))
                    raw2 = float(v2.replace("+", "").replace(",", "").replace("%", ""))
                    delta_val = raw1 - raw2
                except Exception:
                    pass
                st.markdown(
                    f"<span style='color:#2196F3;font-size:1.1em;font-weight:700'>{p1_name[:16]}: {v1}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span style='color:#E91E63;font-size:1.1em;font-weight:700'>{p2_name[:16]}: {v2}</span>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── shot maps ─────────────────────────────────────────────────────────
        map_c1, map_c2 = st.columns(2)

        SAMPLE = 2_000
        with map_c1:
            s1 = df_p1.sample(min(SAMPLE, len(df_p1)), random_state=42)
            fig_p1 = shot_chart_fig(s1, title=f"{p1_name}")
            st.plotly_chart(fig_p1, width='stretch')

        with map_c2:
            s2 = df_p2.sample(min(SAMPLE, len(df_p2)), random_state=42)
            fig_p2 = shot_chart_fig(s2, title=f"{p2_name}")
            st.plotly_chart(fig_p2, width='stretch')

        # ── zone breakdown ────────────────────────────────────────────────────
        st.markdown("### Zone Breakdown")

        z1 = agg_svoe(df_p1, ["SHOT_ZONE_BASIC"]).set_index("SHOT_ZONE_BASIC")
        z2 = agg_svoe(df_p2, ["SHOT_ZONE_BASIC"]).set_index("SHOT_ZONE_BASIC")

        all_zones = sorted(set(z1.index) | set(z2.index))

        z_rows = []
        for zone in all_zones:
            r1 = z1.loc[zone] if zone in z1.index else None
            r2 = z2.loc[zone] if zone in z2.index else None
            z_rows.append({
                "Zone": zone,
                f"{p1_name[:14]} FGA": int(r1["ATTEMPTS"]) if r1 is not None else 0,
                f"{p1_name[:14]} SVOE/100": round(r1["SVOE_PER_100"], 2) if r1 is not None else np.nan,
                f"{p2_name[:14]} FGA": int(r2["ATTEMPTS"]) if r2 is not None else 0,
                f"{p2_name[:14]} SVOE/100": round(r2["SVOE_PER_100"], 2) if r2 is not None else np.nan,
            })
        z_df = pd.DataFrame(z_rows).set_index("Zone")
        st.dataframe(
            z_df.style.format(na_rep="—").background_gradient(
                cmap="RdYlGn",
                subset=[c for c in z_df.columns if "SVOE/100" in c],
                vmin=-8, vmax=8,
            ),
            width='stretch',
        )

        # ── SVOE/100 comparison bar ───────────────────────────────────────────
        st.markdown("### SVOE/100 by Zone")

        fig_cz = go.Figure()
        p1_z = [
            (z1.loc[zone]["SVOE_PER_100"] if zone in z1.index else 0)
            for zone in all_zones
        ]
        p2_z = [
            (z2.loc[zone]["SVOE_PER_100"] if zone in z2.index else 0)
            for zone in all_zones
        ]

        fig_cz.add_trace(go.Bar(
            name=p1_name, x=all_zones, y=p1_z,
            marker_color="#2196F3", opacity=0.85,
            text=[f"{v:+.2f}" for v in p1_z], textposition="outside",
        ))
        fig_cz.add_trace(go.Bar(
            name=p2_name, x=all_zones, y=p2_z,
            marker_color="#E91E63", opacity=0.85,
            text=[f"{v:+.2f}" for v in p2_z], textposition="outside",
        ))
        fig_cz.add_hline(y=0, line_dash="dash", line_color="#aaa")
        fig_cz.update_layout(
            barmode="group",
            title="SVOE/100 by Shot Zone",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#E0E0E0", title=""),
            yaxis=dict(gridcolor="#E0E0E0", title="SVOE/100"),
            height=380,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )
        st.plotly_chart(fig_cz, width='stretch')

        # ── action type ───────────────────────────────────────────────────────
        st.markdown("### Top Action Types")

        at1 = agg_svoe(df_p1, ["ACTION_TYPE"]).nlargest(8, "ATTEMPTS")
        at2 = agg_svoe(df_p2, ["ACTION_TYPE"]).nlargest(8, "ATTEMPTS")

        at_c1, at_c2 = st.columns(2)
        with at_c1:
            st.markdown(f"**{p1_name}**")
            st.dataframe(
                at1[["ACTION_TYPE", "ATTEMPTS", "EP_PER_SHOT", "AP_PER_SHOT", "SVOE_PER_100"]]
                .rename(columns={
                    "ACTION_TYPE": "Action", "ATTEMPTS": "FGA",
                    "EP_PER_SHOT": "EP/Shot", "AP_PER_SHOT": "AP/Shot",
                    "SVOE_PER_100": "SVOE/100",
                })
                .style.format({
                    "EP/Shot": "{:.3f}", "AP/Shot": "{:.3f}",
                    "SVOE/100": "{:+.2f}", "FGA": "{:,}"
                })
                .background_gradient(subset=["SVOE/100"], cmap="RdYlGn", vmin=-8, vmax=8),
                hide_index=True, width='stretch',
            )
        with at_c2:
            st.markdown(f"**{p2_name}**")
            st.dataframe(
                at2[["ACTION_TYPE", "ATTEMPTS", "EP_PER_SHOT", "AP_PER_SHOT", "SVOE_PER_100"]]
                .rename(columns={
                    "ACTION_TYPE": "Action", "ATTEMPTS": "FGA",
                    "EP_PER_SHOT": "EP/Shot", "AP_PER_SHOT": "AP/Shot",
                    "SVOE_PER_100": "SVOE/100",
                })
                .style.format({
                    "EP/Shot": "{:.3f}", "AP/Shot": "{:.3f}",
                    "SVOE/100": "{:+.2f}", "FGA": "{:,}"
                })
                .background_gradient(subset=["SVOE/100"], cmap="RdYlGn", vmin=-8, vmax=8),
                hide_index=True, width='stretch',
            )

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "SVOE Analytics · Data via NBA Stats API (nba_api) · "
    f"{len(df):,} shots loaded · "
    "Model: " + metrics.get("best_model", "?").replace("_", " ").title()
)
