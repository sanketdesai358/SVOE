"""
Model evaluation metrics and SVOE calculation.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")


# ── model evaluation ─────────────────────────────────────────────────────────

def evaluate_model(
    y_true: pd.Series,
    y_prob: np.ndarray,
    name: str = "model",
) -> dict:
    """Return a dict of scalar metrics for one model."""
    return {
        "name": name,
        "log_loss": round(log_loss(y_true, y_prob), 6),
        "brier_score": round(brier_score_loss(y_true, y_prob), 6),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 6),
    }


def evaluate_all_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate every model in *models* against the test split.

    Returns
    -------
    dict with keys = model names, values = metric dicts.
    Also includes 'best_model' key pointing to the model with the lowest
    Brier score (best calibration).
    """
    from src.modeling import predict_proba  # avoid circular at module level

    results: dict = {}
    for name, model in models.items():
        y_prob = predict_proba(model, X_test)
        results[name] = evaluate_model(y_test, y_prob, name)
        print(
            f"  {name:25s} | log_loss={results[name]['log_loss']:.4f}"
            f"  brier={results[name]['brier_score']:.4f}"
            f"  AUC={results[name]['roc_auc']:.4f}"
        )

    best = min(results, key=lambda k: results[k]["brier_score"])
    results["best_model"] = best
    print(f"\n  ✓  Best model (lowest Brier score): {best}")
    return results


def save_metrics(metrics: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", path)


def load_metrics() -> dict:
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── calibration plot ──────────────────────────────────────────────────────────

def calibration_figure(
    models_probs: dict,          # {name: y_prob ndarray}
    y_true: pd.Series,
    n_bins: int = 12,
) -> go.Figure:
    """Return a Plotly calibration curve figure."""
    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
    fig = go.Figure()

    for i, (name, y_prob) in enumerate(models_probs.items()):
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        fig.add_trace(
            go.Scatter(
                x=mean_pred,
                y=frac_pos,
                mode="lines+markers",
                name=name.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=7),
            )
        )

    # Perfect calibration reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="black", width=1, dash="dash"),
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Model Calibration — Predicted vs Actual Make Rate",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Makes",
        height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#E0E0E0"),
        yaxis=dict(gridcolor="#E0E0E0"),
    )
    return fig


# ── SVOE calculation ──────────────────────────────────────────────────────────

def compute_svoe(df: pd.DataFrame, prob_col: str = "PRED_PROB") -> pd.DataFrame:
    """
    Add EXPECTED_POINTS, ACTUAL_POINTS, and SVOE columns.

    SVOE = Actual Points − Expected Points
         = (SHOT_MADE_FLAG × SHOT_VALUE) − (PRED_PROB × SHOT_VALUE)
    """
    out = df.copy()
    out["EXPECTED_POINTS"] = out[prob_col] * out["SHOT_VALUE"]
    out["ACTUAL_POINTS"] = out["SHOT_MADE_FLAG"] * out["SHOT_VALUE"]
    out["SVOE"] = out["ACTUAL_POINTS"] - out["EXPECTED_POINTS"]
    return out


# ── convenience aggregations ──────────────────────────────────────────────────

def aggregate_svoe(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Aggregate SVOE metrics by any combination of grouping columns.

    Returns columns: ATTEMPTS, EXPECTED_POINTS, ACTUAL_POINTS, TOTAL_SVOE,
                     SVOE_PER_100, EP_PER_SHOT, AP_PER_SHOT
    """
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
    agg["SVOE_PER_100"] = (agg["TOTAL_SVOE"] / agg["ATTEMPTS"] * 100).round(3)
    agg["EP_PER_SHOT"] = (agg["EXPECTED_POINTS"] / agg["ATTEMPTS"]).round(4)
    agg["AP_PER_SHOT"] = (agg["ACTUAL_POINTS"] / agg["ATTEMPTS"]).round(4)
    agg["TOTAL_SVOE"] = agg["TOTAL_SVOE"].round(3)
    agg["EXPECTED_POINTS"] = agg["EXPECTED_POINTS"].round(3)
    agg["ACTUAL_POINTS"] = agg["ACTUAL_POINTS"].round(3)
    return agg
