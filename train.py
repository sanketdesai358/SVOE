"""
End-to-end SVOE training pipeline.

Usage
-----
    # First run — fetches data, trains models, saves everything
    python train.py

    # Specific seasons
    python train.py --seasons 2023-24

    # Force re-download from NBA API (ignores cache)
    python train.py --refresh-data

    # Force re-train even if models already exist
    python train.py --retrain

    # Both
    python train.py --refresh-data --retrain
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── arg parsing ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train SVOE shot-make models.")
parser.add_argument(
    "--seasons",
    nargs="+",
    default=["2022-23", "2023-24"],
    help="NBA seasons in YYYY-YY format (default: 2022-23 2023-24)",
)
parser.add_argument(
    "--season-types",
    nargs="+",
    default=["Regular Season", "Playoffs"],
    help='Season types (default: "Regular Season" "Playoffs")',
)
parser.add_argument(
    "--refresh-data",
    action="store_true",
    help="Delete cached data and re-fetch from the NBA API.",
)
parser.add_argument(
    "--retrain",
    action="store_true",
    help="Re-train models even if saved models already exist.",
)
args = parser.parse_args()

SVOE_PATH = Path("data/processed/shots_with_svoe.parquet")
MODEL_DIR = Path("models")

# ── imports (after sys.path is set up correctly) ──────────────────────────────

from src.data_loader import load_all_shots, purge_cache
from src.features import extract_X_y, feature_summary
from src.metrics import (
    calibration_figure,
    compute_svoe,
    evaluate_all_models,
    save_metrics,
)
from src.modeling import predict_proba, save_model, train_all_models
from src.preprocessing import clean_shots


# ── step 1: data ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SVOE Training Pipeline")
print("=" * 60)

if args.refresh_data:
    print("\n[1/5] Purging data cache …")
    purge_cache(seasons=args.seasons, season_types=args.season_types)

print("\n[1/5] Loading shot data …")
raw = load_all_shots(seasons=args.seasons, season_types=args.season_types)

if raw.empty:
    print("ERROR: No data loaded. Check your internet connection or NBA API status.")
    sys.exit(1)

print(f"  Raw rows: {len(raw):,}")

# ── step 2: preprocessing ─────────────────────────────────────────────────────

print("\n[2/5] Cleaning & preprocessing …")
clean = clean_shots(raw)
print(f"  Clean rows: {len(clean):,}")
print(f"  Seasons   : {sorted(clean['SEASON'].unique())}")
print(f"  Make rate : {clean['SHOT_MADE_FLAG'].mean():.3f}")
print(f"\n  Feature summary:\n{feature_summary(clean).to_string(index=False)}")

# ── step 3: train / test split ────────────────────────────────────────────────

print("\n[3/5] Building feature matrix & splitting …")
X, y = extract_X_y(clean)

seasons_available = sorted(clean["SEASON"].unique())

if len(seasons_available) >= 2:
    # Chronological: train on all-but-last, test on last season
    last_season = seasons_available[-1]
    mask_test = clean["SEASON"] == last_season
    X_train_full, X_test = X[~mask_test], X[mask_test]
    y_train_full, y_test = y[~mask_test], y[mask_test]
else:
    # Single season: random 70/15/15 split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

# Reserve 20 % of training set for post-hoc calibration
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

print(f"  Train  : {len(X_train):,} shots")
print(f"  Cal    : {len(X_cal):,} shots (calibration)")
print(f"  Test   : {len(X_test):,} shots")

# ── step 4: model training ────────────────────────────────────────────────────

models_exist = (MODEL_DIR / "random_forest.joblib").exists()

if models_exist and not args.retrain:
    print("\n[4/5] Models already trained (use --retrain to re-train).")
    from src.modeling import load_model
    from src.metrics import load_metrics
    models = {}
    meta = load_metrics()
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        try:
            models[name] = load_model(name)
        except FileNotFoundError:
            pass
else:
    print("\n[4/5] Training models …")
    models = train_all_models(X_train, y_train, X_cal, y_cal)

    print("\n  Saving models …")
    for name, model in models.items():
        save_model(model, name)

    print("\n  Evaluating on test set …")
    meta = evaluate_all_models(models, X_test, y_test)
    save_metrics(meta)

    # Save calibration figure (requires kaleido; skip gracefully if missing)
    probs = {n: predict_proba(m, X_test) for n, m in models.items()}
    cal_fig = calibration_figure(probs, y_test)
    try:
        cal_fig.write_image(MODEL_DIR / "calibration.png")
        print("  Calibration plot → models/calibration.png")
    except Exception:
        cal_fig.write_html(str(MODEL_DIR / "calibration.html"))
        print("  Calibration plot → models/calibration.html (install kaleido for PNG)")

# ── step 5: score all shots → compute SVOE ────────────────────────────────────

print("\n[5/5] Scoring all shots & computing SVOE …")

best_name = meta.get("best_model", "random_forest")
best_model = models.get(best_name)
if best_model is None:
    from src.modeling import load_model
    best_model = load_model(best_name)

print(f"  Using best model: {best_name}")

clean["PRED_PROB"] = predict_proba(best_model, X)
final = compute_svoe(clean, prob_col="PRED_PROB")

SVOE_PATH.parent.mkdir(parents=True, exist_ok=True)
final.to_parquet(SVOE_PATH, index=False)
print(f"  Saved {len(final):,} rows → {SVOE_PATH}")

# Summary
print("\n" + "=" * 60)
print("  Done! Summary")
print("=" * 60)
print(f"  Best model     : {best_name}")
best_metrics = {k: v for k, v in meta.get(best_name, {}).items() if k != "name"}
for metric, val in best_metrics.items():
    print(f"  {metric:20s}: {val:.4f}")
print(f"\n  League SVOE    : {final['SVOE'].sum():+.1f} pts")
print(f"  Avg SVOE/shot  : {final['SVOE'].mean():+.4f} pts")
print("\n  Run the dashboard: streamlit run app.py")
print("=" * 60 + "\n")
