"""
Model training, persistence, and prediction for SVOE.

Models trained
--------------
1. Logistic Regression   (baseline; fast; interpretable)
2. Random Forest         (strong; well-calibrated with isotonic post-hoc)
3. XGBoost               (best in class; optional — skipped if not installed)

Each model is wrapped in a sklearn Pipeline so a single joblib file contains
the full preprocessing → prediction chain.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import make_preprocessor

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.info("XGBoost not installed — will train LR and RF only.")


# ── isotonic calibration wrapper ─────────────────────────────────────────────
# sklearn 1.4 removed cv="prefit" from CalibratedClassifierCV.
# This thin wrapper replicates the same behaviour: fit the base model on the
# training set, then fit an IsotonicRegression on the calibration set.

class _IsotonicCalibrated:
    """Base pipeline + isotonic post-calibration, sklearn 1.4+ compatible."""

    def __init__(self, base):
        self.base = base
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X_cal, y_cal):
        raw = self.base.predict_proba(X_cal)[:, 1]
        self._iso.fit(raw, y_cal)
        return self

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self._iso.predict(raw)
        return np.column_stack([1.0 - cal, cal])


# ── helpers ─────────────────────────────────────────────────────────────────

def _model_path(name: str) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / f"{name}.joblib"


def save_model(model, name: str) -> None:
    path = _model_path(name)
    joblib.dump(model, path)
    log.info("Saved %s → %s", name, path)


def load_model(name: str):
    path = _model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_best_model():
    meta_path = MODEL_DIR / "metrics.json"
    if not meta_path.exists():
        raise FileNotFoundError("No metrics.json found — run train.py first.")
    with open(meta_path) as f:
        meta = json.load(f)
    best = meta.get("best_model", "random_forest")
    return load_model(best), best


# ── training ─────────────────────────────────────────────────────────────────

def _build_lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=0.5,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _build_rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=9,
                    min_samples_leaf=30,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_xgb_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                ),
            ),
        ]
    )


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> dict:
    """
    Fit each model on X_train / y_train, then isotonic-calibrate on X_cal / y_cal.

    Returns dict: {model_name: fitted_pipeline}
    """
    trained: dict = {}

    # 1. Logistic Regression — already well-calibrated by design
    print("  Training Logistic Regression …")
    lr = _build_lr_pipeline()
    lr.fit(X_train, y_train)
    trained["logistic_regression"] = lr

    # 2. Random Forest — isotonic post-calibration on held-out cal set
    print("  Training Random Forest …")
    rf_base = _build_rf_pipeline()
    rf_base.fit(X_train, y_train)
    rf_cal = _IsotonicCalibrated(rf_base).fit(X_cal, y_cal)
    trained["random_forest"] = rf_cal

    # 3. XGBoost (optional)
    if HAS_XGB:
        print("  Training XGBoost …")
        xgb_base = _build_xgb_pipeline()
        xgb_base.fit(X_train, y_train)
        xgb_cal = _IsotonicCalibrated(xgb_base).fit(X_cal, y_cal)
        trained["xgboost"] = xgb_cal

    return trained


# ── inference ────────────────────────────────────────────────────────────────

def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(make) for each shot in X."""
    return model.predict_proba(X)[:, 1]
