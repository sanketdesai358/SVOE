"""
Feature matrix construction for SVOE models.

Design principle: all features must be knowable *before* the shot is taken.
No SHOT_MADE_FLAG, ACTUAL_POINTS, or any derived result column is included.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── feature column lists ────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "PERIOD",
    "TIME_REMAINING_SECS",
    "IS_HOME",
    "TEAM_ID",
    "OPPONENT_TEAM_ID",
]

CATEGORICAL_FEATURES = [
    "SHOT_TYPE",        # '2PT Field Goal' | '3PT Field Goal'
    "ACTION_TYPE",      # 'Jump Shot', 'Layup', 'Dunk', …
    "SHOT_ZONE_BASIC",  # 'Restricted Area', 'Mid-Range', …
    "SHOT_ZONE_AREA",   # 'Center(C)', 'Left Side(L)', …
    "SHOT_ZONE_RANGE",  # 'Less Than 8 ft.', '8-16 ft.', …
]

ALL_FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COL = "SHOT_MADE_FLAG"


def make_preprocessor() -> ColumnTransformer:
    """
    Return a fitted-ready ColumnTransformer:
      - StandardScaler for numeric columns (harmless for trees, helps LR)
      - OrdinalEncoder for categoricals (works for trees; LR gets a meaningful
        ordering that still improves over raw strings)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def extract_X_y(df: pd.DataFrame):
    """
    Return (X, y) for the full DataFrame.

    X is a plain DataFrame of raw feature columns (un-encoded).
    The encoding happens inside each model's Pipeline.
    """
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    X = df[ALL_FEATURE_COLS].copy()

    # Ensure numeric columns are float so the scaler won't complain
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # Ensure categoricals are string
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).fillna("Unknown")

    y = df[TARGET_COL].astype(int)
    return X, y


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick stats table for the feature columns — useful for inspection."""
    rows = []
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append(
            {
                "feature": col,
                "type": "numeric" if col in NUMERIC_FEATURES else "categorical",
                "n_unique": s.nunique(),
                "null_pct": round(s.isna().mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)
