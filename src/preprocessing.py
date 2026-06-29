"""
Clean raw ShotChartDetail data and derive all pre-shot columns.

No result-leaking columns (make/miss) are used to engineer features.
"""

import logging

import numpy as np
import pandas as pd

from src.data_loader import TEAM_ABB_TO_ID, TEAM_ID_TO_ABB

log = logging.getLogger(__name__)

# Columns that must exist in raw data
_REQUIRED = [
    "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
    "TEAM_ID", "TEAM_NAME", "PERIOD",
    "MINUTES_REMAINING", "SECONDS_REMAINING",
    "SHOT_MADE_FLAG", "SHOT_TYPE", "ACTION_TYPE",
    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE", "LOC_X", "LOC_Y",
    "HTM", "VTM",
]

_KEEP = _REQUIRED + ["SEASON", "SEASON_TYPE"]


def clean_shots(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned DataFrame with derived columns:
      SHOT_VALUE, TIME_REMAINING_SECS, IS_HOME, OPPONENT_TEAM_ID,
      GAME_HALF (for sustainability tab).

    Only keeps shots with complete location / result data.
    """
    if raw.empty:
        return raw.copy()

    # ── keep only what we need ──────────────────────────────────────────────
    available = [c for c in _KEEP if c in raw.columns]
    df = raw[available].copy()

    # ── drop incomplete rows ─────────────────────────────────────────────────
    df.dropna(subset=["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_MADE_FLAG"], inplace=True)

    # ── coerce numeric types ─────────────────────────────────────────────────
    for col in ["LOC_X", "LOC_Y", "SHOT_DISTANCE",
                "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
                "SHOT_MADE_FLAG", "PLAYER_ID", "TEAM_ID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["SHOT_MADE_FLAG"], inplace=True)
    df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

    # ── game date ───────────────────────────────────────────────────────────
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d", errors="coerce")

    # ── shot value ──────────────────────────────────────────────────────────
    df["SHOT_VALUE"] = df["SHOT_TYPE"].apply(
        lambda x: 3 if isinstance(x, str) and "3PT" in x else 2
    )

    # ── time remaining in period (seconds) ──────────────────────────────────
    df["TIME_REMAINING_SECS"] = (
        df["MINUTES_REMAINING"].fillna(0) * 60
        + df["SECONDS_REMAINING"].fillna(0)
    )

    # ── home / away ──────────────────────────────────────────────────────────
    df["TEAM_ABB"] = df["TEAM_ID"].map(TEAM_ID_TO_ABB)
    df["IS_HOME"] = (df["TEAM_ABB"] == df["HTM"]).astype(int)

    # ── opponent abbreviation / ID ───────────────────────────────────────────
    df["OPPONENT_ABB"] = np.where(df["IS_HOME"] == 1, df["VTM"], df["HTM"])
    df["OPPONENT_TEAM_ID"] = df["OPPONENT_ABB"].map(TEAM_ABB_TO_ID).fillna(0).astype(int)

    # ── game half (for sustainability analysis) ──────────────────────────────
    # Avoid groupby().apply() — pandas 2.2+ strips group-key columns from
    # the slices passed to the function, dropping SEASON from the result.
    df["GAME_HALF"] = "H1"
    if "GAME_DATE" in df.columns and "SEASON" in df.columns:
        for season_val in df["SEASON"].unique():
            mask = df["SEASON"] == season_val
            dates = df.loc[mask, "GAME_DATE"].dropna().sort_values()
            if not dates.empty:
                mid_date = dates.iloc[len(dates) // 2]
                df.loc[mask & (df["GAME_DATE"] > mid_date), "GAME_HALF"] = "H2"

    # ── tidy up ─────────────────────────────────────────────────────────────
    if "SEASON_TYPE" not in df.columns:
        df["SEASON_TYPE"] = "Regular Season"

    df["ACTION_TYPE"] = df["ACTION_TYPE"].fillna("Unknown").str.strip()
    df["SHOT_ZONE_BASIC"] = df["SHOT_ZONE_BASIC"].fillna("Unknown").str.strip()
    df["SHOT_ZONE_AREA"] = df["SHOT_ZONE_AREA"].fillna("Unknown").str.strip()
    df["SHOT_ZONE_RANGE"] = df["SHOT_ZONE_RANGE"].fillna("Unknown").str.strip()
    df["SHOT_TYPE"] = df["SHOT_TYPE"].fillna("2PT Field Goal").str.strip()

    log.info("clean_shots: %d rows after cleaning", len(df))
    return df.reset_index(drop=True)
