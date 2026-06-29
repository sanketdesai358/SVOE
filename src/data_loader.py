"""
NBA shot data loader with local file caching.

Fetches ShotChartDetail for every team in each requested season/type.
Results are cached at two levels:
  1. Per-team raw CSV  →  data/raw/shots_{team_id}_{season}_{type}.csv
  2. Merged Parquet    →  data/processed/shots_{season}_{type}.parquet

Usage
-----
    from src.data_loader import load_all_shots
    df = load_all_shots(seasons=["2023-24"], season_types=["Regular Season"])
"""

import time
import logging
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.static import teams as nba_teams_static

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

_ALL_TEAMS = nba_teams_static.get_teams()
TEAM_ID_TO_ABB: dict[int, str] = {t["id"]: t["abbreviation"] for t in _ALL_TEAMS}
TEAM_ABB_TO_ID: dict[str, int] = {t["abbreviation"]: t["id"] for t in _ALL_TEAMS}
TEAM_ID_TO_NAME: dict[int, str] = {t["id"]: t["full_name"] for t in _ALL_TEAMS}


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_team_season(
    team_id: int,
    season: str,
    season_type: str,
    delay: float,
) -> pd.DataFrame:
    """Return shot chart rows for one team/season/type; reads cache when available."""
    safe_type = season_type.replace(" ", "_")
    cache = RAW_DIR / f"shots_{team_id}_{season}_{safe_type}.csv"

    if cache.exists():
        return pd.read_csv(cache, dtype={"GAME_ID": str})

    try:
        resp = ShotChartDetail(
            team_id=team_id,
            player_id=0,
            season_nullable=season,
            season_type_all_star=season_type,
            context_measure_simple="FGA",
        )
        df = resp.get_data_frames()[0]
        time.sleep(delay)
        if not df.empty:
            df.to_csv(cache, index=False)
        return df
    except Exception as exc:
        log.warning("team=%s season=%s type=%s → %s", team_id, season, season_type, exc)
        time.sleep(delay)
        return pd.DataFrame()


def load_all_shots(
    seasons=None,
    season_types=None,
    delay: float = 0.75,
) -> pd.DataFrame:
    """
    Load shots for every NBA team across the given seasons and season types.

    Parameters
    ----------
    seasons : list of str, default ["2022-23", "2023-24"]
    season_types : list of str, default ["Regular Season", "Playoffs"]
    delay : float
        Seconds to wait between API calls to respect rate limits.

    Returns
    -------
    pd.DataFrame with raw ShotChartDetail columns plus SEASON / SEASON_TYPE.
    """
    if seasons is None:
        seasons = ["2022-23", "2023-24"]
    if season_types is None:
        season_types = ["Regular Season", "Playoffs"]

    _ensure_dirs()
    all_frames: list[pd.DataFrame] = []

    for season in seasons:
        for stype in season_types:
            safe_type = stype.replace(" ", "_")
            master = PROCESSED_DIR / f"shots_{season}_{safe_type}.parquet"

            if master.exists():
                print(f"  [cache] {season} {stype}")
                chunk = pd.read_parquet(master)
                # Back-fill metadata columns in case cache predates them
                if "SEASON" not in chunk.columns:
                    chunk["SEASON"] = season
                if "SEASON_TYPE" not in chunk.columns:
                    chunk["SEASON_TYPE"] = stype
                all_frames.append(chunk)
                continue

            print(f"\n  Fetching {season} {stype} …")
            frames: list[pd.DataFrame] = []

            for i, team in enumerate(_ALL_TEAMS, 1):
                print(f"    [{i:2d}/30] {team['full_name']}")
                tdf = _fetch_team_season(team["id"], season, stype, delay)
                if not tdf.empty:
                    frames.append(tdf)

            if not frames:
                print(f"    No data returned for {season} {stype}.")
                continue

            combined = pd.concat(frames, ignore_index=True)
            combined.drop_duplicates(subset=["GAME_ID", "GAME_EVENT_ID"], inplace=True)
            combined["SEASON"] = season
            combined["SEASON_TYPE"] = stype
            combined.to_parquet(master, index=False)
            print(f"  Saved {len(combined):,} shots → {master}")
            all_frames.append(combined)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def purge_cache(seasons=None, season_types=None) -> None:
    """Delete cached files to force a fresh API pull on next run."""
    if seasons is None:
        seasons = ["2022-23", "2023-24"]
    if season_types is None:
        season_types = ["Regular Season", "Playoffs"]

    _ensure_dirs()
    for season in seasons:
        for stype in season_types:
            safe_type = stype.replace(" ", "_")
            master = PROCESSED_DIR / f"shots_{season}_{safe_type}.parquet"
            if master.exists():
                master.unlink()
                print(f"  Deleted {master}")
            for team in _ALL_TEAMS:
                raw = RAW_DIR / f"shots_{team['id']}_{season}_{safe_type}.csv"
                if raw.exists():
                    raw.unlink()
    print("Cache cleared.")
