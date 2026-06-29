# Shot Value Over Expected (SVOE)

An NBA analytics project that estimates the expected value of every shot using pre-shot context only, then compares actual performance to expectation.

---

## What is SVOE?

| Term | Definition |
|------|-----------|
| **Shot Value** | 2 for 2PT attempts · 3 for 3PT attempts |
| **Expected Points** | Predicted Make Probability × Shot Value |
| **Actual Points** | Shot Made Flag × Shot Value |
| **SVOE** | Actual Points − Expected Points |

Positive SVOE = the player/team converted **better than expected** given their exact shot locations, contexts, and opponents.

SVOE separates **shot-making** from **shot selection** — a player who only takes corner threes and restricted-area layups will naturally have a high Expected Points per shot. SVOE measures whether they actually convert above that baseline.

---

## Project Structure

```
svoe/
├── data/
│   ├── raw/           # per-team CSV cache (auto-created)
│   └── processed/     # merged Parquet + SVOE output
├── models/            # saved model files + metrics.json
├── src/
│   ├── data_loader.py    # NBA API calls + caching
│   ├── preprocessing.py  # data cleaning + derived columns
│   ├── features.py       # feature matrix construction
│   ├── modeling.py       # model training + persistence
│   ├── metrics.py        # evaluation + SVOE computation
│   └── visuals.py        # Plotly court charts + helpers
├── app.py             # Streamlit dashboard (7 tabs)
├── train.py           # end-to-end training script
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone / navigate to the project
cd svoe

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **Optional:** XGBoost is included in requirements.txt.
> If installation fails (e.g., on ARM Macs), remove the `xgboost` line and the pipeline
> will automatically use Random Forest as the best model.

---

## Running the Pipeline

### Step 1 — Fetch data + train models

```bash
python train.py
```

What it does:
1. Fetches shot chart data for every NBA team via `ShotChartDetail` (cached locally)
2. Cleans and engineers features
3. Trains Logistic Regression, Random Forest, and XGBoost (if available)
4. Calibrates models with isotonic regression
5. Selects the best model by Brier score
6. Scores every shot with SVOE
7. Saves `data/processed/shots_with_svoe.parquet` and `models/`

**Estimated runtime:**
- First run: ~5 min (API) + ~10 min (training) for 2 seasons
- Subsequent runs: <30 sec (everything cached)

### Step 2 — Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Advanced Options

### Fetch specific seasons only
```bash
python train.py --seasons 2023-24
python train.py --seasons 2021-22 2022-23 2023-24
```

### Include / exclude Playoffs
```bash
python train.py --season-types "Regular Season"
python train.py --season-types "Regular Season" "Playoffs"
```

### Force fresh data download (ignores cache)
```bash
python train.py --refresh-data
```

### Force re-train models (keeps existing data)
```bash
python train.py --retrain
```

### Full refresh
```bash
python train.py --refresh-data --retrain
```

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Formula explainer, league-wide KPIs, SVOE distribution, model performance |
| **Player Leaderboard** | Filterable table + bar chart sorted by SVOE/100 |
| **Team Leaderboard** | Team SVOE with shot quality vs shot making scatter |
| **Shot Map** | Interactive court — makes/misses, EP heatmap, SVOE heatmap |
| **Shot Profile** | SVOE breakdown by zone or action type for league / team / player |
| **Sustainability** | H1 vs H2 regression analysis + year-over-year stability |
| **Player Comparison** | Side-by-side shot maps, zone tables, action type breakdown |

---

## Model Features

All features are known **before the shot is taken** — no result leakage:

- `LOC_X`, `LOC_Y` — court coordinates
- `SHOT_DISTANCE` — distance in feet
- `SHOT_TYPE` — 2PT or 3PT
- `ACTION_TYPE` — jump shot, layup, dunk, hook, etc.
- `SHOT_ZONE_BASIC` / `SHOT_ZONE_AREA` / `SHOT_ZONE_RANGE`
- `PERIOD` — quarter / overtime
- `TIME_REMAINING_SECS` — seconds left in period
- `IS_HOME` — home / away
- `TEAM_ID` — shooting team
- `OPPONENT_TEAM_ID` — defending team

`PLAYER_ID` is intentionally excluded: we want to measure shot-making **above the contextual baseline**, not have the model absorb each player's historical make rate.

---

## Model Evaluation

Models are evaluated on a held-out test set (the most recent season, or 15% if only one season is available):

- **Log Loss** — overall probability quality
- **Brier Score** — calibration quality (lower = better probability estimates)
- **ROC-AUC** — discrimination (how well the model separates makes from misses)
- **Calibration plot** — saved to `models/calibration.png`

The model with the lowest Brier score is selected and used for all SVOE calculations.

---

## Data Source

Shot data is fetched from the [NBA Stats API](https://www.nba.com/stats) via the
[nba_api](https://github.com/swar/nba_api) Python library (ShotChartDetail endpoint).

The API does not require authentication. Requests are rate-limited to ~1 per 0.75 seconds.
All data is cached locally so the API is only hit once per team/season/type combination.

---

## Requirements

```
nba_api>=1.4.1
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.28.0
plotly>=5.17.0
matplotlib>=3.7.0
joblib>=1.3.0
requests>=2.31.0
pyarrow>=13.0.0
xgboost>=2.0.0   # optional — remove if install fails
```
