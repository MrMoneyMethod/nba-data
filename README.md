# nba-data — dataset + NBA prediction engine

Historical NBA data (1946 → 2026 Finals) plus a leakage-safe prediction
engine for **game winners, margins/spreads, totals, and player props
(points / rebounds / assists)**.

> ⚠️ The raw files in the repo root total **~2.8 GB uploaded as ~25 MB
> byte-split parts** (GitHub web-upload limit). They are already committed
> to git history, which makes cloning heavy. Recommended follow-up: move
> the `*.part*` blobs to a GitHub Release / Git LFS and keep git lean.
> Reassembled and derived files are git-ignored and must not be committed.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/reassemble.py       # byte-join *.part* -> data/raw_reassembled/ (~2.8 GB)
.venv/bin/python scripts/audit_data.py       # -> reports/data_quality.md
.venv/bin/python scripts/build_features.py   # -> data/processed/*.parquet
.venv/bin/python scripts/train.py            # -> models/*.joblib, reports/backtest.md, reports/metrics.json
.venv/bin/python -m unittest discover -s tests -v   # leakage-guard tests

# Predict
.venv/bin/python scripts/predict.py game --home "Boston Celtics" --away "New York Knicks" \
    --date 2026-10-25 --spread -4.5 --total 224.5 --home-ml -180 --away-ml 155
.venv/bin/python scripts/predict.py player --name "Nikola Jokic"
```

## Data inventory (measured, see `reports/data_quality.md`)

| Source | Size | Rows | Coverage |
|---|---|---|---|
| `Games.csv` | 11 MB | 73,279 games | 1946-11-26 → 2026-06-13 |
| `TeamStatistics.csv.part1-2` | 36 MB | 146,560 team-games (2/game) | same |
| `PlayerStatistics.csv.part01-16` | 390 MB | 1,669,922 player-games | 1946 → 2026 |
| `TeamStatisticsExtended…part1-2` | 38 MB | 79,724 team-games | **1996 →** only (advanced stats) |
| `PlayerStatisticsExtended…part01-19` | 453 MB | 838,804 player-games | modern era |
| `PlayByPlay.parquet.part01-38` | 933 MB | 18,727,295 plays, 89 cols | reassembles to valid parquet |
| `nba_game_scores_1g_2025_26_sample.csv` | 0.7 MB | 1,267 games w/ odds | 2025-26 only |

Known quirks found in the audit: 4 tied games in the 1940s (dropped — the NBA
cannot tie), `attendance` is 98% null (unused), advanced (`Extended`) stats
only exist from 1996 (so v1 features use box-score stats available for all
eras), 13.1% of player rows are zero-minute DNPs.

> **Leakage warning (odds file):** `nba_game_scores_…csv` contains columns
> computed *including the current game* (`total_points`, `point_margin_game`,
> `won_on_spread`, cover/over streaks, …). Using any of them as features
> would leak the outcome. This engine uses that file **only** for market
> evaluation (pre-game spread/total/moneyline), never as model inputs.

## How the engine works

1. **Reassembly** (`scripts/reassemble.py`) — the parts are a raw byte split
   (only part01 has the parquet `PAR1` header), so reassembly is an ordered
   streaming concatenation with contiguity + parquet-magic validation.
2. **Features** (`src/nba_predict/features.py`) — every feature is computed
   from games *strictly before* the game being predicted:
   - per-team rolling form (last 5 / last 10 / season-to-date): points for &
     against, margin, win%, offensive/defensive rating & pace (possessions
     estimated as FGA + 0.44·FTA − ORB + TOV), eFG%, TOV%, ORB%, FT rate, AST;
   - sequential **Elo** carried into the game (K=20 × ln(|margin|+1), +100 home edge);
   - rest days, back-to-back flag, games in the prior 7 days;
   - home-minus-away differentials for every form stat.
   The same discipline applies per player (last-5/10 + season averages for
   pts/reb/ast/minutes, starter rate, rest). Tests in `tests/` prove a game's
   own box score cannot influence its features.
3. **Models** (`src/nba_predict/train.py`) — HistGradientBoosting
   classifier/regressors, trained on **all history since 1946 with recency
   weighting** (exponential decay, 3-season half-life for games, 2-season for
   players). Raw HGB underperformed plain Elo, so win probability and margin
   are **blended with Elo** (blend weights grid-searched on the validation
   season only). Margin/total are converted to cover/over probabilities with
   a Normal model using validation residual σ (14.3 / 18.9 pts).
4. **Prediction** (`src/nba_predict/predict.py`) — `team_form` /
   `player_form` snapshots hold each team/player's form *including* their
   last game (= prior state for the next game), so future matchups can be
   priced without rebuilding history.

## Honest backtest (time split: train < 2023-10 · valid 2023-24 · test ≥ 2024-10, never trained on)

From `reports/backtest.md` (regenerate with `scripts/train.py`):

- **Winner (test, n=2,635): accuracy 0.658, log-loss 0.618, AUC 0.718** —
  Elo alone: 0.668 / 0.635 / 0.720; raw HGB alone: 0.640 / 0.637 / 0.686;
  always-home baseline: 0.551.
- **Margin MAE 11.5 pts** (baseline 13.1), **total MAE 15.3 pts** (baseline 23.9).
- **Player props (rotation players, test):** points MAE 5.09 (last-10-avg
  baseline 5.24), rebounds 2.05 (2.12), assists 1.44 (1.49).
- **vs. the 2025-26 closing market (all 1,267 odds rows matched):** model
  moneyline accuracy 0.674 vs market-favourite 0.665, and 0.523 on the 239
  games where it disagrees with the market; but model log-loss 0.608 still
  trails no-vig market 0.591, and ATS (0.484) / O-U (0.482) picks are
  coin-flips against closing lines. **The market — which prices injuries,
  lineups and news this box-score model cannot see — remains better
  calibrated. This engine is a strong independent baseline, not a proven
  market-beater.** Per-game market comparisons: `reports/market_backtest_games.csv`.

## Repo layout

```
Games.csv, Players.csv, TeamHistories.csv, LeagueSchedule*.csv   small raw files (tracked)
*.part*                                                          split raw files (tracked, ~2.8 GB)
nba_game_scores_1g_2025_26_sample.csv                            odds sample (evaluation only)
scripts/    reassemble.py · audit_data.py · build_features.py · train.py · predict.py
src/nba_predict/  features.py · train.py · predict.py
tests/      test_no_leakage.py
reports/    data_quality.md · backtest.md · metrics.json · market_backtest_games.csv
data/processed/, data/raw_reassembled/, models/*.joblib          generated, git-ignored (models small enough to keep locally)
```

## Limitations & next steps

- No injury/lineup, betting-news, or referee/travel features — the main
  gap vs. the market. Play-by-play (18.7M rows) is reassembled and profiled
  but unused in v1; lineup +/- and clutch/pace splits are the natural v2.
- `games_last_7d_prior` in future predictions is approximated from rest
  days (the form snapshot keeps only the last game's date).
- Player projections assume normal minutes/availability — always check
  injury reports.
- Retire the split-file workflow: publish raw files as a Release asset or
  Git LFS and add a download script, so the git repo returns to a few MB.
