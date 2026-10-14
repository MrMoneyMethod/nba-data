# Backtest results

Time-based split — train < 2023-10-01 · valid 2023-24 · **test ≥ 2024-10-01 (never trained on)**. Recency-weighted (3-season half-life, games; 2-season, players). All features are prior-only (see `src/nba_predict/features.py`).

## Game winner (test)

- **winner_test**: acc=0.658 logloss=0.618 brier=0.215 auc=0.718 (n=2,635)
- **winner_test_elo_baseline**: acc=0.668 logloss=0.635 brier=0.218 auc=0.720 (n=2,635)
- Home-team-always baseline accuracy: 0.551

## Margin / total (test)

- Margin MAE=11.47 (zero-margin baseline 13.07), residual σ=14.3
- Total MAE=15.27 (train-mean baseline 23.85), residual σ=18.9

## vs. the market (2025-26 odds sample, test-window games only)

- Matched 1267 of 1267 odds rows.
- Moneyline: model acc=0.674 vs market-favourite acc=0.665; model logloss=0.608 vs no-vig market logloss=0.591; disagrees with market on 239 games (acc there: 0.523)
- Spread (model margin → cover prob): ATS acc=0.484 excl. 20 pushes
- Totals (model total vs O/U): acc=0.482 excl. 15 pushes

_Honest expectation: a box-score form model should be close to, but not beat, the closing market, which also prices injuries/lineups/news. Gaps vs. the market are reported, not hidden._

## Player props (test, rotation players ≥12 min/game prior-10 avg)

- **points**: model MAE=5.09 vs last-10-avg baseline MAE=5.24 (n=51,878)
- **rebounds**: model MAE=2.05 vs last-10-avg baseline MAE=2.12 (n=51,878)
- **assists**: model MAE=1.44 vs last-10-avg baseline MAE=1.49 (n=51,878)
