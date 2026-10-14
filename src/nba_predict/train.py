"""Train game-outcome + player-prop models with a strict time split.

Split (no shuffling -- shuffling basketball games leaks future form):
    train : date <  2023-10-01
    valid : 2023-10-01 .. 2024-09-30   (early-stopping / sanity only)
    test  : date >= 2024-10-01         (reported, never trained on)

Recency weighting (per user choice): sample_weight = 0.5 ** (age / half-life),
half-life = 3 seasons for games, 2 seasons for players, age measured back
from the end of the training window.

Market backtest: predictions for the test window are joined to the
2025-26 odds sample on (date, home/away team slug). Only the *pre-game
lines* (spread, total, moneyline) are used, and only for evaluation --
never as features, and none of that file's same-game engineered columns
(`total_points`, streaks including the current game, ...) are touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             mean_absolute_error, mean_squared_error, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TRAIN_END = pd.Timestamp("2023-10-01")
VALID_END = pd.Timestamp("2024-10-01")
GAME_HALF_LIFE_DAYS = 3 * 365
PLAYER_HALF_LIFE_DAYS = 2 * 365

GAME_FEATURES_EXCLUDE = {
    "gameId", "date", "hometeamId", "awayteamId", "homeScore", "awayScore",
    "gameType", "home_win", "margin", "total",
    "home_prior_games", "away_prior_games",  # franchise age, not form; breaks future prediction
}
PLAYER_FEATURES = ["points_l5", "points_l10", "points_season",
                   "rebounds_l5", "rebounds_l10", "rebounds_season",
                   "assists_l5", "assists_l10", "assists_season",
                   "numMinutes_l5", "numMinutes_l10", "numMinutes_season",
                   "started_l5", "started_l10", "home", "rest_days"]
PLAYER_TARGETS = ["points", "rebounds", "assists"]


def recency_weights(dates: pd.Series, ref: pd.Timestamp, half_life: int) -> np.ndarray:
    age_days = (ref - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    return 0.5 ** (age_days / half_life)


def elo_prob(elo_diff: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def _cls_metrics(y, p) -> dict:
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def _reg_metrics(y, p) -> dict:
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def train_games() -> tuple[dict, pd.DataFrame, list[str]]:
    df = pd.read_parquet(PROC / "game_features.parquet")
    df = df[df.home_win_l10.notna() & df.away_win_l10.notna()].copy()  # >=10 prior team-games warmup
    df = df.replace([np.inf, -np.inf], np.nan)
    feats = [c for c in df.columns if c not in GAME_FEATURES_EXCLUDE
             and pd.api.types.is_numeric_dtype(df[c])]
    train = df[df.date < TRAIN_END]
    valid = df[(df.date >= TRAIN_END) & (df.date < VALID_END)]
    test = df[df.date >= VALID_END]
    w = recency_weights(train.date, train.date.max(), GAME_HALF_LIFE_DAYS)

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=0.1, random_state=42)
    clf.fit(train[feats], train.home_win, sample_weight=w)
    reg_margin = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                               max_leaf_nodes=31, min_samples_leaf=40,
                                               l2_regularization=0.1, random_state=42)
    reg_margin.fit(train[feats], train.margin, sample_weight=w)
    reg_total = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                              max_leaf_nodes=31, min_samples_leaf=40,
                                              l2_regularization=0.1, random_state=42)
    reg_total.fit(train[feats], train.total, sample_weight=w)

    # --- Blend with Elo, weights chosen on VALID only (never on test) ---
    # Measured motivation: raw HGB (acc .640 / AUC .686 on test) underperforms
    # plain Elo (.668 / .720); a 50-60% Elo blend beats both. Grid-search the
    # blend on the validation season so the choice itself cannot leak test.
    p_valid_hgb = clf.predict_proba(valid[feats])[:, 1]
    p_valid_elo = elo_prob(valid.elo_diff).to_numpy()
    grid = np.arange(0.0, 1.01, 0.05)
    win_elo_weight = float(grid[np.argmin(
        [log_loss(valid.home_win, (1 - w_) * p_valid_hgb + w_ * p_valid_elo) for w_ in grid])])
    elo_margin_coef = np.polyfit(train.elo_diff, train.margin, 1)  # margin ~ elo_diff
    m_valid_hgb = reg_margin.predict(valid[feats])
    m_valid_elo = np.polyval(elo_margin_coef, valid.elo_diff)
    margin_elo_weight = float(grid[np.argmin(
        [mean_absolute_error(valid.margin, (1 - w_) * m_valid_hgb + w_ * m_valid_elo) for w_ in grid])])

    def blended(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = ((1 - win_elo_weight) * clf.predict_proba(part[feats])[:, 1]
             + win_elo_weight * elo_prob(part.elo_diff).to_numpy())
        m = ((1 - margin_elo_weight) * reg_margin.predict(part[feats])
             + margin_elo_weight * np.polyval(elo_margin_coef, part.elo_diff))
        return p, m

    metrics: dict = {"split": {"train_end": str(TRAIN_END.date()), "valid_end": str(VALID_END.date()),
                               "train_n": int(len(train)), "valid_n": int(len(valid)), "test_n": int(len(test))},
                     "features": feats,
                     "blend": {"win_elo_weight_chosen_on_valid": win_elo_weight,
                               "margin_elo_weight_chosen_on_valid": margin_elo_weight,
                               "elo_margin_coef": [float(c) for c in elo_margin_coef]}}
    for name, part in [("valid", valid), ("test", test)]:
        p_raw = clf.predict_proba(part[feats])[:, 1]
        p, m = blended(part)
        metrics[f"winner_{name}"] = _cls_metrics(part.home_win, p)  # headline = blended
        metrics[f"winner_{name}_hgb_raw"] = _cls_metrics(part.home_win, p_raw)
        metrics[f"winner_{name}_elo_baseline"] = _cls_metrics(part.home_win, elo_prob(part.elo_diff))
        metrics[f"winner_{name}_home_baseline_accuracy"] = float((part.home_win == 1).mean())
        metrics[f"margin_{name}"] = _reg_metrics(part.margin, m)
        metrics[f"total_{name}"] = _reg_metrics(part.total, reg_total.predict(part[feats]))
        metrics[f"margin_{name}_zero_baseline"] = _reg_metrics(part.margin, np.zeros(len(part)))
        metrics[f"total_{name}_mean_baseline"] = _reg_metrics(part.total, np.full(len(part), train.total.mean()))

    test_pred = test[["gameId", "date", "hometeamId", "awayteamId", "season",
                      "home_win", "margin", "total"]].copy()
    p_test, m_test = blended(test)
    test_pred["p_home_win"] = p_test
    test_pred["p_home_win_elo"] = elo_prob(test.elo_diff)
    test_pred["pred_margin"] = m_test
    test_pred["pred_total"] = reg_total.predict(test[feats])
    _, m_valid = blended(valid)
    metrics["margin_resid_sigma"] = float(np.std(valid.margin - m_valid))
    metrics["total_resid_sigma"] = float(np.std(valid.total - reg_total.predict(valid[feats])))

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "reg_margin": reg_margin, "reg_total": reg_total,
                 "features": feats,
                 "win_elo_weight": win_elo_weight,
                 "margin_elo_weight": margin_elo_weight,
                 "elo_margin_coef": elo_margin_coef,
                 "margin_resid_sigma": metrics["margin_resid_sigma"],
                 "total_resid_sigma": metrics["total_resid_sigma"]},
                MODELS / "game_models.joblib")
    return metrics, test_pred, feats


CLOSING_ODDS_FILES = ["nba_2024_25_closing_lines_ats.csv", "nba_2025_26_closing_lines_ats.csv"]
# NOTE: nba_2026_27_odds.csv is deliberately NOT used: it is a mislabeled
# duplicate of the 2025-26 file (identical game_id/date/scores in every row,
# only micro-differences in odds from a different bookmaker snapshot) -- it
# contains no 2026-27 fixtures. NBA_Closing_Odds_SAMPLE.csv (20 games, 2008,
# no spread/total) is likewise excluded from the market test.


def _team_name_to_id() -> dict[str, int]:
    """Latest full 'City Name' -> NBA teamId, from Games.csv (both columns)."""
    g = pd.read_csv(ROOT / "Games.csv", low_memory=False,
                    usecols=["gameDateTimeEst", "hometeamId", "hometeamCity", "hometeamName",
                             "awayteamId", "awayteamCity", "awayteamName"],
                    parse_dates=["gameDateTimeEst"]).sort_values("gameDateTimeEst")
    name2id: dict[str, int] = {}
    for _, r in g.iterrows():  # ascending date: latest name for an id wins
        name2id[f"{r.hometeamCity} {r.hometeamName}"] = int(r.hometeamId)
        name2id[f"{r.awayteamCity} {r.awayteamName}"] = int(r.awayteamId)
    # odds files say "Los Angeles Clippers"; Games.csv switched to "LA Clippers" in 2025-26
    if "LA Clippers" in name2id:
        name2id.setdefault("Los Angeles Clippers", name2id["LA Clippers"])
    return name2id


def market_backtest(test_pred: pd.DataFrame, margin_sigma: float = 14.6,
                    total_sigma: float = 18.9) -> dict:
    from scipy.stats import norm

    name2id = _team_name_to_id()
    frames = []
    for f in CLOSING_ODDS_FILES:
        d = pd.read_csv(ROOT / f, parse_dates=["date_et"])
        d["date"] = d["date_et"].dt.normalize()
        frames.append(d)
    odds = pd.concat(frames, ignore_index=True)
    odds["hometeamId"] = odds.home_team.map(name2id)
    odds["awayteamId"] = odds.away_team.map(name2id)

    left = test_pred.copy()
    left["date"] = left["date"].dt.normalize()
    m = left.merge(odds, on=["date", "hometeamId", "awayteamId"], how="inner",
                   suffixes=("", "_odds"))
    if "season_odds" in m.columns:  # odds season label, e.g. '2024-25' (suffix="_odds" merge)
        m = m.rename(columns={"season_odds": "odds_season"})
    out: dict = {"source_files": CLOSING_ODDS_FILES,
                 "matched_games": int(len(m)),
                 "odds_rows": int(len(odds)),
                 "unmatched_odds_rows": int(len(odds) - len(m)),
                 "by_odds_season": ({str(s): int(n) for s, n in m.groupby("odds_season").size().items()}
                                    if len(m) else {})}
    if len(m) < 50:
        out["note"] = "Too few matched games for a market comparison."
        return out
    fav_home = m.close_home_win_prob >= 0.5
    disagree = (m.p_home_win >= 0.5) != fav_home
    edge = (m.p_home_win - m.close_home_win_prob).abs() >= 0.05
    out["moneyline"] = {
        "market_favourite_accuracy": float((fav_home == (m.home_win == 1)).mean()),
        "model_accuracy": float(((m.p_home_win >= 0.5) == (m.home_win == 1)).mean()),
        "model_accuracy_when_disagreeing_with_market":
            float((((m.p_home_win >= 0.5) == (m.home_win == 1))[disagree]).mean()) if disagree.any() else None,
        "n_disagreements": int(disagree.sum()),
        "model_log_loss": float(log_loss(m.home_win, m.p_home_win)),
        "market_log_loss_close_no_vig": float(log_loss(m.home_win, m.close_home_win_prob)),
        "market_log_loss_open_no_vig": float(log_loss(m.home_win, m.open_home_win_prob)),
        "model_accuracy_when_edge_ge_5pts": float((((m.p_home_win >= 0.5) == (m.home_win == 1))[edge]).mean()) if edge.any() else None,
        "n_edge_ge_5pts": int(edge.sum()),
        "model_mean_prob": float(m.p_home_win.mean()),
        "market_mean_prob": float(m.close_home_win_prob.mean()),
        "actual_home_win_rate": float(m.home_win.mean()),
    }
    p_cover = 1 - norm.cdf(-m.close_spread_home, loc=m.pred_margin, scale=margin_sigma)
    actual_cover = m.margin > (-m.close_spread_home)
    push = m.margin == (-m.close_spread_home)
    out["spread"] = {"model_ats_accuracy_excl_pushes": float(((p_cover >= 0.5)[~push] == actual_cover[~push]).mean()),
                     "home_cover_rate_excl_pushes": float(actual_cover[~push].mean()),
                     "pushes": int(push.sum()), "n": int(len(m))}
    p_over = 1 - norm.cdf(m.close_total, loc=m.pred_total, scale=total_sigma)
    actual_over = m.total > m.close_total
    push_t = m.total == m.close_total
    out["total_market"] = {"model_ou_accuracy_excl_pushes": float(((p_over >= 0.5)[~push_t] == actual_over[~push_t]).mean()),
                           "over_rate_excl_pushes": float(actual_over[~push_t].mean()),
                           "pushes": int(push_t.sum()), "n": int(len(m))}
    out["margin_mae_on_market_games"] = float(mean_absolute_error(m.margin, m.pred_margin))
    out["total_mae_on_market_games"] = float(mean_absolute_error(m.total, m.pred_total))
    keep = ["date", "home_team", "away_team", "home_score", "away_score", "home_win",
            "p_home_win", "open_home_win_prob", "close_home_win_prob",
            "pred_margin", "close_spread_home", "pred_total", "close_total"]
    m[keep].to_csv(REPORTS / "market_backtest_games.csv", index=False)
    return out


def train_players() -> dict:
    df = pd.read_parquet(PROC / "player_features.parquet")
    df = df[df.prior_games >= 5].copy()
    feats = PLAYER_FEATURES  # rolling cols are numMinutes_*; target cols are points/rebounds/assists
    train = df[df.date < TRAIN_END]
    test = df[df.date >= VALID_END]
    # evaluation population: established rotation players only (pre-game info)
    test_eval = test[test.numMinutes_l10 >= 12]
    w = recency_weights(train.date, train.date.max(), PLAYER_HALF_LIFE_DAYS)
    models, metrics = {}, {}
    base_col = {"points": "points_l10", "rebounds": "rebounds_l10", "assists": "assists_l10"}
    for target in PLAYER_TARGETS:
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.07,
                                            max_leaf_nodes=31, min_samples_leaf=40,
                                            l2_regularization=0.1, random_state=42)
        reg.fit(train[feats], train[target], sample_weight=w)
        models[target] = reg
        metrics[target] = {
            "test_all_prior5": _reg_metrics(test[target], reg.predict(test[feats])),
            "test_rotation_min12": _reg_metrics(test_eval[target], reg.predict(test_eval[feats])),
            "test_rotation_baseline_last10avg": _reg_metrics(test_eval[target], test_eval[base_col[target]]),
            "test_rotation_n": int(len(test_eval)),
        }
    joblib.dump({"models": models, "features": feats}, MODELS / "player_models.joblib")
    return metrics


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    game_metrics, test_pred, _ = train_games()
    market = market_backtest(test_pred, game_metrics['margin_resid_sigma'], game_metrics['total_resid_sigma'])
    player_metrics = train_players()
    all_metrics = {"game": game_metrics, "market_backtest_closing_2024_26": market, "player_props": player_metrics}
    (REPORTS / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    md = ["# Backtest results", "",
          "Time-based split — train < 2023-10-01 · valid 2023-24 · **test ≥ 2024-10-01 (never trained on)**. "
          "Recency-weighted (3-season half-life, games; 2-season, players). "
          "All features are prior-only (see `src/nba_predict/features.py`).", "",
          "## Game winner (test)", ""]
    for k in ["winner_test", "winner_test_elo_baseline"]:
        m = game_metrics[k]
        md.append(f"- **{k}**: acc={m['accuracy']:.3f} logloss={m['log_loss']:.3f} "
                  f"brier={m['brier']:.3f} auc={m['roc_auc']:.3f} (n={m['n']:,})")
    md.append(f"- Home-team-always baseline accuracy: {game_metrics['winner_test_home_baseline_accuracy']:.3f}")
    md += ["", "## Margin / total (test)", "",
           f"- Margin MAE={game_metrics['margin_test']['mae']:.2f} (zero-margin baseline "
           f"{game_metrics['margin_test_zero_baseline']['mae']:.2f}), residual σ={game_metrics['margin_resid_sigma']:.1f}",
           f"- Total MAE={game_metrics['total_test']['mae']:.2f} (train-mean baseline "
           f"{game_metrics['total_test_mean_baseline']['mae']:.2f}), residual σ={game_metrics['total_resid_sigma']:.1f}",
           "", "## vs. the market (closing lines 2024-25 + 2025-26, test-window games only)", "",
           f"- Matched {market.get('matched_games', 0)} of {market.get('odds_rows', 0)} closing-line rows "
           f"(by season: {market.get('by_odds_season', {})}). The mislabeled duplicate "
           "`nba_2026_27_odds.csv` (same game_ids/scores as 2025-26) is excluded -- see audit.",
           "- Market inputs are evaluation-only: the files' outcome columns "
           "(`ats_result`, `favorite_won`, ...) are never used as features."]
    if "moneyline" in market:
        ml, sp, tm = market["moneyline"], market["spread"], market["total_market"]
        md += [f"- Moneyline: model acc={ml['model_accuracy']:.3f} vs market-favourite acc="
               f"{ml['market_favourite_accuracy']:.3f}; model logloss={ml['model_log_loss']:.3f} vs "
               f"close logloss={ml['market_log_loss_close_no_vig']:.3f} (open: "
               f"{ml['market_log_loss_open_no_vig']:.3f}); disagrees with market on "
               f"{ml['n_disagreements']} games"
               + (f" (acc there: {ml['model_accuracy_when_disagreeing_with_market']:.3f})"
                  if ml["model_accuracy_when_disagreeing_with_market"] is not None else "")
               + (f"; when |model-market| >= 5 pts (n={ml['n_edge_ge_5pts']}): acc="
                  f"{ml['model_accuracy_when_edge_ge_5pts']:.3f}"
                  if ml["model_accuracy_when_edge_ge_5pts"] is not None else ""),
               f"- Spread (model margin vs close_spread_home): ATS acc={sp['model_ats_accuracy_excl_pushes']:.3f} "
               f"excl. {sp['pushes']} pushes (home covers {sp['home_cover_rate_excl_pushes']:.3f} of the time)",
               f"- Totals (model total vs close_total): acc={tm['model_ou_accuracy_excl_pushes']:.3f} "
               f"excl. {tm['pushes']} pushes (over hits {tm['over_rate_excl_pushes']:.3f})",
               "", "_Honest expectation: a box-score form model should be close to, but not beat, "
               "the closing market, which also prices injuries/lineups/news. Gaps vs. the market are "
               "reported, not hidden._"]
    md += ["", "## Player props (test, rotation players ≥12 min/game prior-10 avg)", ""]
    for t, m in player_metrics.items():
        md.append(f"- **{t}**: model MAE={m['test_rotation_min12']['mae']:.2f} vs last-10-avg baseline "
                  f"MAE={m['test_rotation_baseline_last10avg']['mae']:.2f} (n={m['test_rotation_n']:,})")
    (REPORTS / "backtest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nFull metrics -> {REPORTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
