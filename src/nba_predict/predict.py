"""Predict future games / player props from the current form snapshots.

The snapshots (`team_form.parquet`, `player_form.parquet`) hold each
team/player's form *including* their last played game, which is exactly
the prior-only feature state for their next game. Rest days are computed
against the requested game date; `games_last_7d_prior` is approximated
from rest (documented limitation: the snapshot does not retain the full
recent schedule density).

CLI:
    .venv/bin/python scripts/predict.py game --home "Boston Celtics" --away "New York Knicks"
    .venv/bin/python scripts/predict.py game --home Celtics --away Knicks --date 2026-10-25 \
        --spread -4.5 --total 224.5 --home-ml -180 --away-ml 155
    .venv/bin/python scripts/predict.py player --name "Jayson Tatum" --home
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ROLL_STATS = ["pts", "opp_pts", "margin", "win", "off_rtg", "def_rtg",
              "pace", "efg", "tov_pct", "orb_pct", "ft_rate", "ast"]


def _norm(s):
    if isinstance(s, pd.Series):
        return s.str.lower().str.replace(".", "", regex=False).str.strip()
    return s.lower().replace(".", "").strip()


def _find_team(form: pd.DataFrame, query: str) -> pd.Series:
    q = _norm(query)
    full = (_norm(form.teamCity.fillna("")) + " " + _norm(form.teamName.fillna("")))
    for mask in [full == q, _norm(form.teamName.fillna("")) == q, full.str.contains(q, regex=False)]:
        hits = form[mask]
        if len(hits) == 1:
            return hits.iloc[0]
        if len(hits) > 1:
            raise ValueError(f"'{query}' matches multiple teams: "
                             f"{(hits.teamCity + ' ' + hits.teamName).tolist()}")
    raise ValueError(f"Team '{query}' not found. Teams: {(form.teamCity + ' ' + form.teamName).tolist()}")


def predict_game(home_query: str, away_query: str,
                 game_date: dt.date | None = None,
                 market_spread: float | None = None,   # home spread, e.g. -4.5
                 market_total: float | None = None,
                 home_ml: float | None = None, away_ml: float | None = None) -> dict:
    bundle = joblib.load(MODELS / "game_models.joblib")
    form = pd.read_parquet(PROC / "team_form.parquet")
    home, away = _find_team(form, home_query), _find_team(form, away_query)
    if game_date is None:
        game_date = dt.date.today()
    gd = pd.Timestamp(game_date)

    row: dict[str, float] = {"is_playoffs": 0.0, "season": float(gd.year - (gd.month < 10))}
    for prefix, t in [("home", home), ("away", away)]:
        rest = int(np.clip((gd - pd.Timestamp(t["date"])).days, 0, 30))
        row[f"{prefix}_rest_days"] = float(rest)
        row[f"{prefix}_b2b"] = float(rest <= 1)
        row[f"{prefix}_games_last_7d_prior"] = float(1 if rest <= 3 else 0)  # approx, see docstring
        for c in [f"{s}_l{w}" for w in (5, 10) for s in ROLL_STATS] + [f"{s}_season" for s in ROLL_STATS]:
            row[f"{prefix}_{c}"] = float(t[c])
    row["elo_home_prior"], row["elo_away_prior"] = float(home.elo_now), float(away.elo_now)
    row["elo_diff"] = row["elo_home_prior"] - row["elo_away_prior"] + 100.0
    row["rest_diff"] = row["home_rest_days"] - row["away_rest_days"]
    for c in [f"{s}_l{w}" for w in (5, 10) for s in ROLL_STATS] + [f"{s}_season" for s in ROLL_STATS]:
        row[f"diff_{c}"] = row[f"home_{c}"] - row[f"away_{c}"]

    X = pd.DataFrame([row]).reindex(columns=bundle["features"])
    p_elo = 1.0 / (1.0 + 10 ** (-row["elo_diff"] / 400.0))
    p_home = float((1 - bundle["win_elo_weight"]) * bundle["clf"].predict_proba(X)[:, 1][0]
                   + bundle["win_elo_weight"] * p_elo)
    pred_margin = float((1 - bundle["margin_elo_weight"]) * bundle["reg_margin"].predict(X)[0]
                        + bundle["margin_elo_weight"] * np.polyval(bundle["elo_margin_coef"], row["elo_diff"]))
    pred_total = float(bundle["reg_total"].predict(X)[0])

    out: dict = {
        "home_team": f"{home.teamCity} {home.teamName}",
        "away_team": f"{away.teamCity} {away.teamName}",
        "game_date": str(game_date),
        "data_as_of": str(max(pd.Timestamp(home["date"]), pd.Timestamp(away["date"])).date()),
        "p_home_win": round(p_home, 4),
        "p_away_win": round(1 - p_home, 4),
        "pred_margin_home": round(pred_margin, 1),       # >0 = home wins by this much
        "fair_spread_home": round(-pred_margin, 1),      # sportsbook-style home spread
        "pred_total": round(pred_total, 1),
    }
    ms, ts = bundle["margin_resid_sigma"], bundle["total_resid_sigma"]
    if market_spread is not None:
        p_cover = float(1 - norm.cdf(-market_spread, loc=pred_margin, scale=ms))
        out["market_spread_home"] = market_spread
        out["p_home_covers"] = round(p_cover, 4)
        out["spread_pick"] = "home" if p_cover >= 0.5 else "away"
    if market_total is not None:
        p_over = float(1 - norm.cdf(market_total, loc=pred_total, scale=ts))
        out["market_total"] = market_total
        out["p_over"] = round(p_over, 4)
        out["total_pick"] = "over" if p_over >= 0.5 else "under"
    if home_ml is not None and away_ml is not None:
        def _imp(ml: float) -> float:
            return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)
        mkt = _imp(home_ml) / (_imp(home_ml) + _imp(away_ml))
        out["market_p_home_win_no_vig"] = round(float(mkt), 4)
        out["edge_home_win_prob"] = round(p_home - float(mkt), 4)
    return out


def predict_player(name_query: str, home: bool = True,
                   game_date: dt.date | None = None) -> dict:
    bundle = joblib.load(MODELS / "player_models.joblib")
    form = pd.read_parquet(PROC / "player_form.parquet")
    full = (_norm(form.firstName.fillna("")) + " " + _norm(form.lastName.fillna("")))
    q = _norm(name_query)
    hits = form[(full == q) | full.str.contains(q, regex=False)]
    if len(hits) == 0:
        raise ValueError(f"Player '{name_query}' not found.")
    if len(hits) > 1:  # prefer the most recently active match
        hits = hits.sort_values("date", ascending=False)
    p = hits.iloc[0]
    gd = pd.Timestamp(game_date or dt.date.today())
    row = {
        "points_l5": p.points_l5, "points_l10": p.points_l10, "points_season": p.points_season,
        "rebounds_l5": p.rebounds_l5, "rebounds_l10": p.rebounds_l10, "rebounds_season": p.rebounds_season,
        "assists_l5": p.assists_l5, "assists_l10": p.assists_l10, "assists_season": p.assists_season,
        "numMinutes_l5": p.minutes_l5, "numMinutes_l10": p.minutes_l10,
        "numMinutes_season": p.numMinutes_season,
        "started_l5": p.started_l5, "started_l10": p.started_l10,
        "home": float(home),
        "rest_days": float(np.clip((gd - pd.Timestamp(p["date"])).days, 0, 30)),
    }
    X = pd.DataFrame([row]).reindex(columns=bundle["features"])
    out = {"player": f"{p.firstName} {p.lastName}", "data_as_of": str(pd.Timestamp(p["date"]).date()),
           "proj_minutes": round(float(p.minutes_l10), 1)}
    for stat in ["points", "rebounds", "assists"]:
        out[f"proj_{stat}"] = round(float(bundle["models"][stat].predict(X)[0]), 1)
    out["note"] = ("Projection uses prior-only form features (last-5/10 + season averages). "
                   "Assumes the player is active/healthy and plays normal rotation minutes "
                   "-- always check injury reports before betting a prop.")
    return out
