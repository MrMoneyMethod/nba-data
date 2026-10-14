"""Leakage-safe feature engineering.

Iron rule: every feature for a game is computed from games *strictly
before* that game (shifted rolling / expanding windows, Elo carried into
the game, rest days from the previous game's date). Current-game box
scores are only ever used as targets, never as inputs.

Public API:
    build_game_features()  -> writes data/processed/game_features.parquet
                              + data/processed/team_form.parquet
    build_player_features()-> writes data/processed/player_features.parquet
                              + data/processed/player_form.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_reassembled"
PROC = ROOT / "data" / "processed"

GAME_TYPES_KEEP = ["Regular Season", "Playoffs", "Play-in Tournament",
                   "NBA Emirates Cup", "Emirates NBA Cup", "NBA Cup"]
ELO_K, ELO_HOME_ADV, ELO_START = 20.0, 100.0, 1500.0

TEAM_STAT_COLS = [
    "gameId", "gameDateTimeEst", "teamId", "opponentTeamId", "home", "win",
    "teamScore", "opponentScore", "assists", "blocks", "steals",
    "fieldGoalsAttempted", "fieldGoalsMade", "threePointersAttempted",
    "threePointersMade", "freeThrowsAttempted", "freeThrowsMade",
    "reboundsOffensive", "reboundsDefensive", "reboundsTotal",
    "turnovers", "turnoversTeam", "foulsPersonal", "numMinutes", "gameType",
]
ROLL_STATS = ["pts", "opp_pts", "margin", "win", "off_rtg", "def_rtg",
              "pace", "efg", "tov_pct", "orb_pct", "ft_rate", "ast"]


def season_start_year(dates: pd.Series) -> pd.Series:
    return dates.dt.year - (dates.dt.month < 10).astype(int)


def _add_derived(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["margin"] = t.teamScore - t.opponentScore
    tov = t.turnovers.fillna(0) + t.turnoversTeam.fillna(0)
    poss = t.fieldGoalsAttempted + 0.44 * t.freeThrowsAttempted - t.reboundsOffensive + tov
    t["poss"] = poss.replace(0, np.nan)
    t["off_rtg"] = 100 * t.teamScore / t["poss"]
    t["def_rtg"] = 100 * t.opponentScore / t["poss"]
    t["pace"] = t["poss"]  # per-game possessions (~48 min); OT inflates slightly, acceptable
    t["efg"] = (t.fieldGoalsMade + 0.5 * t.threePointersMade) / t.fieldGoalsAttempted.replace(0, np.nan)
    t["tov_pct"] = tov / t["poss"]
    t["orb_pct"] = t.reboundsOffensive / (t.reboundsOffensive + t.reboundsDefensive).replace(0, np.nan)
    t["ft_rate"] = t.freeThrowsAttempted / t.fieldGoalsAttempted.replace(0, np.nan)
    t["ast"] = t.assists
    t["pts"] = t.teamScore
    t["opp_pts"] = t.opponentScore
    return t


def _prior_rolling(t: pd.DataFrame) -> pd.DataFrame:
    """Add shifted (prior-only) rolling + season-expanding features per team."""
    t = t.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)
    g = t.groupby("teamId", sort=False)
    # rest is known pre-game: days since this team's previous game
    prev_date = g["date"].shift(1)
    t["rest_days"] = (t["date"] - prev_date).dt.days.clip(0, 30).fillna(7)
    t["b2b"] = (t["rest_days"] <= 1).astype(int)
    # games in prior 7 days (prior only) via searchsorted per team
    t["games_last_7d_prior"] = 0
    for _, idx in t.groupby("teamId", sort=False).groups.items():
        sub = t.loc[idx]
        d = sub["date"].to_numpy()
        lo = np.searchsorted(d, d - np.timedelta64(7, "D"), side="left")
        hi = np.searchsorted(d, d, side="left")
        t.loc[idx, "games_last_7d_prior"] = hi - lo

    for w in (5, 10):
        rolled = g[ROLL_STATS].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        # rolling above includes current game -> shift by one game within team
        rolled = rolled.groupby(t["teamId"]).shift(1)
        rolled.columns = [f"{c}_l{w}" for c in rolled.columns]
        t = pd.concat([t, rolled], axis=1)
    # season-to-date prior (expanding, reset each season)
    gs = t.groupby(["teamId", "season"], sort=False)[ROLL_STATS]
    seas = gs.expanding(min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    seas = seas.groupby([t["teamId"], t["season"]]).shift(1)
    seas.columns = [f"{c}_season" for c in seas.columns]
    t = pd.concat([t, seas], axis=1)
    t["prior_games"] = g.cumcount()  # games this team played before this one (all-time)
    return t


def _elo_prior(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Sequential Elo over all games; returns elo carried *into* each game."""
    elo: dict[int, float] = {}
    rows = []
    ordered = games.sort_values(["date", "gameId"])
    for r in ordered.itertuples():
        h, a = int(r.hometeamId), int(r.awayteamId)
        eh, ea = elo.get(h, ELO_START), elo.get(a, ELO_START)
        rows.append((int(r.gameId), eh, ea))
        exp_h = 1.0 / (1.0 + 10 ** (-((eh + ELO_HOME_ADV) - ea) / 400.0))
        actual_h = 1.0 if r.homeScore > r.awayScore else 0.0
        mov = np.log(abs(r.homeScore - r.awayScore) + 1.0)  # margin-of-victory multiplier
        k = ELO_K * mov
        elo[h] = eh + k * (actual_h - exp_h)
        elo[a] = ea + k * ((1 - actual_h) - (1 - exp_h))
    elo_df = pd.DataFrame(rows, columns=["gameId", "elo_home_prior", "elo_away_prior"])
    # final ratings snapshot
    form = pd.DataFrame(sorted(elo.items()), columns=["teamId", "elo_now"])
    return elo_df, form


def build_game_features() -> pd.DataFrame:
    PROC.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(ROOT / "Games.csv", low_memory=False,
                        usecols=["gameId", "gameDateTimeEst", "hometeamId", "awayteamId",
                                 "homeScore", "awayScore", "gameType"],
                        parse_dates=["gameDateTimeEst"])
    games = games.rename(columns={"gameDateTimeEst": "date"})
    games = games[games.gameType.isin(GAME_TYPES_KEEP)].copy()
    games = games[games.homeScore != games.awayScore]  # 4 tied 1940s rows: no winner, unusable
    games["season"] = season_start_year(games["date"])
    games["home_win"] = (games.homeScore > games.awayScore).astype(int)
    games["margin"] = games.homeScore - games.awayScore
    games["total"] = games.homeScore + games.awayScore
    games["is_playoffs"] = games.gameType.isin(["Playoffs"]).astype(int)

    team = pd.read_csv(RAW / "TeamStatistics.csv", usecols=TEAM_STAT_COLS,
                       parse_dates=["gameDateTimeEst"], low_memory=False)
    team = team.rename(columns={"gameDateTimeEst": "date", "opponentTeamId": "oppId"})
    team = team[team.gameType.isin(GAME_TYPES_KEEP)].copy()
    team["season"] = season_start_year(team["date"])
    team = _add_derived(team)
    team = _prior_rolling(team)

    elo_df, elo_now = _elo_prior(games, team)

    prior_cols = (["rest_days", "b2b", "games_last_7d_prior", "prior_games"] +
                  [f"{c}_l{w}" for w in (5, 10) for c in ROLL_STATS] +
                  [f"{c}_season" for c in ROLL_STATS])
    home = team[team.home == 1].set_index("gameId")[prior_cols].add_prefix("home_")
    away = team[team.home == 0].set_index("gameId")[prior_cols].add_prefix("away_")

    df = (games.set_index("gameId")
          .join(home, how="left").join(away, how="left")
          .join(elo_df.set_index("gameId"), how="left")
          .reset_index())
    df["elo_diff"] = df.elo_home_prior - df.elo_away_prior + ELO_HOME_ADV
    for c in [f"{s}_l{w}" for w in (5, 10) for s in ROLL_STATS] + [f"{s}_season" for s in ROLL_STATS]:
        df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]
    df["rest_diff"] = df.home_rest_days - df.away_rest_days
    df = df.sort_values(["date", "gameId"]).reset_index(drop=True)

    out = PROC / "game_features.parquet"
    df.to_parquet(out, index=False)
    print(f"game_features: {len(df):,} games -> {out}")

    # ---- current team form snapshot (for predicting future games) ----
    last = (team.sort_values(["teamId", "date", "gameId"])
            .groupby("teamId", as_index=False).tail(1))
    form_cols = ["teamId", "date"] + [c for c in team.columns if c.endswith(("_l5", "_l10", "_season"))]
    # form snapshot must reflect form *including* the last played game:
    # recompute unshifted rolling for the final row per team
    t2 = team.sort_values(["teamId", "date", "gameId"]).copy()
    g2 = t2.groupby("teamId", sort=False)
    snap = g2.tail(1)[["teamId", "date"]].copy()
    for w in (5, 10):
        r = g2[ROLL_STATS].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        r.columns = [f"{c}_l{w}" for c in r.columns]
        snap = snap.join(r.loc[snap.index])
    seas = (t2.groupby(["teamId", "season"], sort=False)[ROLL_STATS]
            .expanding(min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    seas.columns = [f"{c}_season" for c in seas.columns]
    snap = snap.join(seas.loc[snap.index])
    snap = snap.merge(elo_now, on="teamId", how="left")
    snap["prior_games"] = snap["teamId"].map(t2.groupby("teamId").size()).astype(int)
    # team identity lookup
    _g = pd.read_csv(ROOT / "Games.csv", low_memory=False,
                     usecols=["hometeamId", "hometeamCity", "hometeamName", "gameDateTimeEst"],
                     parse_dates=["gameDateTimeEst"]).sort_values("gameDateTimeEst")
    names = (_g.drop_duplicates("hometeamId", keep="last")
             .rename(columns={"hometeamId": "teamId", "hometeamCity": "teamCity",
                              "hometeamName": "teamName"})
             [["teamId", "teamCity", "teamName"]])
    snap = snap.merge(names, on="teamId", how="left")
    snap.to_parquet(PROC / "team_form.parquet", index=False)
    print(f"team_form: {len(snap)} teams (as of {snap.date.max()})")
    return df


PLAYER_COLS = ["personId", "gameId", "gameDateTimeEst", "playerteamId", "opponentteamId",
               "home", "numMinutes", "points", "assists", "reboundsTotal",
               "reboundsOffensive", "reboundsDefensive", "fieldGoalsAttempted",
               "fieldGoalsMade", "threePointersAttempted", "threePointersMade",
               "turnovers", "steals", "blocks", "startingPosition", "gameType"]
PROP_STATS = ["points", "reboundsTotal", "assists", "numMinutes"]


def build_player_features() -> pd.DataFrame:
    PROC.mkdir(parents=True, exist_ok=True)
    frames = []
    for chunk in pd.read_csv(RAW / "PlayerStatistics.csv", usecols=PLAYER_COLS,
                             parse_dates=["gameDateTimeEst"], chunksize=400_000,
                             low_memory=False):
        chunk = chunk[chunk.gameType.isin(GAME_TYPES_KEEP)]
        frames.append(chunk)
    p = pd.concat(frames, ignore_index=True)
    for c in ["numMinutes", "points", "assists", "reboundsTotal", "reboundsOffensive",
              "reboundsDefensive", "fieldGoalsAttempted", "fieldGoalsMade",
              "threePointersAttempted", "threePointersMade", "turnovers", "steals", "blocks"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.rename(columns={"gameDateTimeEst": "date", "playerteamId": "teamId",
                          "opponentteamId": "oppId", "reboundsTotal": "rebounds"})
    p["season"] = season_start_year(p["date"])
    p["started"] = p.startingPosition.notna().astype(int)
    stats = ["points", "rebounds", "assists", "numMinutes"]
    p = p.sort_values(["personId", "date", "gameId"]).reset_index(drop=True)
    g = p.groupby("personId", sort=False)
    p["rest_days"] = (p["date"] - g["date"].shift(1)).dt.days.clip(0, 30).fillna(7)
    for w in (5, 10):
        r = g[stats + ["started"]].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        r = r.groupby(p["personId"]).shift(1)
        r.columns = [f"{c}_l{w}" for c in r.columns]
        p = pd.concat([p, r], axis=1)
    seas = (p.groupby(["personId", "season"], sort=False)[stats]
            .expanding(min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    seas = seas.groupby([p["personId"], p["season"]]).shift(1)
    seas.columns = [f"{c}_season" for c in seas.columns]
    p = pd.concat([p, seas], axis=1)
    p["prior_games"] = g.cumcount()
    p = p.rename(columns={"numMinutes": "minutes"})
    out = PROC / "player_features.parquet"
    p.to_parquet(out, index=False)
    print(f"player_features: {len(p):,} player-games -> {out}")

    # current player form (including last game) for future prop projections
    snap_stats = ["points", "rebounds", "assists", "minutes"]
    t2 = p.sort_values(["personId", "date", "gameId"]).copy()
    g2 = t2.groupby("personId", sort=False)
    snap = g2.tail(1)[["personId", "teamId", "date"]].copy()
    for w in (5, 10):
        r = g2[snap_stats + ["started"]].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        r.columns = [f"{c}_l{w}" for c in r.columns]
        snap = snap.join(r.loc[snap.index])
    snap = snap.rename(columns={"minutes_l5": "numMinutes_l5", "minutes_l10": "numMinutes_l10"})
    _seas = (t2.groupby(["personId", "season"], sort=False)[["points", "rebounds", "assists", "minutes"]]
             .expanding(min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    _seas.columns = ["points_season", "rebounds_season", "assists_season", "numMinutes_season"]
    snap = snap.join(_seas.loc[snap.index])
    snap = snap.rename(columns={"numMinutes_l5": "minutes_l5", "numMinutes_l10": "minutes_l10"})
    snap["prior_games"] = snap["personId"].map(t2.groupby("personId").size()).astype(int)
    snap = snap.merge(
        pd.read_csv(ROOT / "Players.csv", usecols=["personId", "firstName", "lastName"]),
        on="personId", how="left")
    snap.to_parquet(PROC / "player_form.parquet", index=False)
    print(f"player_form: {len(snap):,} players (as of {snap.date.max()})")
    return p


if __name__ == "__main__":
    build_game_features()
    build_player_features()
