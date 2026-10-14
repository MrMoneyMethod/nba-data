"""FantasyNerds NBA injuries integration.

Tested 2026-10-13 against
``https://api.fantasynerds.com/v1/nba/injuries?apikey=TEST``:

* The endpoint works and returns clean JSON:
  ``{"season": ..., "teams": {"ATL": [{"playerId", "name", "position",
  "injury", "game_status", "last_update"}, ...], ...}}``
* BUT the public ``TEST`` key returns a **frozen demo snapshot**
  (``season: 2021``, every ``last_update`` = 2021-04-13). It is five
  seasons stale and cannot be used for predictions or backtests.
* The API is a *current snapshot only* -- there is no historical
  (per-date) endpoint, so it cannot create leakage-safe historical
  features for the 2024-26 test window. With a real API key it is
  useful for (a) live player-prop gating and (b) accumulating daily
  snapshots so a future retrain can learn an injury coefficient.
* ``game_status`` is free text and must be parsed:
  "Out for the season" / "Expected to be out until at least Apr 28" /
  "Game Time Decision".

What this module does with a snapshot:
* parses statuses into an availability weight (0 = plays, 1 = out,
  0.5 = game-time decision -- an explicit, documented assumption);
* maps FantasyNerds team abbreviations + player names onto this repo's
  teams/players and current form snapshot;
* computes each team's *missing production* (status-weighted prior
  minutes/points/rebounds/assists) for display in game predictions;
* gates player-prop projections (OUT -> 0 expected output,
  GTD -> expected value scaled by availability).

It deliberately does **not** numerically adjust game win probabilities:
there is no historical injury dataset here to learn that coefficient
from, and inventing one would be fabrication. Daily snapshots saved by
``scripts/fetch_injuries.py`` are how that dataset gets built.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# FantasyNerds uses a few non-standard abbreviations.
TEAM_ABBREV = {
    "ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GS": "Golden State Warriors", "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets", "IND": "Indiana Pacers", "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NO": "New Orleans Pelicans", "NOP": "New Orleans Pelicans",
    "NY": "New York Knicks", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SA": "San Antonio Spurs", "SAS": "San Antonio Spurs",
    "SAC": "Sacramento Kings", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "UTAH": "Utah Jazz", "WAS": "Washington Wizards",
}

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_UNTIL_RE = re.compile(r"out until at least ([A-Z][a-z]{2}) (\d{1,2})", re.IGNORECASE)


def parse_status(game_status: str, game_date: dt.date | None = None,
                 snapshot_year: int | None = None) -> tuple[str, float]:
    """Return (status_class, availability_loss_weight in [0, 1])."""
    s = (game_status or "").strip()
    low = s.lower()
    if "out for the season" in low:
        return "OUT_SEASON", 1.0
    m = _UNTIL_RE.search(s)
    if m:
        if game_date is not None:
            year = game_date.year
            # 'out until Apr 28' for a game in Oct-Dec means next calendar year
            ret = dt.date(year, _MONTHS[m.group(1).title()], int(m.group(2)))
            if ret < game_date - dt.timedelta(days=120):
                ret = dt.date(year + 1, ret.month, ret.day)
            return ("OUT", 1.0) if game_date < ret else ("AVAILABLE", 0.0)
        return "OUT", 1.0
    if "game time decision" in low or "game-time decision" in low:
        return "GTD", 0.5
    if low.startswith("out"):
        return "OUT", 1.0
    return "OTHER", 0.5  # unknown free text: treat cautiously as GTD-like


def load_snapshot(path: str | Path) -> tuple[pd.DataFrame, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for abbrev, players in raw.get("teams", {}).items():
        for p in players:
            rows.append({
                "team_abbrev": abbrev,
                "team_name": TEAM_ABBREV.get(abbrev, abbrev),
                "player_name": p.get("name", ""),
                "position": p.get("position", ""),
                "injury": p.get("injury", ""),
                "game_status": p.get("game_status", ""),
                "last_update": p.get("last_update", ""),
            })
    df = pd.DataFrame(rows)
    if len(df):
        parsed = df.game_status.map(lambda s: parse_status(s))
        df["status_class"] = [p[0] for p in parsed]
        df["weight"] = [p[1] for p in parsed]
        df["last_update"] = pd.to_datetime(df.last_update, errors="coerce")
    meta = {"season_field": raw.get("season"),
            "n_teams": int(df.team_abbrev.nunique()) if len(df) else 0,
            "n_players": int(len(df)),
            "max_last_update": (str(df.last_update.max().date())
                                if len(df) and df.last_update.notna().any() else None)}
    return df, meta


def freshness(meta: dict, game_date: dt.date | None = None) -> dict:
    ref = game_date or dt.date.today()
    max_up = meta.get("max_last_update")
    if not max_up:
        return {"stale": True, "reason": "snapshot has no parseable last_update dates"}
    age = (ref - dt.date.fromisoformat(max_up)).days
    return {"stale": age > 7, "max_last_update": max_up, "age_days_vs_reference": int(age),
            "reason": ("snapshot is fresh" if age <= 7 else
                       f"snapshot is {age} days old vs {ref} -- the public TEST key returns a "
                       "frozen 2021 demo; a real FantasyNerds API key is required for live data")}


def _norm(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(".", "", regex=False).str.strip()


def team_injury_load(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Status-weighted missing production per team, from current player form."""
    form = pd.read_parquet(PROC / "player_form.parquet")
    form["full"] = _norm(form.firstName.fillna("") + " " + form.lastName.fillna(""))
    snap = snapshot.copy()
    snap["full"] = _norm(snap.player_name)
    merged = snap.merge(form[["full", "minutes_l10", "points_l10", "rebounds_l10", "assists_l10"]],
                        on="full", how="left")
    matched = merged.minutes_l10.notna()
    for c in ["minutes_l10", "points_l10", "rebounds_l10", "assists_l10"]:
        merged[f"miss_{c}"] = merged.weight * merged[c].fillna(0)
    load = (merged.groupby("team_name")
            .agg(players_listed=("player_name", "count"),
                 players_matched_to_form=("minutes_l10", lambda s: int(s.notna().sum())),
                 weighted_missing_minutes=("miss_minutes_l10", "sum"),
                 weighted_missing_points=("miss_points_l10", "sum"),
                 weighted_missing_rebounds=("miss_rebounds_l10", "sum"),
                 weighted_missing_assists=("miss_assists_l10", "sum"))
            .reset_index())
    load.attrs["player_match_rate"] = float(matched.mean()) if len(merged) else 0.0
    return load.sort_values("weighted_missing_minutes", ascending=False)


def player_injury(snapshot: pd.DataFrame, player_full_name: str) -> dict | None:
    snap = snapshot.copy()
    snap["full"] = _norm(snap.player_name)
    hits = snap[snap.full == _norm(pd.Series([player_full_name])).iloc[0]]
    if not len(hits):
        hits = snap[snap.full.str.contains(_norm(pd.Series([player_full_name])).iloc[0], regex=False)]
    if not len(hits):
        return None
    r = hits.iloc[0]
    return {"player": r.player_name, "team": r.team_name, "injury": r.injury,
            "game_status": r.game_status, "status_class": r.status_class,
            "availability_loss_weight": float(r.weight),
            "availability_factor": float(1 - r.weight),
            "last_update": (str(r.last_update.date()) if pd.notna(r.last_update) else None)}
