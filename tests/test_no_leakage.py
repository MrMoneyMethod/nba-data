"""Leakage guard tests (stdlib unittest, no extra deps).

Run:  .venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predict.features import ROLL_STATS, _add_derived, _prior_rolling  # noqa: E402


def _synthetic_team_games() -> pd.DataFrame:
    rows = []
    gid = 1
    for team, base in [(1, 100), (2, 90)]:
        for i in range(12):
            rows.append({
                "gameId": gid, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=2 * i),
                "teamId": team, "oppId": 3 - team, "season": 2023,
                "home": 1, "win": 1, "teamScore": base + i, "opponentScore": 80,
                "assists": 20, "blocks": 5, "steals": 5,
                "fieldGoalsAttempted": 80, "fieldGoalsMade": 40,
                "threePointersAttempted": 30, "threePointersMade": 10,
                "freeThrowsAttempted": 20, "freeThrowsMade": 15,
                "reboundsOffensive": 10, "reboundsDefensive": 30, "reboundsTotal": 40,
                "turnovers": 12, "turnoversTeam": 0, "foulsPersonal": 18,
                "numMinutes": 240, "gameType": "Regular Season",
            })
            gid += 1
    return _add_derived(pd.DataFrame(rows))


class TestPriorOnlyRolling(unittest.TestCase):
    def test_first_game_has_no_form(self):
        t = _prior_rolling(_synthetic_team_games())
        first = t.sort_values(["teamId", "date"]).groupby("teamId").head(1)
        self.assertTrue(first["pts_l5"].isna().all(), "first game must have no prior form")

    def test_rolling_uses_only_previous_games(self):
        t = _prior_rolling(_synthetic_team_games())
        t1 = t[t.teamId == 1].sort_values("date").reset_index(drop=True)
        # game index 5 (0-based): prior 5 scores are 100..104 -> mean 102
        self.assertAlmostEqual(t1.loc[5, "pts_l5"], 102.0)
        # mutating the current game's score must not change its own features
        raw = _synthetic_team_games()
        raw.loc[(raw.teamId == 1) & (raw.gameId == t1.loc[5, "gameId"]), "teamScore"] = 9999
        t2 = _prior_rolling(raw)
        t2 = t2[t2.teamId == 1].sort_values("date").reset_index(drop=True)
        self.assertAlmostEqual(t2.loc[5, "pts_l5"], 102.0)

    def test_rest_days_prior_only(self):
        t = _prior_rolling(_synthetic_team_games())
        t1 = t[t.teamId == 1].sort_values("date")
        self.assertTrue((t1["rest_days"].iloc[1:] == 2).all())


class TestProcessedArtefacts(unittest.TestCase):
    """Integration checks -- skipped if features have not been built yet."""

    def setUp(self):
        self.path = Path(__file__).resolve().parents[1] / "data" / "processed" / "game_features.parquet"
        if not self.path.exists():
            self.skipTest("run scripts/build_features.py first")

    def test_no_future_elo_and_targets_sane(self):
        df = pd.read_parquet(self.path, columns=["date", "home_win", "margin", "total", "elo_diff"])
        self.assertTrue(df.elo_diff.notna().all())
        self.assertTrue(set(df.home_win.unique()) <= {0, 1})
        self.assertTrue((df.total > 0).all())
        self.assertTrue(df.date.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
