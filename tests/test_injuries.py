"""FantasyNerds injuries parsing tests (offline, fixture = real TEST response excerpt).

Run:  .venv/bin/python -m unittest discover -s tests -v
"""
import datetime as dt
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predict.injuries import (freshness, load_snapshot, parse_status,  # noqa: E402
                                  player_injury)

FIXTURE = Path(__file__).parent / "fixtures" / "fantasynerds_test_snapshot.json"


class TestParseStatus(unittest.TestCase):
    def test_out_for_season(self):
        self.assertEqual(parse_status("Out for the season"), ("OUT_SEASON", 1.0))

    def test_out_until_after_game(self):
        cls, w = parse_status("Expected to be out until at least May 2",
                              game_date=dt.date(2021, 4, 20))
        self.assertEqual((cls, w), ("OUT", 1.0))

    def test_out_until_before_game(self):
        cls, w = parse_status("Expected to be out until at least Apr 22",
                              game_date=dt.date(2021, 5, 1))
        self.assertEqual((cls, w), ("AVAILABLE", 0.0))

    def test_gtd(self):
        self.assertEqual(parse_status("Game Time Decision"), ("GTD", 0.5))


class TestSnapshot(unittest.TestCase):
    def test_load_and_freshness_flags_test_demo_as_stale(self):
        df, meta = load_snapshot(FIXTURE)
        self.assertEqual(meta["season_field"], 2021)
        self.assertEqual(meta["n_players"], 6)
        self.assertTrue(freshness(meta, dt.date(2026, 10, 13))["stale"])

    def test_player_lookup(self):
        df, _ = load_snapshot(FIXTURE)
        info = player_injury(df, "Giannis Antetokounmpo")
        self.assertIsNotNone(info)
        self.assertEqual(info["status_class"], "GTD")
        self.assertAlmostEqual(info["availability_factor"], 0.5)
        self.assertIsNone(player_injury(df, "Nikola Jokic"))


if __name__ == "__main__":
    unittest.main()
