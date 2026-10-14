#!/usr/bin/env python3
"""Prediction CLI.

Examples:
    .venv/bin/python scripts/predict.py game --home "Boston Celtics" --away "New York Knicks"
    .venv/bin/python scripts/predict.py game --home Celtics --away Knicks --date 2026-10-25 \
        --spread -4.5 --total 224.5 --home-ml -180 --away-ml 155
    .venv/bin/python scripts/predict.py player --name "Jayson Tatum"
    .venv/bin/python scripts/predict.py player --name "Nikola Jokic" --away
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predict.predict import predict_game, predict_player


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("game", help="predict a matchup")
    g.add_argument("--home", required=True)
    g.add_argument("--away", required=True)
    g.add_argument("--date", type=dt.date.fromisoformat, default=None,
                   help="YYYY-MM-DD (default: today)")
    g.add_argument("--spread", type=float, default=None, help="home spread, e.g. -4.5")
    g.add_argument("--total", type=float, default=None, help="market over/under, e.g. 224.5")
    g.add_argument("--home-ml", type=float, default=None)
    g.add_argument("--away-ml", type=float, default=None)

    p = sub.add_parser("player", help="project a player's pts/reb/ast")
    p.add_argument("--name", required=True)
    p.add_argument("--date", type=dt.date.fromisoformat, default=None)
    loc = p.add_mutually_exclusive_group()
    loc.add_argument("--home", action="store_true", default=True)
    loc.add_argument("--away", action="store_false", dest="home")

    args = ap.parse_args()
    try:
        if args.cmd == "game":
            out = predict_game(args.home, args.away, args.date, args.spread,
                               args.total, args.home_ml, args.away_ml)
        else:
            out = predict_player(args.name, home=args.home, game_date=args.date)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
