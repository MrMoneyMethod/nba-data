#!/usr/bin/env python3
"""Build leakage-safe game + player feature tables.

    .venv/bin/python scripts/build_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predict.features import build_game_features, build_player_features

if __name__ == "__main__":
    build_game_features()
    build_player_features()
