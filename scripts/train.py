#!/usr/bin/env python3
"""Train all models + backtest (incl. vs. the 2025-26 market sample).

    .venv/bin/python scripts/reassemble.py      # once
    .venv/bin/python scripts/build_features.py
    .venv/bin/python scripts/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_predict.train import main

if __name__ == "__main__":
    main()
