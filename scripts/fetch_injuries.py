#!/usr/bin/env python3
"""Fetch a FantasyNerds NBA injuries snapshot and validate it.

    export FANTASYNERDS_API_KEY=<your key>     # 'TEST' = frozen 2021 demo data
    .venv/bin/python scripts/fetch_injuries.py [--apikey KEY] [--out DIR]

Saves the raw JSON to data/injuries/fantasynerds_YYYY-MM-DD.json (git-ignored)
so snapshots accumulate into a historical dataset that a future retrain can
learn an injury coefficient from. There is no historical endpoint -- today's
snapshot is only useful if it is actually saved today.

Sandbox note: api.fantasynerds.com is not on this sandbox's allowlist, so
direct fetching fails here ('ConnectionError'); the endpoint itself was
verified working on 2026-10-13 via an external fetcher (TEST key returns
season 2021, all last_update 2021-04-13). Run this where the API is reachable.
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

URL = "https://api.fantasynerds.com/v1/nba/injuries"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apikey", default=os.environ.get("FANTASYNERDS_API_KEY", "TEST"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "injuries"))
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        print("requests is not installed in this venv (it is not needed by the core engine). "
              "Install with: .venv/bin/pip install requests", file=sys.stderr)
        return 2
    try:
        resp = requests.get(URL, params={"apikey": args.apikey}, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:  # network blocked, bad key, non-JSON, ...
        print(f"Fetch failed: {exc}\n"
              "If you are in the Arena sandbox: api.fantasynerds.com is not allowlisted there, "
              "so this must run in an environment with normal internet access.", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fantasynerds_{dt.date.today().isoformat()}.json"
    out.write_text(json.dumps(raw), encoding="utf-8")

    from nba_predict.injuries import freshness, load_snapshot

    tmp = out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    df, meta = load_snapshot(tmp)
    tmp.unlink()
    print(json.dumps({"saved": str(out), **meta, "freshness": freshness(meta)}, indent=2))
    if meta["season_field"] is not None and int(meta["season_field"]) < dt.date.today().year - 1:
        print("WARNING: this looks like the frozen TEST-key demo (season "
              f"{meta['season_field']}). Do not use it for predictions.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
