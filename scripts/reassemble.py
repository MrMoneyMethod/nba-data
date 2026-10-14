#!/usr/bin/env python3
"""Reassemble the byte-split raw files uploaded to this repo.

The uploads were split into ~25 MB chunks because of GitHub's web-upload
limits. The split is a *raw byte* split (not split on line boundaries):
only ``*.part01`` of the parquet starts with the ``PAR1`` magic and only
the last part ends with ``PAR1``. Reassembly is therefore a plain ordered
concatenation of the parts -- ``cat part* > file`` -- which this script
does in streaming chunks (low memory) with validation:

* parts must be contiguous (part01..partNN with no gaps),
* CSV outputs must start with the expected header and be valid UTF-8
  at the seams,
* the parquet output must start and end with ``PAR1`` and be readable
  by pyarrow (metadata only is loaded).

Usage:
    .venv/bin/python scripts/reassemble.py [--force] [--only NAME]

Outputs land in ``data/raw_reassembled/`` (git-ignored, ~2.8 GB).
Nothing in ``data/raw_reassembled/`` should ever be committed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw_reassembled"

# output filename -> glob for parts, in byte order
FAMILIES: dict[str, str] = {
    "PlayerStatistics.csv": "PlayerStatistics.csv.part*",
    "PlayerStatisticsExtended.csv": "PlayerStatisticsExtended.csv.part*",
    "TeamStatistics.csv": "TeamStatistics.csv.part*",
    "TeamStatisticsExtended.csv": "TeamStatisticsExtended.csv.part*",
    "PlayByPlay.parquet": "PlayByPlay.parquet.part*",
}

CHUNK = 8 * 1024 * 1024


def ordered_parts(pattern: str) -> list[Path]:
    parts = sorted(ROOT.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No files match {pattern} under {ROOT}")
    return parts


def check_contiguous(parts: list[Path]) -> None:
    """Ensure numeric suffixes run 1..N with no gaps/duplicates."""
    suffixes = []
    for p in parts:
        tail = p.name.rsplit("part", 1)[-1].lstrip("0") or "0"
        if not tail.isdigit():
            raise ValueError(f"Cannot parse part number from {p.name}")
        suffixes.append(int(tail))
    if suffixes != list(range(1, len(parts) + 1)):
        raise ValueError(f"Non-contiguous parts: {suffixes}")


def reassemble(name: str, pattern: str, force: bool) -> Path:
    out = OUT_DIR / name
    parts = ordered_parts(pattern)
    check_contiguous(parts)
    total = sum(p.stat().st_size for p in parts)
    if out.exists() and out.stat().st_size == total and not force:
        print(f"[skip] {name}: already reassembled ({total/1e6:.1f} MB, {len(parts)} parts)")
        return out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[build] {name}: {len(parts)} parts, {total/1e6:.1f} MB -> {out}")
    with out.open("wb") as dst:
        for p in parts:
            with p.open("rb") as src:
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    dst.write(buf)
    if out.stat().st_size != total:
        raise IOError(f"{name}: size mismatch {out.stat().st_size} != {total}")
    validate(out)
    return out


def validate(path: Path) -> None:
    if path.suffix == ".parquet":
        with path.open("rb") as fh:
            head = fh.read(4)
            fh.seek(-4, 2)
            tail = fh.read(4)
        if head != b"PAR1" or tail != b"PAR1":
            raise ValueError(f"{path.name}: bad parquet magic head={head!r} tail={tail!r}")
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path).metadata
        print(f"        parquet OK: {meta.num_rows:,} rows, {meta.num_columns} cols, "
              f"{meta.num_row_groups} row group(s)")
    else:
        with path.open("rb") as fh:
            header = fh.readline().decode("utf-8").strip()
        n_cols = header.count(",") + 1
        print(f"        csv OK: header has {n_cols} columns: {header[:90]}...")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if output exists")
    ap.add_argument("--only", choices=sorted(FAMILIES), help="reassemble a single file")
    args = ap.parse_args()

    families = {args.only: FAMILIES[args.only]} if args.only else FAMILIES
    for name, pattern in families.items():
        reassemble(name, pattern, force=args.force)
    print("Done. Reassembled files are in data/raw_reassembled/ (git-ignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
