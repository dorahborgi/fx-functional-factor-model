"""
Build the (trading day × 5-min) log-return matrices used by the notebooks.

For each pair:
    1. load HistData 1-min bars        data/raw/histdata/<PAIR>/*.zip
       + Dukascopy 5-min bars          data/raw/dukascopy/<PAIR>_*.parquet   (if any)
       HistData is used on every day it covers; Dukascopy only fills the other days.
    2. resample to a 5-min grid (gaps ≤ 15 min linearly interpolated, longer gaps → NaN)
    3. 5-min log-returns
    4. pivot to a (trading day × 288) matrix, trading day = 17:00 → 16:55 (EST, no DST)
    5. split in-sample / out-of-sample at --oos-start and save

Outputs
-------
    data/<pair>_matrix_lg.parquet               in-sample   (trading days <  --oos-start)
    data/2026/<pair>_2026_matrix_lg.parquet     out-of-sample (trading days ≥ --oos-start)

Example
-------
    python scripts/build_dataset.py                      # all 44 pairs
    python scripts/build_dataset.py --pairs eurusd usdjpy
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from data_processing import (  # noqa: E402
    FX_PAIRS, combine_sources, load_dukascopy_bars, load_histdata_m1,
    log_returns_5min, resample_5min, trading_day_matrix,
)

DATA_DIR = REPO_ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"


def build_pair(pair: str, oos_start: str, start: str | None) -> None:
    code  = pair.upper()
    hist  = load_histdata_m1(RAW_DIR / "histdata" / code)
    duka_files = sorted((RAW_DIR / "dukascopy").glob(f"{code}_*.parquet"))
    duka  = load_dukascopy_bars(duka_files) if duka_files else None
    prices = combine_sources(hist, duka)

    if prices.empty:
        print(f"  ⚠ {code}: no raw data found — skipped")
        return
    if start is not None:
        prices = prices.loc[pd.Timestamp(start) - pd.Timedelta(days=1):]

    matrix = trading_day_matrix(log_returns_5min(resample_5min(prices)))
    matrix.index.name = "trading_day"

    in_sample  = matrix[matrix.index <  oos_start]
    out_sample = matrix[matrix.index >= oos_start]

    in_sample.to_parquet(DATA_DIR / f"{pair}_matrix_lg.parquet", engine="pyarrow")
    if not out_sample.empty:
        (DATA_DIR / "2026").mkdir(parents=True, exist_ok=True)
        out_sample.to_parquet(DATA_DIR / "2026" / f"{pair}_2026_matrix_lg.parquet", engine="pyarrow")

    src = "HistData" + (" + Dukascopy" if duka is not None else "")
    print(f"  ✓ {code:7s} [{src}]  in-sample {in_sample.index.min()} → {in_sample.index.max()} "
          f"({len(in_sample)} days) | out-of-sample {len(out_sample)} days")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", nargs="+", default=FX_PAIRS,
                        help="pairs in lower case (default: the 44 panel pairs)")
    parser.add_argument("--oos-start", default="2026-01-01",
                        help="first out-of-sample trading day, YYYY-MM-DD (default 2026-01-01)")
    parser.add_argument("--start", default=None,
                        help="optional first trading day to keep, YYYY-MM-DD (default: all data)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for pair in [p.lower() for p in args.pairs]:
        build_pair(pair, args.oos_start, args.start)
