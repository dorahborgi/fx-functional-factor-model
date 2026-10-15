"""
Download HistData.com 1-minute bars (MetaTrader format) for a list of FX pairs.

Files are saved as the original ZIP archives:

    data/raw/histdata/<PAIR>/DAT_MT_<PAIR>_M1_<YYYY or YYYYMM>.zip

HistData serves full years for past years and one file per month for the
current year; both cases are handled. Timestamps are EST without daylight
saving (fixed UTC−5), as documented by HistData.

Example
-------
    python scripts/download_histdata.py                       # all 44 pairs, 2014-2026
    python scripts/download_histdata.py --pairs eurusd usdjpy --start-year 2020
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from histdata import download_hist_data
from histdata.api import Platform, TimeFrame

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from data_processing import FX_PAIRS  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "histdata"

# Pairs not available on HistData: download them with scripts/download_dukascopy.py
NOT_ON_HISTDATA = {"usdcnh"}


def download_pair(pair: str, start_year: int, end_year: int, end_month: int, pause: float) -> None:
    out_dir = RAW_DIR / pair.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    current_year = datetime.now().year

    for year in range(start_year, end_year + 1):
        # Past years: one yearly archive. Current year: monthly archives only.
        periods = [None] if year < current_year else range(1, (end_month if year == end_year else 12) + 1)
        for month in periods:
            tag = f"{year}" if month is None else f"{year}{month:02d}"
            if any(out_dir.glob(f"*_{pair.upper()}_M1_{tag}.zip")):
                continue
            try:
                download_hist_data(
                    year=str(year),
                    month=month,
                    pair=pair,
                    platform=Platform.META_TRADER,
                    time_frame=TimeFrame.ONE_MINUTE,
                    output_directory=str(out_dir),
                    verbose=False,
                )
                print(f"  ✓ {pair.upper()} {tag}")
            except AssertionError as err:          # no file for this pair / period
                print(f"  – {pair.upper()} {tag}: not available ({err})")
            time.sleep(pause)                      # be gentle with the server


def parse_args():
    now = datetime.now()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", nargs="+", default=FX_PAIRS,
                        help="pairs in lower case, e.g. eurusd usdjpy (default: the 44 panel pairs)")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=now.year)
    parser.add_argument("--end-month", type=int, default=now.month,
                        help="last month to download in the current year")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for pair in [p.lower() for p in args.pairs]:
        if pair in NOT_ON_HISTDATA:
            print(f"{pair.upper()}: not on HistData — use scripts/download_dukascopy.py")
            continue
        print(f"{pair.upper()}")
        download_pair(pair, args.start_year, args.end_year, args.end_month, args.pause)
    print(f"Done. Raw files in: {RAW_DIR}")
