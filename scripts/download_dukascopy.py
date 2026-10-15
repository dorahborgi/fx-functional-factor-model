"""
Dukascopy historical tick downloader.

Downloads hourly tick files from the public Dukascopy datafeed, decodes them,
resamples the mid-price into 5-minute OHLCV bars (UTC, labelled by bar start)
and saves one Parquet file per pair:

    data/raw/dukascopy/<PAIR>_<start>_<end>.parquet

Used for the pairs / periods not covered by HistData (USD/CNH, and the
2008-2010 history of USD/MXN, USD/TRY and ZAR/JPY).

Example
-------
    python scripts/download_dukascopy.py --pairs USDCNH --start 2014-01-01 --end 2026-01-31
"""

import argparse
import asyncio
import logging
import lzma
import struct
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm

REPO_ROOT   = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = REPO_ROOT / "data" / "raw" / "dukascopy"
TIMEFRAME   = "5min"     # pandas resampling frequency
MAX_RETRIES = 6          # attempts per hour
CONCURRENCY = 4          # default number of simultaneous requests (the server rate-limits)
TIMEOUT_SEC = 15

BASE_URL = ("https://datafeed.dukascopy.com/datafeed/{inst}/{year}/{month:02d}/"
            "{day:02d}/{hour:02d}h_ticks.bi5")

# Price scale of the .bi5 integer prices (JPY-quoted and some EM pairs use 1e3)
POINT_SCALE = {"JPY": 1e3}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def iter_hours(start: datetime, end: datetime):
    """Yield every hourly timestamp between start and end (inclusive)."""
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur <= end:
        yield cur
        cur += timedelta(hours=1)


def parse_bi5(raw_bytes: bytes, dt: datetime, scale: float) -> pd.DataFrame | None:
    """
    Decode a .bi5 file (LZMA-compressed binary records).
    Dukascopy format: 5 × 4 bytes per tick (big-endian)
        [0]  ms_offset : milliseconds since the start of the hour (uint32)
        [1]  ask       : integer ask price × scale                (uint32)
        [2]  bid       : integer bid price × scale                (uint32)
        [3]  ask_vol   : ask volume                               (float32)
        [4]  bid_vol   : bid volume                               (float32)
    """
    try:
        raw = lzma.decompress(raw_bytes)
    except lzma.LZMAError:
        return None

    n_ticks = len(raw) // 20
    if n_ticks == 0:
        return None

    records = []
    for i in range(n_ticks):
        ms, ask_i, bid_i, ask_v, bid_v = struct.unpack(">IIIff", raw[i * 20: i * 20 + 20])
        ts  = dt + timedelta(milliseconds=int(ms))
        ask = ask_i / scale
        bid = bid_i / scale
        records.append((ts, ask, bid, (ask + bid) / 2, float(ask_v), float(bid_v)))

    return pd.DataFrame(records, columns=["time", "ask", "bid", "mid", "ask_vol", "bid_vol"])


def ticks_to_bars(df_ticks: pd.DataFrame) -> pd.DataFrame:
    """Resample ticks into 5-minute OHLCV bars of the mid-price (plus mean ask / bid)."""
    df = df_ticks.set_index("time").sort_index()

    mid = df["mid"].resample(TIMEFRAME)
    ask = df["ask"].resample(TIMEFRAME)
    bid = df["bid"].resample(TIMEFRAME)
    vol = (df["ask_vol"] + df["bid_vol"]).resample(TIMEFRAME)

    bars = pd.DataFrame({
        "open":     mid.first(),
        "high":     mid.max(),
        "low":      mid.min(),
        "close":    mid.last(),
        "volume":   vol.sum(),
        "ask_mean": ask.mean(),
        "bid_mean": bid.mean(),
    }).dropna(subset=["open"])

    return bars


# ─────────────────────────────────────────────
# ASYNCHRONOUS DOWNLOAD
# ─────────────────────────────────────────────

async def fetch_hour(session, semaphore, instrument: str, dt: datetime, scale: float):
    """Download and decode one hour of ticks, with retries."""
    url = BASE_URL.format(
        inst=instrument,
        year=dt.year,
        month=dt.month - 1,   # Dukascopy months are 0-indexed
        day=dt.day,
        hour=dt.hour,
    )

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC)) as resp:
                    if resp.status == 404:
                        return None          # no data for this hour (weekend, holiday…)
                    if resp.status == 429:   # rate-limited: wait longer and retry
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if resp.status != 200:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    content = await resp.read()
                    if len(content) == 0:
                        return None
                    return parse_bi5(content, dt, scale)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES:
                    log.warning(f"Failed {instrument} {dt} after {MAX_RETRIES} attempts: {e}")
                    return None
                await asyncio.sleep(1.5 * attempt)   # light exponential back-off

    return None


async def download_pair(pair: str, start: datetime, end: datetime, session, semaphore) -> None:
    """Download the whole period for one pair and save it as Parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pair}_{start.date()}_{end.date()}.parquet"

    if out_path.exists():
        log.info(f"[{pair}] Already downloaded → {out_path.name}. Skipped.")
        return

    scale = POINT_SCALE.get(pair[-3:], 1e5)
    hours = list(iter_hours(start, end))
    log.info(f"[{pair}] {len(hours):,} hours to download…")

    results = await tqdm.gather(
        *[fetch_hour(session, semaphore, pair, dt, scale) for dt in hours],
        desc=f"{pair:8s}", unit="h",
    )
    tick_frames = [r for r in results if r is not None]

    if not tick_frames:
        log.warning(f"[{pair}] No data retrieved!")
        return

    all_ticks = pd.concat(tick_frames, ignore_index=True).sort_values("time")
    bars = ticks_to_bars(all_ticks)
    bars.to_parquet(out_path, index=True, engine="pyarrow", compression="snappy")
    log.info(f"[{pair}] ✓ {len(bars):,} bars saved → {out_path.name} "
             f"({out_path.stat().st_size / 1e6:.1f} MB)")


async def main(pairs, start, end, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)
    headers   = {"User-Agent": "Mozilla/5.0 (compatible; FX-research-downloader/1.0)"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        await asyncio.gather(*[download_pair(p, start, end, session, semaphore) for p in pairs])

    log.info(f"Done. Files in: {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", nargs="+", default=["USDCNH"],
                        help="Dukascopy instrument codes, e.g. USDCNH USDMXN (default: USDCNH)")
    parser.add_argument("--start", default="2014-01-01", help="first day, YYYY-MM-DD (UTC)")
    parser.add_argument("--end", default="2026-01-31", help="last day, YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"simultaneous requests (default {CONCURRENCY})")
    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    start = datetime.fromisoformat(args.start)
    end   = datetime.fromisoformat(args.end).replace(hour=23)
    asyncio.run(main([p.upper() for p in args.pairs], start, end, args.concurrency))
