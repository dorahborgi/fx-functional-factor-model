"""
Raw FX data → (trading day × 5-min) log-return matrices.

Pipeline (see ``scripts/build_dataset.py``):

    load_histdata_m1 / load_dukascopy_bars   1-min (HistData) or 5-min (Dukascopy) close prices
        → combine_sources                     HistData first, Dukascopy only on days HistData lacks
        → resample_5min                       5-min grid, linear interpolation of gaps ≤ 15 min
        → log_returns_5min                    r_t = log P_t − log P_{t−1}
        → trading_day_matrix                  (trading day × 288) matrix, day = 17:00 → 16:55

Time convention: HistData timestamps are in EST *without* daylight saving
(fixed UTC−5). Dukascopy bars (UTC) are converted to the same clock, so a
trading day always runs from 17:00 to 16:55 on this fixed clock.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# HistData clock: Eastern Standard Time, no DST shift (POSIX sign convention: Etc/GMT+5 = UTC−5)
HISTDATA_TZ = "Etc/GMT+5"

# Start of the FX trading day on the HistData clock
TRADING_DAY_START_HOUR = 17

# The 44 currency pairs of the final panel (EUR/HUF and EUR/TRY were dropped
# for insufficient coverage; pegged pairs such as USD/HKD were never included).
FX_PAIRS = [
    'audcad', 'audchf', 'audjpy', 'audnzd', 'audusd', 'cadchf', 'cadjpy', 'chfjpy',
    'euraud', 'eurcad', 'eurchf', 'eurczk', 'eurgbp', 'eurjpy', 'eurnok', 'eurnzd',
    'eurpln', 'eursek', 'eurusd', 'gbpaud', 'gbpcad', 'gbpchf', 'gbpjpy', 'gbpnzd',
    'gbpusd', 'nzdcad', 'nzdchf', 'nzdjpy', 'nzdusd', 'sgdjpy', 'usdcad', 'usdchf',
    'usdcnh', 'usdczk', 'usdhuf', 'usdjpy', 'usdmxn', 'usdnok', 'usdpln', 'usdsek',
    'usdsgd', 'usdtry', 'usdzar', 'zarjpy',
]


# =============================================================================
# LOADING RAW DATA
# =============================================================================

def _read_metatrader_csv(source) -> pd.DataFrame:
    """Read one HistData MetaTrader M1 file: 'YYYY.MM.DD,HH:MM,open,high,low,close,volume'."""
    raw = pd.read_csv(
        source,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        usecols=["date", "time", "close"],
        dtype={"date": str, "time": str, "close": "float64"},
    )
    index = pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y.%m.%d %H:%M")
    return pd.DataFrame({"close": raw["close"].to_numpy()}, index=index)


def load_histdata_m1(folder) -> pd.DataFrame:
    """
    Load all HistData MetaTrader M1 files (``.zip`` or extracted ``.csv``) of one pair.

    Parameters
    ----------
    folder : path to ``data/raw/histdata/<PAIR>/``

    Returns
    -------
    DataFrame with a single column ``close``, indexed by minute timestamps on the
    HistData clock (EST, no DST), sorted, without duplicated timestamps.
    """
    folder = Path(folder)
    frames = []
    for path in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    with zf.open(name) as fh:
                        frames.append(_read_metatrader_csv(fh))
    for path in sorted(folder.glob("*.csv")):
        frames.append(_read_metatrader_csv(path))

    if not frames:
        return pd.DataFrame(columns=["close"], index=pd.DatetimeIndex([], name="datetime"))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "datetime"
    return df


def load_dukascopy_bars(paths, tz: str = HISTDATA_TZ) -> pd.DataFrame:
    """
    Load 5-min Dukascopy bars written by ``scripts/download_dukascopy.py``.

    Dukascopy timestamps are UTC and label the *start* of each bar. They are
    shifted by +5 min (bar end, consistent with ``resample_5min``) and converted
    to the HistData clock.

    Parameters
    ----------
    paths : path or list of paths to ``<PAIR>_<start>_<end>.parquet`` files
    tz    : target clock (default: HistData EST without DST)
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [pd.read_parquet(p, columns=["close"]) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["close"], index=pd.DatetimeIndex([], name="datetime"))

    df = pd.concat(frames).sort_index()
    idx = pd.DatetimeIndex(df.index) + pd.Timedelta(minutes=5)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(tz).tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "datetime"
    return df[["close"]].astype("float64")


def combine_sources(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two price sources day by day: ``primary`` is used whenever it has data
    on a calendar day, ``secondary`` only fills the days that ``primary`` lacks.
    This avoids mixing two feeds within the same day.
    """
    if secondary is None or secondary.empty:
        return primary
    if primary is None or primary.empty:
        return secondary
    primary_days = primary.index.normalize().unique()
    fill = secondary[~secondary.index.normalize().isin(primary_days)]
    return pd.concat([primary, fill]).sort_index()


# =============================================================================
# 5-MIN GRID, LOG-RETURNS, TRADING-DAY MATRIX
# =============================================================================

def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute data to 5-min bars with linear interpolation for gaps <= 15 min.
    Input: DataFrame with a 'close' column and a datetime index.
    """
    close_5 = (
        df["close"]
        .resample("5min", closed="right", label="right")
        .last()
    )

    full_grid = pd.date_range(
        start=close_5.index.min(),
        end=close_5.index.max(),
        freq="5min",
    )
    close_5 = close_5.reindex(full_grid)

    is_nan    = close_5.isna()
    group_ids = (is_nan != is_nan.shift()).cumsum()
    gap_size  = is_nan.groupby(group_ids).transform("sum")

    interpolable = is_nan & (gap_size <= 3)

    close_interp = close_5.interpolate(method="time")
    close_5[interpolable] = close_interp[interpolable]
    close_5[is_nan & ~interpolable] = np.nan

    result = pd.DataFrame(index=close_5.index)
    result.index.name = "datetime"
    result["day"]   = result.index.date.astype(str)
    result["time"]  = result.index.strftime("%H:%M")
    result["close"] = close_5.values

    return result


def log_returns_5min(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-min log-returns from resample_5min() output."""
    log_price = np.log(df_5min["close"])
    lr = log_price.diff()
    lr.name = "log_return"

    result = df_5min[["day", "time"]].copy()
    result["close"]      = df_5min["close"]
    result["log_return"] = lr

    return result


def trading_day_matrix(df_lr: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (trading_day x intraday_time) matrix of 5-min log-returns.
    FX trading day convention: 17:00 → 17:00 (next calendar day cutoff).
    Columns ordered: 17:00, 17:05, ..., 23:55, 00:00, ..., 16:55 (288 cols).
    """
    df = df_lr[["log_return"]].copy()

    # A timestamp at or after 17:00 belongs to the next calendar day's session:
    # shifting by (24 - 17) hours maps 17:00 to midnight of the next day.
    shift = pd.Timedelta(hours=24 - TRADING_DAY_START_HOUR)
    df["trading_day"]   = (df.index + shift).strftime("%Y-%m-%d")
    df["intraday_time"] = df.index.strftime("%H:%M")

    matrix = df.pivot(index="trading_day", columns="intraday_time", values="log_return")

    all_times = pd.date_range(f"2000-01-01 {TRADING_DAY_START_HOUR:02d}:00", periods=288, freq="5min")
    col_order = all_times.strftime("%H:%M").tolist()
    col_order = [c for c in col_order if c in matrix.columns]
    matrix = matrix[col_order]

    matrix.index = matrix.index.astype(str)

    return matrix


def panel_to_dict(X: np.ndarray,
                  valid_days: pd.DatetimeIndex,
                  pairs: list,
                  col_names: list = None) -> dict:
    """
    Rebuild a dict {pair: DataFrame(T x M)} from a 3D panel array.

    Parameters
    ----------
    X         : np.ndarray (N, T, M)
    valid_days: pd.DatetimeIndex (T,)
    pairs     : list[str] (N,)
    col_names : list[str] (M,) — column names; defaults to 0..M-1
    """
    N, T, M = X.shape

    index = pd.DatetimeIndex(valid_days).normalize()
    columns = col_names if col_names is not None else list(range(M))

    return {
        pair: pd.DataFrame(X[i], index=index, columns=columns)
        for i, pair in enumerate(pairs)
    }
