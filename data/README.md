# Data

Raw and processed data are **not** versioned (several GB, and the providers' terms
do not allow redistribution). Everything in this folder is rebuilt from the public
sources with the scripts in [`scripts/`](../scripts).

## Sources

| Source | Content | Pairs | Clock |
|---|---|---|---|
| [HistData.com](https://www.histdata.com) | 1-minute bars, MetaTrader format | 43 of the 44 pairs | EST, **no** daylight saving (UTC−5 all year) |
| [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | ticks → 5-minute mid-price bars | USD/CNH (not on HistData); optionally early history of USD/MXN, USD/TRY, ZAR/JPY | UTC, converted to the HistData clock |

## Reproducing the dataset

```bash
# 1. HistData 1-min bars for the 44 panel pairs (USD/CNH is skipped automatically)
python scripts/download_histdata.py --start-year 2014

# 2. USD/CNH from Dukascopy
python scripts/download_dukascopy.py --pairs USDCNH --start 2014-01-01 --end 2026-01-31

# 3. Clean, resample and build the (trading day × 5-min) matrices
python scripts/build_dataset.py
```

Downloading every pair takes a while (HistData serves one archive per pair and year).
For a quick test, restrict the pairs, e.g. `--pairs eurusd usdjpy gbpusd`.

## Cleaning and construction

Implemented in [`src/data_processing.py`](../src/data_processing.py):

1. **Load & merge** — all HistData archives of a pair are concatenated, sorted and
   de-duplicated on the timestamp. Dukascopy bars (UTC, labelled by bar start) are shifted to
   the bar end and converted to the HistData clock; they are only used on calendar days that
   HistData does not cover, so two feeds are never mixed within a day.
2. **5-minute grid** — last price in each 5-minute interval `(t−5min, t]` on a complete grid.
   Gaps of up to 3 missing bars (≤ 15 min) are linearly interpolated; longer gaps stay `NaN`.
3. **Log-returns** — `r_t = log P_t − log P_{t−1}`.
4. **Trading-day matrix** — a trading day runs from **17:00 to 16:55** (New York close
   convention on the HistData clock), i.e. **M = 288** intraday points per day.
5. **Downstream (notebooks)** — weekends and 1 January are removed, the volatility proxy
   `log(r² + ε_i)` is computed (`ε_i` = 5th percentile of the positive `r²` of pair `i`),
   each daily curve is smoothed with penalised cubic B-splines (20 basis functions, `λ_i` by GCV),
   and only the days on which **every** pair has at least 70% of its intraday observations are kept.

## Pair selection

48 candidate pairs were considered. Pegged pairs (EUR/DKK, USD/HKD — near-zero volatility)
and pairs with insufficient coverage (EUR/HUF, EUR/TRY) were dropped, leaving **N = 44 pairs**
and **T = 3,062 trading days** over 2014–2025. The list is `FX_PAIRS` in `src/data_processing.py`.

## Layout after the build

```
data/
├── raw/
│   ├── histdata/<PAIR>/DAT_MT_<PAIR>_M1_<YYYY>.zip      # one archive per year (per month for the current year)
│   └── dukascopy/<PAIR>_<start>_<end>.parquet
├── <pair>_matrix_lg.parquet              # in-sample 5-min log-returns (trading day × 288)
├── 2026/<pair>_2026_matrix_lg.parquet    # out-of-sample (from 2026-01-01), used for forecasting
├── shared/                               # objects cached by notebook 01 for notebooks 02–04
└── results/                              # rolling-window outputs (Grassmann distances, simulations…)
```
