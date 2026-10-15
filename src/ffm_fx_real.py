"""
=============================================================================
FUNCTIONAL FACTOR MODEL (FFM) — Intraday FX Volatility
Adapted to existing day × minute matrices
=============================================================================
Reference: Hallin, Nisol & Tavakoli (2023), Parts I & II

INPUT DATA:
    - N DataFrame matrices: index = date (day), columns = intraday time (minute)
    - Values = transformed 5-min volatility proxy (e.g. smoothed log(r² + ε))
    - Names: eurusd_vol_matrix, usdjpy_vol_matrix, ...

FFM PANEL STRUCTURE:
    X[i, t, τ] = volatility of pair i, on day t, at intraday interval τ
    shape: (N pairs, T = number of common days, M = number of intervals/day)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import warnings
from paths import fig_dir
warnings.filterwarnings("ignore")

# Output directory for all figures produced by this module
FIGURES_DIR = fig_dir('FFM')


def _fig(name):
    """Return full path for a figure file inside FIGURES_DIR."""
    return os.path.join(FIGURES_DIR, name)


def _format_pair(pair: str) -> str:
    """Convert 'audcad' → 'AUD/CAD', keeping '/' if already present."""
    p = pair.strip().upper().replace('_', '').replace(' ', '')
    if '/' in p:
        return p
    if len(p) == 6:
        return f"{p[:3]}/{p[3:]}"
    return p

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — adjust here if needed
# ─────────────────────────────────────────────────────────────────────────────


# Extended palette for N up to 40 pairs
# Generated from tab20 + tab20b for maximum readability
def _make_colors(n=40):
    import matplotlib.cm as cm
    tab20  = [cm.tab20(i)  for i in np.linspace(0, 1, 20, endpoint=False)]
    tab20b = [cm.tab20b(i) for i in np.linspace(0, 1, 20, endpoint=False)]
    palette = tab20 + tab20b
    return ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
            for r, g, b, _ in palette[:n]]

COLORS = _make_colors(40)


# =============================================================================
# STEP 0: BUILD THE 3D PANEL FROM THE (DAY × TIME) MATRICES
# =============================================================================

def build_panel_from_matrices(vol_matrices: dict, min_coverage: float = 0.8,
                              force_dates: list = None):
    """
    Convert N (day × minute) matrices into a 3D panel (N, T, M).

    MISSING VALUE HANDLING — fully vectorised pipeline:
    ──────────────────────────────────────────────────────────────
    1. Days common to all pairs (index union, normalised to date)
    2. Weekends (Sat + Sun) removed
    3. Numpy stack (N, T_raw, M_raw) built in one pass — fast even
       for N=36, T=4300, M=288
    4. Inactive columns removed: interval dropped if fill rate
       mean (over N pairs × T_raw days) < 10%  →  removes residual night/weekend slots
    5. Days retained: a day is kept if ALL pairs have
       ≥ min_coverage of the active columns non-NaN
    6. Residual NaN imputed by column median
       (median computed per pair over valid days)

    Parameters
    ----------
    vol_matrices : dict {str: DataFrame(days × minutes)}
    min_coverage : float — coverage threshold per day (default 0.8)

    Returns
    --------
    X    : np.ndarray (N, T, M)
    days : pd.DatetimeIndex (T,)
    tau  : np.ndarray (M,) ∈ [0,1]
    pairs: list[str]
    """
    import time
    t0 = time.time()

    pairs = list(vol_matrices.keys())
    N     = len(pairs)

    # ── 1. Days common + suppression week-ends ──────────────────────────
    common_days = None
    for key, mat in vol_matrices.items():
        idx = pd.to_datetime(mat.index).normalize()
        d   = set(idx)
        if common_days is None:
            common_days = d
        else:
            common_days = common_days & d

    if not common_days:
        # Diagnose: show index samples per pair
        print("  ⚠ common_days is empty — index mismatch between pairs!")
        for key, mat in list(vol_matrices.items())[:3]:
            idx = pd.to_datetime(mat.index).normalize()
            print(f"    {key}: {len(idx)} days, first={idx[0]}, last={idx[-1]}")
        raise ValueError(
            "No common days found across all pairs. "
            "Check that all DataFrames have overlapping DatetimeIndex. "
            "Tip: make sure filter_session and log_transform preserve the index."
        )

    common_days = sorted(common_days)
    print(f"  Common days (all pairs)        : {len(common_days)}")

    weekdays = [d for d in common_days if pd.Timestamp(d).dayofweek < 5]
    print(f"  Weekends removed             : {len(common_days) - len(weekdays)}"
          f"  ({len(weekdays)} trading days)")
    common_days = weekdays

    # ── 2. Common columns — PRESERVE ORDER from input matrices ──────────────
    # Do NOT sort: if filter_session produced an overnight session
    # (e.g. 18:00→17:55), the columns are already ordered τ=0=18:00.
    # sorted() would reset to 00:00 and destroy that ordering.
    common_cols_set = None
    for mat in vol_matrices.values():
        c = set(mat.columns)
        common_cols_set = c if common_cols_set is None else common_cols_set & c

    # Preserve the column order of the first matrix (respects filter_session)
    first_mat   = next(iter(vol_matrices.values()))
    common_cols = [c for c in first_mat.columns if c in common_cols_set]
    M_brut = len(common_cols)

    # ── Filter FX holidays ───────────────────────────────────────────────
    # A) Fixed holidays
    universal_holidays = set()
    for d in common_days:
        ts = pd.Timestamp(d)
        mm, dd = ts.month, ts.day
        if (mm == 12 and dd == 25) or \
           (mm == 12 and dd == 26) or \
           (mm ==  1 and dd ==  1):
            universal_holidays.add(d)

    trading_days = [d for d in common_days if d not in universal_holidays]
    n_fixed = len(common_days) - len(trading_days)

    # B) Automatic detection of thin-market days (floating holidays)
    # Mean fill rate on 5 representative pairs
    fill_by_day = np.zeros(len(trading_days))
    sample_pairs = list(vol_matrices.items())[:min(5, N)]
    for _, mat in sample_pairs:
        mat2 = mat.copy()
        mat2.index = pd.to_datetime(mat2.index).normalize()
        aligned = mat2.reindex(
            index=pd.DatetimeIndex(trading_days),
            columns=common_cols
        )
        fill_by_day += aligned.notna().mean(axis=1).values
    fill_by_day /= len(sample_pairs)

    active_days = [
        d for d, f in zip(trading_days, fill_by_day)
        if f >= 0.20
    ]
    n_thin = len(trading_days) - len(active_days)

    if n_fixed + n_thin > 0:
        print(f"  Holidays removed                : "
              f"{n_fixed} fixed (Dec 25/26, Jan 1)"
              f"  +  {n_thin} thin-market days detected")
    common_days = active_days

    T_raw    = len(common_days)
    days_idx = pd.DatetimeIndex(common_days)

    print(f"  Common intervals (raw)         : {M_brut}")

    # ── 3. Numpy stack (N, T_raw, M_raw) in a single pass ────────────────
    # Each pair is re-indexed on (days_idx × common_cols) vectorially.
    # Days missing for a pair → NaN (short-history pairs, 2010+).
    print(f"  Building numpy stack...", end=' ', flush=True)
    stack = np.full((N, T_raw, M_brut), np.nan, dtype=np.float32)

    for i, (pair, mat) in enumerate(vol_matrices.items()):
        mat2 = mat.copy()
        mat2.index = pd.to_datetime(mat2.index).normalize()
        # reindex: clean alignment without a Python loop
        aligned = mat2.reindex(index=days_idx, columns=common_cols)
        stack[i] = aligned.values.astype(np.float32)

    print(f"ok  ({time.time()-t0:.1f}s)")

    # ── 4. Inactive columns ──────────────────────────────────────────────
    # fill_rate[j] = fraction of non-NaN cells in column j
    # averaged over N pairs AND T_raw days
    fill_rate   = (~np.isnan(stack)).mean(axis=(0, 1))   # (M_brut,)
    active_mask = fill_rate >= 0.10
    active_idx  = np.where(active_mask)[0]
    M           = len(active_idx)
    active_cols = [common_cols[j] for j in active_idx]
    print(f"  Inactive columns removed       : {M_brut - M}"
          f"  ({M} active = {M*5} min of session)")

    # Reduce stack to active columns
    stack = stack[:, :, active_idx]   # (N, T_raw, M)

    # ── 5. Day filtering by coverage ─────────────────────────────────────
    # coverage[i, t] = fraction of active non-NaN columns for pair i
    # on day t. The cross-pair minimum is taken for each day.
    coverage    = (~np.isnan(stack)).mean(axis=2)        # (N, T_raw)
    min_cov     = coverage.min(axis=0)                   # (T_raw,)
    valid_mask  = min_cov >= min_coverage

    # force_dates: keep specific dates regardless of coverage
    # (useful for known crisis days that have low fill due to market stress)
    if force_dates is not None:
        forced = set(pd.Timestamp(d).date() for d in force_dates)
        for j, d in enumerate(days_idx):
            if pd.Timestamp(d).date() in forced and not valid_mask[j]:
                valid_mask[j] = True
                print(f"  ⚑ force_date: {str(d)[:10]} kept despite "
                      f"coverage={min_cov[j]:.1%} < {min_coverage:.0%}")
    T           = int(valid_mask.sum())

    print(f"  Days rejected (cov < {min_coverage:.0%})      : {T_raw - T}")

    # Most constraining pairs
    mean_cov  = coverage.mean(axis=1)   # (N,)
    worst_idx = np.argsort(mean_cov)[:5]
    print(f"  Least-covered pairs:")
    for idx in worst_idx:
        print(f"    {pairs[idx]:15s}  mean coverage = {mean_cov[idx]:.1%}")

    print(f"  Days retained                  : {T}")

    # Select valid days
    X_raw      = stack[:, valid_mask, :]   # (N, T, M) float32
    valid_days = days_idx[valid_mask]

    # ── 6. Residual NaN imputation ───────────────────────────────────────
    # Two-pass strategy:
    #
    # Pass A — linear interpolation along the intraday time axis (M)
    #   For each pair i and day t, isolated NaN (early close,
    #   liquidity gap) are interpolated between non-NaN neighbours.
    #   This preserves the CONTINUITY of the functional curve — crucial for
    #   the FFM, which estimates smooth loadings.
    #   Without interpolation, the global median creates a constant value
    #   artificial constant value that creates a discontinuity → exploding loadings.
    #
    # Pass B — median per pair/column for non-interpolable NaN
    #   (curve start or end with no valid neighbour).

    nan_mask = np.isnan(X_raw)
    n_nan_total = int(nan_mask.sum())

    if n_nan_total > 0:
        # Pass A: linear interpolation along the M axis (intraday) — vectorised
        # Reshape to (N*T, M) to use pd.DataFrame.interpolate
        NT, M_cols = N * T, X_raw.shape[2]
        flat = X_raw.reshape(NT, M_cols).astype(np.float64)

        # pd.DataFrame.interpolate(axis=1) interpolates each row independently
        df_flat    = pd.DataFrame(flat)
        df_interp  = df_flat.interpolate(method='linear', axis=1,
                                          limit_direction='both')
        X_interp   = df_interp.values.reshape(N, T, M_cols).astype(np.float32)

        # Pass B: median per (pair, column) for remaining NaN (edges)
        still_nan = np.isnan(X_interp)
        if still_nan.any():
            medians  = np.nanmedian(X_interp, axis=1, keepdims=True)
            medians  = np.where(np.isnan(medians), 0.0, medians)
            X_interp = np.where(still_nan, medians, X_interp)

        n_interp = n_nan_total - int(np.isnan(X_interp).sum())
        n_median = int(still_nan.sum()) if still_nan.any() else 0
        print(f"  NaN imputed by interpolation   : {n_interp}")
        if n_median > 0:
            print(f"  NaN imputed by median (edges)  : {n_median}")

        X = X_interp.astype(np.float64)
    else:
        X = X_raw.astype(np.float64)

    n_nan = int(np.isnan(X).sum())
    if n_nan > 0:
        print(f"  ⚠ {n_nan} Residual NaN → forced to 0")
        X = np.nan_to_num(X, nan=0.0)

    n_nan = int(np.isnan(X).sum())
    if n_nan > 0:
        print(f"  ⚠ {n_nan} Residual NaN → forced to 0")
        X = np.nan_to_num(X, nan=0.0)

    tau = np.linspace(0, 1, M)
    print(f"  Final panel X.shape: {X.shape}  "
          f"(N pairs × T days × M intervals)  "
          f"[{time.time()-t0:.1f}s total]")

    return X, valid_days, tau, pairs


def filter_session(vol_matrices: dict,
                   session_start: str = None,
                   session_end: str   = None,
                   min_fill: float    = 0.3) -> dict:
    """
    Restrict columns (5-min intervals) to active trading hours.

    Two modes:
    ─────────────
    A) Explicit hours — if session_start and session_end are provided:
        Keep columns whose time falls in [session_start, session_end].

    B) Automatic detection — if no hours are provided (default):
        Compute the mean fill rate of each column across all
        pairs. Keep columns with rate ≥ min_fill (default 30%).
        → Removes overnight hours without knowing the timezone.

    Parameters
    ----------
    vol_matrices   : dict {str: DataFrame(days × minutes)}
    session_start  : str  e.g. '00:00' or '07:00' (format HH:MM)
    session_end    : str  e.g. '21:55' or '20:55'
    min_fill       : float — fill threshold for automatic mode

    Returns
    --------
    vol_filtered : dict {str: DataFrame} — same index, filtered columns
    """
    # ── Collect all common columns ────────────────────────────
    common_cols = None
    for df in vol_matrices.values():
        c = set(df.columns)
        common_cols = c if common_cols is None else common_cols & c
    common_cols = sorted(common_cols)

    if session_start is not None and session_end is not None:
        # Mode A: explicit time filter
        try:
            t_start = pd.Timestamp(f'2000-01-01 {session_start}').time()
            t_end   = pd.Timestamp(f'2000-01-01 {session_end}').time()
        except Exception:
            raise ValueError(f"Invalid time format: '{session_start}' / '{session_end}'. "
                             f"Use the format 'HH:MM'.")

        cols_time = pd.to_datetime(
            [str(c) for c in common_cols], format='%H:%M:%S'
        ).time

        overnight = t_start > t_end   # e.g. 18:00 → 17:55 crosses midnight

        if overnight:
            # Keep columns OUTSIDE [t_end+1min, t_start-1min]
            # i.e.  t >= 18:00  OR  t <= 17:55
            mask = np.array([
                (t >= t_start) or (t <= t_end) for t in cols_time
            ])
        else:
            # Normal intraday session: t_start <= t <= t_end
            mask = np.array([
                (t >= t_start) and (t <= t_end) for t in cols_time
            ])
        active_cols = [c for c, ok in zip(common_cols, mask) if ok]
        mode = f"hours [{session_start} → {session_end}]"

    else:
        # Mode B: automatic detection by fill rate
        N = len(vol_matrices)
        fill = np.zeros(len(common_cols))
        for df in vol_matrices.values():
            aligned = df.reindex(columns=common_cols)
            fill   += aligned.notna().mean(axis=0).values
        fill /= N

        mask        = fill >= min_fill
        active_cols = [c for c, ok in zip(common_cols, mask) if ok]
        mode        = f"auto (fill ≥ {min_fill:.0%})"

    n_kept    = len(active_cols)
    n_dropped = len(common_cols) - n_kept
    print(f"  filter_session [{mode}] : "
          f"{n_dropped} columns removed → {n_kept} active "
          f"= {n_kept * 5} min of session")

    # For overnight sessions (e.g. 18:00 → 17:55), reorder columns so
    # that τ=0 corresponds to session_start (18:00) and τ=1 to session_end.
    # Standard (non-overnight) sessions are already in chronological order.
    if (session_start is not None and session_end is not None
            and t_start > t_end):
        # Split at midnight and reorder: [18:00..23:55] + [00:00..17:55]
        after_midnight  = [c for c in active_cols
                           if pd.Timestamp(f'2000-01-01 {c}').time() >= t_start]
        before_midnight = [c for c in active_cols
                           if pd.Timestamp(f'2000-01-01 {c}').time() <= t_end]
        active_cols = after_midnight + before_midnight

    return {k: df.reindex(columns=active_cols)
            for k, df in vol_matrices.items()}


# =============================================================================
# DATA CHECKS
# =============================================================================

def diagnostic_panel(X, pairs, tau):
    """
    Print descriptive statistics of the panel before estimation.
    Detects anomalies (extreme outliers, empty days...).
    """
    N, T, M = X.shape
    print("\n── Panel descriptive statistics ──────────────────────────")
    print(f"  Shape : {X.shape}")
    print(f"  Remaining NaN: {np.isnan(X).sum()}")
    print(f"  Global min   : {np.nanmin(X):.2e}")
    print(f"  Global max   : {np.nanmax(X):.2e}")
    print(f"  Median       : {np.nanmedian(X):.2e}")
    print()
    print(f"  {'Pair':12s}  {'Mean':>10s}  {'Std':>10s}  {'Max':>10s}")
    print("  " + "-" * 48)
    for i, pair in enumerate(pairs):
        print(f"  {pair:12s}  {np.nanmean(X[i]):.4e}  "
              f"{np.nanstd(X[i]):.4e}  {np.nanmax(X[i]):.4e}")
    print()

    # Detect extreme outliers (> 99.9th percentile)
    p999 = np.nanpercentile(X, 99.9)
    n_outliers = (X > p999).sum()
    if n_outliers > 0:
        print(f"  ⚠ {n_outliers} values > 99.9th percentile ({p999:.2e})")
        print(f"    → Consider winsorize_panel() before estimation")


def winsorize_panel(X, upper_pct=99.5):
    """
    Winsorise extreme values of the panel.

    Handles both cases:
    - rv² (positive values)  → clip to [0, cap_high]
    - log(rv²) (real values, can be negative)
      → clip to [cap_low, cap_high] (two-sided winsorisation)
    """
    # Detect if data are log-transformed (presence of values < 0)
    has_negative = np.nanmin(X) < 0

    cap_high = np.nanpercentile(X, upper_pct)
    low_pct  = 100 - upper_pct          # e.g. 0.5 if upper_pct=99.5

    if has_negative:
        cap_low = np.nanpercentile(X, low_pct)
        X_wins  = np.clip(X, cap_low, cap_high)
        n_capped = ((X > cap_high) | (X < cap_low)).sum()
        print(f"  Bilateral winsorisation [{cap_low:.2f}, {cap_high:.2f}] "
              f": {n_capped} values clipped  (log mode)")
    else:
        X_wins  = np.clip(X, 0, cap_high)
        n_capped = (X > cap_high).sum()
        print(f"  Winsorisation at {upper_pct}th pct ({cap_high:.2e}) "
              f": {n_capped} values capped")

    return X_wins


# =============================================================================
# STEP 1: CENTRING AND NORMALISATION OF THE PANEL
# =============================================================================

def center_panel(X, mu=None):
    """
    μ̂_i(τ) = (1/T) Σ_t x_{it}(τ)   — mean intraday profile of pair i
    y_{it}(τ) = x_{it}(τ) − μ̂_i(τ)  — centred panel

    μ̂_i(τ) captures the structural pattern of a typical day:
    U-shape for EUR/USD (high vol at the London / New York open and close),
    asymmetric for AUD/JPY (Asian morning spike).
    """
    if mu is None:
        mu = X.mean(axis=1)          # (N, M)
    Y  = X - mu[:, np.newaxis, :]    # (N, T, M)
    return Y, mu


def normalize_panel(Y):
    """
    Global normalisation by the std of the entire centred panel.

    WHY THIS IS NEEDED:
    ---------------------------
    The FFM loading formula (Proposition 2.1, FFM-II) is:

        b̂_{il}(τ) = (1 / (T √λ̂_l)) Σ_t û_{lt} y_{it}(τ)

    It is designed to be a consistent estimator of the true loadings
    UNDER the identification assumption that λ̂_l = O(1) (or at least that
    λ̂_l is bounded away from 0).

    With rv² data of order 10⁻¹²:
        F_{st} ~ (rv)² ~ 10⁻¹²   →   λ̂_l ~ 10⁻¹²

    This gives 1/√λ̂_l ~ 10⁶ → exploding loadings → R² >> 100%.

    The normalisation Y → Y/σ gives:
        F_norm ~ (rv/σ)² ~ 1     →   λ̂_l_norm ~ σ²/σ² ~ 1  ✓

    In practice: σ = std(Y) ~ 10⁻⁶ and λ̂_1 ~ 10⁻¹² → λ̂_1/σ² ~ 1.

    σ is returned so that the outputs (Chi, Xi, B_hat) can be rescaled
    after estimation by multiplying by σ.
    """
    sigma = float(np.std(Y))
    return Y / sigma, sigma


# =============================================================================
# STEP 2: GRAM MATRIX
# =============================================================================

def build_gram_matrix(Y):
    """
    F_{st} = (1/N) * (1/M) * Σ_i Σ_τ y_{is}(τ) y_{it}(τ)

    = discrete approximation of:
      (1/N) * Σ_i ∫ y_{is}(τ) y_{it}(τ) dτ

    F ∈ R^{T×T} encodes the temporal dependence structure (between days)
    of the panel. Its spectral decomposition directly yields the factors.
    """
    N, T, M = Y.shape
    # einsum 'ism,itm->st': for each pair of days (s, t),
    # sum over i (FX pairs) and m (minutes) of y_{is,m} * y_{it,m}
    F = np.einsum('ism,itm->st', Y, Y) / (N * M)
    F = (F + F.T) / 2   # numerical symmetry
    return F


# =============================================================================
# STEP 3: ESTIMATION OF FACTORS, LOADINGS AND COMPONENTS
# =============================================================================

def decompose_gram(F, k_max):
    """
    Decompose F ONCE and return the first k_max
    eigenvectors/eigenvalues sorted in descending order.

    Key optimisation: instead of rerunning eigh(F) for each k in
    select_r(), we do it once here and reuse the result.

    eigh() on a (T, T) matrix is O(T³).
    With T=252 and k_max=10, recomputing it for each k costs 10 × O(T³);
    decomposing once costs 1 × O(T³) + 10 light O(T) operations → ~10× faster.

    Parameters
    ----------
    F     : np.ndarray (T, T) — symmetric Gram matrix
    k_max : int — number of eigenvectors to keep

    Returns
    --------
    eigenvalues  : np.ndarray (k_max,) — eigenvalues sorted ↓
    eigenvectors : np.ndarray (T, k_max) — associated eigenvectors
    """
    N_eig = F.shape[0]
    # eigh returns the eigenvalues in ASCENDING order;
    # subset_by_index=[N_eig-k_max, N_eig-1] only computes the k_max
    # largest ones → faster than the full decomposition
    eigenvalues, eigenvectors = eigh(
        F,
        subset_by_index=[N_eig - k_max, N_eig - 1]
    )
    # Reverse for descending order
    eigenvalues  = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]   # (T, k_max)
    return eigenvalues, eigenvectors


def estimate_ffm(Y, F, k, _eigen=None):
    """
    Proposition 2.1 (FFM-II): least-squares estimation.

    (i)  Spectral decomposition of F → (λ̃_l, f̃_l), l=1..k
    (ii) Normalisation:
            û_{lt}  = √T * f̃_{l,t} / ‖f̃_l‖      ← scalar factors
            λ̂_l    = λ̃_l / √T                    ← normalised eigenvalues
    (iii) Functional loadings:
            b̂_{il}(τ) = (1 / T) * Σ_t û_{lt} * y_{it}(τ)
    (iv) Common component        : χ̂_{it}(τ) = Σ_l b̂_{il}(τ) * û_{lt}
         Idiosyncratic component : ξ̂_{it}(τ) = y_{it}(τ) − χ̂_{it}(τ)

    Parameters
    ----------
    Y      : np.ndarray (N, T, M)
    F      : np.ndarray (T, T)
    k      : int — number of factors
    _eigen : optional tuple (eigenvalues, eigenvectors) pre-computed
             → passed from select_r() to avoid recomputing eigh()
    """
    N, T, M = Y.shape

    # Spectral decomposition — reuse _eigen if provided
    if _eigen is not None:
        eigenvalues, eigenvectors = _eigen
        # Keep only the first k (already sorted ↓)
        lambda_tilde = eigenvalues[:k]
        f_tilde      = eigenvectors[:, :k]
    else:
        # Standalone call (not from select_r): decompose F here
        eigenvalues, eigenvectors = decompose_gram(F, k_max=k)
        lambda_tilde = eigenvalues
        f_tilde      = eigenvectors

    # Normalisation
    lambda_hat = lambda_tilde / np.sqrt(T)  # (k,)

    U_hat = np.zeros((k, T))
    for l in range(k):
        f_l      = f_tilde[:, l]
        U_hat[l] = np.sqrt(T) * f_l / np.linalg.norm(f_l)

    # Loadings — Proposition 2.1 (FFM-II), step (iii)+(iv):
    #
    # Step (iii): ê_l = λ̂_l^{-1/2} T^{-1} Σ_t û_{lt} y_t   ← normalised DIRECTION
    # Step (iv):  b̂_{il}(τ) = λ̂_l^{1/2} ê_{il}(τ)          ← actual LOADING
    #
    # Expanding:
    #   b̂_{il}(τ) = λ̂_l^{1/2} · λ̂_l^{-1/2} · T^{-1} Σ_t û_{lt} y_{it}(τ)
    #             = T^{-1} Σ_t û_{lt} y_{it}(τ)
    #
    # This is the OLS projection of y_i onto span(û_1,...,û_k) — maximises R².
    # DO NOT divide by sqrt(λ̂_l): that would compute ê_l, not b̂_{il}.
    #
    B_hat = np.einsum('kt,itm->kim', U_hat, Y).transpose(1, 0, 2) / T  # (N, k, M)

    # Sign convention: force the mean loading to be > 0
    # (the sign of û_l and b̂_il is unidentified — flipping both
    #  simultaneously leaves χ̂ unchanged, but makes the plots readable)
    for l in range(k):
        if np.mean(B_hat[:, l, :]) < 0:
            B_hat[:, l, :] *= -1
            U_hat[l]       *= -1

    # Components
    Chi_hat = np.einsum('ilm,lt->itm', B_hat, U_hat)  # (N, T, M)
    Xi_hat  = Y - Chi_hat                              # (N, T, M)

    return U_hat, B_hat, Chi_hat, Xi_hat, lambda_hat


# =============================================================================
# STEP 4: SELECTING THE NUMBER OF FACTORS r̂
# =============================================================================

def select_r(Y, F, k_max=5, criterion='IC2a'):
    """
    IC(k) = V(k) + k * g(N, T)   with   V(k) = ‖Y - Ŷ^(k)‖² / (NTM)

    LOOP-FREE ALGORITHM:
    ---------------------
    Exploits orthogonality of F's eigenvectors.
    The common component with k factors is the projection of Y onto
    the subspace spanned by {û_1, ..., û_k}:

        ‖ξ̂^(k)‖² = ‖Y‖² - Σ_{l=1}^{k} ‖proj_l‖²

    where  proj_l[i, τ] = (1/T) * Σ_t û_{lt} * y_{it}(τ)

    This means V(k) is computed INCREMENTALLY:
        V(k) = V(k-1) - contrib_k / (N·T·M)

    All contributions are pre-computed in a single vectorised pass
    before the IC loop → the loop only does scalar additions.

    Complexity:
        Old (k_max × eigh + k_max × einsum (N,T,M)) : O(k·T³ + k²·NTM)
        New (1 × eigh_subset + 1 × einsum (k,N,M))  : O(T³ + k·NTM)
        Typical speedup (T=500, N=12, M=96, k=10)   : ~10×

    Penalties (FFM-II, Section 2.2):
        IC1a : g = √((N+T)/NT) * log(NT/(N+T))
        IC2a : g = √((N+T)/NT) * log(min(√N, √T)²)   ← recommended default
        IC3a : g = log(min(N,T)) / min(N,T)            ← smallest penalty

    NOTE — IC1a ≈ IC2a when T >> N:
        When T >> N,  NT/(N+T) ≈ N = min(N,T),
        so  log(NT/(N+T)) ≈ log(N) = log(min(N,T)).
        With N=34, T=3977: g_IC1a = 0.1044,  g_IC2a = 0.1046  (ratio = 0.9976).
        Both criteria will give the same r̂ in this regime.

    WARNING — Small N / Large T regime:
        IC1a and IC2a are calibrated for N, T → ∞ at comparable rates.
        With N=34 and T=3977, g ≈ 0.104 per factor, while the marginal
        reduction in V is ~0.008. Both IC criteria will always choose r̂=1.
        Use criterion='ratio' (spectral elbow criterion) in this regime.
    """
    N, T, M = Y.shape
    C_NT = min(np.sqrt(N), np.sqrt(T))

    # ── 1. Decomposition of F — ONCE ────────────────────────────────────
    eigenvalues, eigenvectors = decompose_gram(F, k_max=k_max)

    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    r_90 = np.argmax(cumvar >= 0.90) + 1
    print(f'90% of the variance explained by {r_90} PC')

    # ── Spectral ratio criterion (robust when N << T) ──────────────────────
    # r̂_ratio = argmax_l λ̂_l / λ̂_{l+1}  (elbow of the scree plot)
    # Corresponds to ED estimator (Ahn & Horenstein 2013)
    ratios  = eigenvalues[:-1] / (eigenvalues[1:] + 1e-14)
    r_ratio = int(np.argmax(ratios)) + 1

    if criterion in ('ratio', 'GR'):
        norms      = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        U_all      = np.sqrt(T) * (eigenvectors / norms).T
        projs      = np.tensordot(U_all, Y, axes=([1], [1])) / T
        lambda_hat = eigenvalues / np.sqrt(T)
        P_sq       = np.sum(projs ** 2, axis=(1, 2)) / (N * M)
        delta      = P_sq * (1.0 / lambda_hat - 2.0 / np.sqrt(lambda_hat))
        V0         = np.sum(Y ** 2) / (N * T * M)
        V_vals     = np.maximum(V0 + np.cumsum(delta), 0.0)

        if criterion == 'ratio':
            ic_vals = ratios[:k_max] if k_max <= len(ratios) else \
                  np.pad(ratios, (0, k_max - len(ratios)))
            return r_ratio, ic_vals, V_vals, (eigenvalues, eigenvectors)

        if criterion == 'GR':
            eigs_sum = eigenvalues[:k_max] 
            V_AH = np.array([
            np.sum(eigs_sum[k:]) for k in range(k_max + 1)
                    ])   # V_AH[0] = V(0), V_AH[1] = V(1), etc.
    
            GR_vals = np.ones(k_max - 1)
            for k in range(1, k_max):
                num = np.log(V_AH[k-1] / (V_AH[k]   + 1e-14) + 1e-14)
                den = np.log(V_AH[k]   / (V_AH[k+1] + 1e-14) + 1e-14)
                GR_vals[k-1] = num / (den + 1e-14) if den > 1e-10 else 1.0
    
            r_GR = int(np.argmax(GR_vals)) + 1
            return r_GR, GR_vals, V_vals, (eigenvalues, eigenvectors)
    
    V0     = np.sum(Y ** 2) / (N * T * M)
    penalties = {
        'IC2a': np.sqrt((N + T) / (N * T)) * np.log(C_NT**2),
        'IC1a': np.sqrt((N + T) / (N * T)) * np.log(N * T / (N + T)),
        'IC2' : ((N + T) / (N * T)) * np.log(C_NT**2),
        'IC1' : ((N + T) / (N * T)) * np.log(N * T / (N + T)),
        'BIC3': V0*(N+T)*np.log(N*T) / (N*T)
    }
    g = penalties[criterion]
    print(f"    [IC] penalty g = {g:.4f} per factor  (N={N}, T={T})")

    # ── 2. Normalise all k_max factors in one pass ───────────────────────
    # û_l = √T * f̃_l / ‖f̃_l‖   →  U_all shape (k_max, T)
    norms   = np.linalg.norm(eigenvectors, axis=0, keepdims=True)   # (1, k_max)
    U_all   = np.sqrt(T) * (eigenvectors / norms).T                 # (k_max, T)

    # ── 3. Projections of Y on each factor ───────────────────────────────
    # P_l[i, τ] = (1/T) * Σ_t û_{lt} * y_{it}(τ)   →  shape (k_max, N, M)
    projs = np.tensordot(U_all, Y, axes=([1], [1])) / T    # (k_max, N, M)

    # Normalised eigenvalues λ̂_l = λ̃_l / √T
    lambda_hat = eigenvalues / np.sqrt(T)   # (k_max,)

    # ── 4. Correct incremental formula for V(k) ──────────────────────────
    #
    # Expansion of ‖Y - χ̂^(k)‖² using the orthogonality of the û_l:
    #
    #   Σ_t û_{lt} û_{l't} = T * δ_{ll'}   →  the χ̂_l are orthogonal
    #
    #   <Y, χ̂_l> = T/√λ̂_l  * Σ_{i,τ} P_l[i,τ]²   ← cross term
    #   ‖χ̂_l‖²   = T/λ̂_l   * Σ_{i,τ} P_l[i,τ]²   ← squared norm
    #
    #   V(k) = V(0) + Σ_{l=1}^k P_sq[l] * (1/λ̂_l - 2/√λ̂_l)
    #
    # where P_sq[l] = Σ_{i,τ} P_l[i,τ]² / (N*M)
    #
    # Note: if λ̂_l >> 1 (strong factor), the term is negative → V decreases.
    #       if λ̂_l << 1 (noise),         the term is positive → V increases.
    #        The IC criterion penalises k to avoid including noise factors.

    P_sq     = np.sum(projs ** 2, axis=(1, 2)) / (N * M)          # (k_max,)
    delta    = P_sq * (1.0 / lambda_hat - 2.0 / np.sqrt(lambda_hat))

    # ── 5. Incremental V(k) ──────────────────────────────────────────────
    
    V_vals = np.maximum(V0 + np.cumsum(delta), 0.0)               # (k_max,)

    # ── 6. IC criterion ─────────────────────────────────────────────────────
    ks      = np.arange(1, k_max + 1)
    ic_vals = V_vals + ks * g

    r_hat = int(np.argmin(ic_vals)) + 1

    # -- 7. BIC3 criterion (FFM-II, Section 2.2) — smallest penalty (log(N)/N) --
    # r̂_BIC3 = argmin_k V(k) + k * sigma_hat *(N+T-k)*log(N*T) / N*T

    if criterion =='BIC3':
        V_vals_bic3  = np.zeros(k_max)
        sigma2_hat = V_vals[-1] 
        bic3_vals  = np.array([
            V_vals_bic3[k] + (k+1) * sigma2_hat * (N + T - (k+1)) * np.log(N*T) / (N*T)
            for k in range(k_max)
        ])
        r_hat_bic3 = int(np.argmin(bic3_vals)) + 1

        r_hat = r_hat_bic3
        ic_vals = bic3_vals
        V_vals = V_vals_bic3

    return r_hat, ic_vals, V_vals, (eigenvalues, eigenvectors)



# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_mean_profiles(mu, tau, pairs, days, X=None):
    """
    Plot the mean intraday profiles μ̂_i(τ).

    If data are log-transformed (mu contains negative values),
    two curves are shown per pair:
      - Main curve: back-transformed to real volatility units
        vol(τ) = exp(μ_i(τ)/2)  ≈ geometric mean of |r_t(τ)|
      - Shading: ±1 intraday std computed on X if provided

    If data are raw rv², display directly.
    """
    N = len(pairs)

    # Detect if log-transformed (negative values in mu)
    is_log = mu.min() < 0

    if is_log:
        mu_display = np.exp(mu / 2)
        ylabel     = "|r| mean (geom.)"
        scale      = 1
        title_note = "Intraday volatility  exp(μ̂/2)  — units |return|"
    else:
        mu_display = mu
        ylabel     = "rv²"
        scale      = 1
        title_note = "Intraday volatility  μ̂(τ)  — units rv²"

    ncols = 4
    nrows = (N + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, pair in enumerate(pairs):
        ax  = axes[i]
        y   = mu_display[i] * scale

        # ±1 std shading if X is provided
        if X is not None and is_log:
            vol_daily = np.exp(X[i] / 2) * scale   # (T, M)
            std_daily = np.std(vol_daily, axis=0)   # (M,)
            ax.fill_between(tau, np.maximum(y - std_daily, 0), y + std_daily,
                            alpha=0.12, color=COLORS[i % len(COLORS)])
        elif X is not None:
            std_daily = np.std(X[i] * scale, axis=0)
            ax.fill_between(tau, np.maximum(y - std_daily, 0), y + std_daily,
                            alpha=0.12, color=COLORS[i % len(COLORS)])

        ax.fill_between(tau, y, alpha=0.20, color=COLORS[i % len(COLORS)])
        ax.plot(tau, y, color=COLORS[i % len(COLORS)], linewidth=1.8)
        ax.set_title(_format_pair(pair), fontsize=11, fontweight='bold')
        ax.set_xlabel("τ [0=open → 1=close]", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        # Annotate the intraday peak
        peak_idx = np.argmax(y)
        ax.axvline(tau[peak_idx], color=COLORS[i % len(COLORS)],
                   linewidth=0.8, linestyle=':', alpha=0.6)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    period = f"{str(days[0])[:10]} → {str(days[-1])[:10]}"
    fig.suptitle(f"{title_note}\n{period}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_fig('01_mean_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_mean_profiles.png")


def plot_mean_profiles_log(mu, tau, pairs, days, X=None):
    """
    Mean profiles in log-variance space, split into two thesis-ready images.

    - Alphabetical order across both images
    - First image: 24 pairs  |  second image: remaining pairs
    - Shared y-axis (global min/max across all pairs and ±1 std bands)
    - Only mean line + ±1 std shading; no fill-to-zero, no peak marker
    """
    N = len(pairs)

    # Sort alphabetically
    order   = np.argsort([p.lower() for p in pairs])
    pairs_s = [pairs[k] for k in order]
    mu_s    = mu[order]                              # (N, M)
    X_s     = X[order] if X is not None else None   # (N, T, M) or None

    # Global y limits — computed from ALL pairs so both images share the same scale
    y_min_g, y_max_g = np.inf, -np.inf
    for k in range(N):
        y = mu_s[k]
        if X_s is not None:
            std = np.nanstd(X_s[k], axis=0)
            y_min_g = min(y_min_g, float(np.nanmin(y - std)))
            y_max_g = max(y_max_g, float(np.nanmax(y + std)))
        else:
            y_min_g = min(y_min_g, float(np.nanmin(y)))
            y_max_g = max(y_max_g, float(np.nanmax(y)))
    pad  = (y_max_g - y_min_g) * 0.06
    ylim = (y_min_g - pad, y_max_g + pad)

    period = f"{str(days[0])[:10]} → {str(days[-1])[:10]}"
    n1     = min(24, N)

    splits = [
        (pairs_s[:n1],  mu_s[:n1],  X_s[:n1]  if X_s is not None else None,
         0,  '01a_mean_profiles_log_part1.png'),
    ]
    if N > n1:
        splits.append(
            (pairs_s[n1:],  mu_s[n1:],  X_s[n1:]  if X_s is not None else None,
             n1, '01a_mean_profiles_log_part2.png')
        )

    for pairs_g, mu_g, X_g, offset, fname in splits:
        n     = len(pairs_g)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        axes = axes.flatten()

        rng = np.random.default_rng(42)
        for j, pair in enumerate(pairs_g):
            ax         = axes[j]
            color_idx  = offset + j
            y          = mu_g[j]

            if X_g is not None:
                curves = X_g[j]                        # (T, M)
                T_cur  = curves.shape[0]
                idx    = rng.choice(T_cur, size=min(50, T_cur), replace=False)
                for t in idx:
                    ax.plot(tau, curves[t], color='#aaaaaa', linewidth=0.4,
                            alpha=0.35, zorder=1)

                std_daily = np.nanstd(X_g[j], axis=0)
                ax.fill_between(tau, y - std_daily, y + std_daily,
                                alpha=0.20, color=COLORS[color_idx % len(COLORS)],
                                label='$\\pm 1\\,\\sigma$', zorder=2)

            ax.plot(tau, y, color=COLORS[color_idx % len(COLORS)], linewidth=1.8,
                    label='$\\hat{\\mu}_i(\\tau)$', zorder=3)
            ax.set_title(_format_pair(pair), fontsize=11, fontweight='bold')
            ax.set_xlabel("$\\tau$  [0 = open $\\rightarrow$ 1 = close]", fontsize=8)
            ax.set_ylabel("$\\log(r^2 + \\varepsilon)$", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(ylim)
            if j == 0:
                ax.legend(fontsize=7, loc='lower center')

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Mean profiles  $\\hat{{\\mu}}_i(\\tau)$\n{period}",
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(_fig(fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {fname}")


def plot_scree(F, T, k_max=15):
    # Preventive check
    if not np.isfinite(F).all():
        n_bad = (~np.isfinite(F)).sum()
        raise ValueError(
            f"The Gram matrix F contains {n_bad} non-finite values "
            f"(NaN/inf). This indicates NaN or -inf in Y_norm, "
            f"probably from a log transform on zero values. "
            f"Check your ε in log(rv² + ε)."
        )
    eigenvalues, _ = eigh(F)
    eigenvalues    = np.sort(eigenvalues)[::-1][:k_max] / np.sqrt(T)
    ks             = np.arange(1, len(eigenvalues) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(ks, eigenvalues, color=COLORS[0], alpha=0.8)
    axes[0].set_xlabel("Rank $l$")
    axes[0].set_ylabel(r"$\hat{\lambda}_l$")
    axes[0].set_title("Eigenvalues of $F$\nLarge values $\\rightarrow$ common factors")
    axes[0].grid(True, alpha=0.3, axis='y')

    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-14)
    axes[1].plot(ks[:-1], ratios, 'o-', color=COLORS[1], linewidth=2, markersize=8)
    axes[1].axhline(1, color='gray', linestyle='--')
    axes[1].set_xlabel("Rank $l$")
    axes[1].set_ylabel(r"$\hat{\lambda}_l \,/\, \hat{\lambda}_{l+1}$")
    axes[1].set_title("Eigenvalue ratio\nPeak $\\rightarrow$ number of factors $r$")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Scree plot — Spectral structure of the FX panel",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_fig('02_scree_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_scree_plot.png")


def plot_ic(ic_vals, V_vals, r_hat, ic_vals2=None, label1='IC1a', label2='IC2a'):
    """
    Plot information criterion and residual variance curves.

    Parameters
    ----------
    ic_vals  : IC values for primary criterion (plotted in colour)
    V_vals   : residual variance V(k)
    r_hat    : selected number of factors
    ic_vals2 : optional second IC series to overlay (e.g. IC2a alongside IC1a)
    label1   : label for ic_vals
    label2   : label for ic_vals2
    """
    ks = np.arange(1, len(ic_vals) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: IC curves ───────────────────────────────────────────────────
    axes[0].plot(ks, ic_vals, 'o-', color=COLORS[2], linewidth=2,
                 markersize=8, label=label1)
    if ic_vals2 is not None:
        ks2 = np.arange(1, len(ic_vals2) + 1)
        axes[0].plot(ks2, ic_vals2, 's--', color=COLORS[4], linewidth=1.8,
                     markersize=7, alpha=0.8, label=label2)
    axes[0].axvline(r_hat, color='red', linestyle='--', linewidth=2,
                    label=f'r̂ = {r_hat}')
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("IC(k)")
    axes[0].set_title(f"{label1} vs {label2 if ic_vals2 is not None else 'V(k)'}"
                      f" — minimum = r̂")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # ── Right: residual V(k) ──────────────────────────────────────────────
    axes[1].plot(ks, V_vals, 's-', color=COLORS[3], linewidth=2, markersize=8)
    axes[1].axvline(r_hat, color='red', linestyle='--', linewidth=2,
                    label=f'r̂ = {r_hat}')
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("V(k) = residual error")
    axes[1].set_title("Residual error V(k)")
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Number of factors selection  →  r̂ = {r_hat}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_fig('03_information_criterion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_information_criterion.png")


def plot_factors(U_hat, days, r_hat):
    fig, axes = plt.subplots(r_hat, 1, figsize=(14, 3.5 * r_hat), sharex=True)
    if r_hat == 1:
        axes = [axes]

    T = U_hat.shape[1]
    x = np.arange(T)

    # Build yearly tick positions if dates are available
    has_dates = days is not None and len(days) == T
    if has_dates:
        dates_pd  = pd.DatetimeIndex(days)
        years     = sorted(set(dates_pd.year))
        tick_pos  = [int(np.argmax(dates_pd.year == y)) for y in years]
        tick_labs = [str(y) for y in years]
    else:
        tick_pos  = list(range(0, T, max(1, T // 10)))
        tick_labs = [str(p) for p in tick_pos]

    for l in range(r_hat):
        axes[l].plot(x, U_hat[l], color=COLORS[l], linewidth=1.2)
        axes[l].fill_between(x, U_hat[l], alpha=0.15, color=COLORS[l])
        axes[l].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[l].set_ylabel(f"$\\hat{{u}}_{{l={l+1},\\,t}}$", fontsize=11)
        axes[l].set_title(f"Factor {l+1}  —  common systemic shock", fontsize=11)
        axes[l].set_xticks(tick_pos)
        axes[l].set_xticklabels(tick_labs, fontsize=9, rotation=45, ha='right')
        axes[l].set_xlim(0, T - 1)
        axes[l].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Year" if has_dates else "Day $t$")
    fig.suptitle(r"Estimated scalar factors $\hat{u}_{l,t}$ — Common FX market shocks",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_fig('04_factors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_factors.png")


def plot_loadings(B_hat, tau, r_hat, pairs):
    # Split pairs into two readable groups: majors/crosses vs EM/other
    _HIGHLIGHT = {
        'eurusd', 'gbpusd', 'usdjpy', 'eurchf', 'audusd', 'usdcad',
        'audnzd', 'sgdjpy', 'eurgbp',
        'eurnok', 'eursek', 'eurczk',
        'usdpln', 'usdhuf', 'usdtry', 'usdmxn', 'zarjpy', 'usdcnh',
    }
    pairs_lower = [p.lower() for p in pairs]
    grp_a = [p for p in pairs if p.lower() in _HIGHLIGHT]
    grp_b = [p for p in pairs if p.lower() not in _HIGHLIGHT]

    # Index lookup for colour consistency across groups
    pair_idx = {p: i for i, p in enumerate(pairs)}

    fig, axes = plt.subplots(r_hat, 2,
                             figsize=(18, 5 * r_hat),
                             squeeze=False)

    group_labels = ['Majors / Crosses / EM', 'Remaining pairs']
    for l in range(r_hat):
        for col, grp in enumerate([grp_a, grp_b]):
            ax = axes[l][col]
            lines, labels = [], []
            for p in grp:
                i = pair_idx[p]
                ln, = ax.plot(tau, B_hat[i, l, :],
                              label=_format_pair(p), color=COLORS[i % len(COLORS)],
                              linewidth=1.6)
                lines.append(ln)
                labels.append(_format_pair(p))
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel("$\\tau$  [0 = 17:00 NY $\\rightarrow$ 1 = 16:55 NY]")
            ax.set_ylabel(f"$\\hat{{b}}_{{i,{l+1}}}(\\tau)$")
            ax.set_title(f"Factor {l+1} — {group_labels[col]}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(lines, labels,
                      loc='lower right', fontsize=7.5, frameon=True,
                      ncol=2, title='FX pair', title_fontsize=8,
                      columnspacing=0.6, handlelength=1.0)

    fig.suptitle(r"Functional loadings $\hat{b}_{il}(\tau)$ — Intraday factor sensitivity",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(_fig('05_loadings.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_loadings.png")


def plot_decomposition(X, Chi_hat, Xi_hat, mu, tau, pairs,
                       pair_idx=0, day_idx=50,
                       save_as=None,
                       days=None):
    """
    FFM decomposition for a given pair and day.
    Automatically detects if data are log-transformed and
    back-transforms to real volatility exp(x/2) for display.

    Parameters
    ----------
    save_as : filename for the output PNG
    days    : DatetimeIndex — if provided, shows the real date in the title
    """
    if save_as is None:
        save_as = _fig('06_decomposition.png')
    i, t = pair_idx, day_idx

    # Real date label
    if days is not None:
        try:
            date_str = str(days[t])[:10]
        except Exception:
            date_str = f"t={t}"
    else:
        date_str = f"t={t}"

    is_log = mu.min() < 0

    if is_log:
        x_obs  = np.exp(X[i, t, :] / 2)
        mean   = np.exp(mu[i] / 2)
        common = np.exp((mu[i] + Chi_hat[i, t, :]) / 2)
        xi_display = x_obs - common
        ylabel = "|r| avg"
        space_label = "back-transformed log-vol"
    else:
        x_obs      = X[i, t, :]
        mean       = mu[i]
        common     = mean + Chi_hat[i, t, :]
        xi_display = Xi_hat[i, t, :]
        ylabel     = "rv²"
        space_label = "rv²"

    y_raw = X[i, t, :] - mu[i]
    r2    = float(np.clip(
        np.sum(Chi_hat[i, t, :] ** 2) / (np.sum(y_raw ** 2) + 1e-14), 0, 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Style: observed = dashed colour, estimated = solid black
    axes[0, 0].plot(tau, x_obs, color=COLORS[0], linewidth=1.5,
                    linestyle='--', alpha=0.8, label='Observed (dashed)')
    axes[0, 0].plot(tau, mean, 'k-', linewidth=2, label='μ̂_i (mean)')
    axes[0, 0].set_title("Observed curve vs mean profile")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].set_xlabel("τ [0=open → 1=close]")

    axes[0, 1].plot(tau, x_obs, color=COLORS[1], linewidth=1.5,
                    linestyle='--', alpha=0.8, label='Observed (dashed)')
    axes[0, 1].plot(tau, common, 'k-', linewidth=2,
                    label=f'μ̂_i + χ̂_{{it}} (R²={r2:.1%})')
    axes[0, 1].set_title(f"Common component χ̂_{{it}}(τ)\nR² = {r2:.1%}")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylabel(ylabel)
    axes[0, 1].set_xlabel("τ [0=open → 1=close]")

    axes[1, 0].plot(tau, xi_display, color=COLORS[2], linewidth=1.5,
                    linestyle='--', alpha=0.8, label='ξ̂_{it} observed')
    axes[1, 0].plot(tau, np.zeros_like(tau), 'k-', linewidth=1.5, label='zero')
    axes[1, 0].axhline(0, color='black', linewidth=0.8, linestyle='-')
    axes[1, 0].set_title("Idiosyncratic component ξ̂_{it}(τ)\n← microstructure, pair-specific news")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel(ylabel)
    axes[1, 0].set_xlabel("τ [0=open → 1=close]")

    ss_chi = np.sum(Chi_hat[i, t, :] ** 2)
    ss_xi  = np.sum(Xi_hat[i, t, :] ** 2)
    axes[1, 1].bar(['Common\nχ̂', 'Idio\nξ̂'],
                   [ss_chi, ss_xi],
                   color=[COLORS[1], COLORS[2]], alpha=0.8)
    axes[1, 1].set_title(f"Energy decomposition\nR² = {r2:.1%}  ({space_label})")
    axes[1, 1].set_ylabel("Variance")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"FFM Decomposition — {_format_pair(pairs[pair_idx])}  ({date_str})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_as, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_as}")


def plot_decomposition_multipair(X, Chi_hat, Xi_hat, mu, tau, pairs,
                                 pair_indices, day_idx,
                                 days=None,
                                 save_as=None):
    """
    FFM decomposition for multiple pairs on the same day — 4 columns × 3 rows.

    Layout per column (one pair per column):
      Row 0 : Observed (black) + mean profile μ̂_i (dashed)
      Row 1 : Common component μ̂_i + χ̂_{it} (solid) + observed (dotted)
      Row 2 : Idiosyncratic ξ̂_{it} centred at 0

    Title shows the real date and per-column R².
    Designed for presentation: clean, no bar chart, focus on curves.

    Parameters
    ----------
    pair_indices : list of int  — which pairs (by position in `pairs`) to show
    day_idx      : int          — time index of the day to display
    days         : DatetimeIndex or None
    save_as      : output filename
    """
    if save_as is None:
        save_as = _fig('06c_decomposition_multipair.png')
    t = day_idx

    if days is not None:
        try:
            date_str = str(days[t])[:10]
        except Exception:
            date_str = f"t={t}"
    else:
        date_str = f"t={t}"

    is_log = mu.min() < 0
    n_pairs = len(pair_indices)

    fig, axes = plt.subplots(3, n_pairs, figsize=(4.5 * n_pairs, 11))

    for col, i in enumerate(pair_indices):
        pair = _format_pair(pairs[i])

        if is_log:
            x_obs      = np.exp(X[i, t, :] / 2)
            mean       = np.exp(mu[i] / 2)
            common     = np.exp((mu[i] + Chi_hat[i, t, :]) / 2)
            xi_display = x_obs - common
            ylabel     = "|r|"
        else:
            x_obs      = X[i, t, :]
            mean       = mu[i]
            common     = mean + Chi_hat[i, t, :]
            xi_display = Xi_hat[i, t, :]
            ylabel     = "rv²"

        y_raw = X[i, t, :] - mu[i]
        r2    = float(np.clip(
            np.sum(Chi_hat[i, t, :] ** 2) / (np.sum(y_raw ** 2) + 1e-14), 0, 1))

        # ── Row 0: observed (dashed) + mean (solid black) ────────────────
        ax = axes[0, col]
        ax.plot(tau, x_obs, color=COLORS[0], linewidth=1.5,
                linestyle='--', alpha=0.8, label='Observed')
        ax.plot(tau, mean, 'k-', linewidth=2.0, label='Mean  μ̂_i')
        ax.set_title(pair, fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel("Observed vs μ̂_i\n" + ylabel, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_xlabel("τ", fontsize=8)

        # ── Row 1: observed (same colour as row 0) + μ̂+χ̂ (solid black) ──
        ax = axes[1, col]
        ax.plot(tau, x_obs, color=COLORS[0], linewidth=1.2,
                linestyle='--', alpha=0.75, label='Observed')
        ax.plot(tau, common, 'k-', linewidth=2.0,
                label=f'μ̂_i + χ̂_{{it}}  (R²={r2:.1%})')
        if col == 0:
            ax.set_ylabel("μ̂_i + χ̂_{it}\n" + ylabel, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_xlabel("τ", fontsize=8)

        # ── Row 2: idiosyncratic ──────────────────────────────────────────
        ax = axes[2, col]
        ax.plot(tau, xi_display, color=COLORS[2], linewidth=1.4,
                label='Idiosyncratic  ξ̂_{it}')
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.fill_between(tau, xi_display, alpha=0.2, color=COLORS[2])
        if col == 0:
            ax.set_ylabel("Idiosyncratic  ξ̂_{it}\n" + ylabel, fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_xlabel("τ [0=open → 1=close]", fontsize=8)

    # Shared y-scale for row 2 across all pairs
    xi_lines = [axes[2, c].get_lines()[0].get_ydata()
                for c in range(n_pairs)
                if len(axes[2, c].get_lines()) > 0]
    if xi_lines:
        xi_abs_max = max(np.nanmax(np.abs(v)) for v in xi_lines) * 1.1
        for c in range(n_pairs):
            axes[2, c].set_ylim(-xi_abs_max, xi_abs_max)

    fig.suptitle(
        f"FFM Variance Decomposition — Crisis day: {date_str}\n"
        "$x_{{it}}(\\tau) = \\hat{{\\mu}}_i(\\tau) + "
        "\\hat{{\\chi}}_{{it}}(\\tau) + \\hat{{\\xi}}_{{it}}(\\tau)$",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_as, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_as}")


def plot_r2_heatmap(X, Chi_hat, mu, pairs, U_hat=None, days=None):
    """
    Heatmap R²_{it} — rows = FX pairs (sorted by mean R²), columns = days.
    Log10 scale, yearly x-ticks, year separator lines.
    """
    import pandas as pd

    N, T, M = X.shape
    Y = X - mu[:, np.newaxis, :]

    # ── Compute R²_{it} = ||χ̂_{it}||² / ||y_{it}||²  ───────────────────
    # Sum of squares over τ — consistent with the Frobenius R²
    # (vs np.var which centres over τ and gives ~0.8% globally)
    R2 = np.zeros((N, T))
    for i in range(N):
        ss_chi = np.sum(Chi_hat[i] ** 2, axis=1)
        ss_y   = np.sum(Y[i] ** 2,       axis=1)
        R2[i]  = np.clip(ss_chi / (ss_y + 1e-14), 0, 1)

    sort_order   = np.argsort(R2.mean(axis=1))[::-1]
    R2_sorted    = R2[sort_order]
    pairs_sorted = [_format_pair(pairs[i]) for i in sort_order]

    # Append cross-sectional mean row at the bottom
    mean_row = R2.mean(axis=0, keepdims=True)              # (1, T)
    R2_plot  = np.vstack([R2_sorted, mean_row])            # (N+1, T)
    n_rows   = N + 1
    y_labels = pairs_sorted + ['Cross-sect. mean']

    # ── X-axis: yearly ticks + separator positions ────────────────────────
    has_dates = days is not None and len(days) == T
    if has_dates:
        dates_pd    = pd.DatetimeIndex(days)
        years       = sorted(set(dates_pd.year))
        tick_pos    = [int(np.argmax(dates_pd.year == y)) for y in years]
        tick_labels = [str(y) for y in years]
        sep_pos     = tick_pos[1:]
    else:
        tick_pos    = list(range(0, T, max(1, T // 10)))
        tick_labels = [str(p) for p in tick_pos]
        sep_pos     = []

    # ── Figure (landscape) ────────────────────────────────────────────────
    row_h  = max(0.27, min(0.40, 10 / N))
    fig_h  = min(n_rows * row_h + 2.0, 12)
    fig, ax = plt.subplots(figsize=(22, fig_h))

    im = ax.imshow(R2_plot, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1,
                   extent=[0, T, n_rows + 0.5, 0.5],
                   interpolation='nearest')

    # Separator between pairs and mean row
    ax.axhline(N + 0.5, color='white', linewidth=2.5, zorder=4)

    # Year separator lines
    for xp in sep_pos:
        ax.axvline(xp, color='white', linewidth=0.6, alpha=0.5, zorder=3)

    # Pair labels (y-axis) — mean row in bold
    ax.set_yticks(range(1, n_rows + 1))
    tick_objects = ax.set_yticklabels(y_labels, fontsize=8.5)
    tick_objects[-1].set_fontweight('bold')
    tick_objects[-1].set_fontsize(9)
    ax.tick_params(axis='y', length=0)

    # Date labels (x-axis)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.tick_params(axis='x', length=4)
    ax.set_xlim(0, T)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, aspect=35)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
    cbar.set_label('$R^2_{it}$', fontsize=10, labelpad=8)
    cbar.ax.tick_params(length=3)

    ax.set_xlabel('Year', fontsize=11, labelpad=6)
    ax.set_ylabel('FX pair  (sorted by mean $R^2$, high → low)',
                  fontsize=10, labelpad=8)

    fig.suptitle(
        "$R^2_{it} = "
        "\\|\\hat{\\chi}_{it}\\|^2 / \\|y_{it}\\|^2$",
        fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(_fig('07_r2_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_r2_heatmap.png")


def plot_residual_heatmap(X, Chi_hat, mu, pairs, days=None,
                          save_as=None):
    """
    Heatmap: FX pairs × days — colour = ||y_it - χ̂_it||² / ||y_it||²
    i.e. the fraction of variance NOT explained by the factor model.
    Low values (blue) = day well represented by common factors.
    High values (red) = day dominated by idiosyncratic component.

    Rows : FX pairs (sorted by mean unexplained variance, worst on top)
    Cols : time (yearly ticks)
    """
    if save_as is None:
        save_as = _fig('08_residual_heatmap.png')
    N, T, M = X.shape
    Y = X - mu[:, np.newaxis, :]   # centred

    # Unexplained fraction per (pair, day): 1 - R²_it
    unexp = np.zeros((N, T))
    for i in range(N):
        var_y   = np.var(Y[i], axis=1) + 1e-14          # (T,)
        var_chi = np.var(Chi_hat[i], axis=1)             # (T,)
        unexp[i] = 1.0 - np.clip(var_chi / var_y, 0, 1) # (T,)

    # Sort pairs by mean unexplained variance (worst at top)
    order      = np.argsort(unexp.mean(axis=1))[::-1]
    unexp_sort = unexp[order]
    pairs_sort = [_format_pair(pairs[i]) for i in order]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.08})

    # ── Top: heatmap ──────────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(unexp_sort, aspect='auto', cmap='RdYlBu_r',
                   vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Unexplained fraction  (1 − R²_it)',
                 fraction=0.02, pad=0.01)

    ax.set_yticks(range(N))
    ax.set_yticklabels(pairs_sort, fontsize=8)
    ax.set_title("Residual Heatmap — Unexplained variance per (pair, day)\n"
                 "Blue = well captured by factors  |  Red = idiosyncratic dominated",
                 fontsize=11, fontweight='bold')

    # Yearly x-ticks
    if days is not None:
        days_pd = pd.DatetimeIndex(days)
        years   = sorted(set(days_pd.year))
        tpos    = [int(np.argmax(days_pd.year == y)) for y in years]
        ax.set_xticks(tpos)
        ax.set_xticklabels([str(y) for y in years], fontsize=8, rotation=45)
    ax.set_xlabel("Year", fontsize=9)

    # ── Bottom: cross-sectional mean unexplained variance ────────────────────
    ax2 = axes[1]
    mean_unexp = unexp.mean(axis=0)   # (T,)
    ax2.fill_between(range(T), mean_unexp, alpha=0.6, color='#C0392B')
    ax2.plot(range(T), mean_unexp, color='#C0392B', linewidth=0.8)
    ax2.set_xlim(0, T - 1)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Mean\n1−R²", fontsize=8)
    ax2.grid(True, alpha=0.2)
    if days is not None:
        ax2.set_xticks(tpos)
        ax2.set_xticklabels([str(y) for y in years], fontsize=8, rotation=45)

    fig.suptitle("Factor Model Coverage — How well common factors explain each day",
                 fontsize=12, fontweight='bold')
    plt.savefig(save_as, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_as}")



def plot_decomposition_multiday(X, Chi_hat, Xi_hat, mu, tau, pairs,
                                pair_idx=0,
                                day_indices=None,
                                day_labels=None,
                                days=None,
                                save_as=None):
    """
    FFM decomposition for ONE pair across MULTIPLE days — side by side.

    Parameters
    ----------
    pair_idx    : which pair to show
    day_indices : list of day indices. If None, auto-picks 3 calm + 3 crisis.
    day_labels  : optional list of header labels (e.g. ['Crisis: 2008-10-10', 'Calm']).
                  If None, uses the date string from `days`.
    days        : DatetimeIndex — if provided, shows real dates in column headers
    save_as     : output filename
    """
    if save_as is None:
        save_as = _fig('06d_decomposition_multiday.png')
    i    = pair_idx
    pair = _format_pair(pairs[i])
    is_log = mu.min() < 0

    if day_indices is None:
        # auto-select: cannot pick without U_hat here, use variance of Y
        N_, T_, M_ = X.shape
        Y_pair = X[i] - mu[i]
        var_day = np.var(Y_pair, axis=1)           # (T,)
        n_calm   = 3
        n_crisis = 3
        calm_idx   = list(np.argsort(var_day)[:n_calm])
        crisis_idx = list(np.argsort(var_day)[::-1][:n_crisis])
        day_indices = calm_idx + crisis_idx
        labels_auto = (['calm'] * n_calm) + (['crisis'] * n_crisis)
    else:
        labels_auto = None

    n_days = len(day_indices)

    fig, axes = plt.subplots(3, n_days, figsize=(4.2 * n_days, 11))
    if n_days == 1:
        axes = axes[:, np.newaxis]

    for col, t in enumerate(day_indices):
        # Date label
        if days is not None:
            try:
                date_str = str(days[t])[:10]
            except Exception:
                date_str = f"t={t}"
        else:
            date_str = f"t={t}"

        # Column title — priority: day_labels > labels_auto > date_str
        if day_labels is not None and col < len(day_labels):
            lbl       = day_labels[col]
            col_title = f"{lbl}\n{date_str}"
            is_crisis = 'crisis' in lbl.lower() or 'Crisis' in lbl
            title_col = '#C0392B' if is_crisis else '#2C3E50'
        elif labels_auto is not None:
            day_type  = labels_auto[col]
            col_title = f"{date_str}\n({day_type})"
            title_col = '#E74C3C' if day_type == 'crisis' else '#2C3E50'
        else:
            col_title = date_str
            title_col = '#2C3E50'

        if is_log:
            x_obs      = np.exp(X[i, t, :] / 2)
            mean       = np.exp(mu[i] / 2)
            common     = np.exp((mu[i] + Chi_hat[i, t, :]) / 2)
            xi_display = x_obs - common
            ylabel     = "|r|"
        else:
            x_obs      = X[i, t, :]
            mean       = mu[i]
            common     = mean + Chi_hat[i, t, :]
            xi_display = Xi_hat[i, t, :]
            ylabel     = "rv²"

        y_raw = X[i, t, :] - mu[i]
        r2    = float(np.clip(
            np.sum(Chi_hat[i, t, :] ** 2) / (np.sum(y_raw ** 2) + 1e-14), 0, 1))

        # ── Row 0: observed (dashed colour) + mean (solid black) ─────────
        ax = axes[0, col]
        ax.plot(tau, x_obs, color=COLORS[0], linewidth=1.4,
                linestyle='--', alpha=0.8, label='Observed')
        ax.plot(tau, mean, 'k-', linewidth=1.8, label='μ̂_i')
        ax.set_title(col_title, fontsize=9, fontweight='bold', color=title_col)
        if col == 0:
            ax.set_ylabel(f"Observed vs μ̂_i\n{ylabel}", fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        ax.set_xlabel("τ", fontsize=7)
        ax.tick_params(labelsize=7)

        # ── Row 1: observed (dashed, same colour as row 0) + μ̂+χ̂ (solid black)
        ax = axes[1, col]
        ax.plot(tau, x_obs, color=COLORS[0], linewidth=1.0,
                linestyle='--', alpha=0.75, label='Observed')
        ax.plot(tau, common, 'k-', linewidth=1.8,
                label=f'μ̂_i + χ̂_{{it}}  (R²={r2:.0%})')
        if col == 0:
            ax.set_ylabel(f"μ̂_i + χ̂_{{it}}\n{ylabel}", fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        ax.set_xlabel("τ", fontsize=7)
        ax.tick_params(labelsize=7)

        # ── Row 2: idiosyncratic (dashed) + zero (solid black) ────────────
        ax = axes[2, col]
        ax.plot(tau, xi_display, color=COLORS[2], linewidth=1.2,
                linestyle='--', alpha=0.8)
        ax.fill_between(tau, xi_display, alpha=0.12, color=COLORS[2])
        ax.plot(tau, np.zeros_like(tau), 'k-', linewidth=1.2, alpha=0.7)
        if col == 0:
            ax.set_ylabel(f"Idiosyncratic ξ̂_{{it}}\n{ylabel}", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("τ [0=open → 1=close]", fontsize=7)
        ax.tick_params(labelsize=7)

    # Shared y-scale for row 0 (observed vs mean) across all columns
    obs_all = [axes[0, c].get_lines()[0].get_ydata() for c in range(n_days)
               if len(axes[0, c].get_lines()) > 0]
    if obs_all:
        obs_min = min(np.nanmin(v) for v in obs_all)
        obs_max = max(np.nanmax(v) for v in obs_all)
        pad = (obs_max - obs_min) * 0.08
        for c in range(n_days):
            axes[0, c].set_ylim(obs_min - pad, obs_max + pad)

    # Shared y-scale for row 2 (idiosyncratic) across all columns
    xi_all = [axes[2, c].get_lines()[0].get_ydata() for c in range(n_days)
              if len(axes[2, c].get_lines()) > 0]
    if xi_all:
        xi_abs_max = max(np.nanmax(np.abs(v)) for v in xi_all) * 1.1
        for c in range(n_days):
            axes[2, c].set_ylim(-xi_abs_max, xi_abs_max)

    fig.suptitle(
        f"FFM Decomposition — {pair}  across {n_days} days\n"
        "$x_{{it}}(\\tau) = \\hat{{\\mu}}_i(\\tau) + "
        "\\hat{{\\chi}}_{{it}}(\\tau) + \\hat{{\\xi}}_{{it}}(\\tau)$\n"
        "Red headers = crisis days  |  Dark headers = calm days",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_as, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_as}")


def plot_decomp_crisis_calm_logspace(
    X, Chi_hat, Xi_hat, mu, tau, pairs,
    pair_name, day_indices, day_labels,
    days=None, save_as=None
):
    """
    Decomposition plot staying in log(r²+ε) space — for thesis inclusion.

    Layout: 3 rows × n_days columns
      Row 0: Observed x_it(τ)  vs  mean profile μ̂_i(τ)
      Row 1: Observed x_it(τ)  vs  factor model μ̂_i(τ) + χ̂_it(τ)
      Row 2: Idiosyncratic component ξ̂_it(τ)  (centred at 0)

    Each row shares a common y-axis across all columns.
    Under each column (in the bottom xlabel): ‖ξ̂_it‖₂.

    Parameters
    ----------
    pair_name   : str — e.g. 'eurusd' or 'EUR/USD'
    day_indices : list[int] — day positions in the panel [crisis1, crisis2, calm1, calm2]
    day_labels  : list[str] — column headers, e.g. ['Crisis', 'Crisis', 'Calm', 'Calm']
    """
    # Resolve pair index
    pair_name_key = pair_name.lower().replace('/', '').replace('_', '')
    pair_idx = None
    for j, p in enumerate(pairs):
        if p.lower().replace('/', '').replace('_', '') == pair_name_key:
            pair_idx = j
            break
    if pair_idx is None:
        print(f"  ⚠ Pair '{pair_name}' not found — skipping")
        return

    i        = pair_idx
    n_days   = len(day_indices)
    pair_fmt = _format_pair(pair_name)

    if save_as is None:
        save_as = _fig(f'06e_{pair_name_key}_crisis_calm.png')

    # ── Precompute all curves ─────────────────────────────────────────────────
    obs_curves    = []   # x_it(τ) in log space
    common_curves = []   # μ̂_i + χ̂_it(τ)
    xi_curves     = []   # ξ̂_it(τ)
    norms_xi      = []   # ‖ξ̂_it‖₂
    r2_vals       = []

    for t in day_indices:
        x_obs   = X[i, t, :]                        # observed log-vol
        mean_mu = mu[i]                              # mean profile
        common  = mu[i] + Chi_hat[i, t, :]          # factor model
        xi      = Xi_hat[i, t, :]                   # idiosyncratic

        obs_curves.append(x_obs)
        common_curves.append(common)
        xi_curves.append(xi)
        norms_xi.append(float(np.linalg.norm(xi)))

        y_raw = X[i, t, :] - mu[i]
        r2    = float(np.clip(
            np.var(Chi_hat[i, t, :]) / (np.var(y_raw) + 1e-14), 0, 1))
        r2_vals.append(r2)

    # ── Global y limits per row ───────────────────────────────────────────────
    # Row 0 & 1: observed + mean/factor model — use same range for both rows
    all_row01 = obs_curves + common_curves + [mu[i]]
    y01_min = min(float(np.nanmin(v)) for v in all_row01)
    y01_max = max(float(np.nanmax(v)) for v in all_row01)
    pad01   = (y01_max - y01_min) * 0.08
    ylim01  = (y01_min - pad01, y01_max + pad01)

    # Row 2: idiosyncratic (symmetric around 0)
    xi_abs_max = max(float(np.nanmax(np.abs(v))) for v in xi_curves) * 1.15
    ylim2      = (-xi_abs_max, xi_abs_max)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, n_days, figsize=(4.5 * n_days, 11))
    if n_days == 1:
        axes = axes[:, np.newaxis]

    OBS_COLOR    = '#2980B9'   # blue  — observed curve
    MODEL_COLOR  = 'black'     # black — mean / factor model
    IDIO_COLOR   = '#C0392B'   # red   — idiosyncratic

    for col, t in enumerate(day_indices):
        lbl      = day_labels[col] if col < len(day_labels) else ''
        is_crisis = 'crisis' in lbl.lower()
        title_col = '#C0392B' if is_crisis else '#27AE60'

        date_str = f"t={t}"
        if days is not None:
            try:
                date_str = str(days[t])[:10]
            except Exception:
                pass
        col_title = f"{lbl}\n{date_str}"
        norm_str  = f"$\\|\\hat{{\\xi}}_{{it}}\\|_2 = {norms_xi[col]:.3f}$"

        # ── Row 0: observed vs mean ────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(tau, obs_curves[col], color=OBS_COLOR, linewidth=1.4,
                linestyle='--', alpha=0.85, label='Observed')
        ax.plot(tau, mu[i], color=MODEL_COLOR, linewidth=1.8,
                label='$\\hat{\\mu}_i$')
        ax.set_title(col_title, fontsize=9, fontweight='bold', color=title_col)
        ax.set_ylim(ylim01)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('$\\tau$', fontsize=7)
        if col == 0:
            ax.set_ylabel(
                "Observed vs $\\hat{\\mu}_i$\n$\\log(r^2 + \\varepsilon)$",
                fontsize=8)
            ax.legend(fontsize=7, loc='best')

        # ── Row 1: observed vs factor model ──────────────────────────────
        ax = axes[1, col]
        ax.plot(tau, obs_curves[col], color=OBS_COLOR, linewidth=1.4,
                linestyle='--', alpha=0.85, label='Observed')
        ax.plot(tau, common_curves[col], color=MODEL_COLOR, linewidth=1.8,
                label='Factor model')
        ax.set_ylim(ylim01)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('$\\tau$', fontsize=7)
        if col == 0:
            ax.set_ylabel(
                "Observed vs factor model\n$\\log(r^2 + \\varepsilon)$",
                fontsize=8)
            ax.legend(fontsize=7, loc='best')

        # ── Row 2: idiosyncratic + norm in xlabel ─────────────────────────
        ax = axes[2, col]
        ax.plot(tau, xi_curves[col], color=IDIO_COLOR, linewidth=1.2, alpha=0.85)
        ax.fill_between(tau, xi_curves[col], alpha=0.15, color=IDIO_COLOR)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylim(ylim2)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(
            f"$\\tau$  [0 = open $\\rightarrow$ 1 = close]\n{norm_str}",
            fontsize=7)
        if col == 0:
            ax.set_ylabel(
                "Idiosyncratic $\\hat{\\xi}_{it}$\n$\\log(r^2 + \\varepsilon)$",
                fontsize=8)

    fig.suptitle(
        f"FFM Decomposition — {pair_fmt}\n"
        "$x_{it}(\\tau) = \\hat{\\mu}_i(\\tau) + "
        "\\hat{\\chi}_{it}(\\tau) + \\hat{\\xi}_{it}(\\tau)$  ",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.savefig(save_as, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_as}")


def plot_crisis_calm_fourpairs(results, save_prefix='06e'):
    """
    Produce 4 crisis/calm decomposition plots (log space) for:
    EUR/USD, AUD/CHF, EUR/CHF, USD/TRY

    Crisis dates
    ─────────────
    EUR/CHF  : crisis_1 = 2015-01-15 (SNB floor removal), crisis_2 = 2020-03-09
    Others   : crisis_1 = argmax ‖x_it − μ̂_i‖₂ per pair,  crisis_2 = 2020-03-09

    Calm dates (common to all 4 pairs)
    ────────────────────────────────────
    2 days that minimise the combined normalised L2-norm across the 4 pairs,
    separated by at least 30 trading days and at least 20 days from any crisis.
    """
    X       = results['X']
    Chi_hat = results['Chi_hat']
    Xi_hat  = results['Xi_hat']
    mu      = results['mu']
    tau     = results['tau']
    pairs   = results['pairs']
    days    = results['days']
    T       = X.shape[1]

    TARGET_KEYS    = ['eurusd', 'audjpy', 'eurchf', 'usdtry']
    TARGET_DISPLAY = ['EUR/USD', 'AUD/JPY', 'EUR/CHF', 'USD/TRY']

    # Map to panel indices
    norm_map = {p.lower().replace('/', '').replace('_', ''): j
                for j, p in enumerate(pairs)}
    idx_map = {}
    for tk in TARGET_KEYS:
        if tk in norm_map:
            idx_map[tk] = norm_map[tk]
        else:
            print(f"  ⚠ Pair '{tk}' not found in panel — skipping")

    if not idx_map:
        print("  ⚠ None of the 4 target pairs found — skipping 06e plots")
        return

    # ── Helper: find closest day index to a target date ──────────────────────
    def find_day(target_date):
        ts    = pd.Timestamp(target_date)
        diffs = np.abs((pd.DatetimeIndex(days) - ts).days)
        idx   = int(np.argmin(diffs))
        actual = str(days[idx])[:10]
        if diffs[idx] == 0:
            print(f"    {target_date} → t={idx} (exact)")
        elif diffs[idx] <= 5:
            print(f"    {target_date} → t={idx} ({actual})  "
                  f"[closest, {diffs[idx]}d away]")
        else:
            print(f"    ⚠ {target_date}: closest date is {actual} "
                  f"({diffs[idx]}d away) — using it")
        return idx

    day_covid = find_day('2020-03-09')

    # ── Per-pair L2 norm of centred curves (volatility proxy) ─────────────────
    vol_norm = {}
    for tk, j in idx_map.items():
        vol_norm[tk] = np.linalg.norm(X[j] - mu[j], axis=1)   # (T,)

    # ── crisis_1 per pair ─────────────────────────────────────────────────────
    crisis1 = {}
    for tk, j in idx_map.items():
        if tk == 'eurchf':
            crisis1[tk] = find_day('2015-01-15')
        else:
            crisis1[tk] = int(np.argmax(vol_norm[tk]))
            print(f"    {tk}: crisis_1 = t={crisis1[tk]}"
                  f" ({str(days[crisis1[tk]])[:10]})"
                  f"  [argmax vol]")

    # ── Common calm days ──────────────────────────────────────────────────────
    # Build list of days to exclude (crisis windows ± 20d)
    exclude = set()
    for t_c in list(crisis1.values()) + [day_covid]:
        for d in range(max(0, t_c - 20), min(T, t_c + 21)):
            exclude.add(d)

    # Combined normalised vol (sum over available pairs)
    combined_vol = np.zeros(T)
    for tk, j in idx_map.items():
        v = vol_norm[tk]
        combined_vol += v / (float(np.max(v)) + 1e-14)

    calm_ranked = list(np.argsort(combined_vol))
    calm_days   = []
    for c in calm_ranked:
        if c in exclude:
            continue
        if all(abs(c - prev) > 30 for prev in calm_days):
            calm_days.append(c)
        if len(calm_days) == 2:
            break

    # Fallback: if fewer than 2 found, relax the 20-day exclusion window
    if len(calm_days) < 2:
        for c in calm_ranked:
            if c in crisis1.values() or c == day_covid:
                continue
            if all(abs(c - prev) > 30 for prev in calm_days):
                calm_days.append(c)
            if len(calm_days) == 2:
                break

    print(f"\n  Common calm days:")
    for c in calm_days:
        print(f"    t={c}  ({str(days[c])[:10]})  "
              f"combined_vol={combined_vol[c]:.3f}")

    # ── Produce one image per pair ─────────────────────────────────────────────
    for tk, td in zip(TARGET_KEYS, TARGET_DISPLAY):
        if tk not in idx_map:
            continue

        c1  = crisis1[tk]
        c1_date = str(days[c1])[:10]
        c2_date = str(days[day_covid])[:10]

        day_indices = [c1, day_covid] + calm_days[:2]

        if tk == 'eurchf':
            c1_label = 'Crisis  (SNB, 2015-01-15)'
        else:
            c1_label = f'Crisis  (max vol,  {c1_date})'

        day_labels = [
            c1_label,
            f'Crisis  (COVID-19,  {c2_date})',
            f'Calm  ({str(days[calm_days[0]])[:10]})',
            f'Calm  ({str(days[calm_days[1]])[:10]})' if len(calm_days) > 1
            else 'Calm',
        ]

        save_as = _fig(f'{save_prefix}_{tk}_crisis_calm.png')
        plot_decomp_crisis_calm_logspace(
            X, Chi_hat, Xi_hat, mu, tau, pairs,
            pair_name   = tk,
            day_indices = day_indices,
            day_labels  = day_labels,
            days        = days,
            save_as     = save_as,
        )


def plot_forecast(X, forecast, tau, pairs, pair_idx=0, n_context=10):
    """
    T+1 forecast of the intraday curve.
    Back-transforms exp(x/2) if data are log-transformed.
    """
    i     = pair_idx
    T     = X.shape[1]
    is_log = X.min() < 0

    if is_log:
        to_vol = lambda x: np.exp(x / 2)
        ylabel = "|r| avg  (= exp(x/2))"
        fc_vol   = to_vol(forecast[i])
        fc_lo    = to_vol(forecast[i] * 0.8)
        fc_hi    = to_vol(forecast[i] * 1.2)
        last_vol = to_vol(X[i, -1])
        ctx_vols = [to_vol(X[i, t]) for t in range(T - n_context, T)]
    else:
        to_vol = lambda x: x
        ylabel = "rv²"
        fc_vol  = to_vol(forecast[i])
        fc_lo   = fc_vol * 0.8
        fc_hi   = fc_vol * 1.2
        last_vol = to_vol(X[i, -1])
        ctx_vols = [to_vol(X[i, t]) for t in range(T - n_context, T)]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Context: last 10 days in grey (increasingly opaque)
    for k, vol_k in enumerate(ctx_vols):
        alpha = 0.06 + 0.30 * k / n_context
        ax.plot(tau, vol_k, color='gray', alpha=alpha, linewidth=0.8)

    ax.plot(tau, last_vol, color='black', linewidth=2,
            label='Last observed day', zorder=5)
    ax.plot(tau, fc_vol, color=COLORS[0], linewidth=2.5,
            linestyle='--', label='T+1 Forecast', zorder=6)
    ax.fill_between(tau, fc_lo, fc_hi,
                    alpha=0.2, color=COLORS[0], label='± 20%')

    ax.set_xlabel("τ [0=open → 1=close]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"T+1 intraday curve forecast — {pairs[pair_idx]}",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_fig('08_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_forecast.png")


# =============================================================================
# MAIN PIPELINE — PASTE THIS BLOCK IN YOUR NOTEBOOK
# =============================================================================

def run_ffm(vol_matrices: dict,
            k_max: int = 10,
            winsorize: bool = False,
            upper_pct: float = 99.5,
            min_coverage: float = 0.8,
            run_forecast: bool = False,
            date_start: str = None,
            date_end: str = None,
            force_dates: list = None,
            r_forced: int = None,
            silent : bool = False,
            mu_ref: dict = None
            ):
    """
    Full FFM pipeline on your intraday volatility matrices.

    Parameters
    ----------
    vol_matrices : dict  {str: pd.DataFrame}
    k_max        : int   — maximum number of factors to test (default 10)
    winsorize    : bool  — apply winsorisation (default True)
    upper_pct    : float — winsorisation percentile (default 99.5)
    min_coverage : float — minimum day coverage (default 0.8)
    run_forecast : bool  — run T+1 forecast (slow, default False)
    date_start   : str   — restrict sample start 'YYYY-MM-DD' (default: all)
    date_end     : str   — restrict sample end   'YYYY-MM-DD' (default: all)

    Returns
    --------
    dict with all model outputs.
    """

    print("=" * 65)
    print("  FUNCTIONAL FACTOR MODEL — Intraday FX Volatility")
    print("=" * 65)

    # ── 0. Build panel ────────────────────────────────────────────────────
    print("\n[0] Building panel 3D...")
    X, days, tau, pairs = build_panel_from_matrices(vol_matrices, min_coverage,
                                                     force_dates=force_dates)
    N, T, M = X.shape

    # ── Date range filter ─────────────────────────────────────────────────
    if date_start is not None or date_end is not None:
        mask = np.ones(T, dtype=bool)
        if date_start is not None:
            ts = pd.Timestamp(date_start)
            mask &= pd.DatetimeIndex(days) >= ts
        if date_end is not None:
            te = pd.Timestamp(date_end)
            mask &= pd.DatetimeIndex(days) <= te
        X    = X[:, mask, :]
        days = pd.DatetimeIndex(days)[mask]
        T    = int(mask.sum())
        print(f"  Date filter: {date_start or 'start'} → {date_end or 'end'}"
              f"  →  {T} days retained")

    # ── Winsorisation optionnelle ─────────────────────────────────────────
    if winsorize:
        print("\n[0b] Winsorisation of extreme values...")
        X = winsorize_panel(X, upper_pct)

    # ── Diagnostic ───────────────────────────────────────────────────────
    diagnostic_panel(X, pairs, tau)

    # ── 1. Centrage + normalisation ───────────────────────────────────────
    print("\n[1] Panel centering...")
    if mu_ref is not None:
        mu_aligned = np.array([mu_ref[p] for p in pairs])  # (N, M)
        Y, mu = center_panel(X, mu=mu_aligned)
    else:
        Y, mu = center_panel(X)
    print(f"  Y.shape = {Y.shape}  |  std(Y) = {np.std(Y):.3e}")
    if not silent:
        plot_mean_profiles(mu, tau, pairs, days, X=X)
        plot_mean_profiles_log(mu, tau, pairs, days, X=X)

    # Normalisation: Y_norm = Y / σ  →  λ̂_norm = λ̂ / σ² ~ O(1)
    # REQUIRED: without it λ̂ ~ 10⁻¹² and the loadings are numerically unstable
    Y_norm, sigma = normalize_panel(Y)
    print(f"  σ = {sigma:.3e}  →  Y_norm ~ O(1), λ̂_norm ~ O(1)")

    # ── 2. Gram matrix (on Y_norm) ───────────────────────────────────────
    print("\n[2] Building Gram matrix F (on normalised Y)...")
    F = build_gram_matrix(Y_norm)
    print(f"  F.shape = {F.shape}  |  Trace = {np.trace(F):.4e}  (should be ~ T)")
    if not silent:
        plot_scree(F, T, k_max=min(k_max + 5, 20))

    # ── 3. Factor number selection (on Y_norm) ───────────────────────────
    # Two criteria run in parallel — all results printed for comparison:
    # - 'ratio' : spectral elbow (λ̂_l/λ̂_{l+1}), robust when N << T → PRIMARY
    # - 'IC1a'  : penalty = √((N+T)/NT) × log(NT/(N+T))
    # - 'IC2a'  : penalty = √((N+T)/NT) × log(min(N,T))   ← ≈ IC1a when T >> N
    # NOTE: with N=34, T=3977 → g_IC1a = 0.1044, g_IC2a = 0.1046 (diff = 0.25%)
    #       IC1a and IC2a give the same r̂ in this asymmetric regime.
    #       Both over-penalise vs ΔV ≈ 0.008 → ratio criterion is preferred.
    print(f"\n[3] Factor number selection r̂ (k_max={k_max})...")

    r_ratio, ratios,   V_vals_r, eigen_cache = select_r(Y_norm, F, k_max=k_max, criterion='ratio')
    r_ic1a, ic_vals1a, V_vals,   _           = select_r(Y_norm, F, k_max=k_max, criterion='IC1a')
    r_ic2a, ic_vals2a, _,        _           = select_r(Y_norm, F, k_max=k_max, criterion='IC2a')
    r_bic3, ic_vals_bic3, V_vals_bic3, _     = select_r(Y_norm, F, k_max=k_max, criterion='BIC3')
    r_GR, ic_vals_GR, V_vals_GR, _           = select_r(Y_norm, F, k_max=k_max, criterion='GR')

    r_hat = r_ratio
    print(f"\n  ┌─ Factor selection summary ───────────────────────────")
    print(f"  │  Spectral ratio : r̂ = {r_ratio}")
    print(f"  │  IC1a           : r̂ = {r_ic1a}")
    print(f"  │  IC2a           : r̂ = {r_ic2a}")
    print(f"  │  BIC3            : r̂ = {r_bic3}")
    print(f"  └─ → Using r̂ = {r_hat}  (spectral ratio, primary)")

    if r_forced is not None:
        r_hat = int(r_forced)
        print(f" → Using r̂ = {r_hat}  (FORCED by user)")
    else:
        print(f" → Using r̂ = {r_hat}  (spectral ratio)")

    N_, T_, M_ = Y_norm.shape
    g_ic1a = np.sqrt((N_ + T_) / (N_ * T_)) * np.log(N_ * T_ / (N_ + T_))
    g_ic2a = np.sqrt((N_ + T_) / (N_ * T_)) * np.log(min(N_, T_))
    print(f"  IC1a penalty g = {g_ic1a:.6f}  |  IC2a penalty g = {g_ic2a:.6f}"
          f"  (ratio = {g_ic1a/g_ic2a:.6f})")

    print(f"  Ratios λ̂_l/λ̂_{{l+1}}:")
    for k in range(min(k_max - 1, 8)):
        marker = " ← r̂" if k + 1 == r_hat else ""
        print(f"     k={k+1:2d}: ratio = {ratios[k]:.3f}   V = {V_vals[k]:.6f}{marker}")
    
    if not silent:
        plot_ic(ic_vals1a, V_vals, r_hat, ic_vals2=ic_vals2a,
                label1='IC1a', label2='IC2a')

    # ── 4. Final estimation (on Y_norm, then rescale) ────────────────────
    print(f"\n[4] FFM estimation with r̂ = {r_hat}...")
    U_hat, B_hat_norm, Chi_hat_norm, Xi_hat_norm, lam = estimate_ffm(Y_norm, F, r_hat, _eigen=eigen_cache)

    # Rescale to original Y units (= rv units)
    B_hat   = B_hat_norm   * sigma
    Chi_hat = Chi_hat_norm * sigma
    Xi_hat  = Xi_hat_norm  * sigma

    print(f"  U_hat  : {U_hat.shape}")
    print(f"  B_hat  : {B_hat.shape}")
    print(f"  Chi_hat: {Chi_hat.shape}")

    u_means = U_hat.mean(axis=1)  # mean over T for each factor l
    print(f"\n  Mean of U_hat per factor:")
    for l, m in enumerate(u_means):
        print(f"    Factor {l+1}: {m:.6f}")

    print(f"\n  R² per pair  (||χ̂_i||² / ||y_i||²):")
    print(f"  {'Pair':12s}  {'Mean R²':>10s}  {'Max R² (day)':>14s}")
    print("  " + "-" * 40)
    for i, pair in enumerate(pairs):
        ss_chi   = np.sum(Chi_hat[i] ** 2, axis=1)   # (T,)
        ss_y     = np.sum(Y[i] ** 2,       axis=1)   # (T,)
        r2_serie = np.clip(ss_chi / (ss_y + 1e-14), 0, 1)
        print(f"  {pair:12s}  {r2_serie.mean():>9.2%}  {r2_serie.max():>13.1%}")

    # Find a crisis day for decomposition.
    #
    # IMPORTANT — why not argmax(R2_cross) ?
    # R2_cross[t] = Σᵢ Var_τ(χ̂ᵢₜ) / Var_τ(yᵢₜ)
    # The loadings have a spike at τ≈0.75 (NY close artefact), so
    # Var_τ(χ̂ᵢₜ) ≈ ûₗₜ² · spike².  On a quiet day Var_τ(yᵢₜ) is
    # small → the ratio is large even though the common factor
    # explains little in absolute terms.  This gave 2021-05-31
    # (a quiet Monday) instead of a real crisis day.
    #
    # Fix: use argmax |û₁ₜ|  — picks the day the first factor was
    # most extreme, which corresponds to the largest common shock.
    # This is robust to the loading spike artefact.
    crisis_day = int(np.argmax(np.abs(U_hat[0])))
    print(f"\n  Crisis day (max |û₁ₜ|, primary factor)  : t = {crisis_day}"
          + (f" ({str(days[crisis_day])[:10]})" if days is not None else ""))

    # ── Specific crisis dates ─────────────────────────────────────────────
    def _find_day(target_date):
        """Return day index closest to target_date."""
        if days is None:
            return crisis_day
        ts    = pd.Timestamp(target_date)
        diffs = np.abs((pd.DatetimeIndex(days) - ts).days)
        idx   = int(np.argmin(diffs))
        actual = str(days[idx])[:10]
        if diffs[idx] == 0:
            print(f"  Date {target_date} → t={idx} (exact match)")
        elif diffs[idx] <= 5:
            print(f"  Date {target_date} → t={idx} ({actual})  "
                  f"⚠ exact date not in panel (excluded by min_coverage?), "
                  f"using closest ({diffs[idx]}d away)")
        else:
            print(f"  ⚠ {target_date} too far from any panel date — "
                  f"using crisis_day t={crisis_day}")
            return crisis_day
        return idx

    # ── Crisis dates (Convention A: session labelled by START date) ──────────
    # 2008-10-09 = Thursday session starting at 17:00 NY → GFC peak
    # 2015-01-14 = Wednesday session → day before CHF floor removal (Jan 15)
    # 2020-03-08 = Sunday session → eve of COVID crash (Mar 9)
    day_gfc    = _find_day('2008-10-09')
    day_chf    = _find_day('2015-01-14')
    day_covid  = _find_day('2020-03-08')

    if not silent:
        plot_factors(U_hat, days, r_hat)
        plot_loadings(B_hat, tau, r_hat, pairs)

    # 06 — calm day (day 50), pair 0
    calm_day = min(50, T - 1)
    if not silent:
        plot_decomposition(X, Chi_hat, Xi_hat, mu, tau, pairs,
                       pair_idx=0, day_idx=calm_day,
                       days=days)

    # 06b — 2008-10-09, EUR/USD
    eurusd_keys = [k for k in pairs
                   if k.lower().replace('/', '') in ('eurusd', 'eur/usd', 'eur_usd')]
    pair_06b = pairs.index(eurusd_keys[0]) if eurusd_keys else 0
    if not silent:
        plot_decomposition(X, Chi_hat, Xi_hat, mu, tau, pairs,
                       pair_idx=pair_06b, day_idx=day_gfc,
                       save_as=_fig('06b_decomposition_crisis.png'),
                       days=days)

    # 06c — 4 pairs on 2008-10-09
    selected_4 = ['audjpy', 'eurusd', 'gbpusd', 'usdsgd']
    norm_keys  = {p.lower().replace('/', '').replace('_', ''): p for p in pairs}
    idx_4 = []
    for s in selected_4:
        if s in norm_keys and norm_keys[s] in pairs:
            idx_4.append(pairs.index(norm_keys[s]))
        elif len(idx_4) < 4 and len(idx_4) < N:
            idx_4.append(len(idx_4))
    idx_4 = idx_4[:4] if len(idx_4) >= 4 else list(range(min(4, N)))
    if not silent:
        plot_decomposition_multipair(X, Chi_hat, Xi_hat, mu, tau, pairs,
                                     pair_indices=idx_4,
                                     day_idx=day_gfc,
                                     days=days)

    # 06d — EUR/USD: 3 crisis days + 3 calm days
    pair_06d = pairs.index(eurusd_keys[0]) if eurusd_keys else 0

    day_indices_crisis = [day_gfc, day_chf, day_covid]

    var_eurusd  = np.var(X[pair_06d] - mu[pair_06d], axis=1)
    all_t       = np.arange(T)
    third       = T // 3
    calm_candidates = []
    for t_min, t_max in [(0, third), (third, 2*third), (2*third, T)]:
        mask_period   = (all_t >= t_min) & (all_t < t_max)
        t_in_period   = all_t[mask_period]
        best          = t_in_period[np.argmin(var_eurusd[mask_period])]
        calm_candidates.append(int(best))

    day_indices_06d = day_indices_crisis + calm_candidates
    day_labels_06d  = ['GFC  2008-10-09', 'EUR/CHF  2015-01-14',
                       'COVID  2020-03-08', 'Calm', 'Calm', 'Calm']

    print(f"\n  06d calm days (lowest EUR/USD variance per third):")
    for idx in calm_candidates:
        d = str(days[idx])[:10] if days is not None else str(idx)
        print(f"    t={idx} ({d})  var={var_eurusd[idx]:.6f}")

    if not silent:
        plot_decomposition_multiday(X, Chi_hat, Xi_hat, mu, tau, pairs,
                                pair_idx=pair_06d,
                                day_indices=day_indices_06d,
                                day_labels=day_labels_06d,
                                days=days)

    if not silent:
        plot_r2_heatmap(X, Chi_hat, mu, pairs, U_hat=U_hat, days=days)
        plot_residual_heatmap(X, Chi_hat, mu, pairs, days=days)

    # 06e — crisis/calm decomposition in log space for 4 thesis pairs
    if not silent:
        _res_tmp = {
            'X': X, 'Chi_hat': Chi_hat, 'Xi_hat': Xi_hat,
            'mu': mu, 'tau': tau, 'pairs': pairs, 'days': days,
        }
        plot_crisis_calm_fourpairs(_res_tmp)

    # ── Estimation errors (Hallin, Nisol & Tavakoli 2023, Section 3.2) ──────
    #
    # Three in-sample approximation errors (all normalised to be O(1)):
    #
    # (1) φ_{N,T}  — common component error  (Eq. 4.1)
    #     = (NT)^{-1} Σ_i Σ_t || y_it - χ̂_it ||²_M
    #     = V(r̂) by construction (in-sample residual variance)
    #     Theoretical rate: O_P(C_{N,T}^{-(1+α)})  where C_{N,T}=min(√N,√T)
    #
    # (2) δ²_{N,T}  — factor estimation error (proxy)
    #     True formula requires unknown true factors U_T.
    #     Proxy: orthogonality defect of Û — measures how far Û is from
    #     satisfying the normalisation Û Û^T / T = I_r.
    #     For perfectly estimated factors: defect = 0.
    #     δ_proxy = || Û Û^T / T - I_r ||_F / r
    #
    # (3) ε²_{N,T}  — loading estimation error (proxy)
    #     True formula requires unknown true loadings B_N.
    #     Proxy: cross-sectional variation of loading norms ||b̂_il||²
    #     normalised by their mean — measures dispersion around the
    #     pervasiveness condition (Assumption B).
    #     Small value → loadings well-spread across pairs (Assumption B holds).
    #
    C_NT = min(np.sqrt(N), np.sqrt(T))






    print("\n" + "=" * 65)
    print(f"  r̂ = {r_hat} factor(s)  (spectral ratio)")
    r2_global = np.mean([
        np.clip(np.sum(Chi_hat[i] ** 2, axis=1) / (np.sum(Y[i] ** 2, axis=1) + 1e-14), 0, 1).mean()
        for i in range(N)
    ])
    print(f"  Global mean R²: {r2_global:.2%}")
    r2_crisis = float(np.mean(np.clip(
        np.array([np.sum(Chi_hat[i, crisis_day] ** 2) / (np.sum(Y[i, crisis_day] ** 2) + 1e-14) for i in range(N)]),
        0, 1
    )))
    print(f"  Crisis-day R² (t={crisis_day}) : {r2_crisis:.1%}")
    print("=" * 65)

    return {
        'X': X, 
        'Y': Y, 
        'mu': mu, 
        'F': F,
        'sigma':sigma, 
        'r_hat': r_hat,
        'U_hat': U_hat, 
        'B_hat': B_hat,
        'Chi_hat': Chi_hat, 
        'Xi_hat': Xi_hat,
        'lambda': lam,
        'days': days, 
        'tau': tau,
        'pairs': pairs,
        'ic': ic_vals1a,
        'ic_ic2a': ic_vals2a, 
        'V': V_vals,
        'crisis_day': crisis_day,
        'r_hat_IC1a': r_ic1a,
        'r_hat_IC2a': r_ic2a,
        'r_hat_GR': r_GR,
        'ic_vals_GR': ic_vals_GR,
        # Estimation errors (Hallin et al. 2023, Section 3.2)

    }


def plot_raw_one_day(vol_matrices: dict, date: str):
    """
    Plot the raw intraday curves (original data) of all pairs on a given date.

    Parameters
    ----------
    vol_matrices : dict {str: DataFrame(days × minutes)}
    date         : str  ex. '2008-10-09'
    """
    target = pd.Timestamp(date)
    pairs  = list(vol_matrices.keys())
    N      = len(pairs)

    ncols = 4
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i, (pair, mat) in enumerate(vol_matrices.items()):
        ax = axes[i]

        # Find the date in the index
        mat_idx = pd.to_datetime(mat.index).normalize()
        diffs   = np.abs((mat_idx - target).days)
        t       = int(np.argmin(diffs))
        actual  = str(mat_idx[t])[:10]

        if diffs[t] > 3:
            ax.set_title(f'{pair}\n⚠ {date} absent', fontsize=8, color='red')
            axes[i].set_visible(True)
            continue

        row = mat.iloc[t].values.astype(float)
        x   = np.linspace(0, 1, len(row))

        ax.plot(x, row, lw=1.0, color=COLORS[i % len(COLORS)])
        ax.set_title(pair, fontsize=9, fontweight='bold')
        ax.set_xlabel('τ', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Raw data — all pairs — {actual}',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    fname = _fig(f'raw_{actual}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"✓ {fname}")
    plt.show()


def plot_all_pairs_one_day(results, date: str):
    """
    Plot the intraday curves of all FX pairs on a given date.
    
    Parameters
    ----------
    results : dict returned by run_ffm()
    date    : str  ex. '2008-10-09'
    """
    days  = pd.DatetimeIndex(results['days'])
    X     = results['X']       # (N, T, M) — raw log-vol
    mu    = results['mu']      # (N, M)    — mean per pair
    tau   = results['tau']     # (M,)      — axe intraday [0,1]
    pairs = results['pairs']   # list of pair names
    N     = len(pairs)

    # ── Find the index t of the date ─────────────────────────────────────
    target = pd.Timestamp(date)
    diffs  = np.abs((days - target).days)
    t      = int(np.argmin(diffs))
    actual = str(days[t])[:10]

    if diffs[t] > 3:
        print(f"⚠ Date {date} not found in the panel — closest: {actual}")
    else:
        print(f"Date : {actual}  (t={t})")

    # ── Observed values on that day (X + mu) ─────────────────────────────
    # X holds y_it = x_it - mu_i → add mu back to get x_it
    x_day = X[:, t, :] + mu    # (N, M)

    # ── Plot ────────────────────────────────────────────────────────────────
    ncols = 4
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharey=False)
    axes = axes.flatten()

    for i, pair in enumerate(pairs):
        ax = axes[i]
        ax.plot(tau, x_day[i], lw=1.2, color=COLORS[i])
        ax.set_title(pair, fontsize=9, fontweight='bold')
        ax.set_xlabel('τ  [0=open → 1=close]', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.5)

    # Hide empty axes
    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Intraday log-volatility — all pairs — {actual}',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    fname = _fig(f'all_pairs_{actual}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"✓ {fname}")
    plt.show()





