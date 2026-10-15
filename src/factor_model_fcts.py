"""
=============================================================================
FUNCTIONAL FACTOR MODEL (FFM) — Intraday FX Volatility
Adapted to existing day × minute matrices
=============================================================================
Reference: Hallin, Nisol & Tavakoli (2023), Parts I & II

DATA:
    - N DataFrame matrices: index = date (day), columns = intraday time (minute)
    - Names: eurusd_vol_matrix, usdjpy_vol_matrix, ...
    - Transformation: abs(r_t) => volatilities (log(r_t^2 + eps_i) + smoothing outside)

FFM PANEL STRUCTURE:
    X[i, t, τ] = volatility of pair i, on day t, at intraday interval τ
    shape: (N pairs, T = number of common days, M = number of intervals per day)
=============================================================================
"""

# Pipeline outline
#1. Data cleaning & preparation (input data are |r_t|)
#2. Log-transform log(r_t^2 + eps_i) and smoothing are done upstream (ffm_transform, ffm_smoothing)

#3. Centering
#4. Normalization 
#5. Gram matrix F
#6. Spectral decomposition of F => eigenvalues & eigenvectors
#7. Selection of number of factors r => spectral ratio 
#8. Factors and Loadings estimation 
#9. Common and idiosyncratic components estimation (remove information criteria)
#10. output: r_hat, U_hat, B_hat, Chi_hat, Xi_hat, mu, days, tau, pairs


"""
1. IMPORTS ==============================================================================

"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh
import warnings
warnings.filterwarnings("ignore")

"""
2. DATA CLEANING & PREPARATION ==============================================================================

"""
# Source: Claude IA ----------------------------------------------------------------
#---------------------------------------------------------------------------------

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
    #fill_rate   = (~np.isnan(stack)).mean(axis=(0, 1))   # (M_brut,)
    #active_mask = fill_rate >= 0.10
    #active_idx  = np.where(active_mask)[0]
    #M           = len(active_idx)
    #active_cols = [common_cols[j] for j in active_idx]
    #print(f"  Inactive columns removed       : {M_brut - M}"
    #      f"  ({M} active = {M*5} min of session)")

    # Reduce stack to active columns
    #stack = stack[:, :, active_idx]   # (N, T_raw, M)
    M = M_brut
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

"""
3. CENTERING ===============================================================================================
"""

"""
4. NORMALIZATION =============================================================================
"""
def normalize_panel(Y):
    """
    Global normalisation by the std of the entire centred panel.
    y_{it}(τ) = y_{it}(τ) / σ̂   — normalised panel
     with σ̂ = std(Y) = sqrt( (1/(N*T*M)) Σ_i Σ_t Σ_τ y_{it}(τ)^2 )
     Returns the normalized panel and the scaling factor σ̂ for later rescaling if needed.
    """
    sigma = float(np.std(Y))
    return Y / sigma, sigma

"""
5. GRAM MATRIX F CALCULATION ============================================================================
"""
def build_gram_matrix(Y):
    """
    F_{st} = (1/N) * (1/M) * Σ_i Σ_τ y_{is}(τ) y_{it}(τ)

    F ∈ R^{T×T} encodes the temporal dependence structure (between days) of the panel. 
    """
    N, T, M = Y.shape

    F = np.einsum('ism,itm->st', Y, Y) / (N * M)
    F = (F + F.T) / 2   # numerical symmetry
    return F

"""
6. SPECTRAL DECOMPOSITION OF F => EIGENVALUES & EIGENVECTORS ==========================================================
"""
def decompose_gram(F, k_max):
    """
    Parameters:
    F     : np.ndarray (T, T) — symmetric Gram matrix
    k_max : int — number of eigenvectors to keep

    Returns:
    eigenvalues  : np.ndarray (k_max,)- eigenvalues sorted ↓
    eigenvectors : np.ndarray (T, k_max) - associated eigenvectors
    """
    N_eig = F.shape[0]

    eigenvalues, eigenvectors = eigh(F,subset_by_index=[N_eig - k_max, N_eig - 1])

    # descending order
    eigenvalues  = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]   # (T, k_max)
    return eigenvalues, eigenvectors

"""
7. SELECTION OF NUMBER OF FACTORS r => SPECTRAL RATIO ============================================================================
"""


"""
8. FACTORS AND LOADINGS ESTIMATION + COMMON AND IDIOSYNCRATIC COMPONENTS ESTIMATION=============================
"""
def estimate_ffm(Y, F, k):
    """
    Proposition 2.1 (FFM-II): least-squares estimation.

    (i)  Spectral decomposition of F → (λ̃_l, f̃_l), l=1..k
    (ii) Normalisation :
            û_{lt}  = √T * f̃_{l,t} / ‖f̃_l‖      ← factors scalaires
                   λ̂_l    = λ̃_l / √T                    ← normalised eigenvalues
    (iii) Loadings fonctionnels :
            b̂_{il}(τ) = (1 /T) * Σ_t û_{lt} * y_{it}(τ)
    (iv) Common component    : χ̂_{it}(τ) = Σ_l b̂_{il}(τ) * û_{lt}
         Composante idiosync : ξ̂_{it}(τ) = y_{it}(τ) − χ̂_{it}(τ)

    Parameters
    ----------
    Y      : np.ndarray (N, T, M)
    F      : np.ndarray (T, T)
    k      : int — number of factors
    _eigen : optional tuple (eigenvalues, eigenvectors) pre-computed
                      → passed from select_r() to avoid recomputing eigh()
    """
    N, T, M = Y.shape

    eigenvalues, eigenvectors = decompose_gram(F, k_max=k)
    lambda_tilde = eigenvalues
    f_tilde      = eigenvectors

    lambda_hat = lambda_tilde / np.sqrt(T)  # (k,)  Normalisation

    U_hat = np.zeros((k, T))
    for l in range(k):
        f_l      = f_tilde[:, l]
        U_hat[l] = np.sqrt(T) * f_l / np.linalg.norm(f_l)

    # Loadings — Proposition 2.1 (FFM-II), step (iii)+(iv):
    #
    # Step (iii): ê_l = λ̂_l^{-1/2} T^{-1} Σ_t û_{lt} y_t   ← normalised DIRECTION
    # Step (iv):  b̂_{il}(τ) = λ̂_l^{1/2} ê_{il}(τ)          ← actual LOADING
    #
    # This is the OLS projection of y_i onto span(û_1,...,û_k) 

    B_hat = np.einsum('kt,itm->kim', U_hat, Y).transpose(1, 0, 2) / T  # (N, k, M)

    # Sign convention: force the mean loading to be > 0
    # (flipping û and b̂ simultaneously leaves χ̂ unchanged but makes the plots readable)
    for l in range(k):
        if np.mean(B_hat[:, l, :]) < 0:
            B_hat[:, l, :] *= -1
            U_hat[l]       *= -1

    Chi_hat = np.einsum('ilm,lt->itm', B_hat, U_hat)  # (N, T, M)
    Xi_hat  = Y - Chi_hat                              # (N, T, M)

    return U_hat, B_hat, Chi_hat, Xi_hat, lambda_hat


"""
10. FFM PER PERIOD =================================================================
"""


def ffm_per_period(data, start_date, end_date, n_months, func, step_months=1):
    """
    Parameters:
    ----------
    data       : vol_matrices — dict of DataFrames with a datetime index
    start_date : start of the period (str or Timestamp)
    end_date   : end of the period (str or Timestamp)
    n_months   : block length in months
    func       : function to apply (run_ffm)
    """
    results_list = []
    current_start = pd.to_datetime(start_date)
    end_date      = pd.to_datetime(end_date)

    while current_start < end_date:
        current_end = current_start + pd.DateOffset(months=n_months)

        # we keep only the pairs for which data is available before current_start
        subset = {}
        for pair, df in data.items():

            if current_start.year == 2012 and pair == 'audjpy':  # known data issue for AUDJPY in 2012
                continue

            first_date = pd.to_datetime(df.index).min()
            if first_date <= current_start:
                mask = (pd.to_datetime(df.index) >= current_start) & \
                       (pd.to_datetime(df.index) < current_end)
                subset[pair] = df.loc[mask]

        if len(subset) > 0:
            ffm = func(subset)

            N  = len(ffm['pairs'])
            T  = ffm['X'].shape[1]
            g   = np.sqrt((N + T) / (N * T)) * np.log(N * T / (N + T))
            C_NT = min(np.sqrt(N), np.sqrt(T))
            V_vals = ffm['V']
            if V_vals is not None and len(V_vals) >= 2:
                dV  = float(V_vals[0] - V_vals[1])
                snr = dV / g if g > 0 else np.nan
            else:
                dV  = np.nan
                snr = np.nan

            eigs = ffm['lambda']
            if eigs is not None and len(eigs) >= 2:
                ratio12 = float(eigs[0] / (eigs[1] + 1e-14))
            else:
                ratio12 = np.nan

            results_list.append({
                'start'      : current_start.date(),
                'end'        : current_end.date(),
                'r_hat_ratio'      : ffm['r_hat'],
                'r_hat_IC1a'       : ffm['r_hat_IC1a'],
                'r_hat_IC2a'       : ffm['r_hat_IC2a'],
                'r_hat_GR'       : ffm['r_hat_GR'],
                'V_IC1'        : ffm['V'],
                'ic_GR'       : ffm['ic_vals_GR'],
                'dV'         : dV,
                'g'          : g,
                'snr'        : snr,
                'ratio12'    : ratio12,
                'C_NT'       : C_NT,
                'n_pairs'    : len(ffm['pairs']),
                'n_days'     : ffm['X'].shape[1],   # effective T used by run_ffm
                'pairs_used' : ffm['pairs'],
            })
        print("===================================================")
        print(f"Processed period: {current_start.date()} → {current_end.date()}  | Pairs: {len(subset)}")
        print("===================================================")
        current_start = current_start + pd.DateOffset(months=step_months)

    return pd.DataFrame(results_list)


# =============================================================================
# VOLATILITY UTILITY FUNCTIONS
# =============================================================================

def compute_daily_avg_vol(X: np.ndarray, days: pd.DatetimeIndex,
                          log_space: bool = False) -> pd.Series:
    """
    Compute mean daily volatility aggregated over all FX pairs and intraday intervals.

    Parameters
    ----------
    X        : np.ndarray (N, T, M)
    days     : pd.DatetimeIndex (T,)
    log_space: if True return mean in log space; if False back-transform via exp(x/2)
    """
    if log_space:
        daily_mean = X.mean(axis=(0, 2))
    else:
        daily_mean = np.exp(X / 2).mean(axis=(0, 2))

    return pd.Series(daily_mean, index=pd.DatetimeIndex(days), name="avg_vol")
