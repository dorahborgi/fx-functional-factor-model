"""
=============================================================================
PENALIZED B-SPLINE SMOOTHING — Intraday FX Volatility
=============================================================================
Smooths the log-volatility curves x_{it}(tau) before applying the
Functional Factor Model (FFM) of Hallin, Nisol & Tavakoli (2023).

MODEL:
    For each pair i and day t, replace the raw discrete curve
        x_{it}(tau_1), ..., x_{it}(tau_M)
    by the penalized B-spline fit
        tilde_x_{it}(tau) = sum_{j=1}^{K} c_{itj} phi_j(tau)

    solved via:
        min_{c} || x_{it} - Phi c ||^2  +  lambda * c^T P c

    where:
        Phi_{kj} = phi_j(tau_k)   B-spline design matrix  (M x K)
        P        = D^T D           P-spline penalty matrix (K x K)
        lambda                    smoothing parameter (selected by GCV)

USAGE:
    from ffm_smoothing import smoothing_for_ffm
    vol_ready = smoothing_for_ffm(vol_matrices)
    results   = run_ffm(vol_ready, k_max=10, min_coverage=0.7)
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve
import warnings
warnings.filterwarnings("ignore")
from ffm_transform import log_transform, extract_epsilon


# =============================================================================
# B-SPLINE BASIS
# =============================================================================

def bspline_basis(tau, n_basis=20, order=4):
    """
    Parameters:
        tau     : (M,) grid of evaluation points in [0,1]
        n_basis : number of basis functions K
        order   : spline order (4 = cubic)
    Returns:
        Phi     : (M, K) design matrix
        knots   : full knot vector (for reference)
    """
    K = n_basis
    n_interior = K - order

    if n_interior > 0:
        interior = np.linspace(0, 1, n_interior + 2)[1:-1]
    elif n_interior < 0:
        raise ValueError(f"n_basis={K} must be >= order={order}")
    else:
        interior = np.array([])

    knots_full = np.concatenate([np.zeros(order), interior, np.ones(order)])
    Phi = BSpline.design_matrix(tau, knots_full, order - 1).toarray()

    return Phi, knots_full


# =============================================================================
# PENALTY MATRIX
# =============================================================================

def penalty_matrix(n_basis, order=4):
    """P-spline penalty: P = D^T D where D is the second-order difference matrix (K-2, K)."""
    K = n_basis
    D = np.diff(np.eye(K), n=2, axis=0)
    return D.T @ D


# =============================================================================
# GCV LAMBDA SELECTION
# =============================================================================

def gcv_score(lam, y, Phi, P, PhiTPhi):
    """
    GCV score for a single curve y and a given lambda.

        GCV(lambda) = (RSS / M) / (1 - df/M)^2

    Parameters:
        lam     : smoothing parameter
        y       : (M,) curve to smooth
        Phi     : (M, K) design matrix
        P       : (K, K) penalty matrix
        PhiTPhi : precomputed Phi^T Phi
    Returns:
        GCV score (float), or inf if numerical failure
    """
    M, K = Phi.shape
    A = PhiTPhi + lam * P
    try:
        cf = cho_factor(A)
        c  = cho_solve(cf, Phi.T @ y)
    except Exception:
        return np.inf

    yhat = Phi @ c
    rss  = np.sum((y - yhat) ** 2)

    try:
        A_inv_PhiTPhi = cho_solve(cf, PhiTPhi)
        df = np.trace(A_inv_PhiTPhi)
    except Exception:
        return np.inf

    shrink = 1.0 - df / M
    if shrink <= 0:
        return np.inf

    return (rss / M) / (shrink ** 2)


def select_lambda_gcv(y_sample, Phi, P, lam_grid=None):
    """
    Select optimal lambda by minimising mean GCV over a sample of curves.

    Parameters:
        y_sample : (n_sample, M) subset of curves
        Phi      : (M, K) design matrix
        P        : (K, K) penalty matrix
        lam_grid : grid of lambda values (default: logspace(-4, 4, 30))
    Returns:
        lambda_star : float
    """
    if lam_grid is None:
        lam_grid = np.logspace(-4, 4, 30)

    PhiTPhi = Phi.T @ Phi

    gcv_mean = np.zeros(len(lam_grid))
    for idx, lam in enumerate(lam_grid):
        scores = [gcv_score(lam, y, Phi, P, PhiTPhi) for y in y_sample]
        gcv_mean[idx] = np.nanmean(scores)

    best_idx = int(np.nanargmin(gcv_mean))
    return float(lam_grid[best_idx])


# =============================================================================
# BATCH SMOOTHING
# =============================================================================

def smooth_matrix(mat, tau, Phi, P, lam):
    """
    Smooth all T rows of mat with a fixed lambda via a single Cholesky factorisation.

    Parameters:
        mat  : (T, M) matrix of curves
        tau  : (M,) grid (unused, kept for API consistency)
        Phi  : (M, K) design matrix
        P    : (K, K) penalty matrix
        lam  : smoothing parameter
    Returns:
        mat_smooth : (T, M) smoothed curves
    """
    T, M = mat.shape
    K    = Phi.shape[1]

    PhiTPhi = Phi.T @ Phi
    A       = PhiTPhi + lam * P

    try:
        cf = cho_factor(A)
    except Exception:
        cf = cho_factor(A + 1e-10 * np.eye(K))

    PhiT = Phi.T
    rhs       = PhiT @ mat.T
    C         = cho_solve(cf, rhs)
    mat_smooth = (Phi @ C).T

    return mat_smooth


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================

def smooth_panel(vol_log,
                 n_basis=20,
                 order=4,
                 lam_grid=None,
                 n_gcv_sample=150,
                 lam_common=None,
                 lam_dict=None,
                 verbose=False,
                 return_lambdas=False):
    """
    Apply penalized B-spline smoothing to all pairs in vol_log.

    Parameters:
        vol_log        : dict {pair: DataFrame(days x minutes)}
        n_basis        : number of B-spline basis functions K (default 20)
        order          : spline order (4 = cubic, recommended)
        lam_grid       : grid of lambda values for GCV (default: logspace(-4, 4, 30))
        n_gcv_sample   : curves per pair used for GCV (default 150)
        lam_common     : if provided, skip GCV and use this lambda for all pairs
        lam_dict       : dict {pair: float} — per-pair lambda (takes priority over lam_common)
        verbose        : print progress
        return_lambdas : if True, return (vol_smooth, lam_used) instead of vol_smooth
    Returns:
        vol_smooth : dict {pair: DataFrame} — same index/columns as vol_log
        lam_used   : dict {pair: float} — only if return_lambdas=True
    """
    if lam_grid is None:
        lam_grid = np.logspace(-4, 4, 30)

    first_df = next(iter(vol_log.values()))
    M        = first_df.shape[1]
    tau      = np.linspace(0, 1, M)

    Phi, knots = bspline_basis(tau, n_basis=n_basis, order=order)
    P          = penalty_matrix(n_basis, order=order)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PENALIZED B-SPLINE SMOOTHING")
        print(f"  Basis    : B-splines  K={n_basis}  order={order}")
        print(f"  Grid     : M={M} intervals")
        print(f"  GCV      : {n_gcv_sample} curves/pair, {len(lam_grid)} lambda values")
        if lam_common is not None:
            print(f"  Lambda   : fixed = {lam_common:.2e} (GCV skipped)")
        print(f"{'='*60}")
        print(f"  {'Pair':15s}  {'T':>6s}  {'lambda':>10s}  "
              f"{'GCV score':>12s}  {'RMSE raw vs smooth':>20s}")
        print("  " + "-" * 70)

    vol_smooth = {}
    lam_used   = {}
    np.random.seed(42)

    for pair, df in vol_log.items():
        mat       = df.values.astype(np.float64)
        fully_nan = np.all(np.isnan(mat), axis=1)
        valid_idx = np.where(~fully_nan)[0]

        if len(valid_idx) == 0:
            vol_smooth[pair] = df.copy()
            if verbose:
                print(f"  {pair:15s}  ⚠ all days are NaN — skipped")
            continue

        mat_valid = mat[valid_idx]

        df_filled  = pd.DataFrame(mat_valid).interpolate(
            method='linear', axis=1, limit_direction='both')
        df_filled  = df_filled.ffill(axis=1).bfill(axis=1)
        mat_filled = df_filled.values

        T_valid = len(valid_idx)

        if lam_dict is not None and pair in lam_dict:
            lam_star = float(lam_dict[pair])
            gcv_val  = np.nan
        elif lam_common is not None:
            lam_star = lam_common
            gcv_val  = np.nan
        else:
            n_samp   = min(n_gcv_sample, T_valid)
            idx_samp = np.random.choice(T_valid, n_samp, replace=False)
            y_samp   = mat_filled[idx_samp]

            lam_star = select_lambda_gcv(y_samp, Phi, P, lam_grid)

            PhiTPhi = Phi.T @ Phi
            gcv_val = np.nanmean([
                gcv_score(lam_star, y, Phi, P, PhiTPhi) for y in y_samp
            ])

        lam_used[pair] = lam_star
        mat_smooth_valid = smooth_matrix(mat_filled, tau, Phi, P, lam_star)

        rmse = np.sqrt(np.nanmean((mat_valid - mat_smooth_valid) ** 2))

        mat_out = np.full_like(mat, np.nan)
        mat_out[valid_idx] = mat_smooth_valid

        vol_smooth[pair] = pd.DataFrame(mat_out, index=df.index, columns=df.columns)

        if verbose:
            n_nan_days = int(fully_nan.sum())
            gcv_str    = f"{gcv_val:.4e}" if not np.isnan(gcv_val) else "  (fixed)"
            nan_info   = f"  [{n_nan_days}j NaN]" if n_nan_days > 0 else ""
            print(f"  {pair:15s}  {T_valid:>6d}  {lam_star:>10.2e}  "
                  f"{gcv_str:>12s}  {rmse:>20.4f}{nan_info}")

    if verbose:
        print(f"\n  ✅ Smoothing complete — {len(vol_smooth)} pairs")
        print(f"{'='*60}\n")

    if return_lambdas:
        return vol_smooth, lam_used
    return vol_smooth


# =============================================================================
# PIPELINE
# =============================================================================

def smoothing_for_ffm(vol_matrices, smooth=True, n_basis=20, lam_common=None,
                      n_gcv_sample=150, verbose=False,
                      eps_dict=None, lam_dict=None,
                      drop_sparse=True, max_nan_pct=0.30):
    """
    Full preparation pipeline: log-transform + optional P-spline smoothing.

    Parameters:
        vol_matrices  : dict {pair: DataFrame of |r_t|}
        smooth        : whether to apply smoothing (default True)
        n_basis       : number of B-spline basis functions
        lam_common    : fixed lambda for all pairs (skip GCV); None = use GCV
        n_gcv_sample  : curves per pair for GCV
        verbose       : print progress
        eps_dict      : dict {pair: float} — fixed epsilon per pair (from extract_smoothing_params)
        lam_dict      : dict {pair: float} — fixed lambda per pair (from extract_smoothing_params)
        drop_sparse   : if True, drop days where any pair has > max_nan_pct NaN (default True)
        max_nan_pct   : NaN threshold per day per pair (default 0.30 = 30%)
    Returns:
        vol_ready : dict ready for run_ffm()
    """
    print("[0] Dropping weekends...")
    vol_matrices = {
        k: df[pd.to_datetime(df.index).dayofweek < 5]
        for k, df in vol_matrices.items()
    }
    n_days = next(iter(vol_matrices.values())).shape[0]
    print(f"    → {n_days} weekdays retained")

    if drop_sparse:
        # A day is dropped if ANY pair has more than max_nan_pct NaN on that day
        all_days = next(iter(vol_matrices.values())).index
        M = next(iter(vol_matrices.values())).shape[1]
        bad_days = set()
        for pair, df in vol_matrices.items():
            nan_pct = df.isna().sum(axis=1) / M
            bad_days |= set(df.index[nan_pct > max_nan_pct])
        keep = [d for d in all_days if d not in bad_days]
        n_dropped = len(bad_days)
        vol_matrices = {k: df.loc[keep] for k, df in vol_matrices.items()}
        print(f"[0b] Dropping days with >{max_nan_pct:.0%} NaN in any pair...")
        print(f"    → {n_dropped} days dropped, {len(keep)} days retained")

    print("[1] Squaring |r_t| → r_t²...")
    vol_sq = {k: v ** 2 for k, v in vol_matrices.items()}

    print("[2] Applying log(r_t² + ε)...")
    vol_log = log_transform(vol_sq, eps_dict=eps_dict)

    if not smooth:
        print("Smoothing skipped — returning log-transformed data.\n")
        return vol_log

    print("[3] P-spline smoothing...")
    vol_smooth, lam_used = smooth_panel(
        vol_log,
        n_basis=n_basis,
        lam_common=lam_common,
        lam_dict=lam_dict,
        n_gcv_sample=n_gcv_sample,
        verbose=verbose,
        return_lambdas=True,
    )

    print("\n    Lambda per pair:")
    for pair, lam in lam_used.items():
        source = "fixed" if (lam_dict and pair in lam_dict) or lam_common else "GCV"
        print(f"      {pair:15s}  λ = {lam:.4e}  ({source})")

    return vol_smooth


# =============================================================================
# TRAINING-PARAMETER EXTRACTION
# =============================================================================

def extract_lambda(vol_log: dict, n_basis: int = 20, order: int = 4,
                   lam_grid=None, n_gcv_sample: int = 150) -> dict:
    """
    Compute the GCV-optimal lambda per pair on the training data.

    Parameters
    ----------
    vol_log : dict {pair: DataFrame} — log-transformed (training) data

    Returns
    -------
    lam_dict : dict {pair: float}
    """
    if lam_grid is None:
        lam_grid = np.logspace(-4, 4, 30)

    first_df = next(iter(vol_log.values()))
    M        = first_df.shape[1]
    tau      = np.linspace(0, 1, M)
    Phi, _   = bspline_basis(tau, n_basis=n_basis, order=order)
    P        = penalty_matrix(n_basis, order=order)

    lam_dict = {}
    np.random.seed(42)

    for pair, df in vol_log.items():
        mat       = df.values.astype(np.float64)
        fully_nan = np.all(np.isnan(mat), axis=1)
        valid_idx = np.where(~fully_nan)[0]
        if len(valid_idx) == 0:
            continue

        mat_valid  = mat[valid_idx]
        df_filled  = pd.DataFrame(mat_valid).interpolate(
            method='linear', axis=1, limit_direction='both')
        df_filled  = df_filled.ffill(axis=1).bfill(axis=1)
        mat_filled = df_filled.values

        n_samp   = min(n_gcv_sample, len(valid_idx))
        idx_samp = np.random.choice(len(valid_idx), n_samp, replace=False)
        y_samp   = mat_filled[idx_samp]

        lam_dict[pair] = select_lambda_gcv(y_samp, Phi, P, lam_grid)

    return lam_dict


def extract_smoothing_params(vol_matrices: dict, n_basis: int = 20,
                             eps_pct: float = 5.0,
                             n_gcv_sample: int = 150) -> dict:
    """
    Compute and return all preprocessing parameters from the training data,
    so they can be re-applied to new data (e.g. 2026 out-of-sample).

    Parameters
    ----------
    vol_matrices : dict {pair: DataFrame of |r_t|} — training data

    Returns
    -------
    params : dict with keys 'eps' and 'lam'
        params['eps'] : dict {pair: float} — epsilon per pair
        params['lam'] : dict {pair: float} — GCV lambda per pair

    Usage
    -----
    # On the training data:
    params = extract_smoothing_params(vol_matrices_all)

    # On new data:
    vol_2026_smooth = smoothing_for_ffm(
        vol_matrices_2026,
        smooth=True, n_basis=20,
        eps_dict=params['eps'],
        lam_dict=params['lam'],
    )
    """
    print("[extract] Computing ε from training data...")
    vol_sq   = {k: v ** 2 for k, v in vol_matrices.items()}
    eps_dict = extract_epsilon(vol_sq, eps_pct=eps_pct)

    print("[extract] Computing GCV-optimal λ from training data...")
    vol_log  = log_transform(vol_sq, eps_dict=eps_dict)
    lam_dict = extract_lambda(vol_log, n_basis=n_basis, n_gcv_sample=n_gcv_sample)

    print(f"[extract] Done — {len(eps_dict)} pairs, ε ∈ "
          f"[{min(eps_dict.values()):.2e}, {max(eps_dict.values()):.2e}], "
          f"λ ∈ [{min(lam_dict.values()):.2e}, {max(lam_dict.values()):.2e}]")

    return {'eps': eps_dict, 'lam': lam_dict}
