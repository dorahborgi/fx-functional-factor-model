"""
Dette & Kutta (2021) — Structural Break Tests for Eigensystems of
Functional Time Series.

Reference:
    Dette, H. & Kutta, T. (2021). Detecting structural breaks in
    eigensystems of functional time series.
    Electronic Journal of Statistics.

DISCRETISATION CONVENTION
--------------------------
Δ = 1/M  (uniform grid, M equispaced points on [0,1])
⟨f, g⟩ = Σ_m f_m g_m Δ        (L² inner product)
‖f‖²   = Σ_m f_m² Δ            (L² norm squared)
Cov. operator ≈ Δ · C   where C is the M×M empirical covariance matrix.
L²-normalised eigenvectors: v = w / √Δ  (w = Euclidean-unit eigenvector).
Key identity: ‖v^(1) − v^(2)‖²_L² = ‖w^(1) − w^(2)‖²_2  (Δ cancels).

DEPENDENCY NOTE (Assumption 2.1)
----------------------------------
The test assumes weakly dependent functional observations. FX intraday
volatility curves exhibit strong serial dependence (volatility clustering).
The self-normalisation in V̂/Û absorbs the long-run variance, making the
pivot W free of nuisance parameters even under weak dependence. However,
strong dependence (long memory) may distort finite-sample size; interpret
results accordingly.

PUBLIC API
----------
    estimate_change_point(X, eps=0.05, dates=None) -> dict
    eigenvalue_test(X, j, Delta_tau, alpha=0.05, K=20, eps=0.05) -> dict
    eigenfunction_test(X, j, Delta_v, alpha=0.05, K=20, eps=0.05) -> dict
    simulate_W_quantiles(K, alpha=0.05, n_sim=50000) -> float
    run_validation(N, M, n_reps, theta_0, K, alpha, Delta_v, seed) -> dict
    load_fx_curves(pair, data_path, date_start, date_end) -> (X, days)
    run_fx_analysis(pairs, data_path, j_list, ...) -> dict
    binary_segmentation(X, dates, min_segment, eps, threshold_q) -> list
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from paths import fig_dir, DATA_DIR

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = fig_dir('DK')

# ── Precomputed quantiles — Table 1 of Dette & Kutta (2021) ──────────────────
# W = B(1) / [∫ λ²(B(λ)−λB(1))² ν(dλ)]^{1/2}
# ν = discrete uniform on {l/K : l=1,...,K-1}
_TABLE1 = {
    20: {0.90: 7.097, 0.95: 9.895, 0.99: 16.479},
    30: {0.90: 7.149, 0.95: 9.925, 0.99: 16.248},
}


# =============================================================================
# L² HELPERS
# =============================================================================

def l2_norm_sq(v, delta):
    """‖v‖²_L² = Σ_m v_m² Δ  (public utility)"""
    return float(np.dot(v, v) * delta)


def l2_inner(u, v, delta):
    """⟨u, v⟩_L² = Σ_m u_m v_m Δ  (public utility)"""
    return float(np.dot(u, v) * delta)


def _l2_dist_sq(v1, v2, delta):
    """
    ‖v1 − v2‖²_L² with sign alignment.

    Eigenvectors are sign-ambiguous; we pick the sign that minimises the
    distance:  min(‖v1 − v2‖², ‖v1 + v2‖²) in L² metric.
    """
    d_minus = np.dot(v1 - v2, v1 - v2) * delta
    d_plus  = np.dot(v1 + v2, v1 + v2) * delta
    return float(min(d_minus, d_plus))


def _eig_op(C, delta, j_max=2):
    """
    Eigendecomposition of the integral operator Δ·C.

    Solves  Δ·C·v = τ·v   with  ‖v‖_L² = 1.
    The Euclidean-unit eigenvectors w of Δ·C satisfy
    v = w / √Δ  →  ‖v‖²_L² = ‖w‖²_2 = 1.  ✓

    Parameters
    ----------
    C     : (D, D) symmetric non-negative definite matrix
    delta : float = 1/M  (or 1/D for a generic functional dimension)
    j_max : int   — return only top j_max eigenpairs

    Returns
    -------
    tau : (j_max,) descending eigenvalues of Δ·C
    V   : (D, j_max) L²-normalised eigenvectors  v_j = w_j / √Δ
    """
    D = C.shape[0]
    A = delta * C

    idx_lo = max(0, D - j_max)
    eigenvalues, eigenvectors = eigh(A, subset_by_index=[idx_lo, D - 1])

    eigenvalues  = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]      # (D, j_max)

    V = eigenvectors / np.sqrt(delta)         # L²-normalised

    return eigenvalues[:j_max], V[:, :j_max]


def _eig_op_dual(Z, delta, j_max=2):
    """
    Eigendecomposition of Δ·C via the DUAL (n×n) Gram matrix.

    Use this when D >> n  (many features, few samples).  Avoids materialising
    the D×D covariance matrix and its O(D³) eigendecomposition.

    Given Z ∈ R^{n×D}:
        C = Z.T @ Z / n   [D×D, expensive]
        G = Z @ Z.T / n   [n×n, cheap]

    Non-zero eigenvalues of C and G are identical.
    If  G w_j = σ_j w_j  (Euclidean),  then
        v_j = Z.T @ w_j / √(n σ_j)       (Euclidean-unit eigvec of C)
        v_j^{L²} = v_j / √Δ              (L²-normalised)

    Equivalently:  v_j^{L²} = Z.T @ w_j / √(n σ_j Δ)

    Parameters
    ----------
    Z     : (n, D) — centred data matrix (n < D typically)
    delta : float
    j_max : int

    Returns
    -------
    tau : (j_max,) eigenvalues of Δ·C  (= Δ × eigenvalues of G)
    V   : (D, j_max) L²-normalised eigenvectors
    """
    n, D   = Z.shape
    G      = Z @ Z.T / n              # (n, n)
    idx_lo = max(0, n - j_max)
    sigma, W = eigh(G, subset_by_index=[idx_lo, n - 1])
    sigma    = sigma[::-1]            # descending  (eigenvalues of G)
    W        = W[:, ::-1]             # (n, j_max)

    tau = delta * sigma[:j_max]       # eigenvalues of Δ·C

    # Recover D-dimensional L²-normalised eigenvectors
    V = np.zeros((D, j_max))
    for j in range(j_max):
        if sigma[j] > 1e-14:
            # v_j^{L²} = Z.T @ w_j / sqrt(n · σ_j · Δ)
            V[:, j] = Z.T @ W[:, j] / np.sqrt(n * sigma[j] * delta)

    return tau, V


# =============================================================================
# THRESHOLD CALIBRATION
# =============================================================================

def calibrate_delta_v(
    X, j=1,
    calm_start=None, calm_end=None, dates=None,
    n_blocks=10, q=0.95,
    method='empirical',
):
    """
    Calibrate the relevance threshold Δv for eigenfunction_test.

    Two methods
    -----------
    'empirical'
        Split the calm/baseline period into n_blocks consecutive blocks.
        For each pair of adjacent blocks compute ‖v̂_j^(l) − v̂_j^(l+1)‖²_L².
        Return the q-quantile of those (n_blocks−1) distances.
        Interpretation: threshold = expected L² rotation of v_j between two
        adjacent calm-period windows → anything larger is "relevant".

    'davis_kahan'
        Use the Davis-Kahan perturbation bound:
            Δv = 2 · (ε_op / gap_j)²
        where
            gap_j  = τ_j − τ_{j+1}   (eigenvalue gap, estimated on calm period)
            ε_op   = q-quantile of ‖ΔC_l‖_op across consecutive block pairs
                     (operator-norm perturbation between adjacent covariances)
        Interpretation: worst-case L² rotation of v_j under a perturbation of
        the covariance operator by ε_op (Davis-Kahan theorem, functional form).

    Parameters
    ----------
    X           : (N, M) array  (stacked panel or single-pair curves)
    j           : int — eigenfunction index (1-based, default 1)
    calm_start  : str or None — start of baseline period (e.g. '2014-01-01')
    calm_end    : str or None — end of baseline period   (e.g. '2017-12-31')
    dates       : DatetimeIndex or None
    n_blocks    : int — number of consecutive blocks (default 10)
    q           : float — quantile level (default 0.95)
    method      : 'empirical' or 'davis_kahan'

    Returns
    -------
    dict with keys:
        delta_v   : float — calibrated threshold
        distances : (n_blocks-1,) array — per-pair ‖Δv_j‖² values
        method    : str
        gap_j     : float — eigenvalue gap τ_j − τ_{j+1} (NaN for empirical)
        eps_op    : float — ε_op used in DK bound (NaN for empirical)
    """
    # ── Optional filtering to calm period ────────────────────────────────────
    X_calm = X
    if (calm_start is not None or calm_end is not None) and dates is not None:
        dates_dt = pd.to_datetime(dates)
        mask = np.ones(len(dates_dt), dtype=bool)
        if calm_start is not None:
            mask &= dates_dt >= pd.Timestamp(calm_start)
        if calm_end is not None:
            mask &= dates_dt <= pd.Timestamp(calm_end)
        X_calm = X[mask]

    N_calm, M = X_calm.shape
    delta     = 1.0 / M

    if N_calm < n_blocks * 2:
        raise ValueError(
            f"Calm period has only {N_calm} observations; need at least "
            f"{n_blocks * 2} for {n_blocks} blocks."
        )

    # ── Split into n_blocks consecutive blocks ────────────────────────────────
    edges  = np.linspace(0, N_calm, n_blocks + 1, dtype=int)
    blocks = [X_calm[edges[i]:edges[i + 1]] for i in range(n_blocks)]

    def _eigvec(Z_block, j_idx):
        Z  = Z_block - Z_block.mean(axis=0)
        n, D = Z.shape
        if n < D:
            _, V = _eig_op_dual(Z, delta, j_idx)
        else:
            C = Z.T @ Z / n
            _, V = _eig_op(C, delta, j_idx)
        return V[:, j_idx - 1]   # (M,) or (NM,)

    # ── Compute per-block eigenvectors ────────────────────────────────────────
    vecs = [_eigvec(blk, j) for blk in blocks]

    # ── Consecutive L² distances ──────────────────────────────────────────────
    distances = np.array([
        _l2_dist_sq(vecs[l], vecs[l + 1], delta)
        for l in range(n_blocks - 1)
    ])

    gap_j  = float('nan')
    eps_op = float('nan')

    if method == 'empirical':
        delta_v = float(np.quantile(distances, q))

    elif method == 'davis_kahan':
        # ── Eigenvalue gap from full calm-period covariance ──────────────────
        Z_full = X_calm - X_calm.mean(axis=0)
        n, D   = Z_full.shape
        if n < D:
            tau_full, _ = _eig_op_dual(Z_full, delta, j + 1)
        else:
            C_full       = Z_full.T @ Z_full / n
            tau_full, _  = _eig_op(C_full, delta, j + 1)

        if len(tau_full) >= j + 1:
            gap_j = float(tau_full[j - 1] - tau_full[j])
        else:
            raise ValueError(
                f"Need at least {j+1} eigenvalues to compute the gap for j={j}."
            )

        if gap_j < 1e-14:
            raise ValueError(
                f"Eigenvalue gap near zero (gap_j={gap_j:.2e}); "
                "Davis-Kahan bound is not informative."
            )

        # ── Operator-norm perturbation between consecutive block covariances ──
        def _cov_op(Z_block):
            Z = Z_block - Z_block.mean(axis=0)
            n, D = Z.shape
            return Z.T @ Z / n   # (D, D)

        op_norms = []
        for l in range(n_blocks - 1):
            dC = _cov_op(blocks[l]) - _cov_op(blocks[l + 1])
            # largest singular value of dC (= spectral norm of symmetric matrix)
            op_norms.append(float(np.linalg.norm(dC, ord=2)))

        eps_op  = float(np.quantile(op_norms, q))
        delta_v = 2.0 * (delta * eps_op / gap_j) ** 2

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'empirical' or 'davis_kahan'.")

    return {
        "delta_v":   delta_v,
        "distances": distances,
        "method":    method,
        "gap_j":     gap_j,
        "eps_op":    eps_op,
    }


def calibrate_delta_tau(
    X, j=1,
    calm_start=None, calm_end=None, dates=None,
    n_blocks=10, q=0.95,
):
    """
    Calibrate the relevance threshold Δτ for eigenvalue_test.

    Splits the calm/baseline period into n_blocks consecutive blocks,
    computes the j-th eigenvalue for each block, and returns the q-quantile
    of the squared differences between adjacent blocks:

        Δτ = quantile_q { (τ̂_j^(l) − τ̂_j^(l+1))² : l = 1,...,n_blocks−1 }

    Parameters
    ----------
    X           : (N, M) array
    j           : int — eigenvalue index (1-based, default 1)
    calm_start  : str or None
    calm_end    : str or None
    dates       : DatetimeIndex or None
    n_blocks    : int — number of consecutive blocks (default 10)
    q           : float — quantile level (default 0.95)

    Returns
    -------
    dict with keys:
        delta_tau  : float — calibrated threshold
        sq_diffs   : (n_blocks-1,) array — per-pair (τ_j^(l) − τ_j^(l+1))² values
    """
    X_calm = X
    if (calm_start is not None or calm_end is not None) and dates is not None:
        dates_dt = pd.to_datetime(dates)
        mask = np.ones(len(dates_dt), dtype=bool)
        if calm_start is not None:
            mask &= dates_dt >= pd.Timestamp(calm_start)
        if calm_end is not None:
            mask &= dates_dt <= pd.Timestamp(calm_end)
        X_calm = X[mask]

    N_calm, M = X_calm.shape
    delta     = 1.0 / M

    if N_calm < n_blocks * 2:
        raise ValueError(
            f"Calm period has only {N_calm} observations; need at least "
            f"{n_blocks * 2} for {n_blocks} blocks."
        )

    edges  = np.linspace(0, N_calm, n_blocks + 1, dtype=int)
    blocks = [X_calm[edges[i]:edges[i + 1]] for i in range(n_blocks)]

    def _eigval(Z_block, j_idx):
        Z    = Z_block - Z_block.mean(axis=0)
        n, D = Z.shape
        if n < D:
            tau, _ = _eig_op_dual(Z, delta, j_idx)
        else:
            C      = Z.T @ Z / n
            tau, _ = _eig_op(C, delta, j_idx)
        return float(tau[j_idx - 1])

    eigenvalues = [_eigval(blk, j) for blk in blocks]
    sq_diffs    = np.array([
        (eigenvalues[l] - eigenvalues[l + 1]) ** 2
        for l in range(n_blocks - 1)
    ])

    return {
        "delta_tau": float(np.quantile(sq_diffs, q)),
        "sq_diffs":  sq_diffs,
    }


# =============================================================================
# STEP 1 — CHANGE-POINT ESTIMATION  (Eq. 2.6)
# =============================================================================

def estimate_change_point(X, eps=0.05, dates=None):
    """
    Estimate the dominant change point via the full-operator CUSUM (Eq. 2.6).

    f(k) = [k(N−k)/N²] · ‖A_k − B_k‖²_F · Δ²

    A_k = (1/k) Σ_{n=1}^k X_n X_n^T   (uncentred second moments)
    B_k = (1/(N−k)) Σ_{n=k+1}^N X_n X_n^T

    Remark: uses uncentred moments by design — the test targets changes in
    the full second-moment operator, not just the centred covariance.

    Efficient computation via the Gram matrix G = X @ X.T:
      ‖N·C_k − k·C_N‖²_F  =  N² ‖C_k‖²_F − 2Nk ⟨C_k,C_N⟩_F + k² ‖C_N‖²_F
    where C_k = Σ_{n≤k} x_n x_n^T  and the Frobenius terms reduce to
    prefix sums on G² (see Gram-matrix identities in module docstring).

    Memory: O(N²) — G is (N×N); for N=5000 this is ~200 MB (float64).

    Parameters
    ----------
    X     : (N, M) array
    eps   : float — trim fraction, default 0.05
    dates : DatetimeIndex or None

    Returns
    -------
    dict(theta_hat, k0, f_values, break_date)
    """
    N, M  = X.shape
    delta = 1.0 / M

    k_min = max(2, int(np.ceil(N * eps)))
    k_max = min(N - 2, int(np.floor(N * (1.0 - eps))))

    # ── Gram matrix and its element-wise square ───────────────────────────
    G  = X @ X.T          # (N, N)
    G2 = G ** 2

    FN     = float(G2.sum())                    # ‖C_N‖²_F
    cumA2  = G2.sum(axis=1).cumsum()            # cumA2[k-1] = ⟨C_k, C_N⟩_F

    # ‖C_k‖²_F as a function of k — incremental O(N) per step
    # ‖C_k‖²_F = ‖C_{k-1}‖²_F + 2·sum(G²[:k-1, k-1]) + G²[k-1,k-1]
    nrm2 = np.zeros(N + 1)   # nrm2[k] = ‖C_k‖²_F  (1-indexed)
    for k in range(1, N + 1):
        nrm2[k] = nrm2[k - 1] + 2.0 * G2[:k - 1, k - 1].sum() + G2[k - 1, k - 1]

    # ── f(k) vectorised over the trim window ─────────────────────────────
    ks     = np.arange(k_min, k_max + 1)
    norm_D = (N ** 2 * nrm2[ks]
              - 2.0 * N * ks * cumA2[ks - 1]
              + ks ** 2 * FN)

    f_k = norm_D * (delta ** 2) / (N ** 2 * ks * (N - ks))

    f_values          = np.zeros(N)
    f_values[ks]      = f_k
    k0_hat            = int(np.argmax(f_values))
    theta_hat         = k0_hat / N

    break_date = None
    if dates is not None and 0 < k0_hat < len(dates):
        break_date = pd.Timestamp(dates[k0_hat])

    return {
        "theta_hat":  theta_hat,
        "k0":         k0_hat,
        "f_values":   f_values,
        "break_date": break_date,
    }


def estimate_change_point_eigenvalue(X, j, eps=0.05, step=20, dates=None):
    """
    Estimate the change point by maximising the Bai-Perron weighted squared
    difference in the j-th eigenvalue across candidate splits.

    f_ev(k) = [k(N−k)/N²] · (τ_j^(1)(k) − τ_j^(2)(k))²

    The Bai-Perron weight k(N-k)/N² penalises splits near the boundaries
    where one sub-sample is small and the eigenvalue estimate is noisy.

    Parameters
    ----------
    X     : (N, M) array
    j     : int — eigenvalue index (1-based)
    eps   : float — trim fraction (default 0.05)
    step  : int — grid step in observations (default 20 ≈ 1 month).
            Reduces the number of eigendecompositions from O(N) to O(N/step).
    dates : DatetimeIndex or None

    Returns
    -------
    dict(k0, theta_hat, f_values, break_date)
    """
    N, M  = X.shape
    delta = 1.0 / M

    k_min = max(2, int(np.ceil(N * eps)))
    k_max = min(N - 2, int(np.floor(N * (1.0 - eps))))

    candidates = range(k_min, k_max + 1, step)
    f_values = np.zeros(N)
    for k in candidates:
        tau1, tau2, _, _ = _full_eigensystems(X, k, j, delta)
        w = k * (N - k) / N ** 2
        f_values[k] = w * (tau1[j - 1] - tau2[j - 1]) ** 2

    k0_hat    = int(np.argmax(f_values))
    theta_hat = k0_hat / N

    break_date = None
    if dates is not None and 0 < k0_hat < len(dates):
        break_date = pd.Timestamp(dates[k0_hat])

    return {
        "k0":         k0_hat,
        "theta_hat":  theta_hat,
        "f_values":   f_values,
        "break_date": break_date,
    }


def estimate_change_point_eigenfunction(X, j, eps=0.05, step=20, dates=None):
    """
    Estimate the change point by maximising the Bai-Perron weighted L²
    distance between the j-th eigenfunction across candidate splits.

    f_ef(k) = [k(N−k)/N²] · ‖v_j^(1)(k) − v_j^(2)(k)‖²_L²

    The Bai-Perron weight k(N-k)/N² penalises splits near the boundaries
    where one sub-sample is small and the eigenfunction estimate is noisy.

    Parameters
    ----------
    X     : (N, M) array
    j     : int — eigenfunction index (1-based)
    eps   : float — trim fraction (default 0.05)
    step  : int — grid step in observations (default 20 ≈ 1 month).
            Reduces the number of eigendecompositions from O(N) to O(N/step).
    dates : DatetimeIndex or None

    Returns
    -------
    dict(k0, theta_hat, f_values, break_date)
    """
    N, M  = X.shape
    delta = 1.0 / M

    k_min = max(2, int(np.ceil(N * eps)))
    k_max = min(N - 2, int(np.floor(N * (1.0 - eps))))

    candidates = range(k_min, k_max + 1, step)
    f_values = np.zeros(N)
    for k in candidates:
        _, _, V1, V2 = _full_eigensystems(X, k, j, delta)
        w = k * (N - k) / N ** 2
        f_values[k] = w * _l2_dist_sq(V1[:, j - 1], V2[:, j - 1], delta)

    k0_hat    = int(np.argmax(f_values))
    theta_hat = k0_hat / N

    break_date = None
    if dates is not None and 0 < k0_hat < len(dates):
        break_date = pd.Timestamp(dates[k0_hat])

    return {
        "k0":         k0_hat,
        "theta_hat":  theta_hat,
        "f_values":   f_values,
        "break_date": break_date,
    }


# =============================================================================
# STEPS 2–3 — RUNNING KERNELS & EIGENSYSTEMS  (Eq. 2.9–2.13)
# =============================================================================

def _running_eigensystems(X, k0, K, j_max, delta, min_n=3):
    """
    Compute eigenvalues/eigenvectors of the running covariance operators
    Δ·c^(i)(λ) for both sub-samples at the ν-grid {l/K : l=1,...,K-1}.

    Remark 2.7 (centring): µ^(1) and µ^(2) are the empirical means of the
    FULL sub-samples (λ=1), re-computed even if curves are pre-centred.

    Parameters
    ----------
    X     : (N, M) array
    k0    : int — estimated break index
    K     : int — grid size
    j_max : int — number of eigenpairs needed
    delta : float = 1/M
    min_n : int — minimum sub-sample size to attempt eigendecomposition

    Returns
    -------
    lam_grid : (K-1,)
    tau1     : (K-1, j_max) eigenvalues of Δ·c^(1)(λ)
    tau2     : (K-1, j_max) eigenvalues of Δ·c^(2)(λ)
    V1       : (K-1, j_max, M) L²-normalised eigenvectors
    V2       : (K-1, j_max, M)
    valid    : (K-1,) bool — True when both sub-samples have ≥ min_n points
    """
    N, D  = X.shape           # D = M for single pair, D = N_pairs*M for panel
    X1    = X[:k0]
    X2    = X[k0:]
    k1    = k0
    k2    = N - k0

    mu1 = X1.mean(axis=0)  # full first-sample mean  (Remark 2.7)
    mu2 = X2.mean(axis=0)

    Z1   = X1 - mu1
    Z2   = X2 - mu2

    lam_grid = np.arange(1, K) / K
    n_lam    = K - 1

    tau1  = np.zeros((n_lam, j_max))
    tau2  = np.zeros((n_lam, j_max))
    V1    = np.zeros((n_lam, j_max, D))   # D instead of M: works for panel too
    V2    = np.zeros((n_lam, j_max, D))
    valid = np.ones(n_lam, dtype=bool)

    for l, lam in enumerate(lam_grid):
        n1 = max(1, int(np.floor(k1 * lam)))
        n2 = max(1, int(np.floor(k2 * lam)))

        if n1 < min_n or n2 < min_n:
            valid[l] = False
            continue

        # Use dual when n < D (avoids materialising the large D×D covariance).
        # This is always the case for the stacked panel (D = N·M >> n).
        if n1 < D:
            t1, e1 = _eig_op_dual(Z1[:n1], delta, j_max)
        else:
            C1 = Z1[:n1].T @ Z1[:n1] / n1
            t1, e1 = _eig_op(C1, delta, j_max)

        if n2 < D:
            t2, e2 = _eig_op_dual(Z2[:n2], delta, j_max)
        else:
            C2 = Z2[:n2].T @ Z2[:n2] / n2
            t2, e2 = _eig_op(C2, delta, j_max)

        tau1[l] = t1
        tau2[l] = t2
        V1[l]   = e1.T    # (j_max, M)
        V2[l]   = e2.T

    return lam_grid, tau1, tau2, V1, V2, valid


def _full_eigensystems(X, k0, j_max, delta):
    """
    Eigenstructure at λ=1 (full sub-samples).

    Returns (tau1_j, tau2_j, v1_j, v2_j) for the j_max-th eigenpair.
    Uses the same centring convention as _running_eigensystems.
    """
    X1 = X[:k0]; X2 = X[k0:]
    mu1 = X1.mean(axis=0); mu2 = X2.mean(axis=0)
    Z1  = X1 - mu1; Z2 = X2 - mu2

    D = Z1.shape[1]   # functional dimension (M or N*M for stacked panel)

    if k0 < D:
        tau1, V1 = _eig_op_dual(Z1, delta, j_max)
    else:
        C1 = Z1.T @ Z1 / k0
        tau1, V1 = _eig_op(C1, delta, j_max)

    k2 = len(Z2)
    if k2 < D:
        tau2, V2 = _eig_op_dual(Z2, delta, j_max)
    else:
        C2 = Z2.T @ Z2 / k2
        tau2, V2 = _eig_op(C2, delta, j_max)

    return tau1, tau2, V1, V2   # eigenvalues (j_max,), eigenvectors (M, j_max)


# =============================================================================
# STEP 4 — EIGENVALUE TEST  (Eq. 2.16, 2.20)
# =============================================================================

def eigenvalue_test(X, j, Delta_tau, alpha=0.05, K=20, eps=0.05, k0=None,
                    break_date=None, plot=False, pair_name="", dates=None,
                    start_date=None, end_date=None, delta=None):
    """
    Test for a relevant change in the j-th eigenvalue (Eq. 2.16, 2.20).

    H₀ : (τ^(1)_j − τ^(2)_j)² ≤ Δτ
    H₁ : (τ^(1)_j − τ^(2)_j)² > Δτ

    Reject H₀ iff  (Ê_{j,N} − Δτ) / V̂_{j,N}  >  q_{1−α}

    where
      Ê_{j,N}(λ) = (τ^(1)_{j,λ} − τ^(2)_{j,λ})²
      Ê_{j,N}    = Ê_{j,N}(1)
      V̂_{j,N}   = [(1/(K−1)) Σ_{l=1}^{K−1} (l/K)⁴ (Ê_{j,N}(l/K) − Ê_{j,N})²]^{1/2}

    One-sided confidence interval (Remark 2.7.3):
      I^(τ)_j = [0, Ê_{j,N} + q_{1−α} V̂_{j,N}]

    Parameters
    ----------
    X          : (N, M) array of daily functional observations
    j          : int — eigenvalue index (1-based)
    Delta_tau  : float — relevance threshold  Δτ ≥ 0
    alpha      : float — test level
    K          : int — ν grid size (default 20)
    eps        : float — CUSUM trim fraction
    k0         : int or None — pre-computed break index (avoids recomputing)
    break_date : str/Timestamp or None — candidate break date; converted to k0
                 via nearest index in `dates` (overrides k0 if both provided)
    plot       : bool — generate figures (default False)
    pair_name  : str — label for figures (e.g. 'eurusd' or 'global_panel')
    dates      : DatetimeIndex or None — for x-axis labels

    Returns
    -------
    dict(E_hat, V_hat, stat, q_crit, reject, tau1_j, tau2_j,
         k0, theta_hat, break_date, E_lam, lam_grid, valid, CI_upper)
    """
    # ── Date filtering ────────────────────────────────────────────────────
    if start_date is not None or end_date is not None:
        if dates is None:
            raise ValueError("dates must be provided to use start_date/end_date")
        dates_dt = pd.to_datetime(dates)
        mask = np.ones(len(dates_dt), dtype=bool)
        if start_date is not None:
            mask &= dates_dt >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= dates_dt <= pd.Timestamp(end_date)
        X     = X[mask]
        dates = dates_dt[mask]
        k0    = None  # index refers to original array — recompute on filtered data

    N, M  = X.shape
    if delta is None:
        delta = 1.0 / M

    # ── Change point ──────────────────────────────────────────────────────
    if break_date is not None and dates is not None:
        bd_ts  = pd.Timestamp(break_date)
        dates_dt2 = pd.to_datetime(dates)
        k0 = int(np.argmin(np.abs(dates_dt2 - bd_ts)))
        theta_hat = k0 / N
    elif k0 is None:
        cp        = estimate_change_point(X, eps=eps)
        k0        = cp["k0"]
        theta_hat = cp["theta_hat"]
    else:
        theta_hat = k0 / N

    # ── Full-sample eigenvalues at λ=1 ───────────────────────────────────
    tau1_all, tau2_all, _, _ = _full_eigensystems(X, k0, j, delta)
    tau1_j = float(tau1_all[j - 1])
    tau2_j = float(tau2_all[j - 1])
    E_hat  = (tau1_j - tau2_j) ** 2

    # ── Running eigenvalues on the ν grid ─────────────────────────────────
    lam_grid, tau1_run, tau2_run, _, _, valid = _running_eigensystems(
        X, k0, K, j_max=j, delta=delta
    )

    E_lam = (tau1_run[:, j - 1] - tau2_run[:, j - 1]) ** 2   # (K-1,)

    # ── Self-normaliser V̂  (Eq. 2.20) ────────────────────────────────────
    lam_v   = lam_grid[valid]
    E_v     = E_lam[valid]
    V_hat   = float(np.sqrt(np.mean(lam_v ** 4 * (E_v - E_hat) ** 2)))

    # ── Critical value ────────────────────────────────────────────────────
    q_crit = simulate_W_quantiles(K, alpha)

    # ── Test decision ─────────────────────────────────────────────────────
    if V_hat < 1e-14:
        stat   = np.nan
        reject = False
    else:
        stat   = float((E_hat - Delta_tau) / V_hat)
        reject = bool(stat > q_crit)

    CI_upper = E_hat + q_crit * V_hat
    p_value  = simulate_W_pvalue(K, stat)

    break_date = None
    if dates is not None and 0 < k0 < len(dates):
        break_date = pd.Timestamp(dates[k0])

    if plot:
        _plot_eigenvalue_running(
            tau1_run[:, j - 1], tau2_run[:, j - 1],
            E_lam, lam_grid, valid, E_hat, Delta_tau, CI_upper,
            j, pair_name, k0, dates
        )

    return {
        "E_hat":      E_hat,
        "V_hat":      V_hat,
        "stat":       stat,
        "p_value":    p_value,
        "q_crit":     q_crit,
        "reject":     reject,
        "tau1_j":     tau1_j,
        "tau2_j":     tau2_j,
        "k0":         k0,
        "theta_hat":  theta_hat,
        "break_date": break_date,
        "E_lam":      E_lam,
        "lam_grid":   lam_grid,
        "valid":      valid,
        "CI_upper":   CI_upper,
    }


# =============================================================================
# STEP 5 — EIGENFUNCTION TEST  (Eq. 2.24–2.25)
# =============================================================================

def eigenfunction_test(X, j, Delta_v, alpha=0.05, K=20, eps=0.05, k0=None,
                       break_date=None, plot=False, pair_name="", pairs=None,
                       M_functional=None, dates=None,
                       start_date=None, end_date=None, delta=None):
    """
    Test for a relevant change in the j-th eigenfunction (Eq. 2.24–2.25).

    H₀ : ‖v^(1)_j − v^(2)_j‖²_L² ≤ Δv
    H₁ : ‖v^(1)_j − v^(2)_j‖²_L² > Δv

    Reject H₀ iff  (D̂_{j,N} − Δv) / Û_{j,N}  >  q_{1−α}

    Sign alignment: replace ‖v − v'‖ by min(‖v − v'‖, ‖v + v'‖) at every λ.

    Parameters
    ----------
    X            : (T, D) array  (D=M single pair, D=N*M stacked panel)
    j            : int — eigenfunction index (1-based)
    Delta_v      : float — relevance threshold  Δv ≥ 0
    alpha        : float — test level
    K            : int — ν grid size
    eps          : float — CUSUM trim fraction
    k0           : int or None — pre-computed break index
    break_date   : str/Timestamp or None — candidate break date; converted to k0
                   via nearest index in `dates` (overrides k0 if both provided)
    plot         : bool — generate figures (default False)
    pair_name    : str — label for single-pair figures (e.g. 'eurusd')
    pairs        : list[str] or None — pair names for panel figures
    M_functional : int or None — M to reshape D=N*M into (N_pairs, M)
    dates        : DatetimeIndex or None

    Returns
    -------
    dict(D_hat, U_hat, stat, q_crit, reject, v1_j, v2_j,
         k0, theta_hat, D_lam, lam_grid, valid, CI_upper)
    """
    # ── Date filtering ────────────────────────────────────────────────────
    if start_date is not None or end_date is not None:
        if dates is None:
            raise ValueError("dates must be provided to use start_date/end_date")
        dates_dt = pd.to_datetime(dates)
        mask = np.ones(len(dates_dt), dtype=bool)
        if start_date is not None:
            mask &= dates_dt >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= dates_dt <= pd.Timestamp(end_date)
        X     = X[mask]
        dates = dates_dt[mask]
        k0    = None  # index refers to original array — recompute on filtered data

    T, M  = X.shape
    if delta is None:
        delta = 1.0 / M

    # ── Change point ──────────────────────────────────────────────────────
    if break_date is not None and dates is not None:
        bd_ts  = pd.Timestamp(break_date)
        dates_dt2 = pd.to_datetime(dates)
        k0 = int(np.argmin(np.abs(dates_dt2 - bd_ts)))
        theta_hat = k0 / T
    elif k0 is None:
        cp        = estimate_change_point(X, eps=eps)
        k0        = cp["k0"]
        theta_hat = cp["theta_hat"]
    else:
        theta_hat = k0 / T

    # ── Full-sample eigenfunctions ────────────────────────────────────────
    _, _, V1_full, V2_full = _full_eigensystems(X, k0, j, delta)
    v1_j = V1_full[:, j - 1]   # (M,) L²-normalised
    v2_j = V2_full[:, j - 1]

    D_hat = _l2_dist_sq(v1_j, v2_j, delta)

    # ── Running eigenfunctions on the ν grid ──────────────────────────────
    lam_grid, _, _, V1_run, V2_run, valid = _running_eigensystems(
        X, k0, K, j_max=j, delta=delta
    )

    D_lam = np.full(K - 1, D_hat)   # default: fallback to full-sample value
    for l in range(K - 1):
        if not valid[l]:
            continue
        v1_l = V1_run[l, j - 1]     # (M,)
        v2_l = V2_run[l, j - 1]
        D_lam[l] = _l2_dist_sq(v1_l, v2_l, delta)

    # ── Self-normaliser Û  (Eq. 2.25) ─────────────────────────────────────
    lam_v   = lam_grid[valid]
    D_v     = D_lam[valid]
    U_hat   = float(np.sqrt(np.mean(lam_v ** 4 * (D_v - D_hat) ** 2)))

    q_crit = simulate_W_quantiles(K, alpha)

    if U_hat < 1e-14:
        stat   = np.nan
        reject = False
    else:
        stat   = float((D_hat - Delta_v) / U_hat)
        reject = bool(stat > q_crit)

    CI_upper = D_hat + q_crit * U_hat
    p_value  = simulate_W_pvalue(K, stat)

    break_date_out = None
    if dates is not None and 0 < k0 < len(dates):
        break_date_out = pd.Timestamp(pd.to_datetime(dates)[k0])

    if plot:
        _, D = X.shape
        tau  = np.linspace(0, 1, M_functional if M_functional else D)
        if pairs is not None and M_functional is not None:
            # Panel mode: reshape NM → (N_pairs, M)
            N_p   = len(pairs)
            delta = 1.0 / M_functional
            v1_m  = v1_j.reshape(N_p, M_functional)
            v2_m  = v2_j.reshape(N_p, M_functional)
            norm1 = np.sqrt(np.sum(v1_m ** 2, axis=1) * delta)
            norm2 = np.sqrt(np.sum(v2_m ** 2, axis=1) * delta)
            _plot_global_eigvec(v1_m, v2_m, norm1, norm2,
                                pairs, j, dates, k0)
        else:
            _plot_eigenfunctions(v1_j, v2_j, tau, j,
                                 pair_name or f"j{j}", dates, k0)
        _plot_running_stat(D_lam, lam_grid, valid, D_hat, Delta_v, CI_upper,
                           label="D̂", j=j, pair_name=pair_name or "")

    return {
        "D_hat":      D_hat,
        "U_hat":      U_hat,
        "stat":       stat,
        "p_value":    p_value,
        "q_crit":     q_crit,
        "reject":     reject,
        "v1_j":       v1_j,
        "v2_j":       v2_j,
        "k0":         k0,
        "theta_hat":  theta_hat,
        "break_date": break_date_out,
        "D_lam":      D_lam,
        "lam_grid":   lam_grid,
        "valid":      valid,
        "CI_upper":   CI_upper,
    }


# =============================================================================
# SUBSPACE TEST — joint stability of first r eigenvectors  (Grassmann)
# =============================================================================

def subspace_test(X, r, Delta_sub=0.0, alpha=0.05, K=20, eps=0.05, k0=None,
                  plot=False, pairs=None, M_functional=None, dates=None):
    """
    Joint test for relevant change in the r-dimensional eigenspace (Grassmann).

    Instead of testing each eigenvector j separately (eigenfunction_test),
    this tests whether the SUBSPACE spanned by {v_1,...,v_r} has rotated:

        D_r(λ) = 1 − (1/r)·‖A(λ)‖²_F

    where  A(λ)[j,k] = ⟨v^(1)_j(λ), v^(2)_k(λ)⟩_L²  (r×r cross-Gram matrix).

    Connection to Grassmann geometry:
        D_r(1) = 1 − (1/r)‖A_full‖²_F
               = (1/r) Σ_j sin²(θ_j)   (mean squared principal angle)

    where θ_1,...,θ_r are the principal angles between the two r-dim subspaces.
    This is (1/(2r)) × ‖P_r^(1) − P_r^(2)‖²_{L²⊗L²} (Frobenius of projections).

    Advantages over individual tests:
    - Single decision (no multiple testing issue)
    - Captures joint rotation even when individual eigenvectors are close
    - Natural for the Grassmann/Bai-Perron framework already in the thesis

    Null hypothesis:   D_r ≤ Delta_sub
    Reject if:   (D_r − Delta_sub) / U_r > q_{1−α}

    NOTE: the pivot W is used as in the individual tests (same self-normalisation
    structure). For r > 1 this is an approximation; it is exact for r = 1
    (reduces to eigenfunction_test with a different distance formula).

    Parameters
    ----------
    X         : (N, M) or (T, N*M) — functional time series
    r         : int — number of eigenvectors to test jointly
    Delta_sub : float — relevance threshold in [0,1] (0 = any change)
    alpha     : float — test level
    K         : int — ν grid
    eps       : float — CUSUM trim fraction

    Returns
    -------
    dict(D_r, U_r, stat, q_crit, reject,
         principal_angles, A_full, k0, theta_hat,
         D_lam, lam_grid, valid, CI_upper)
    """
    _T, M_or_D = X.shape
    delta       = 1.0 / M_or_D

    if k0 is None:
        cp        = estimate_change_point(X, eps=eps)
        k0        = cp["k0"]
        theta_hat = cp["theta_hat"]
    else:
        theta_hat = k0 / _T

    # ── Full-sample eigenvectors (λ=1) ────────────────────────────────────
    _, _, V1_full, V2_full = _full_eigensystems(X, k0, r, delta)
    # V1_full, V2_full: (M_or_D, r), each column L²-normalised

    # Cross-Gram matrix A_full[j,k] = ⟨v^(1)_j, v^(2)_k⟩_L²
    A_full = (V1_full.T @ V2_full) * delta   # (r, r)

    # D_r = 1 − (1/r)‖A‖²_F
    D_r = float(1.0 - np.sum(A_full ** 2) / r)

    # Principal angles θ_j = arccos(σ_j)  where σ_j = singular values of A_full
    sv       = np.linalg.svd(A_full, compute_uv=False)
    sv       = np.clip(sv, -1.0, 1.0)
    p_angles = np.degrees(np.arccos(sv))   # (r,) in degrees

    # ── Running D_r(λ) on the ν grid ─────────────────────────────────────
    lam_grid, _, _, V1_run, V2_run, valid = _running_eigensystems(
        X, k0, K, j_max=r, delta=delta
    )

    D_lam = np.full(K - 1, D_r)   # fallback = full-sample value

    for l in range(K - 1):
        if not valid[l]:
            continue
        # V1_run[l]: (r, M_or_D) — eigenvectors as rows
        V1_l = V1_run[l].T   # (M_or_D, r)
        V2_l = V2_run[l].T
        A_l  = (V1_l.T @ V2_l) * delta  # (r, r)
        D_lam[l] = float(1.0 - np.sum(A_l ** 2) / r)

    # ── Self-normaliser U_r ───────────────────────────────────────────────
    lam_v = lam_grid[valid]
    D_v   = D_lam[valid]
    U_r   = float(np.sqrt(np.mean(lam_v ** 4 * (D_v - D_r) ** 2)))

    q_crit = simulate_W_quantiles(K, alpha)

    if U_r < 1e-14:
        stat   = np.nan
        reject = False
    else:
        stat   = float((D_r - Delta_sub) / U_r)
        reject = bool(stat > q_crit)

    CI_upper = D_r + q_crit * U_r

    if plot:
        _plot_running_stat(D_lam, lam_grid, valid, D_r, Delta_sub, CI_upper,
                           label=f"D_r (r={r})", j=r,
                           pair_name="subspace")
        _plot_principal_angles(p_angles, r, k0, dates,
                               pairs=pairs, M_functional=M_functional)

    return {
        "D_r":            D_r,
        "U_r":            U_r,
        "stat":           stat,
        "q_crit":         q_crit,
        "reject":         reject,
        "principal_angles": p_angles,
        "A_full":         A_full,
        "k0":             k0,
        "theta_hat":      theta_hat,
        "D_lam":          D_lam,
        "lam_grid":       lam_grid,
        "valid":          valid,
        "CI_upper":       CI_upper,
        "r":              r,
    }


# =============================================================================
# PIVOT SIMULATION  (Table 1)
# =============================================================================

def simulate_W_quantiles(K, alpha=0.05, n_sim=50_000, seed=0):
    """
    Simulate the (1−α)-quantile of the self-normalised pivot:

        W = B(1) / [ (1/(K−1)) Σ_{l=1}^{K−1} (l/K)² (B(l/K)−(l/K)B(1))² ]^{1/2}

    where B is standard Brownian motion and ν is discrete uniform on {l/K}.

    First checks precomputed Table 1 of Dette & Kutta (2021).
    Falls back to Monte Carlo with n_sim ≥ 50 000 replications.

    Parameters
    ----------
    K     : int   — ν parameter (20 or 30 for Table 1)
    alpha : float — test level  (0.10, 0.05, or 0.01)
    n_sim : int   — replications for Monte Carlo fallback

    Returns
    -------
    q : float — (1−α)-quantile of W
    """
    q_level = round(1.0 - alpha, 4)

    if K in _TABLE1 and q_level in _TABLE1[K]:
        return _TABLE1[K][q_level]

    # ── Monte Carlo simulation ─────────────────────────────────────────────
    rng       = np.random.default_rng(seed)
    lam_grid  = np.arange(1, K) / K          # (K-1,) = {1/K,...,(K-1)/K}

    # Simulate BM on fine grid, sample at lam_grid and t=1
    n_fine    = 10_000
    dt        = 1.0 / n_fine
    lam_idx   = np.round(lam_grid * n_fine).astype(int) - 1   # 0-based

    # Batch simulation: (n_sim, n_fine) increments
    batch = 2000   # process in batches to control memory
    W_samples = np.empty(n_sim)

    for start in range(0, n_sim, batch):
        end   = min(start + batch, n_sim)
        size  = end - start
        incs  = rng.normal(0.0, np.sqrt(dt), size=(size, n_fine))
        B     = np.cumsum(incs, axis=1)          # (size, n_fine)
        B1    = B[:, -1]                          # B(1)
        B_lam = B[:, lam_idx]                     # (size, K-1)

        # denominator: (1/(K-1)) Σ (l/K)² (B(l/K) - (l/K)B(1))²
        bridge   = B_lam - lam_grid[None, :] * B1[:, None]  # (size, K-1)
        denom_sq = np.mean((lam_grid ** 2)[None, :] * bridge ** 2, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            W_samples[start:end] = np.where(
                denom_sq > 1e-14, B1 / np.sqrt(denom_sq), np.nan
            )

    W_finite = W_samples[np.isfinite(W_samples)]
    return float(np.quantile(W_finite, q_level))


def simulate_W_pvalue(K, stat, n_sim=50_000, seed=0):
    """
    Simulate the p-value of the self-normalised pivot W.

    p = P(W > stat)  under H₀, where W is the same pivot as in
    simulate_W_quantiles (Dette & Kutta 2021, Eq. 2.20 / 2.25).

    Always uses Monte Carlo (Table 1 only stores three quantile levels).

    Parameters
    ----------
    K    : int   — ν parameter
    stat : float — observed test statistic
    n_sim : int  — replications (default 50 000)
    seed : int   — RNG seed

    Returns
    -------
    p_value : float  — empirical P(W > stat)
    """
    if np.isnan(stat):
        return float('nan')

    rng      = np.random.default_rng(seed)
    lam_grid = np.arange(1, K) / K

    n_fine  = 10_000
    dt      = 1.0 / n_fine
    lam_idx = np.round(lam_grid * n_fine).astype(int) - 1

    batch     = 2000
    n_exceed  = 0
    n_finite  = 0

    for start in range(0, n_sim, batch):
        end  = min(start + batch, n_sim)
        size = end - start
        incs = rng.normal(0.0, np.sqrt(dt), size=(size, n_fine))
        B    = np.cumsum(incs, axis=1)
        B1   = B[:, -1]
        B_lam = B[:, lam_idx]

        bridge   = B_lam - lam_grid[None, :] * B1[:, None]
        denom_sq = np.mean((lam_grid ** 2)[None, :] * bridge ** 2, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where(denom_sq > 1e-14, B1 / np.sqrt(denom_sq), np.nan)

        finite_mask = np.isfinite(W)
        n_finite   += finite_mask.sum()
        n_exceed   += (W[finite_mask] > stat).sum()

    return float(n_exceed / n_finite) if n_finite > 0 else float('nan')


# =============================================================================
# VALIDATION SIMULATION  (Section 3.1)
# =============================================================================

def _make_functional_ts(N, M, phi_rot, n_basis=8, eigenvalues=None,
                        theta_0=0.5, rho=0.0, seed=None):
    """
    Simulate a functional time series on M grid points with a rotation
    break at index ⌊N·theta_0⌋.

    Model:
        X_n(τ) = Σ_k a_{n,k} √λ_k φ_k(τ)

    Basis: truncated Fourier  φ_k(τ) = √2 sin(k π τ)  (L²-orthonormal)
    Eigenvalues: λ_k = exp(−k)  (default)
    Temporal dependence: a_{n,k} = rho·a_{n−1,k} + √(1−rho²) ε_{n,k}
                         ε_{n,k} ~ iid N(0,1)

    Break: for n > ⌊N·theta_0⌋, the first two basis vectors are rotated
    by angle phi_rot (Givens rotation R_{1,2}(phi_rot)).

    Parameters
    ----------
    phi_rot : float — rotation angle in radians
    rho     : float — AR(1) coefficient for temporal dependence (default 0)

    Returns
    -------
    X : (N, M) array
    """
    rng = np.random.default_rng(seed)

    tau  = np.linspace(0, 1, M, endpoint=False) + 0.5 / M   # (M,) midpoints
    K    = n_basis

    # L²-orthonormal Fourier basis on M grid points
    # φ_k(τ) = √2 sin(kπτ);  normalised to ‖φ_k‖_L² = 1
    PHI  = np.array([np.sqrt(2) * np.sin(k * np.pi * tau) for k in range(1, K + 1)])
    # PHI shape (K, M); L²-normalisation check: (PHI**2).mean(axis=1) ≈ 1

    if eigenvalues is None:
        lam = np.exp(-np.arange(1, K + 1, dtype=float))   # (K,)
    else:
        lam = np.asarray(eigenvalues, dtype=float)[:K]

    k_break = int(np.floor(N * theta_0))

    # Rotated basis vectors for second sub-sample
    c, s = np.cos(phi_rot), np.sin(phi_rot)
    PHI_rot       = PHI.copy()
    PHI_rot[0]    =  c * PHI[0] + s * PHI[1]
    PHI_rot[1]    = -s * PHI[0] + c * PHI[1]

    # AR(1) coefficients  a_{n,k}  (N, K)
    eps  = rng.standard_normal((N, K))
    a    = np.zeros((N, K))
    a[0] = eps[0]
    for n in range(1, N):
        a[n] = rho * a[n - 1] + np.sqrt(1 - rho ** 2) * eps[n]

    # Scale by √λ_k
    a_scaled = a * np.sqrt(lam)[None, :]    # (N, K)

    # Build curves
    X = np.zeros((N, M))
    X[:k_break] = a_scaled[:k_break] @ PHI        # pre-break
    X[k_break:] = a_scaled[k_break:] @ PHI_rot    # post-break

    return X


def run_validation(N=500, M=50, n_reps=1000, theta_0=0.5,
                   K=20, alpha=0.05, j=1, Delta_v=0.02,
                   phi_list=None, n_basis=8, rho=0.0,
                   seed=42, verbose=True):
    """
    Reproduce the Monte Carlo study of Section 3.1.

    Under H₀ (phi = phi_H0 s.t. D_j = Delta_v): level ≈ alpha.
    Under H₁ (phi > phi_H0): power → 1.

    Also verifies simulate_W_quantiles against Table 1.

    Parameters
    ----------
    N       : int — time series length
    M       : int — number of grid points (use 50–100 for speed)
    n_reps  : int — Monte Carlo replications
    theta_0 : float — true break fraction
    K       : int — ν grid
    alpha   : float — test level
    j       : int — eigenvalue/function index to test
    Delta_v : float — relevance threshold for eigenfunction test
    phi_list: list of float — rotation angles [phi_H0, phi_H1, phi_H1_large]
              None → auto-computed
    n_basis : int — number of Fourier basis functions
    rho     : float — AR(1) coefficient
    seed    : int

    Returns
    -------
    dict with rejection tables and quantile comparisons
    """
    rng = np.random.default_rng(seed)

    # ── Critical value ─────────────────────────────────────────────────────
    q_crit = simulate_W_quantiles(K, alpha)
    print(f"\n{'='*65}")
    print(f"  VALIDATION — Dette & Kutta (2021), Section 3.1")
    print(f"  N={N}, M={M}, n_reps={n_reps}, theta_0={theta_0}")
    print(f"  K={K}, alpha={alpha:.2f}, j={j}, Delta_v={Delta_v:.4f}")
    print(f"{'='*65}")

    # ── Verify Table 1 via MC ──────────────────────────────────────────────
    print("\n[0] Verifying simulate_W_quantiles against Table 1:")
    for K_check in [20, 30]:
        for a_check in [0.10, 0.05, 0.01]:
            q_sim   = simulate_W_quantiles(K_check, a_check, n_sim=100_000, seed=99)
            q_table = _TABLE1.get(K_check, {}).get(round(1 - a_check, 2), np.nan)
            flag    = "" if np.isnan(q_table) else (
                " ✓" if abs(q_sim - q_table) < 0.5 else " ✗")
            q_tab_str = f"{q_table:.3f}" if not np.isnan(q_table) else "N/A"
            print(f"  K={K_check}, alpha={a_check:.2f}: "
                  f"simulated={q_sim:.3f}  table={q_tab_str}{flag}")

    # ── Rotation angles ────────────────────────────────────────────────────
    # True D_j for rotation phi: D = 2(1 − cos(phi))  [L²-orthonormal basis]
    # H₀ boundary: phi_0 = arccos(1 − Delta_v/2)
    phi_H0 = float(np.arccos(1.0 - Delta_v / 2.0))
    if phi_list is None:
        phi_list = [phi_H0, phi_H0 * 2.0, phi_H0 * 4.0]

    print(f"\n  phi_H0 = {phi_H0:.4f} rad  (D_j = Delta_v = {Delta_v:.4f})")
    print(f"  phi values tested: {[f'{p:.4f}' for p in phi_list]}")

    # ── Monte Carlo loop ───────────────────────────────────────────────────
    seeds     = rng.integers(0, 2**31, size=n_reps)
    results_v = {p: [] for p in phi_list}   # eigenfunction test rejections
    results_e = {p: [] for p in phi_list}   # eigenvalue test rejections
    theta_hats = {p: [] for p in phi_list}

    for rep in range(n_reps):
        if verbose and rep % 200 == 0:
            print(f"  rep {rep}/{n_reps} ...", flush=True)

        for phi_idx, phi in enumerate(phi_list):
            X = _make_functional_ts(
                N, M, phi_rot=phi, n_basis=n_basis,
                theta_0=theta_0, rho=rho,
                seed=int(seeds[rep]) + phi_idx * 10_000
            )

            # eigenfunction test
            res_v = eigenfunction_test(
                X, j=j, Delta_v=Delta_v, alpha=alpha, K=K, eps=0.05
            )
            # eigenvalue test (Delta_tau = 0 → test for any change)
            # Compute true eigenvalue gap: Δτ = 0 (no change in λ under rotation)
            res_e = eigenvalue_test(
                X, j=j, Delta_tau=0.0, alpha=alpha, K=K, eps=0.05
            )

            results_v[phi].append(int(res_v["reject"]))
            results_e[phi].append(int(res_e["reject"]))
            theta_hats[phi].append(res_v["theta_hat"])

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n  Rejection rates (eigenfunction test, j={j}, Δv={Delta_v:.4f}):")
    print(f"  {'phi':>8s}  {'D_j_true':>10s}  {'Reject%':>10s}  "
          f"{'theta_hat':>12s}  {'Interpretation'}")
    print("  " + "-" * 72)

    phi_labels = {phi_list[0]: "H₀ (level)",
                  **{p: "H₁ (power)" for p in phi_list[1:]}}
    eigenf_rates = {}
    eigenvalue_rates = {}

    for phi in phi_list:
        D_true  = 2.0 * (1.0 - np.cos(phi))
        rr_v    = float(np.mean(results_v[phi]))
        rr_e    = float(np.mean(results_e[phi]))
        th_mean = float(np.mean(theta_hats[phi]))
        lbl     = phi_labels.get(phi, "")
        eigenf_rates[phi]     = rr_v
        eigenvalue_rates[phi] = rr_e
        print(f"  {phi:>8.4f}  {D_true:>10.4f}  {rr_v:>9.3f}  "
              f"{th_mean:>12.4f}  {lbl}")

    print(f"\n  Rejection rates (eigenvalue test, j={j}, Δτ=0):")
    print(f"  {'phi':>8s}  {'Reject%':>10s}")
    print("  " + "-" * 25)
    for phi in phi_list:
        print(f"  {phi:>8.4f}  {eigenvalue_rates[phi]:>9.3f}")

    # ── theta^ distribution under H₁ ─────────────────────────────────────
    if len(phi_list) >= 2:
        th_arr = np.array(theta_hats[phi_list[1]])
        print(f"\n  theta^ under H₁ (phi={phi_list[1]:.4f}):")
        print(f"    mean = {th_arr.mean():.4f}  (true: {theta_0:.4f})")
        print(f"    std  = {th_arr.std():.4f}")
        print(f"    bias = {th_arr.mean() - theta_0:.4f}")

    print(f"\n  Expected level ≈ {alpha:.2f}  (H₀ at phi={phi_list[0]:.4f})")
    print(f"  Observed level  = {eigenf_rates[phi_list[0]]:.3f}")

    return {
        "eigenf_rates":    eigenf_rates,
        "eigenvalue_rates": eigenvalue_rates,
        "theta_hats":      theta_hats,
        "phi_list":        phi_list,
        "phi_H0":          phi_H0,
        "q_crit":          q_crit,
    }


# =============================================================================
# FX DATA LOADING
# =============================================================================

def load_fx_curves(pair="eurusd", data_path=str(DATA_DIR) + "/",
                   date_start=None, date_end=None,
                   n_basis=20, verbose=False):
    """
    Load and preprocess FX intraday log-volatility curves.

    Pipeline (reuses existing preprocessing):
      1. pd.read_parquet({pair}_matrix_lg.parquet)
         → raw 5-min log-return matrix (day × 5-min interval)
      2. smoothing_for_ffm(): squares → log(r² + ε) → P-spline smooth
         → smoothed log-vol curves
      3. build_panel_from_matrices(): NaN handling, active-column filter
         → X (T, M) float64

    Parameters
    ----------
    pair       : str — FX pair name, e.g. 'eurusd'
    data_path  : str — directory containing {pair}_matrix_lg.parquet
    date_start : str or None — e.g. '2014-01-01'
    date_end   : str or None
    n_basis    : int — P-spline basis size (default 20)
    verbose    : bool

    Returns
    -------
    X    : (T, M) float64 array — smoothed log-vol curves
    days : pd.DatetimeIndex (T,)
    """
    sys.path.insert(0, _HERE)
    from ffm_smoothing import smoothing_for_ffm
    from ffm_fx_real   import build_panel_from_matrices

    fpath = os.path.join(data_path, f"{pair}_matrix_lg.parquet")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index)

    if date_start:
        df = df.loc[date_start:]
    if date_end:
        df = df.loc[:date_end]

    if verbose:
        print(f"  Loaded {pair}: {len(df)} rows, {df.shape[1]} columns")

    vol_matrices = {pair: df}
    vol_smooth   = smoothing_for_ffm(
        vol_matrices, smooth=True, n_basis=n_basis, verbose=verbose
    )

    X_3d, days, _, _ = build_panel_from_matrices(
        vol_smooth, min_coverage=0.8
    )

    X = X_3d[0]    # (T, M)
    return X, days


# =============================================================================
# BINARY SEGMENTATION (multiple breaks)
# =============================================================================

def binary_segmentation(X, dates=None, min_segment=100, eps=0.05,
                        threshold_f=None, max_breaks=10):
    """
    Recursive binary segmentation to locate multiple structural breaks.

    At each step, apply estimate_change_point to the current segment.
    A break is accepted if the CUSUM statistic f(k̂) exceeds threshold_f.
    Recurse on left and right sub-segments.

    NOTE: The CUSUM (Eq. 2.6) estimates ONE dominant break per segment.
    Binary segmentation extends this to multiple breaks at the cost of
    assuming breaks are well-separated (gap ≥ min_segment observations).

    Parameters
    ----------
    X            : (N, M) array
    dates        : DatetimeIndex or None
    min_segment  : int — minimum segment length
    eps          : float — CUSUM trim fraction
    threshold_f  : float or None — minimum f(k̂) to accept a break.
                   None → use 0.1 × max f(k̂) on the full series.
    max_breaks   : int — safety cap on number of break points

    Returns
    -------
    breaks : list of dicts with keys (index, theta, date, f_value)
    """
    N = len(X)

    # Set default threshold: 25% of the dominant full-series CUSUM value.
    # This prevents spurious recursion on short sub-segments where finite-
    # sample CUSUM noise exceeds the global break signal.
    if threshold_f is None:
        cp_full = estimate_change_point(X, eps=eps, dates=dates)
        threshold_f = 0.25 * cp_full["f_values"].max()

    breaks = []

    def _recurse(left, right, depth=0):
        if depth >= max_breaks or right - left < 2 * min_segment:
            return
        seg   = X[left:right]
        seg_d = dates[left:right] if dates is not None else None
        cp    = estimate_change_point(seg, eps=eps, dates=seg_d)
        k_loc = cp["k0"]           # index within segment
        f_val = cp["f_values"][k_loc]
        if f_val < threshold_f or k_loc < min_segment or k_loc > (right - left) - min_segment:
            return
        abs_k = left + k_loc
        breaks.append({
            "index":   abs_k,
            "theta":   abs_k / N,
            "date":    cp["break_date"],
            "f_value": f_val,
        })
        _recurse(left, abs_k, depth + 1)
        _recurse(abs_k, right, depth + 1)

    _recurse(0, N)
    breaks.sort(key=lambda b: b["index"])
    return breaks


# =============================================================================
# FX ANALYSIS — MAIN APPLICATION
# =============================================================================

def run_fx_analysis(pairs=None, data_path=str(DATA_DIR) + "/",
                    j_list=(1,), K=20, alpha=0.05, eps=0.05,
                    Delta_v=0.0225, Delta_tau_frac=0.10,
                    date_start=None, date_end=None,
                    Delta_v_grid=(0.0, 0.01, 0.05, 0.0225),
                    do_binary_seg=True, verbose=True):
    """
    Apply the Dette & Kutta (2021) tests to FX intraday volatility curves.

    For each pair and each j in j_list:
      1. estimate_change_point  → θ̂, break date, f(k) plot
      2. eigenvalue_test        (Δτ = Delta_tau_frac × τ^(1)_j)
      3. eigenfunction_test     (Δv default + grid)
      4. Plot v^(1)_j vs v^(2)_j on the intraday clock
      5. Optional binary segmentation for additional breaks

    Δv motivation (Remark 2.7.2):
        √Δv ≤ 0.15  ↔  R ≥ 0.70 (70% explained variance preserved)
        → Δv ≈ 0.0225 (default)

    Assumption 2.3 check for j=2:
        Use j=2 only if spectral gap τ^_2 − τ^_3 is at least 20% of τ^_2.

    Parameters
    ----------
    pairs           : list of str or None → ['eurusd']
    data_path       : str
    j_list          : iterable of ints
    K               : int — ν grid
    alpha           : float — test level
    eps             : float — CUSUM trim fraction
    Delta_v         : float — default eigenfunction relevance threshold
    Delta_tau_frac  : float — Δτ as fraction of τ^(1)_j  (default 0.10 = 10%)
    date_start      : str or None
    date_end        : str or None
    Delta_v_grid    : tuple — extra Δv values to report
    do_binary_seg   : bool — run binary segmentation
    verbose         : bool

    Returns
    -------
    dict {pair: {'X', 'days', 'cp', 'tests', 'breaks'}}
    """
    if pairs is None:
        pairs = ["eurusd"]

    all_results = {}

    for pair in pairs:
        print(f"\n{'='*65}")
        print(f"  DETTE & KUTTA TEST — {pair.upper()}")
        print(f"{'='*65}")

        # ── Load data ──────────────────────────────────────────────────────
        try:
            X, days = load_fx_curves(
                pair, data_path=data_path,
                date_start=date_start, date_end=date_end,
                verbose=verbose
            )
        except Exception as e:
            print(f"  ✗ Could not load {pair}: {e}")
            continue

        N, M  = X.shape
        delta = 1.0 / M
        tau   = np.linspace(0, 1, M)
        print(f"\n  Panel: N={N} days, M={M} intervals, Δ={delta:.4f}")

        # ── Step 1: Change point ───────────────────────────────────────────
        print("\n[1] Estimating dominant change point...")
        cp = estimate_change_point(X, eps=eps, dates=days)
        k0 = cp["k0"]
        print(f"  θ̂ = {cp['theta_hat']:.4f}  (k₀ = {k0})")
        if cp["break_date"]:
            print(f"  Break date: {str(cp['break_date'])[:10]}")

        # ── f(k) plot ──────────────────────────────────────────────────────
        _plot_cusum(cp, days, pair)

        # ── Spectral structure at break ────────────────────────────────────
        n_eigs = max(j_list) + 1
        tau1_all, tau2_all, _, _ = _full_eigensystems(
            X, k0, n_eigs, delta
        )
        print(f"\n  Top eigenvalues before break: "
              f"{', '.join(f'{v:.4f}' for v in tau1_all[:4])}")
        print(f"  Top eigenvalues after break:  "
              f"{', '.join(f'{v:.4f}' for v in tau2_all[:4])}")

        # Spectral gap check for j=2 (Assumption 2.3)
        gap_ok = {}
        for j in j_list:
            if j == 1:
                gap_ok[j] = True
            elif j <= len(tau2_all) - 1:
                gap = float(tau2_all[j - 1] - tau2_all[j])
                frac = gap / (tau2_all[j - 1] + 1e-14)
                gap_ok[j] = frac >= 0.20
                print(f"  Spectral gap τ^_({j})−τ^_({j+1}) = "
                      f"{gap:.5f}  ({frac:.1%} of τ^_({j})) "
                      + ("✓ gap OK" if gap_ok[j] else "⚠ gap small"))
            else:
                gap_ok[j] = False

        pair_tests = {}

        for j in j_list:
            if not gap_ok.get(j, False):
                print(f"\n  Skipping j={j}: spectral gap insufficient "
                      f"(Assumption 2.3 / Davis-Kahan)")
                continue

            print(f"\n[2–5] Tests for j={j}")

            # ── Eigenvalue test ────────────────────────────────────────────
            # Δτ = (r × τ^(1)_j)²: detect if |τ^(1)_j − τ^(2)_j| > r × τ^(1)_j
            # (same units as E_hat = (τ^(1)_j − τ^(2)_j)²)
            Delta_tau = (Delta_tau_frac * float(tau1_all[j - 1])) ** 2
            res_e = eigenvalue_test(X, j=j, Delta_tau=Delta_tau,
                                    alpha=alpha, K=K, eps=eps)

            print(f"\n  EIGENVALUE TEST (j={j})")
            print(f"    τ^(1)_j = {res_e['tau1_j']:.6f}   "
                  f"τ^(2)_j = {res_e['tau2_j']:.6f}")
            print(f"    Ê_{{j,N}} = {res_e['E_hat']:.6f}   "
                  f"Δτ = {Delta_tau:.6f}")
            print(f"    V̂_{{j,N}} = {res_e['V_hat']:.6f}")
            print(f"    Statistic = {res_e['stat']:.4f}   "
                  f"q_crit = {res_e['q_crit']:.4f}")
            print(f"    Reject H₀: {res_e['reject']}  "
                  f"(CI upper: {res_e['CI_upper']:.6f})")

            # ── Eigenfunction test across Δv grid ─────────────────────────
            print(f"\n  EIGENFUNCTION TEST (j={j})")
            res_v_main = None
            for dv in Delta_v_grid:
                res_v = eigenfunction_test(X, j=j, Delta_v=float(dv),
                                           alpha=alpha, K=K, eps=eps)
                flag = "REJECT" if res_v["reject"] else "no reject"
                if dv == Delta_v:
                    res_v_main = res_v
                print(f"    Δv={dv:.4f}: "
                      f"D̂={res_v['D_hat']:.5f}  "
                      f"Û={res_v['U_hat']:.5f}  "
                      f"stat={res_v['stat']:.3f}  [{flag}]")
                if dv == Delta_v:
                    print(f"    CI: [0, {res_v['CI_upper']:.5f}]  "
                          f"(q_crit={res_v['q_crit']:.3f})")

            # ── Eigenfunction plot ─────────────────────────────────────────
            if res_v_main is not None:
                _plot_eigenfunctions(
                    res_v_main["v1_j"], res_v_main["v2_j"],
                    tau, j, pair, days, k0
                )

            pair_tests[j] = {"eigenvalue": res_e, "eigenfunction": res_v_main}

        # ── Binary segmentation ────────────────────────────────────────────
        breaks = []
        if do_binary_seg:
            print("\n[6] Binary segmentation for additional breaks...")
            breaks = binary_segmentation(X, dates=days, min_segment=100, eps=eps)
            print(f"  Found {len(breaks)} break(s):")
            for b in breaks:
                d = str(b["date"])[:10] if b["date"] else f"k={b['index']}"
                print(f"    θ={b['theta']:.4f}  k={b['index']}  "
                      f"date={d}  f={b['f_value']:.6f}")
            if breaks:
                _plot_binary_seg(cp, breaks, days, pair)

        all_results[pair] = {
            "X":       X,
            "days":    days,
            "cp":      cp,
            "tests":   pair_tests,
            "breaks":  breaks,
            "tau1":    tau1_all,
            "tau2":    tau2_all,
        }

    return all_results


# =============================================================================
# GLOBAL PANEL TEST — NM×NM covariance (joint loading structure)
# =============================================================================

def stack_panel(X_3d):
    """
    Stack the N-pair panel (N, T, M) into a single functional time series
    (T, N·M) suitable for the global covariance test.

    Each day t becomes one observation Y_t ∈ R^{N·M}:
        Y_t = [X_{1,t,0}, ..., X_{1,t,M-1}, X_{2,t,0}, ..., X_{N,t,M-1}]

    The inner product uses delta = 1/M  so that:
        ⟨Y_s, Y_t⟩ = Σ_i ⟨X_{i,s}, X_{i,t}⟩_{L²[0,1]}
    i.e. the sum of per-pair L² inner products.

    Returns
    -------
    Y     : (T, N*M) array
    delta : float = 1/M
    """
    N, T, M = X_3d.shape
    Y       = X_3d.transpose(1, 0, 2).reshape(T, N * M)   # (T, N·M)
    delta   = 1.0 / M      # spacing of the functional grid (same as single pair)
    return Y, delta


def run_global_panel_test(X_3d, days, pairs,
                          j_list=(1, 2), K=20, alpha=0.05, eps=0.05,
                          Delta_v=0.0225, Delta_tau_frac=0.10,
                          Delta_v_grid=(0.0, 0.01, 0.0225, 0.05),
                          do_binary_seg=False, plot=True):
    """
    Test the stability of the GLOBAL loading structure via the NM×NM
    joint covariance operator of the full panel.

    RATIONALE
    ----------
    Each day t is treated as ONE functional observation:
        Y_t = (X_{1,t}, ..., X_{N,t}) ∈ L²([0,1])^N  ≅  R^{N·M}

    The covariance operator is NM×NM.  Its j-th eigenvector v_j ∈ R^{NM}
    can be reshaped to (N, M): it encodes BOTH the intraday shape AND the
    cross-pair pattern of the j-th global loading.

    The eigenvalues τ_j are the factor variances (same as FFM eigenvalues).
    Testing stability of v_j  =  testing if the j-th GLOBAL loading direction
    (i.e. the full N-pair loading function) has rotated significantly.

    This is the correct object for the Grassmann-geometry framework:
        ‖v^(1)_j − v^(2)_j‖² is the squared Grassmann distance on the
        unit sphere in R^{NM} (= rotation of the j-th loading shape
        across all pairs simultaneously).

    IMPLEMENTATION
    ---------------
    The NM×NM covariance is never materialised. All eigendecompositions
    use the DUAL n×n Gram matrix  G^(i)(λ) = Z^(i)_λ @ Z^(i)_λ.T / n_i(λ),
    and eigenvectors are recovered via  v_j = Z.T @ w_j / √(n σ_j Δ).
    Cost: O(T · N · M) for the Gram matrix + O(n² · N·M) for running kernels.

    Parameters
    ----------
    X_3d   : (N_pairs, T, M) — smoothed log-vol panel
    days   : DatetimeIndex of length T
    pairs  : list[str] of length N_pairs
    j_list : eigenvalue/function indices to test (1-based)
    K      : ν grid size
    alpha  : test level
    eps    : CUSUM trim fraction
    Delta_v         : eigenfunction relevance threshold (default 0.0225)
    Delta_tau_frac  : eigenvalue threshold fraction (default 0.10)
    Delta_v_grid    : tuple of Δv values to report
    do_binary_seg   : bool — also run binary segmentation on the stacked series
    plot            : bool — save figures

    Returns
    -------
    dict with keys:
        Y         : (T, N*M) stacked series
        cp        : change-point output
        tests     : {j: {eigenvalue: ..., eigenfunction: ...}}
        breaks    : binary segmentation results
        v1_reshaped, v2_reshaped : (N, M) reshaped global eigenvectors
    """
    N_pairs, _, M = X_3d.shape
    Y, delta      = stack_panel(X_3d)

    print(f"\n{'='*65}")
    print(f"  GLOBAL PANEL TEST  ({N_pairs} pairs × {M} intervals = {N_pairs*M} dims)")
    print(f"  Inner product: δ=1/M={delta:.4f}  (sum of L² inner products)")
    print(f"{'='*65}")

    # ── Change point ──────────────────────────────────────────────────────
    print("\n[1] Dominant change point (NM×NM operator)...")
    cp = estimate_change_point(Y, eps=eps, dates=days)
    k0 = cp["k0"]
    print(f"  θ̂ = {cp['theta_hat']:.4f}   k₀ = {k0}")
    if cp["break_date"]:
        print(f"  Break date: {str(cp['break_date'])[:10]}")
    if plot:
        _plot_cusum(cp, days, "global_panel")

    # ── Full-sample eigenstructure ─────────────────────────────────────────
    n_eigs         = max(j_list) + 1
    tau1_all, tau2_all, _, _ = _full_eigensystems(Y, k0, n_eigs, delta)
    print(f"\n  Top {min(4, n_eigs)} eigenvalues (full panel NM×NM):")
    print(f"  Pre-break  : {', '.join(f'{v:.4f}' for v in tau1_all[:4])}")
    print(f"  Post-break : {', '.join(f'{v:.4f}' for v in tau2_all[:4])}")

    tests = {}

    for j in j_list:
        # Spectral gap check
        if j >= 2 and j <= len(tau2_all) - 1:
            gap  = float(tau2_all[j-1] - tau2_all[j])
            frac = gap / (tau2_all[j-1] + 1e-14)
            ok   = frac >= 0.20
            print(f"\n  j={j}: spectral gap = {frac:.1%}  {'✓' if ok else '⚠ <20%'}")
            if not ok:
                continue

        print(f"\n[2–5] Tests for j={j} (global NM×NM)...")

        # Eigenvalue test
        Delta_tau = (Delta_tau_frac * float(tau1_all[j-1])) ** 2
        res_e = eigenvalue_test(Y, j=j, Delta_tau=Delta_tau,
                                alpha=alpha, K=K, eps=eps)
        print(f"  EIGENVALUE : τ^(1)={res_e['tau1_j']:.5f}  "
              f"τ^(2)={res_e['tau2_j']:.5f}  "
              f"stat={res_e['stat']:.3f}  reject={res_e['reject']}")

        # Eigenfunction test
        res_v_all = {}
        for dv in Delta_v_grid:
            res_v_all[dv] = eigenfunction_test(
                Y, j=j, Delta_v=float(dv), alpha=alpha, K=K, eps=eps
            )
        res_v = res_v_all[Delta_v]
        for dv, rv in res_v_all.items():
            flag = "REJECT" if rv["reject"] else "no rej"
            print(f"  EIGFUNC Δv={dv:.4f}: D̂={rv['D_hat']:.5f}  "
                  f"stat={rv['stat']:.3f}  [{flag}]")
        print(f"  CI: [0, {res_v['CI_upper']:.5f}]")

        # Reshape eigenvectors (N, M) for interpretation
        v1_flat = res_v["v1_j"]   # (N*M,)
        v2_flat = res_v["v2_j"]
        v1_mat  = v1_flat.reshape(N_pairs, M)
        v2_mat  = v2_flat.reshape(N_pairs, M)

        # L²-norms per pair (how much each pair contributes to the j-th direction)
        norm1 = np.sqrt(np.sum(v1_mat**2, axis=1) * delta)   # (N,)
        norm2 = np.sqrt(np.sum(v2_mat**2, axis=1) * delta)
        print(f"\n  Per-pair contribution to j={j} global eigenfunction:")
        print(f"  {'Pair':12s}  {'‖v^(1)‖_L²':>12s}  {'‖v^(2)‖_L²':>12s}  {'Change':>8s}")
        print("  " + "-" * 52)
        for i, p in enumerate(pairs):
            chg = norm2[i] - norm1[i]
            print(f"  {p:12s}  {norm1[i]:>12.4f}  {norm2[i]:>12.4f}  {chg:>+8.4f}")

        if plot:
            _plot_global_eigvec(v1_mat, v2_mat, norm1, norm2,
                                pairs, j, days, k0)

        tests[j] = {
            "eigenvalue":   res_e,
            "eigenfunction": res_v,
            "v1_reshaped":  v1_mat,
            "v2_reshaped":  v2_mat,
        }

    # ── Binary segmentation ────────────────────────────────────────────────
    breaks = []
    if do_binary_seg:
        print("\n[6] Binary segmentation on stacked series...")
        breaks = binary_segmentation(Y, dates=days, min_segment=100, eps=eps)
        print(f"  Found {len(breaks)} break(s):")
        for b in breaks:
            d = str(b["date"])[:10] if b["date"] else f"k={b['index']}"
            print(f"    θ={b['theta']:.4f}  {d}  f={b['f_value']:.4f}")

    return {
        "Y":             Y,
        "cp":            cp,
        "tests":         tests,
        "breaks":        breaks,
        "tau1":          tau1_all,
        "tau2":          tau2_all,
    }


def _plot_global_eigvec(v1_mat, v2_mat, norm1, norm2, pairs, j, days, k0):
    """
    Plot the global j-th eigenfunction: heatmap (N pairs × M intervals)
    for pre- and post-break, plus per-pair contribution norms.
    """
    N = len(pairs)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, N * 0.35 + 2)),
                             gridspec_kw={"width_ratios": [4, 4, 2]})

    vmax = max(np.abs(v1_mat).max(), np.abs(v2_mat).max())

    for ax, vmat, lbl in [(axes[0], v1_mat, "Pre-break"),
                          (axes[1], v2_mat, "Post-break")]:
        im = ax.imshow(vmat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[0, 1, N, 0])
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_yticks(np.arange(N) + 0.5)
        ax.set_yticklabels(pairs, fontsize=7)
        ax.set_xlabel("τ  [0=open → 1=close]", fontsize=9)
        if days is not None and k0 > 0:
            period = ("before " + str(days[k0])[:10]
                      if lbl == "Pre-break"
                      else "after  " + str(days[k0])[:10])
        else:
            period = lbl
        ax.set_title(f"v^{lbl.split('-')[0].lower()[:1]}_{j}(τ)\n{period}",
                     fontsize=10, fontweight="bold")

    # Per-pair norms bar chart
    ax3 = axes[2]
    y   = np.arange(N)
    ax3.barh(y - 0.2, norm1, 0.35, label="pre",  color="#2980B9", alpha=0.8)
    ax3.barh(y + 0.2, norm2, 0.35, label="post", color="#E74C3C", alpha=0.8)
    ax3.set_yticks(y); ax3.set_yticklabels(pairs, fontsize=7)
    ax3.set_xlabel("‖v_j‖_L² per pair", fontsize=9)
    ax3.set_title("Pair contribution\n(larger = more affected)", fontsize=9)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="x")

    d_break = str(days[k0])[:10] if days is not None and k0 < len(days) else ""
    fig.suptitle(f"Global j={j} eigenfunction — break: {d_break}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"DK_global_eigvec_j{j}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


# =============================================================================
# PANEL API — run directly on vol_ffm_all (already smoothed)
# =============================================================================

def run_panel_analysis(vol_ffm_all, pairs=None,
                       j_list=(1,), K=20, alpha=0.05, eps=0.05,
                       Delta_v=0.0225, Delta_tau_frac=0.10,
                       Delta_v_grid=(0.0, 0.01, 0.05, 0.0225),
                       do_binary_seg=False, min_coverage=0.8,
                       summary_only=False):
    """
    Apply the Dette & Kutta (2021) tests to an already-smoothed panel.

    Accepts the dict produced by smoothing_for_ffm() directly — no reload,
    no re-smoothing.  Each pair's DataFrame is converted to a (T, M) array
    via build_panel_from_matrices (handles NaN, active columns, date alignment).

    Typical usage
    -------------
    # In your notebook, after:
    #   vol_ffm_all = smoothing_for_ffm(vol_matrices_all, smooth=True)
    #
    from dette_kutta import run_panel_analysis
    dk_results = run_panel_analysis(
        vol_ffm_all,
        pairs=['eurusd', 'usdjpy', 'gbpusd'],   # None → all pairs
        j_list=[1, 2],
        K=20, alpha=0.05,
        Delta_v=0.0225,
        Delta_tau_frac=0.10,
    )

    Parameters
    ----------
    vol_ffm_all   : dict {str: DataFrame(days × minutes)} — smoothed log-vol
    pairs         : list[str] or None — subset of pairs; None = all
    j_list        : iterable[int] — eigenvalue/function indices (1-based)
    K             : int — ν grid size
    alpha         : float — test level
    eps           : float — CUSUM trim fraction
    Delta_v       : float — eigenfunction relevance threshold (default 0.0225)
    Delta_tau_frac: float — eigenvalue threshold as fraction of τ^(1)_j (10%)
    Delta_v_grid  : tuple — Δv values to report in the eigenfunction test
    do_binary_seg : bool — run binary segmentation (slow for large N; default False)
    min_coverage  : float — day coverage threshold for build_panel_from_matrices
    summary_only  : bool — print only the summary table, skip per-pair details

    Returns
    -------
    dict {pair: {
        'X'       : (T, M) array,
        'days'    : DatetimeIndex,
        'cp'      : estimate_change_point output,
        'tests'   : {j: {'eigenvalue': ..., 'eigenfunction': ...}},
        'breaks'  : list of binary-seg breaks,
        'tau1'    : eigenvalues pre-break,
        'tau2'    : eigenvalues post-break,
    }}

    Also prints a compact summary table across all pairs.
    """
    sys.path.insert(0, _HERE)
    from ffm_fx_real import build_panel_from_matrices

    if pairs is None:
        pairs = list(vol_ffm_all.keys())

    # ── Build aligned panel (T, M) shared across all pairs ───────────────────
    sub_dict = {p: vol_ffm_all[p] for p in pairs if p in vol_ffm_all}
    missing  = [p for p in pairs if p not in vol_ffm_all]
    if missing:
        print(f"  ⚠ Pairs not found in vol_ffm_all: {missing}")
    if not sub_dict:
        raise ValueError("No valid pairs found in vol_ffm_all.")

    print(f"\n  Building aligned panel for {len(sub_dict)} pairs...")
    X_3d, days, tau, pair_keys = build_panel_from_matrices(
        sub_dict, min_coverage=min_coverage
    )
    N, T, M = X_3d.shape   # N = n_pairs, T = n_days, M = n_intervals
    delta   = 1.0 / M
    print(f"  Panel: {N} pairs × {T} days × {M} intervals")

    all_results = {}

    # Summary accumulators
    summary_rows = []

    for pair_idx, pair in enumerate(pair_keys):
        X    = X_3d[pair_idx]   # (T, M)

        if not summary_only:
            print(f"\n{'─'*60}")
            print(f"  {pair.upper():10s}  (pair {pair_idx+1}/{N})")

        # ── Change point ──────────────────────────────────────────────────
        cp    = estimate_change_point(X, eps=eps, dates=days)
        k0    = cp["k0"]
        bd    = str(cp["break_date"])[:10] if cp["break_date"] else "?"

        if not summary_only:
            print(f"  Break: θ̂={cp['theta_hat']:.3f}  k₀={k0}  date={bd}")
            _plot_cusum(cp, days, pair)

        # ── Spectral structure ────────────────────────────────────────────
        n_eigs         = max(j_list) + 1
        tau1_all, tau2_all, _, _ = _full_eigensystems(
            X, k0, n_eigs, delta
        )

        pair_tests = {}

        for j in j_list:
            # Spectral gap check (Assumption 2.3)
            if j >= 2 and j <= len(tau2_all) - 1:
                gap  = float(tau2_all[j - 1] - tau2_all[j])
                frac = gap / (tau2_all[j - 1] + 1e-14)
                if frac < 0.20:
                    if not summary_only:
                        print(f"  j={j}: skip (spectral gap {frac:.1%} < 20%)")
                    continue

            # Eigenvalue test
            Delta_tau = (Delta_tau_frac * float(tau1_all[j - 1])) ** 2
            res_e = eigenvalue_test(X, j=j, Delta_tau=Delta_tau,
                                    alpha=alpha, K=K, eps=eps)

            # Eigenfunction test (main Δv + grid)
            res_v_list = {}
            for dv in Delta_v_grid:
                res_v_list[dv] = eigenfunction_test(
                    X, j=j, Delta_v=float(dv), alpha=alpha, K=K, eps=eps
                )
            res_v_main = res_v_list[Delta_v]

            if not summary_only:
                print(f"\n  j={j}")
                print(f"    τ^(1)={res_e['tau1_j']:.4f}  "
                      f"τ^(2)={res_e['tau2_j']:.4f}  "
                      f"Δτ={Delta_tau:.5f}")
                print(f"    Eigenvalue : stat={res_e['stat']:.3f}  "
                      f"reject={res_e['reject']}")
                for dv, rv in res_v_list.items():
                    flag = "REJECT" if rv["reject"] else "no rej"
                    print(f"    Eigfunc Δv={dv:.4f}: "
                          f"D̂={rv['D_hat']:.5f}  stat={rv['stat']:.3f}  [{flag}]")

                _plot_eigenfunctions(
                    res_v_main["v1_j"], res_v_main["v2_j"],
                    tau, j, pair, days, k0
                )

            pair_tests[j] = {"eigenvalue": res_e, "eigenfunction": res_v_main}

            # Row for summary table
            summary_rows.append({
                "pair":       pair,
                "j":          j,
                "break_date": bd,
                "theta_hat":  round(cp["theta_hat"], 3),
                "tau1_j":     round(float(tau1_all[j-1]), 4),
                "tau2_j":     round(float(tau2_all[j-1]), 4),
                "rel_chg_%":  round(abs(tau1_all[j-1]-tau2_all[j-1])/tau1_all[j-1]*100, 1),
                "E_hat":      round(res_e["E_hat"], 5),
                "stat_eig":   round(res_e["stat"], 2) if not np.isnan(res_e["stat"]) else np.nan,
                "rej_eig":    res_e["reject"],
                "D_hat":      round(res_v_main["D_hat"], 5),
                "stat_efn":   round(res_v_main["stat"], 2) if not np.isnan(res_v_main["stat"]) else np.nan,
                "rej_efn":    res_v_main["reject"],
            })

        # ── Binary segmentation (optional) ───────────────────────────────
        breaks = []
        if do_binary_seg:
            breaks = binary_segmentation(X, dates=days, min_segment=100, eps=eps)

        all_results[pair] = {
            "X":       X,
            "days":    days,
            "cp":      cp,
            "tests":   pair_tests,
            "breaks":  breaks,
            "tau1":    tau1_all,
            "tau2":    tau2_all,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        print(f"\n{'='*90}")
        print(f"  SUMMARY — Dette & Kutta tests  (K={K}, α={alpha}, Δv={Delta_v})")
        print(f"{'='*90}")
        cols_show = ["pair", "j", "break_date", "theta_hat",
                     "tau1_j", "tau2_j", "rel_chg_%",
                     "D_hat", "stat_efn", "rej_efn",
                     "E_hat", "stat_eig", "rej_eig"]
        print(df_sum[cols_show].to_string(index=False))

        # Count rejections
        rej_efn = df_sum["rej_efn"].sum()
        rej_eig = df_sum["rej_eig"].sum()
        total   = len(df_sum)
        print(f"\n  Eigenfunction test : {rej_efn}/{total} rejections "
              f"({rej_efn/total:.0%})  [Δv={Delta_v}]")
        print(f"  Eigenvalue test    : {rej_eig}/{total} rejections "
              f"({rej_eig/total:.0%})  [Δτ = (10%×τ^(1)_j)²]")

        all_results["_summary"] = df_sum

    return all_results


# =============================================================================
# PLOTTING
# =============================================================================

def _plot_cusum(cp, days, pair):
    """Plot CUSUM f(k) with the argmax marked."""
    f    = cp["f_values"]
    k0   = cp["k0"]
    N    = len(f)
    _, ax = plt.subplots(figsize=(12, 4))

    if days is not None and len(days) == N:
        x = pd.DatetimeIndex(days)
        ax.plot(x, f, lw=1.2, color="#2980B9")
        ax.axvline(x[k0], color="#E74C3C", lw=2, linestyle="--",
                   label=f"Break: {str(x[k0])[:10]}")
        ax.set_xlabel("Date")
    else:
        ax.plot(f, lw=1.2, color="#2980B9")
        ax.axvline(k0, color="#E74C3C", lw=2, linestyle="--",
                   label=f"k₀={k0}")
        ax.set_xlabel("k")

    ax.set_ylabel("f(k)  [CUSUM statistic]")
    ax.set_title(f"Full-operator CUSUM — {pair.upper()}  "
                 f"(θ̂ = {cp['theta_hat']:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"DK_cusum_{pair}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


def _plot_eigenfunctions(v1, v2, tau, j, pair, days, k0):
    """Plot v^(1)_j vs v^(2)_j on the intraday clock [0,1]."""
    _, ax = plt.subplots(figsize=(10, 4))
    d_label = ""
    if days is not None and k0 < len(days):
        d1 = f"{str(days[0])[:10]}–{str(days[k0-1])[:10]}"
        d2 = f"{str(days[k0])[:10]}–{str(days[-1])[:10]}"
        d_label = f"\nPre: {d1}  |  Post: {d2}"

    ax.plot(tau, v1, lw=2, color="#2980B9", label=r"$\hat{v}^{(1)}_j$  (pre-break)")
    ax.plot(tau, v2, lw=2, color="#E74C3C", linestyle="--",
            label=r"$\hat{v}^{(2)}_j$  (post-break)")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("τ  [0 = session open → 1 = session close]")
    ax.set_ylabel("Eigenfunction value (L²-normalised)")
    ax.set_title(f"Eigenfunctions  j={j} — {pair.upper()}{d_label}")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"DK_eigfunc_{pair}_j{j}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


def _plot_binary_seg(cp, breaks, days, pair):
    """Plot CUSUM f(k) with all binary-segmentation break points marked."""
    f  = cp["f_values"]
    N  = len(f)
    _, ax = plt.subplots(figsize=(13, 4))

    if days is not None and len(days) == N:
        x = pd.DatetimeIndex(days)
        ax.plot(x, f, lw=1.0, color="#2980B9")
        for b in breaks:
            bd = x[b["index"]] if b["index"] < len(x) else None
            if bd is not None:
                ax.axvline(bd, color="#E74C3C", lw=1.5, linestyle="--", alpha=0.8)
                ax.text(bd, f.max() * 0.9, str(bd)[:10],
                        fontsize=6, rotation=90, va="top", ha="right", color="#C0392B")
        ax.set_xlabel("Date")
    else:
        ax.plot(f, lw=1.0, color="#2980B9")
        for b in breaks:
            ax.axvline(b["index"], color="#E74C3C", lw=1.5, linestyle="--")
        ax.set_xlabel("k")

    ax.set_ylabel("f(k)")
    ax.set_title(f"Binary segmentation — {pair.upper()} ({len(breaks)} breaks)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"DK_binseg_{pair}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


def _plot_eigenvalue_running(tau1_lam, tau2_lam, E_lam, lam_grid, valid,
                             E_hat, Delta_tau, CI_upper,
                             j, pair_name, k0, dates):
    """
    Two-panel figure for eigenvalue_test:
      Left  — running eigenvalues τ^(1)(λ) and τ^(2)(λ) vs λ
      Right — running statistic Ê(λ) = (τ^(1)(λ)−τ^(2)(λ))² vs λ,
              with Ê(1) and Δτ as horizontal lines
    """
    lam_v = lam_grid[valid]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: τ trajectories
    axes[0].plot(lam_v, tau1_lam[valid], "o-", lw=1.5, ms=4,
                 color="#2980B9", label=r"$\hat{\tau}^{(1)}_{j,\lambda}$")
    axes[0].plot(lam_v, tau2_lam[valid], "s--", lw=1.5, ms=4,
                 color="#E74C3C", label=r"$\hat{\tau}^{(2)}_{j,\lambda}$")
    axes[0].axvline(1.0, color="gray", lw=0.8, ls=":")
    axes[0].set_xlabel("λ"); axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title(f"Running eigenvalues  j={j}")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Right: Ê(λ)
    axes[1].plot(lam_v, E_lam[valid], "o-", lw=1.5, ms=4,
                 color="#8E44AD", label=r"$\hat{E}_{j,N}(\lambda)$")
    axes[1].axhline(E_hat,    color="#E74C3C", lw=1.5, ls="--",
                    label=f"Ê(1) = {E_hat:.5f}")
    axes[1].axhline(Delta_tau, color="gray", lw=1.2, ls=":",
                    label=f"Δτ = {Delta_tau:.5f}")
    axes[1].axhspan(0, CI_upper, alpha=0.08, color="#E74C3C",
                    label=f"CI upper = {CI_upper:.5f}")
    axes[1].set_xlabel("λ"); axes[1].set_ylabel("Ê(λ)")
    axes[1].set_title(f"Running statistic Ê(λ)  j={j}")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    bd = (f"  break k={k0}" + (f" ({str(dates[k0])[:10]})"
          if dates is not None and k0 < len(dates) else ""))
    fig.suptitle(f"Eigenvalue test  j={j}  —  {pair_name}{bd}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"DK_eigenvalue_running_{pair_name}_j{j}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


def _plot_running_stat(stat_lam, lam_grid, valid, stat_hat,
                       Delta, CI_upper, label, j, pair_name):
    """
    Single-panel figure: running statistic (D̂(λ) or D_r(λ)) vs λ.
    Shows the full-sample value, the relevance threshold Δ, and the CI.
    """
    lam_v  = lam_grid[valid]
    stat_v = stat_lam[valid]

    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lam_v, stat_v, "o-", lw=1.5, ms=4,
            color="#27AE60", label=f"{label}(λ)")
    ax.axhline(stat_hat, color="#E74C3C", lw=1.8, ls="--",
               label=f"{label}(1) = {stat_hat:.5f}")
    if Delta > 0:
        ax.axhline(Delta, color="gray", lw=1.2, ls=":",
                   label=f"Δ = {Delta:.5f}")
    ax.axhspan(0, CI_upper, alpha=0.07, color="#E74C3C",
               label=f"CI upper = {CI_upper:.5f}")
    ax.set_xlabel("λ  [fraction of sub-sample]")
    ax.set_ylabel(label)
    ax.set_title(f"Running {label}  j={j}  —  {pair_name}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    tag = f"j{j}" if j != 0 else f"r{j}"
    fname = os.path.join(FIGURES_DIR,
                         f"DK_running_{label.replace('(','').replace(')','').replace(' ','_')}_{pair_name}_{tag}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


def _plot_principal_angles(p_angles, r, k0, dates,
                           pairs=None, M_functional=None):
    """
    Bar chart of the r principal angles between the two eigenspaces.
    """
    _, ax = plt.subplots(figsize=(max(5, r * 1.2), 4))
    colors = ["#E74C3C" if a > 10 else "#2980B9" for a in p_angles]
    ax.bar(range(1, r + 1), p_angles, color=colors, alpha=0.85)
    ax.axhline(10, color="gray", lw=1, ls="--",
               label="10° reference")
    ax.set_xticks(range(1, r + 1))
    ax.set_xticklabels([f"θ_{i}" for i in range(1, r + 1)])
    ax.set_xlabel("Principal angle index")
    ax.set_ylabel("Degrees")
    ax.set_ylim(0, max(p_angles) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    bd = ""
    if dates is not None and k0 is not None and k0 < len(dates):
        bd = f"  (break: {str(dates[k0])[:10]})"
    label = f"r={r}"
    if pairs is not None and M_functional is not None:
        label += f"  ({len(pairs)} pairs × {M_functional})"
    ax.set_title(f"Principal angles — subspace test  {label}{bd}",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"DK_principal_angles_r{r}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {fname}")


# =============================================================================
# MAIN — standalone execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dette & Kutta (2021) break tests for FX volatility."
    )
    parser.add_argument("--validate",  action="store_true",
                        help="Run Monte Carlo validation (Section 3.1)")
    parser.add_argument("--fx",        action="store_true",
                        help="Run FX application (EURUSD default)")
    parser.add_argument("--pairs",     nargs="+", default=["eurusd"],
                        help="FX pair(s) to analyse")
    parser.add_argument("--data_path", default=str(DATA_DIR) + "/",
                        help="Path to parquet data files")
    parser.add_argument("--N",        type=int, default=500,
                        help="N for validation (default 500)")
    parser.add_argument("--n_reps",   type=int, default=1000,
                        help="Replications for validation")
    parser.add_argument("--alpha",    type=float, default=0.05)
    parser.add_argument("--K",        type=int,   default=20)
    parser.add_argument("--j",        type=int,   default=1)
    args = parser.parse_args()

    if args.validate:
        run_validation(
            N=args.N, M=50, n_reps=args.n_reps,
            K=args.K, alpha=args.alpha, j=args.j,
            Delta_v=0.02
        )

    if args.fx:
        run_fx_analysis(
            pairs=args.pairs,
            data_path=args.data_path,
            j_list=[args.j],
            K=args.K, alpha=args.alpha,
        )

    if not args.validate and not args.fx:
        # Quick smoke test
        print("Running quick smoke test (N=200, M=30, n_reps=100)...")
        run_validation(N=200, M=30, n_reps=100, n_basis=4,
                       K=20, alpha=0.05, j=1, Delta_v=0.02, verbose=True)
