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
    estimate_change_point_eigenvalue / _eigenfunction(X, j, ...) -> dict
    stack_panel(X_3d) -> (Y, delta)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from paths import fig_dir

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
# PLOTTING
# =============================================================================


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
