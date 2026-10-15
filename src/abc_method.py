"""
ABC method (Alessi, Barigozzi & Capasso, 2010) for selecting the number of factors.

Goal: data-driven choice of the number r of factors, robust to the arbitrary
finite-sample scale of the Bai-Ng penalty (Hallin, Nisol & Tavakoli, 2023, Sec. 4.2).

    IC(c, X_{N,T}, k) = V(k, U_T^(k)) + c * k * g(N, T)

Algorithm
---------
0. Grid of penalty scalings C = {c_i : i = 1, ..., I}.
1. Shuffle the cross-sectional order (rows) P times.
2. For each permutation p, build nested sub-panels {(N_j, T_j), j = 1, ..., J}
   from the top-left block, with (N_J, T_J) = (N, T).
3. For each (i, j, p):  r_hat(c_i, j, p) = argmin_k IC(c_i, X_{N_j, T_j}, k).
4. Form the stability curve  c_i -> Var_j[ r_hat(c_i, j, p) ].
5. Pick c_hat_p at the mid-point of the *second* zero-variance plateau
   (the first plateau, c -> 0, is the trivial over-fit r_hat = k_max);
   set r_hat = median_p r_hat(c_hat_p, J, p).
"""

import numpy as np
from factor_model_fcts import build_gram_matrix, decompose_gram
from scipy.linalg import eigh


def g_IC1a(N, T):
    """Penalty g(N,T) for IC1a — Hallin, Nisol & Tavakoli (2023), Sec 4.2."""
    return np.sqrt((N + T) / (N * T)) * np.log(N * T / (N + T))

def g_IC2a(N, T):
    """Penalty g(N,T) for IC2a — Hallin, Nisol & Tavakoli (2023), Sec 4.2."""
    C_NT = min(np.sqrt(N), np.sqrt(T))
    return np.sqrt((N + T) / (N * T)) * np.log(C_NT ** 2)

def shuffle_rows(Y, seed=None):
    """Shuffle the rows of Y (cross-sectional order)"""
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(Y.shape[0])
    return Y[idx, :, :]

def nested_subpanels(N, T, J=10, frac_min=0.8):
    """Generate nested sub-panels (N_j, T_j) with N_j increasing from floor(frac_min * N) to N."""
    N_min = int(np.floor(frac_min * N))
    N_j = np.floor(np.linspace(N_min, N, J)).astype(int)
    T_j = np.full(J, T)  # T fixed
    return list(zip(N_j, T_j))

def _precompute_for_panel(Y, k_max):
    """
    Compute V(k) for panel Y. Called ONCE per sub-panel.

    V(k) = V0 - sum_{l=1}^k lambda_l / T
    where lambda_l = eigenvalues of F = Y_mat Y_mat^T / (N*M).
    Each factor l explains lambda_l/T of variance (orthogonal factors).
    """
    N, T, M = Y.shape
    G = build_gram_matrix(Y)
    eigenvalues, _ = decompose_gram(G, k_max=k_max)
    V0 = np.sum(Y ** 2) / (N * T * M)
    V_vals = np.maximum(V0 - np.cumsum(eigenvalues) / T, 0.0)   # (k_max,)
    return V_vals

def _r_hat_from_V(V_vals, k_max, c, g_val):
    """argmin IC(c,k) given pre-computed V(k) and g(N,T). O(k_max) — cheap."""
    ks = np.arange(1, k_max + 1)
    return int(np.argmin(V_vals + c * ks * g_val)) + 1

def compute_r_hat(Y, k_max, c, g_func):
    """Compute r̂(c) = argmin_k IC(c, Y, k) for a given panel Y (single call)."""
    N, T, M = Y.shape
    V_vals = _precompute_for_panel(Y, k_max)
    return _r_hat_from_V(V_vals, k_max, c, g_func(N, T))

def select_c_from_stability_curve(var_curve, c_grid, r_curve=None, k_max=None,
                                   min_plateau=3):
    """Select c* at the mid-point of the first valid zero-variance plateau.

    Algorithm:
      1. Find all contiguous runs of exact-zero variance.
      2. Skip trivial runs where r̂ = k_max throughout (overfit at c ≈ 0).
         Falls back to skipping just the first run if r_curve is unavailable.
      3. Among remaining runs, pick the first one with >= min_plateau points
         and return its mid-point.
      4. Fallbacks (in order): longest non-trivial run → last zero point.
    """
    is_zero = (var_curve == 0)

    # Build list of (start_idx, end_idx) for every contiguous zero run
    runs = []
    i = 0
    while i < len(is_zero):
        if is_zero[i]:
            j = i
            while j < len(is_zero) and is_zero[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if not runs:
        return c_grid[len(c_grid) // 2]

    def _is_trivial(start, end):
        """True when the run is the trivial k_max overfit plateau."""
        if r_curve is not None and k_max is not None:
            mid = (start + end) // 2
            return int(r_curve[mid]) == int(k_max)
        # Without r_curve: assume the first run is trivial
        return (start, end) == runs[0]

    # Filter to non-trivial runs
    candidate_runs = [(s, e) for s, e in runs if not _is_trivial(s, e)]

    if not candidate_runs:
        candidate_runs = runs  # safety: keep all if filtering removed everything

    # Pick first run with enough consecutive zeros
    for start, end in candidate_runs:
        if end - start + 1 >= min_plateau:
            return (c_grid[start] + c_grid[end]) / 2

    # Fallback: midpoint of the longest candidate run (even if short)
    start, end = max(candidate_runs, key=lambda x: x[1] - x[0])
    return (c_grid[start] + c_grid[end]) / 2

def plot_stability_curves(var_curves, c_grid, c_star_list, r_hat_per_perm, r_hat,
                          r_curves=None, save_path=None):
    """
    Plot ABC stability curves c -> Var_j[r̂] for each permutation.
    If r_curves is provided, overlays r̂(c) on a twin right axis.

    Parameters
    ----------
    var_curves      : (P, len(c_grid))
    c_grid          : (len(c_grid),)
    c_star_list     : (P,) — optimal ĉ per permutation
    r_hat_per_perm  : (P,) — r̂ per permutation
    r_hat           : int  — final r̂ (median)
    r_curves        : (P, len(c_grid)) or None — r̂(c) on the full panel
    save_path       : str or None
    """
    import matplotlib.pyplot as plt

    P = var_curves.shape[0]
    fig, axes = plt.subplots(1, P, figsize=(4.5 * P, 4), sharey='row')
    if P == 1:
        axes = [axes]

    r_max = int(r_curves.max()) + 1 if r_curves is not None else None

    for p, ax in enumerate(axes):
        # ── Variance curve (left axis) ──────────────────────────────────────
        ax.plot(c_grid, var_curves[p], color='steelblue', linewidth=1.5,
                label='Var$_j$[$\\hat{r}$]')

        # Highlight the selected plateau (consecutive zeros around c_star)
        is_zero = var_curves[p] == 0
        c_star  = c_star_list[p]
        # Find the zero run that contains c_star
        i = 0
        while i < len(c_grid):
            if is_zero[i]:
                j = i
                while j < len(c_grid) and is_zero[j]:
                    j += 1
                if c_grid[i] <= c_star <= c_grid[j - 1]:
                    ax.axvspan(c_grid[i], c_grid[j - 1],
                               color='limegreen', alpha=0.15, zorder=0,
                               label='Selected plateau')
                i = j
            else:
                i += 1

        ax.axvline(c_star, color='crimson', linestyle='--', linewidth=1.3,
                   label=f'ĉ = {c_star:.3f}')
        ax.set_xlabel('c', fontsize=9)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # ── r̂(c) curve (right axis) ────────────────────────────────────────
        if r_curves is not None:
            ax2 = ax.twinx()
            ax2.step(c_grid, r_curves[p], where='post',
                     color='darkorange', linewidth=1.4, alpha=0.85,
                     label='$\\hat{r}$(c)')
            ax2.axhline(r_hat_per_perm[p], color='darkorange',
                        linestyle=':', linewidth=0.9, alpha=0.6)
            ax2.set_ylabel('$\\hat{r}$(c)', color='darkorange', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=8)
            ax2.set_ylim(0, r_max + 0.5)
            ax2.yaxis.set_major_locator(
                plt.MaxNLocator(integer=True, nbins=r_max))
            # Merge legends
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7,
                      loc='upper right')
        else:
            ax.legend(fontsize=7)

        ax.set_title(
            f'Permutation {p + 1} — ĉ = {c_star:.3f},  r̂ = {r_hat_per_perm[p]}',
            fontsize=9)

    axes[0].set_ylabel('Var$_j$[$\\hat{r}$]', fontsize=9)
    fig.suptitle(f'ABC stability curves  —  final r̂ (median) = {r_hat}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def abc_select_r(Y_full, k_max=10,
                 c_grid=None, P=5, n_subpanels=10,
                 frac_min=0.8, seed=42, verbose=True):
    """
    ABC method (Alessi-Barigozzi-Capasso) — Hallin, Nisol & Tavakoli (2023), Sec 4.2.

    IC(c, k) = V(k) + c * k * g(N, T)      k = 1, ..., k_max

    Algorithm:
    - sweep a grid of c ∈ {0, 0.05, ..., 10}
    - nested sub-panels of size N_j = floor(4N/5 + N(j-1)/45), T_j = T fixed
    - P permutations of the cross-sectional order
    - for each c: Var_j[r̂] across sub-panels
    - ĉ = mid-point of the 2nd plateau where Var = 0  (1st plateau = over-fit at c≈0)
    - r̂ = median of r̂(ĉ) over permutations

    Parameters
    ----------
    Y_full      : ndarray (N, T, M)
    k_max       : maximum number of factors to test
    c_grid      : grid of c values; default {0, 0.05, ..., 10}
    P           : number of permutations
    n_subpanels : number of sub-panels (J = 10 in the paper)
    frac_min    : minimal fraction of N for the sub-panels (4/5 = 0.8 in the paper)
    seed        : random seed

    Returns
    --------
    r_hat           : int   — estimated number of factors (median over permutations)
    r_hat_per_perm  : (P,) — r̂ for each permutation
    c_star_list     : (P,) — optimal ĉ for each permutation
    c_grid          : grid of c used
    var_curves      : (P, len(c_grid)) — Var_j[r̂] for each (p, c)
    """

    if c_grid is None:
        c_grid = np.arange(0, 10.05, 0.05)
    g_func = g_IC1a

    # Centering + normalisation — idempotent if Y is already normalised (std≈1)
    Y_full = Y_full - Y_full.mean(axis=1, keepdims=True)
    sigma  = float(np.std(Y_full))
    if sigma > 0:
        Y_full = Y_full / sigma

    N, T, M = Y_full.shape
    var_curves     = np.zeros((P, len(c_grid)))
    r_curves       = np.zeros((P, len(c_grid)), dtype=int)
    r_hat_per_perm = np.zeros(P, dtype=int)
    c_star_list    = np.zeros(P)

    for p in range(P):
        if verbose:
            print(f"Permutation {p+1}/{P}")
        Y_perm    = shuffle_rows(Y_full, seed=seed + p)
        subpanels = nested_subpanels(N, T, J=n_subpanels, frac_min=frac_min)

        # Pre-compute V(k) once per sub-panel (not once per c value)
        cache = []
        for (N_j, T_j) in subpanels:
            Y_sub  = Y_perm[:N_j, :T_j, :]
            V_vals = _precompute_for_panel(Y_sub, k_max)
            g_val  = g_func(N_j, T_j)
            cache.append((V_vals, g_val))

        # Sweep c grid using cached V(k) — O(k_max) per (subpanel, c)
        for i, c in enumerate(c_grid):
            r_hats = [_r_hat_from_V(V_vals, k_max, c, g_val)
                      for V_vals, g_val in cache]
            var_curves[p, i] = np.var(r_hats)
            r_curves[p, i]   = r_hats[-1]   # r̂ at the full panel (largest subpanel)

        c_star_list[p]    = select_c_from_stability_curve(
            var_curves[p], c_grid, r_curve=r_curves[p], k_max=k_max)
        r_hat_per_perm[p] = compute_r_hat(Y_perm, k_max, c_star_list[p], g_func)
    r_hat = int(np.median(r_hat_per_perm))
    return r_hat, r_hat_per_perm, c_star_list, c_grid, var_curves, r_curves


def abc_per_period(data, start_date, end_date, n_months,
                   step_months=1, k_max=6,
                   c_grid=None, P=3, n_subpanels=10,
                   frac_min=0.8, seed=42, min_coverage=0.8):
    """
    Rolling ABC estimation of r — same structure as ffm_per_period.

    For each window [start, start + n_months):
      1. build_panel_from_matrices  →  X
      2. centering_data             →  Y
      3. normalize_panel            →  Y_norm  (V0 ≈ 1)
      4. abc_select_r(Y_norm)       →  r̂_ABC

    No need for run_ffm — lightweight preprocessing only.

    Parameters
    ----------
    data         : dict {str: pd.DataFrame} — vol_matrices
    start_date   : str or Timestamp
    end_date     : str or Timestamp
    n_months     : int   — window length in months
    step_months  : int   — rolling step (default 1)
    k_max        : int   — maximum number of factors (default 6)
    c_grid       : array — grid of c; default [0, 2] with step 0.005
    P            : int   — permutations (default 3, fast for rolling)
    n_subpanels  : int   — sub-panels J (default 10)
    frac_min     : float — minimal fraction of N for the sub-panels (default 0.8)
    seed         : int   — random seed
    min_coverage : float — minimal daily coverage for build_panel (default 0.8)

    Returns
    -------
    pd.DataFrame : start, end, r_hat_abc, c_star_median, n_pairs, n_days
    """
    import pandas as pd
    import io, contextlib
    from factor_model_fcts import build_panel_from_matrices

    if c_grid is None:
        # Must cover the whole descent of r̂(c) down to ER.
        # The final transition to r̂=1 happens at c ~ λ_2/(g·T).
        # For N≈44, T≈3062: c_thresh ≈ 0.17 → c_max=2.0 is more than enough.
        # A step of 0.005 resolves wide plateaus without over-resolving narrow ones.
        c_grid = np.arange(0, 2.005, 0.005)

    results_list  = []
    current_start = pd.to_datetime(start_date)
    end_date_ts   = pd.to_datetime(end_date)
    period_num    = 0

    while current_start < end_date_ts:
        current_end = current_start + pd.DateOffset(months=n_months)
        period_num += 1

        # Same pair filter as ffm_per_period
        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = ((pd.to_datetime(df.index) >= current_start) &
                        (pd.to_datetime(df.index) <  current_end))
                subset[pair] = df.loc[mask]
        r_hat_abc  = np.nan
        c_star_med = np.nan
        n_pairs    = len(subset)
        n_days     = np.nan

        if len(subset) >= 2:
            try:
                # Build panel only — centering/normalisation happens in abc_select_r
                with contextlib.redirect_stdout(io.StringIO()):
                    X, _, _, _ = build_panel_from_matrices(subset, min_coverage)
                n_days = X.shape[1]

                if X.shape[0] >= 2 and n_days >= 20:
                    r_hat_abc, _, c_star_list, _, _, _ = abc_select_r(
                        X, k_max=k_max, c_grid=c_grid,
                        P=P, n_subpanels=n_subpanels,
                        frac_min=frac_min, seed=seed,
                        verbose=False
                    )
                    c_star_med = float(np.median(c_star_list))
            except Exception as e:
                print(f"  ⚠  {current_start.date()} → {current_end.date()} : {e}")

        print(f"[{period_num:>3}] {current_start.date()} → {current_end.date()} "
              f"| N={n_pairs:>2}  T={int(n_days) if not np.isnan(n_days) else '?':>4} "
              f"| r̂_ABC = {r_hat_abc}")
        print(f"      median c* over permutations = {c_star_med:.3f}")

        results_list.append({
            'start'        : current_start.date(),
            'end'          : current_end.date(),
            'r_hat_abc'    : r_hat_abc,
            'c_star_median': c_star_med,
            'n_pairs'      : n_pairs,
            'n_days'       : n_days,
        })

        current_start = current_start + pd.DateOffset(months=step_months)

    return pd.DataFrame(results_list)