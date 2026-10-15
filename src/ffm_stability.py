#Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from ffm_fx_real import run_ffm
from paths import FIGURES_ROOT
#==================================================================================================================
# GRASSMANN DISTANCE FUNCTIONS
#================================================================================================================
'''
1. stack the loadings + orthonormal basis → Q_t
2. take Q_t and Q_{t+1} and compute the Grassmann distance between the two
    subspaces spanned by the columns of Q_t and Q_{t+1}; return the angle
    cos(theta)= singular value max
    return angle
3. 
'''

# global structure : all factors and all FX pairs
def empilation_loadings(B_hat):
    # B_hat : N x r x M 
    # return : B_hat_stacked : (N*r) x M
    #          Vt : r x r 
    N, r, M = B_hat.shape
    B_hat_stacked = B_hat.transpose(0, 2, 1).reshape(N * M, r)

    U, _, _ = np.linalg.svd(B_hat_stacked, full_matrices=False)
    print(f"Empilation → B_hat_stacked shape: {B_hat_stacked.shape}, U shape: {U.shape}" )
    return B_hat_stacked, U

def grassmann_distance(Q1, Q2):
    # Singular values of Q1^T Q2
    M = np.dot(Q1.T, Q2)
    _, S, _ = np.linalg.svd(M)
    
    theta = np.arccos(np.clip(S, -1.0, 1.0))
    distance = np.sqrt(np.sum(theta**2)) 

    return distance, theta

def projector_distance(Q1, Q2):
    P1 = Q1 @ Q1.T
    P2 = Q2 @ Q2.T
    return np.linalg.norm(P1 - P2, 'fro') / np.sqrt(2)

def align_signs(U_old, U_new):
    """Align the signs of U_new on U_old, column by column."""
    signs = np.sign(np.diag(U_old.T @ U_new))
    signs[signs == 0] = 1
    return U_new * signs

from functools import partial
run_ffm_r1 = partial(run_ffm, r_forced=1, silent= True)

def grassman_rolling_window( # compare each window with the previous one
        data , 
        window_size = 6 ,  
        step = 1 , 
        func = run_ffm_r1, 
        start_date = '2008-01-01', 
        end_date = '2025-12-31'
        ):
    '''
    Parameters: 
    -----------
    data       : vol_matrices — dict of DataFrames with a datetime index
    window_size : window size in months
    step        : step size in months
    start_date  : start date of the rolling window (str or Timestamp)
    end_date    : end date of the rolling window (str or Timestamp)

    Returns:
    ---------
    A Df month x distance
    '''
    results = []
    current_start = pd.to_datetime(start_date)
    end_date      = pd.to_datetime(end_date)
    U_old         = None  # no previous window at the start

    while current_start < end_date:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = (pd.to_datetime(df.index) >= current_start) & \
                       (pd.to_datetime(df.index) <= current_end)
                subset[pair] = df.loc[mask]
        print(f"Processing → window: {current_start.date()} to {current_end.date()}")
        if len(subset) > 0:
            ffm         = func(subset)
            B_hat_new   = ffm['B_hat']             # (N, r, M)
            _, U_new    = empilation_loadings(B_hat_new)  # (NM, r)

            if U_old is not None:                
                U_new_aligned = align_signs(U_old, U_new)
                distance, theta = grassmann_distance(U_old, U_new_aligned)
                #d_proj = projector_distance(U_old, U_new)
                results.append({
                    'start'    : current_start.date(),
                    'end'      : current_end.date(),
                    'distance_grass' : distance,
                    'angles'   : theta.tolist(),
                    'Nb of days': ffm['X'].shape[1]
                })
                print(f"{current_start.date()} → {current_end.date()} | d_G = {distance:.4f}")

            U_old = U_new 

        current_start += pd.DateOffset(months=step)

    results = pd.DataFrame(results)
    results['speed'] = results['distance_grass'].diff()
    results['acceleration'] = results['speed'].diff()

    return results

def grassman_rolling_window_ref( # compare each window with the first 3/6/12 months of 2014
        data , 
        window_size = 6 ,  
        step = 1 , 
        func = run_ffm_r1, 
        start_date = '2014-01-01', 
        end_date = '2025-12-31'
        ):
    '''
    Parameters: 
    -----------
    data       : vol_matrices — dict of DataFrames with a datetime index
    window_size : window size in months
    step        : step size in months
    start_date  : start date of the rolling window (str or Timestamp)
    end_date    : end date of the rolling window (str or Timestamp)

    Returns:
    ---------
    A Df month x distance
    '''
    results = []
    start_date   = pd.to_datetime(start_date)
    end_date     = pd.to_datetime(end_date)
    current_start = start_date

    # ── Reference: first window [start_date, start_date + window_size] ──
    ref_end = current_start + pd.DateOffset(months=window_size)
    subset_ref = {}
    for pair, df in data.items():
        first_date = pd.to_datetime(df.index).min()
        if first_date < ref_end:
            mask = (pd.to_datetime(df.index) >= current_start) & \
                   (pd.to_datetime(df.index) <= ref_end)
            subset_ref[pair] = df.loc[mask]

    U_ref = None
    mu_ref_dict = None
    if len(subset_ref) > 0:
        ffm_ref     = func(subset_ref)
        B_hat_ref   = ffm_ref['B_hat']
        _, U_ref    = empilation_loadings(B_hat_ref)
        mu_ref_dict = dict(zip(ffm_ref['pairs'], ffm_ref['mu']))
        print(f"Reference window: {current_start.date()} → {ref_end.date()} "
              f"| N={len(subset_ref)} pairs")

    current_start = ref_end

    while current_start < end_date:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = (pd.to_datetime(df.index) >= current_start) & \
                       (pd.to_datetime(df.index) <= current_end)
                subset[pair] = df.loc[mask]
        print(f"Processing → window: {current_start.date()} to {current_end.date()}")
        if len(subset) > 0:
            ffm         = func(subset, mu_ref=mu_ref_dict)
            B_hat_new   = ffm['B_hat']             # (N, r, M)
            _, U_new    = empilation_loadings(B_hat_new)  # (NM, r)


            U_new_aligned = align_signs(U_ref, U_new)
            distance, theta = grassmann_distance(U_ref, U_new_aligned)
                #d_proj = projector_distance(U_ref, U_new)
            results.append({
                    'start'    : current_start.date(),
                    'end'      : current_end.date(),
                    'distance_grass' : distance,
                    'angles'   : theta.tolist(),
                    #distance_proj' : d_proj,
                    'Nb of days': ffm['X'].shape[1]
                })
            print(f"{current_start.date()} → {current_end.date()} | d_G = {distance:.4f}")

        current_start += pd.DateOffset(days=step)

    results = pd.DataFrame(results)
    results['speed'] = results['distance_grass'].diff()
    results['acceleration'] = results['speed'].diff()

    return results


#=================================================================================================================
# GRASSMANN PAIRWISE HEATMAP
#=================================================================================================================

def grassmann_distance_matrix(
        data,
        window_size=6,
        step=1,
        func=run_ffm_r1,
        start_date='2008-01-01',
        end_date='2025-12-31',
):
    '''
    Compute all 6-month rolling windows (step days, same grid as
    grassman_rolling_window_ref), fit FFM on each (all windows centred
    on the mean of the first window via mu_ref), then compute all
    pairwise Grassmann distances → symmetric matrix.

    Returns
    -------
    dist_matrix : np.ndarray (K x K)
    labels      : list of str  (start date of each window, "YYYY-MM-DD")
    '''
    current_start = pd.to_datetime(start_date)
    end_date_ts   = pd.to_datetime(end_date)

    # ── Reference window: compute mu_ref from [start_date, start_date+window] ──
    ref_end = current_start + pd.DateOffset(months=window_size)
    subset_ref = {}
    for pair, df in data.items():
        first_date = pd.to_datetime(df.index).min()
        if first_date < ref_end:                          # fixed: < ref_end
            mask = (pd.to_datetime(df.index) >= current_start) & \
                   (pd.to_datetime(df.index) <= ref_end)
            subset_ref[pair] = df.loc[mask]

    mu_ref_dict = None
    U_list = []
    labels = []
    if len(subset_ref) > 0:
        ffm_ref     = func(subset_ref)
        mu_ref_dict = dict(zip(ffm_ref['pairs'], ffm_ref['mu']))
        _, U_ref    = empilation_loadings(ffm_ref['B_hat'])
        U_list.append(U_ref)
        labels.append(current_start.strftime('%Y-%m'))
        print(f"Reference window: {current_start.date()} → {ref_end.date()} "
              f"| N={len(subset_ref)} pairs")

    current_start = current_start + pd.DateOffset(months=step)

    while current_start < end_date_ts:                    # fixed: allow last window past end_date
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:                  # fixed: < current_end
                mask = (pd.to_datetime(df.index) >= current_start) & \
                       (pd.to_datetime(df.index) <= current_end)
                subset[pair] = df.loc[mask]

        if len(subset) > 0:
            try:
                print(f"Processing → {current_start.date()} to {current_end.date()}")
                ffm       = func(subset, mu_ref=mu_ref_dict)
                _, U_new  = empilation_loadings(ffm['B_hat'])
                U_list.append(U_new)
                labels.append(current_start.strftime('%Y-%m'))
            except Exception as e:
                print(f"  Skipped: {e}")

        current_start += pd.DateOffset(months=step)

    K = len(U_list)
    dist_matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(i + 1, K):
            U_j_aligned = align_signs(U_list[i], U_list[j])
            d, _        = grassmann_distance(U_list[i], U_j_aligned)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix, labels


def plot_grassmann_heatmap(dist_matrix, labels, figsize=(14, 11), save_path=None):
    '''
    Plot a symmetric heatmap of pairwise Grassmann distances.

    Parameters
    ----------
    dist_matrix : np.ndarray (K x K)
    labels      : list of str  (one per window, e.g. "2014-01")
    save_path   : optional path to save the figure
    '''
    # Keep only a readable subset of tick labels (every 6 months)
    n = len(labels)
    tick_step = max(1, n // 20)
    tick_labels = [l if i % tick_step == 0 else '' for i, l in enumerate(labels)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto',
                   vmin=0, vmax=dist_matrix.max())

    plt.colorbar(im, ax=ax, label='Grassmann distance $d_G$')

    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_xlabel('Period $s$ (6-month window)', fontsize=11)
    ax.set_ylabel('Period $t$ (6-month window)', fontsize=11)
    ax.set_title('Pairwise Grassmann distance between all 6-month windows', fontsize=13)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return fig, ax


#=================================================================================================================
# BAI-PERRON TEST
#=================================================================================================================

def _dp_breaks(ssr_table, m):
    """
    Dynamic programming: m break points minimising total SSR.
    V[k, t] = min SSR with k breaks in y[0..t]  (k+1 segments).
    Returns (total_SSR, list of end-indices of the first m segments).
    """
    T  = ssr_table.shape[0]
    V  = np.full((m + 1, T), np.inf)
    bp = np.full((m + 1, T), -1, dtype=int)

    V[0, :] = ssr_table[0, :]          # 0 breaks: single segment [0..t]

    for k in range(1, m + 1):
        for t in range(1, T):
            s_vec     = np.arange(t)
            cands     = V[k - 1, s_vec] + ssr_table[s_vec + 1, t]
            best      = int(np.argmin(cands))
            V[k, t]   = cands[best]
            bp[k, t]  = best

    # Backtrack: bp[k, t] = end of (k-th from last) segment boundary
    breaks = []
    t = T - 1
    for k in range(m, 0, -1):
        s = bp[k, t]
        breaks.append(s)
        t = s
    breaks.reverse()
    return float(V[m, T - 1]), breaks


#=================================================================================================================
# BAI-PERRON ON A SCALAR SERIES OF DISTANCES
#=================================================================================================================

def _build_ssr_table_scalar(y, h):
    """
    SSR(i,j) = Σ_{t=i}^j (y_t − ȳ_{ij})²
             = Σy²[i:j+1] − (Σy[i:j+1])² / (j−i+1)
    O(1) per cell via prefix sums. Full table: O(T²).
    """
    T      = len(y)
    cum_y  = np.concatenate([[0.0], np.cumsum(y)])
    cum_y2 = np.concatenate([[0.0], np.cumsum(y ** 2)])
    ssr    = np.full((T, T), np.inf)
    for i in range(T - h + 1):
        j_arr         = np.arange(i + h - 1, T)
        n_arr         = (j_arr - i + 1).astype(float)
        sy            = cum_y[j_arr + 1]  - cum_y[i]
        sy2           = cum_y2[j_arr + 1] - cum_y2[i]
        ssr[i, j_arr] = sy2 - sy ** 2 / n_arr
    return ssr

_BP_CV_5PCT = {
    0.05: {1: 8.58,  2: 7.22,  3: 5.96,  4: 5.26,  5: 4.84},
    0.10: {1: 9.63,  2: 7.59,  3: 6.05,  4: 5.25,  5: 4.77},
    0.15: {1: 10.13, 2: 7.87,  3: 6.14,  4: 5.25,  5: 4.71},
    0.20: {1: 10.74, 2: 8.12,  3: 6.20,  4: 5.22,  5: 4.65},
    0.25: {1: 11.35, 2: 8.36,  3: 6.24,  4: 5.19,  5: 4.59},
}

def _long_run_var_nw(u: np.ndarray, bandwidth: int) -> float:
    """
    Newey-West long-run variance with a Bartlett kernel.
 
    σ²_LR = γ(0) + 2 Σ_{k=1}^{H} w_k γ(k)
    w_k   = 1 − k/(H+1)   (Bartlett weights, decreasing)
 
    Notes:
      - bandwidth must be ≥ the MA order to cover all the autocorrelation.
        For MA(5), use bandwidth ≥ 6.
      - u is demeaned internally.
      - Returns max(lrv, 1e-12) to avoid a division by zero.
    """
    u  = u - u.mean()          # demean
    T  = len(u)
    g0 = np.dot(u, u) / T     # empirical variance
    lrv = g0
    for k in range(1, bandwidth + 1):
        w_k = 1.0 - k / (bandwidth + 1)        # Bartlett weight
        g_k = np.dot(u[k:], u[:-k]) / T        # autocovariance of order k
        lrv += 2.0 * w_k * g_k
    return max(lrv, 1e-12)

def _get_cv(h_frac: float, m: int) -> float:
    """
    Return the 5% critical value for m breaks at the closest h_frac.
    If m > 5, use the value for m=5 (conservative).
    """
    eps_keys = sorted(_BP_CV_5PCT.keys())
    eps      = min(eps_keys, key=lambda e: abs(e - h_frac))
    return _BP_CV_5PCT[eps].get(min(m, 5), 4.5)

def bai_perron_distances(df, col='distance_grass', date_col='start',
                          max_breaks=5, h_frac=0.15, criterion='bic',
                          verbose=True):
    """
    Bai-Perron test on a scalar series of Grassmann distances.

    Finds the break dates that minimise the total within-segment
    sum of squares (break in the mean):

        (τ̂₁,…,τ̂_m) = argmin  Σ_j SSR(segment j)
        SSR(i,j) = Σ_{t=i}^j (y_t − ȳ_{ij})²

    Same dynamic programming as Bai-Perron (1998), q=1 (constant).

    Note: if the series is z-scored and the windows overlap,
    the observations are autocorrelated → approximate supF.

    Parameters
    ----------
    df         : DataFrame with columns [date_col, col]
    col        : column to test  (default 'distance_grass')
    date_col   : date column  (default 'start')
    max_breaks : maximum number of breaks  (default 5)
    h_frac     : minimal fraction of T per segment  (default 0.15)
    criterion  : 'bic' | 'lwz' | 'supF'
    verbose    : print the results table

    Returns
    -------
    dict : n_breaks, break_dates, boundaries, segment_means,
           segment_stds, ssr_by_m, bic, lwz, supF_stats, y, dates, col
    """
    y     = df[col].dropna().values
    dates = pd.to_datetime(df.loc[df[col].notna(), date_col].values)
    T     = len(y)
    h     = max(int(np.ceil(h_frac * T)), 2)
    max_breaks = min(max_breaks, T // h - 1)

    if max_breaks < 1:
        raise ValueError(f"T={T} too small for h_frac={h_frac}.")

    if verbose:
        print(f"  Building SSR table : T={T}, h={h}...")

    ssr_table   = _build_ssr_table_scalar(y, h)
    ssr_by_m    = [float(ssr_table[0, T - 1])]
    breaks_by_m = [[]]
    for m in range(1, max_breaks + 1):
        ssr_m, brks = _dp_breaks(ssr_table, m)
        ssr_by_m.append(ssr_m)
        breaks_by_m.append(brks)

    bic = [T * np.log(s / T) + (k + 1) * np.log(T)
           for k, s in enumerate(ssr_by_m)]
    lwz = [T * np.log(s / T) + (k + 1) * 0.299 * (np.log(T) ** 2.1)
           for k, s in enumerate(ssr_by_m)]

    ssr0 = ssr_by_m[0]
    supF = []
    for m in range(1, max_breaks + 1):
        ssrm   = ssr_by_m[m]
        F_stat = ((T - (m + 1)) / m) * (ssr0 - ssrm) / ssrm if ssrm > 0 else np.inf
        supF.append(F_stat)

    _CV5 = {1: 8.58, 2: 7.22, 3: 5.96, 4: 5.26, 5: 4.84}

    if criterion == 'bic':
        n_breaks = int(np.argmin(bic))
    elif criterion == 'lwz':
        n_breaks = int(np.argmin(lwz))
    elif criterion == 'supF':
        n_breaks = 0
        for idx, Fm in enumerate(supF):
            if Fm > _CV5.get(idx + 1, 4.5):
                n_breaks = idx + 1
    else:
        raise ValueError("criterion must be 'bic', 'lwz' or 'supF'")

    opt_breaks = breaks_by_m[n_breaks]
    boundaries = [0] + [b + 1 for b in opt_breaks] + [T]

    segment_means, segment_stds = [], []
    for k in range(n_breaks + 1):
        seg = y[boundaries[k]:boundaries[k + 1]]
        segment_means.append(float(np.mean(seg)))
        segment_stds.append(float(np.std(seg, ddof=max(len(seg) - 1, 1))))

    break_dates = [dates[b] for b in opt_breaks] if opt_breaks else []

    if verbose:
        print("\n" + "=" * 65)
        print("  BAI-PERRON — GRASSMANN DISTANCE (scalar series)")
        print("=" * 65)
        print(f"  T={T}   h={h} ({h_frac:.0%})   criterion={criterion}")
        print(f"  Selected breaks : {n_breaks}\n")
        print(f"  {'m':>3}  {'SSR':>12}  {'BIC':>10}  {'LWZ':>10}  "
              f"{'Sup-F':>9}  {'CV5%':>6}")
        print("  " + "-" * 57)
        for m in range(max_breaks + 1):
            F_str  = f"{supF[m-1]:9.3f}" if m > 0 else "        —"
            cv_str = f"{_CV5.get(m, '—'):>6}" if m > 0 else "      —"
            mark   = "  ◄" if m == n_breaks else ""
            print(f"  {m:>3d}  {ssr_by_m[m]:>12.4f}  {bic[m]:>10.3f}  "
                  f"{lwz[m]:>10.3f}  {F_str}  {cv_str}{mark}")
        if n_breaks > 0:
            print(f"\n  Break dates:")
            for k, b_idx in enumerate(opt_breaks):
                print(f"    [{k+1}]  {dates[b_idx].date()}"
                      f"   μ_before={segment_means[k]:.4f}"
                      f"   μ_after={segment_means[k+1]:.4f}"
                      f"   Δμ={segment_means[k+1]-segment_means[k]:+.4f}")
        else:
            print(f"\n  No break.  μ={segment_means[0]:.4f}  "
                  f"σ={segment_stds[0]:.4f}")
        print("=" * 65 + "\n")

    return dict(
        n_breaks      = n_breaks,
        break_idx     = opt_breaks,
        break_dates   = break_dates,
        boundaries    = boundaries,
        segment_means = segment_means,
        segment_stds  = segment_stds,
        ssr_by_m      = ssr_by_m,
        bic           = bic,
        lwz           = lwz,
        supF_stats    = supF,
        criterion     = criterion,
        y             = y,
        dates         = dates,
        col           = col,
    )

def bai_perron_distances_hac(
    df,
    col          = 'distance_grass',
    date_col     = 'start',
    max_breaks   = 5,
    h_frac       = 0.15,
    criterion    = 'bic',
    nw_bandwidth = 6,
    verbose      = True,
):
    """
    Bai-Perron test on a scalar series of Grassmann distances (HAC-corrected).
 
    Tested model:  d_t = μ_j + u_t  for t in segment j,
    where u_t can be MA(p) with p ≤ nw_bandwidth.
 
    Parameters
    ----------
    df           : DataFrame with at least the columns [date_col, col]
    col          : column holding the Grassmann distances
    date_col     : date column (window start)
    max_breaks   : maximum number of breaks to test
    h_frac       : minimal fraction of T per segment (ε in Bai-Perron)
                   Recommended: 0.10–0.15 for T ~ 100–200
    criterion    : 'bic' | 'lwz' | 'supF'
                   BIC/LWZ → minimisation  |  supF → first significant break
    nw_bandwidth : Newey-West kernel bandwidth for σ²_LR
                   Must be ≥ the MA order of the residuals (MA(5) → use 6)
    verbose      : print the results table
 
    Returns
    -------
    dict with keys:
      n_breaks      : selected number of breaks
      break_idx     : indices of the breaks in y
      break_dates   : corresponding dates
      boundaries    : [0, i1, i2, ..., T]  (segment bounds)
      segment_means : mean per segment
      segment_stds  : standard deviation per segment
      ssr_by_m      : [SSR(0), SSR(1), ..., SSR(max_breaks)]
      bic           : BIC values for m = 0..max_breaks
      lwz           : LWZ values for m = 0..max_breaks
      supF_hac      : HAC-corrected F statistics for m = 1..max_breaks
      sigma2_lr     : long-run variance estimated under H0
      y             : series used (array)
      dates         : dates (array of Timestamps)
      col           : column name
    """
 
    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    y     = df[col].dropna().values.astype(float)
    dates = pd.to_datetime(df.loc[df[col].notna(), date_col].values)
    T     = len(y)
    h     = max(int(np.ceil(h_frac * T)), 2)
    max_breaks = min(max_breaks, T // h - 1)
 
    if max_breaks < 1:
        raise ValueError(
            f"T={T} too small for h_frac={h_frac}. "
            f"Reduce h_frac or use a longer series."
        )
 
    if verbose:
        print(f"\n  Preparation  :  T={T},  h_min={h} (ε={h_frac:.0%}),  "
              f"max_breaks={max_breaks},  NW bandwidth={nw_bandwidth}")
        print(f"  Building the SSR table ({T}×{T})...")
 
    # ------------------------------------------------------------------
    # 1. SSR table + DP for each m
    # ------------------------------------------------------------------
    ssr_table   = _build_ssr_table_scalar(y, h)
    ssr_by_m    = [float(ssr_table[0, T - 1])]   # m=0: a single segment
    breaks_by_m = [[]]
    for m in range(1, max_breaks + 1):
        ssr_m, brks = _dp_breaks(ssr_table, m)
        ssr_by_m.append(ssr_m)
        breaks_by_m.append(brks)
 
    # ------------------------------------------------------------------
    # 2. HAC long-run variance under H0 (demeaned full-sample residuals)
    #
    #    σ²_LR is estimated under the null of no break.
    #    Under a positive MA(5), σ²_LR > σ²_OLS → F_HAC < F_OLS.
    #    Without this correction the F statistics would be inflated and
    #    would detect too many breaks.
    # ------------------------------------------------------------------
    u_null    = y - y.mean()
    sigma2_lr = _long_run_var_nw(u_null, bandwidth=nw_bandwidth)
 
    # ------------------------------------------------------------------
    # 3. HAC-corrected supF statistics
    #
    #    F_m^HAC = ((T−(m+1))/m) × (SSR_0 − SSR_m) / (T × σ²_LR)
    #
    #    Rationale: under H0, (SSR_0 − SSR_m) / σ²_LR →_d χ²(mq)/m
    #    with q=1 (one changing parameter = the mean).
    #    The Bai-Perron (1998) Tab. 1 critical values are calibrated for this ratio.
    # ------------------------------------------------------------------
    ssr0     = ssr_by_m[0]
    supF_hac = []
    for m in range(1, max_breaks + 1):
        ssrm  = ssr_by_m[m]
        denom = T * sigma2_lr
        F_hac = ((T - (m + 1)) / m) * (ssr0 - ssrm) / denom
        supF_hac.append(float(F_hac))
 
    # ------------------------------------------------------------------
    # 4. BIC and LWZ
    #    Formules standard (Bai & Perron 2003, Eq. 8-9).
    #    Selection by SSR is super-consistent even under MA(p)
    #    → BIC/LWZ remain valid for choosing m.
    # ------------------------------------------------------------------
    bic = [T * np.log(s / T) + (k + 1) * np.log(T)
           for k, s in enumerate(ssr_by_m)]
    lwz = [T * np.log(s / T) + (k + 1) * 0.299 * (np.log(T) ** 2.1)
           for k, s in enumerate(ssr_by_m)]
 
    # ------------------------------------------------------------------
    # 5. Selecting the number of breaks
    # ------------------------------------------------------------------
    if criterion == 'bic':
        n_breaks = int(np.argmin(bic))
    elif criterion == 'lwz':
        n_breaks = int(np.argmin(lwz))
    elif criterion == 'supF':
        # First m for which F_HAC exceeds the 5% critical value
        n_breaks = 0
        for idx, Fm in enumerate(supF_hac):
            cv = _get_cv(h_frac, idx + 1)
            if Fm > cv:
                n_breaks = idx + 1
    else:
        raise ValueError("criterion must be 'bic', 'lwz' or 'supF'")
 
    # ------------------------------------------------------------------
    # 6. Optimal segments for the selected m
    # ------------------------------------------------------------------
    opt_breaks = breaks_by_m[n_breaks]
    boundaries = [0] + [b + 1 for b in opt_breaks] + [T]
 
    segment_means, segment_stds = [], []
    for k in range(n_breaks + 1):
        seg = y[boundaries[k]:boundaries[k + 1]]
        segment_means.append(float(np.mean(seg)))
        segment_stds.append(float(np.std(seg, ddof=max(len(seg) - 1, 1))))
 
    break_dates = [dates[b] for b in opt_breaks] if opt_breaks else []
 
    # ------------------------------------------------------------------
    # 7. Affichage
    # ------------------------------------------------------------------
    if verbose:
        W = 74
        print("\n" + "=" * W)
        print("  BAI-PERRON — GRASSMANN DISTANCE  (HAC-corrected supF)")
        print("=" * W)
        print(f"  T={T}  ε={h_frac:.0%}  h_min={h}  "
              f"criterion={criterion}  NW_bandwidth={nw_bandwidth}")
        print(f"  σ²_LR (Newey-West, H0) = {sigma2_lr:.6f}   "
              f"σ_LR = {sigma2_lr**0.5:.6f}")
        print(f"  Selected breaks: {n_breaks}\n")
 
        hdr = (f"  {'m':>3}  {'SSR':>12}  {'BIC':>10}  {'LWZ':>10}"
               f"  {'F_HAC':>9}  {'CV(5%)':>8}  {'sig?':>5}")
        print(hdr)
        print("  " + "-" * (W - 2))
 
        for m in range(max_breaks + 1):
            if m > 0:
                F_val  = supF_hac[m - 1]
                cv_val = _get_cv(h_frac, m)
                F_str  = f"{F_val:9.3f}"
                cv_str = f"{cv_val:8.2f}"
                sig    = "  *" if F_val > cv_val else "   "
            else:
                F_str  = "        —"
                cv_str = "       —"
                sig    = "   "
            mark = "  ◄" if m == n_breaks else ""
            print(f"  {m:>3d}  {ssr_by_m[m]:>12.4f}  {bic[m]:>10.3f}"
                  f"  {lwz[m]:>10.3f}  {F_str}  {cv_str}{sig}{mark}")
 
        if n_breaks > 0:
            print(f"\n  Break dates (super-consistent):")
            for k, b_idx in enumerate(opt_breaks):
                d_before = dates[b_idx].date()
                mu_bef   = segment_means[k]
                mu_aft   = segment_means[k + 1]
                print(f"    [{k+1}]  {d_before}"
                      f"   μ_before={mu_bef:.4f}"
                      f"   μ_after={mu_aft:.4f}"
                      f"   Δμ={mu_aft - mu_bef:+.4f}")
        else:
            print(f"\n  No break detected."
                  f"  μ={segment_means[0]:.4f}  σ={segment_stds[0]:.4f}")
 
        print("=" * W + "\n")
 
    return dict(
        n_breaks      = n_breaks,
        break_idx     = opt_breaks,
        break_dates   = break_dates,
        boundaries    = boundaries,
        segment_means = segment_means,
        segment_stds  = segment_stds,
        ssr_by_m      = ssr_by_m,
        bic           = bic,
        lwz           = lwz,
        supF_hac      = supF_hac,
        sigma2_lr     = float(sigma2_lr),
        y             = y,
        dates         = dates,
        col           = col,
        criterion     = criterion
    )


def plot_bai_perron_distances(bp_result, crisis_dates=None, figsize=(14, 5)):
    """
    Plot the results of bai_perron_distances:
      - scalar series (line)
      - mean per segment (horizontal line) + ±1σ band
      - break dates (red dashed lines)
      - crisis windows (grey background)
    """
    import matplotlib.dates as mdates

    y      = bp_result['y']
    dates  = bp_result['dates']
    bounds = bp_result['boundaries']
    seg_mu = bp_result['segment_means']
    seg_sd = bp_result['segment_stds']
    col    = bp_result.get('col', 'distance_grass')

    default_crises = {
        'CHF floor':  ('2015-01-01', '2015-03-01'),
        'Brexit':     ('2016-06-01', '2016-09-01'),
        'TRY crisis': ('2018-07-01', '2019-01-01'),
        'COVID-19':   ('2020-02-01', '2020-06-01'),
        'Ukraine':    ('2022-02-01', '2022-06-01'),
    }
    crises = crisis_dates if crisis_dates is not None else default_crises

    fig, ax = plt.subplots(figsize=figsize)

    for name, (s, e) in crises.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.10, color='grey', zorder=0)

    ax.plot(dates, y, color='steelblue', lw=1.0, alpha=0.8,
            label=col, zorder=2)

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(seg_mu)))
    for k in range(bp_result['n_breaks'] + 1):
        a, b   = bounds[k], bounds[k + 1] - 1
        d_a    = dates[a]
        d_b    = dates[b]
        mu, sd = seg_mu[k], seg_sd[k]
        ax.hlines(mu, d_a, d_b, colors=colors[k], lw=2.5, zorder=3,
                  label=f'Regime {k+1}: μ={mu:.3f}, σ={sd:.3f}')
        ax.fill_between([d_a, d_b], mu - sd, mu + sd,
                        alpha=0.10, color=colors[k], zorder=1)

    for k, bd in enumerate(bp_result['break_dates']):
        ax.axvline(pd.to_datetime(str(bd)), color='tomato',
                   lw=1.8, linestyle='--', alpha=0.9, zorder=4,
                   label='Breakpoint' if k == 0 else '_nolegend_')

    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.set_title(
        f"Bai-Perron — Grassmann Distance  |  "
        f"{bp_result['n_breaks']} breaks, {bp_result['criterion'].upper()}",
        fontsize=11
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(
        f'{FIGURES_ROOT}/STABILITY/bai_perron_dist_{bp_result["criterion"]}.png',
        dpi=160, bbox_inches='tight')
    plt.show()


#=================================================================================================================
# BAI-PERRON DIRECTLY ON THE FFM OBJECTIVE  (minimises the idiosyncratic variance)
#=================================================================================================================

def _top_r_eigenvalues(A, r, n_iter=60):
    """
    Top-r eigenvalues of symmetric PSD matrix A via subspace iteration.

    Much faster than full eigh for small r (r=1 or 2):
        cost = O(n² × r × n_iter)  vs  O(n³) for eigh.

    Convergence is guaranteed for PSD matrices; 60 iterations is sufficient
    for eigenvalue gaps typical of Gram matrices.
    """
    n = A.shape[0]
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, r)))
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(A @ Q)
    return np.diag(Q.T @ A @ Q)   # (r,) — unsorted Rayleigh quotients


def _build_ssr_table_ffm(X, F, grid_idx, r, h):
    """
    Precompute SSR(g_i, g_j) for all valid grid segments.

    For segment [g_i, g_j] (grid units) covering days [a, b]:

        SSR(g_i, g_j) = ||Y~^[a:b]||_F^2 − N*M * Σ_{l=1}^r λ_l(F[a:b, a:b])

    Y~ is the normalized centered panel (Y_norm = Y/σ, stored as ffm_result['F']),
    NOT the raw X.  F[a:b, a:b] is the Gram submatrix restricted to days [a, b].
    Variance is computed as N*M * diag(F) to stay in Y_norm units (consistent
    with N*M * eigenvalues on the same side of the SSR formula).

    The Gram submatrix trick works because global centering is fixed:
    F[a:b, a:b]_st = (1/NM) Σ_i <Y~_is, Y~_it>  for s,t ∈ [a,b].

    Cost: O(G² × n_avg² × r × n_iter)  with prefix sums for ||Y~||_F².
    """
    N, _T, M = X.shape
    scale    = float(N * M)          # F is normalised by 1/(N*M); undo to match Y_norm units
    T        = F.shape[0]
    G        = len(grid_idx)
    grid_ext = np.concatenate([grid_idx, [T]])    # sentinel: month G starts at T

    # F is built on Y_norm, not on X.  diag(F)[t] = (1/NM)||Y_norm[:,t,:]||^2,
    # so scale * diag(F)[t] = ||Y_norm[:,t,:]||^2  — consistent with scale * lams.
    yt_sq  = scale * np.diagonal(F)               # (T,) — per-day variance in Y_norm units
    cum_sq = np.concatenate([[0.0], np.cumsum(yt_sq)])   # prefix sums

    ssr = np.full((G, G), np.inf)

    for g_i in range(G - h + 1):
        a = int(grid_ext[g_i])
        for g_j in range(g_i + h - 1, G):
            b = int(grid_ext[g_j + 1]) - 1      # inclusive end (days)
            n = b - a + 1

            y_sq = float(cum_sq[b + 1] - cum_sq[a])

            if n <= r:                           # not enough days to fit r factors
                ssr[g_i, g_j] = y_sq
                continue

            lams = _top_r_eigenvalues(F[a:b+1, a:b+1], r=r)
            # eigenvalues of F are in units of y_sq/(N*M) — rescale before subtracting
            ssr[g_i, g_j] = max(y_sq - scale * float(np.sum(np.maximum(lams, 0))), 0.0)

    return ssr


def bai_perron_ffm(X, F, dates=None, r=1, grid='monthly',
                   max_breaks=5, h_frac=0.15, criterion='bic', verbose=True):
    """
    Bai-Perron test directly on the FFM objective function.

    Finds break dates (τ̂₁, …, τ̂_m) that minimise total idiosyncratic variance:

        (τ̂₁,…,τ̂_m) = argmin  Σ_{j=1}^{m+1} ||Ξ̂^(j)||_F²
                        τ₁<…<τ_m

    For segment j covering days [a, b]:

        SSR(a,b) = ||Y~^[a:b]||_F²  −  N*M * Σ_{l=1}^r λ_l( F[a:b, a:b] )
                 = total variance − explained variance on that segment
                 (F is normalised by 1/(N*M), so eigenvalues must be rescaled)

    The same dynamic programming as Bai-Perron (1998) is applied, but the
    SSR is the FFM idiosyncratic residual rather than a scalar regression
    residual.

    Parameters
    ----------
    X          : (N, T, M) normalized centered panel  — ffm_result['X']
    F          : (T, T) Gram matrix                   — ffm_result['F']
                 If not in ffm_result, compute:
                 F = np.einsum('ism,itm->st', X, X) / (N * M)
    dates      : (T,) array of trading dates
    r          : number of factors
    grid       : 'monthly' | 'quarterly' — granularity of candidate break dates
    max_breaks : max number of breaks to test
    h_frac     : min segment fraction of G (trimming, default 0.15)
    criterion  : 'bic' | 'lwz' | 'supF'
    verbose    : print result table

    Returns
    -------
    dict with: n_breaks, break_dates, boundaries_g, segment_r2,
               ssr_by_m, bic, lwz, supF_stats, grid_idx
    """
    N, T, M = X.shape

    # ── Temporal grid ────────────────────────────────────────────────────
    if dates is not None:
        dates_pd = pd.to_datetime(dates)
        freq     = 'M' if grid == 'monthly' else 'Q'
        df_tmp   = pd.DataFrame({'idx': range(T),
                                 'period': dates_pd.to_period(freq)})
        grid_idx = df_tmp.groupby('period')['idx'].first().values.astype(int)
    else:
        grid_idx = np.arange(T)

    G          = len(grid_idx)
    h          = max(int(np.ceil(h_frac * G)), 2)
    max_breaks = min(max_breaks, G // h - 1)

    if max_breaks < 1:
        raise ValueError(f"G={G} grid points too few for h_frac={h_frac}.")

    if verbose:
        print(f"\n  Building SSR table: G={G} {grid} segments, "
              f"r={r}, T={T} days, h={h}...")

    ssr_table = _build_ssr_table_ffm(X, F, grid_idx, r, h)

    # ── Dynamic programming ──────────────────────────────────────────────
    ssr_by_m    = [float(ssr_table[0, G - 1])]
    breaks_by_m = [[]]
    for m in range(1, max_breaks + 1):
        ssr_m, brks = _dp_breaks(ssr_table, m)
        ssr_by_m.append(ssr_m)
        breaks_by_m.append(brks)

    # ── Model selection criteria ─────────────────────────────────────────
    bic = [G * np.log(s / G) + (k + 1) * np.log(G)
           for k, s in enumerate(ssr_by_m)]
    lwz = [G * np.log(s / G) + (k + 1) * 0.299 * (np.log(G) ** 2.1)
           for k, s in enumerate(ssr_by_m)]

    ssr0 = ssr_by_m[0]
    supF = []
    for m in range(1, max_breaks + 1):
        ssrm  = ssr_by_m[m]
        F_stat = ((G - (m+1)) / m) * (ssr0 - ssrm) / ssrm if ssrm > 0 else np.inf
        supF.append(F_stat)

    _CV5 = {1: 8.58, 2: 7.22, 3: 5.96, 4: 5.26, 5: 4.84}

    if criterion == 'bic':
        n_breaks = int(np.argmin(bic))
    elif criterion == 'lwz':
        n_breaks = int(np.argmin(lwz))
    elif criterion == 'supF':
        n_breaks = 0
        for idx, Fm in enumerate(supF):
            if Fm > _CV5.get(idx + 1, 4.5):
                n_breaks = idx + 1
    else:
        raise ValueError(f"criterion must be 'bic', 'lwz' or 'supF'")

    # ── Segment statistics ───────────────────────────────────────────────
    opt_breaks_g = breaks_by_m[n_breaks]
    boundaries_g = [0] + [b + 1 for b in opt_breaks_g] + [G]
    grid_ext     = np.concatenate([grid_idx, [T]])

    # Consistent with _build_ssr_table_ffm: use scale * diag(F) for variance
    N_bp, _, M_bp = X.shape
    scale_bp      = float(N_bp * M_bp)
    yt_sq_full    = scale_bp * np.diagonal(F)          # (T,) in Y_norm units
    cum_sq_full   = np.concatenate([[0.0], np.cumsum(yt_sq_full)])

    segment_r2 = []
    for k in range(n_breaks + 1):
        g_a     = boundaries_g[k]
        g_b     = boundaries_g[k + 1] - 1
        a, b    = int(grid_ext[g_a]), int(grid_ext[g_b + 1]) - 1
        y_sq    = float(cum_sq_full[b + 1] - cum_sq_full[a])
        ssr_seg = ssr_table[g_a, g_b]
        segment_r2.append(1.0 - ssr_seg / y_sq if y_sq > 0 else 0.0)

    break_dates = None
    if dates is not None and opt_breaks_g:
        dates_arr   = list(dates)
        break_dates = [dates_arr[int(grid_ext[g + 1])] for g in opt_breaks_g]

    # ── Verbose output ───────────────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 68)
        print("  BAI-PERRON — FFM IDIOSYNCRATIC VARIANCE")
        print("=" * 68)
        print(f"  T={T} days, G={G} {grid} periods, r={r}, criterion={criterion}")
        print(f"  Selected number of breaks: {n_breaks}\n")
        print(f"  {'m':>3s}  {'SSR':>14s}  {'BIC':>10s}  {'LWZ':>10s}  "
              f"{'Sup-F':>9s}  {'CV5%':>6s}")
        print("  " + "-" * 60)
        for m in range(max_breaks + 1):
            F_str  = f"{supF[m-1]:9.3f}" if m > 0 else "        —"
            cv_str = f"{_CV5.get(m, '—'):>6}" if m > 0 else "      —"
            mark   = "  ◄" if m == n_breaks else ""
            print(f"  {m:>3d}  {ssr_by_m[m]:>14.4f}  {bic[m]:>10.3f}  "
                  f"{lwz[m]:>10.3f}  {F_str}  {cv_str}{mark}")
        if n_breaks > 0:
            print(f"\n  Break dates:")
            for k, g in enumerate(opt_breaks_g):
                bd    = break_dates[k] if break_dates else f"grid[{g}]"
                print(f"    [{k+1}]  {bd}"
                      f"   R²_before={segment_r2[k]:.4f}"
                      f"   R²_after={segment_r2[k+1]:.4f}"
                      f"   ΔR²={segment_r2[k+1]-segment_r2[k]:+.4f}")
        else:
            print(f"\n  No break detected.  Overall R² = {segment_r2[0]:.4f}")
        print("=" * 68 + "\n")

    return dict(
        n_breaks     = n_breaks,
        break_idx_g  = opt_breaks_g,
        break_dates  = break_dates,
        boundaries_g = boundaries_g,
        segment_r2   = segment_r2,
        ssr_by_m     = ssr_by_m,
        bic          = bic,
        lwz          = lwz,
        supF_stats   = supF,
        criterion    = criterion,
        grid_idx     = grid_idx,
        r            = r,
        yt_sq        = yt_sq_full,   # (T,) per-day variance in Y_norm units, for plotting
    )


def plot_bai_perron_ffm(X, dates, bp_result, figsize=(14, 5)):
    """
    Plot segment R² with break dates and per-segment horizontal lines.

    R²(segment j) = 1 - ||Ξ̂^(j)||_F² / ||Y~^(j)||_F²
    measures what fraction of the variance the FFM explains in each regime.
    """
    T          = X.shape[1]
    grid_idx   = bp_result['grid_idx']
    grid_ext   = np.concatenate([grid_idx, [T]])
    boundaries = bp_result['boundaries_g']
    seg_r2     = bp_result['segment_r2']
    dates_plot = pd.to_datetime(dates) if dates is not None else np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)

    # Per-grid-period variance for background series — use Y_norm units from bp_result
    yt_sq    = bp_result.get('yt_sq', np.sum(X ** 2, axis=(0, 2)))   # (T,)
    period_r2, period_dates = [], []
    for g in range(len(grid_idx)):
        a = int(grid_ext[g])
        b = int(grid_ext[g + 1]) - 1
        y_sq = float(yt_sq[a:b+1].sum())
        if y_sq > 0:
            period_dates.append(dates_plot[a])
            period_r2.append(y_sq)               # total variance per period (proxy)

    # Normalised total variance per period (to show temporal variation)
    pv = np.array(period_r2)
    pv = (pv - pv.mean()) / pv.std()
    ax.plot(period_dates, pv, color='#AAAAAA', lw=0.8, label='Total variance (normalized)', zorder=1)

    # Segment R² as horizontal lines
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(seg_r2)))
    for k in range(bp_result['n_breaks'] + 1):
        g_a  = boundaries[k]
        g_b  = boundaries[k + 1] - 1
        d_a  = dates_plot[int(grid_ext[g_a])]
        d_b  = dates_plot[int(grid_ext[g_b + 1]) - 1]
        ax.hlines(seg_r2[k], d_a, d_b,
                  colors=colors[k], lw=2.5, zorder=3,
                  label=f'Regime {k+1}: R²={seg_r2[k]:.3f}')

    if bp_result['break_dates']:
        for k, bd in enumerate(bp_result['break_dates']):
            ax.axvline(pd.to_datetime(str(bd)), color='tomato',
                       lw=1.8, linestyle='--', alpha=0.85, zorder=4,
                       label='Breakpoint' if k == 0 else '_nolegend_')

    ax.set_title(
        f"Bai-Perron FFM — {bp_result['n_breaks']} break(s), "
        f"{bp_result['criterion'].upper()}",
        fontsize=11)
    ax.set_ylabel("FFM R² per segment")
    ax.set_xlabel("Date")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/bai_perron_ffm_{bp_result["criterion"]}.png',
                dpi=160, bbox_inches='tight')
    plt.show()


#=================================================================================================================
# BAI-PERRON DIRECTLY ON LOADING SUBSPACES  (Grassmannian)
#=================================================================================================================

def _gram_projectors(U_list):
    """
    K[s,t] = ||U_s^T U_t||_F^2  =  Σ_l cos²(θ_l)  ∈ [0, r]

    Measures the similarity between two subspaces:
        K[s,t] = r  ↔  identical subspaces
        K[s,t] = 0  ↔  completely orthogonal subspaces

    Invariant to sign flips and factor permutations.
    Cost: O(W² · d · r)  with d = dim of U columns.
    """
    W = len(U_list)
    K = np.zeros((W, W))
    for s in range(W):
        r        = U_list[s].shape[1]
        K[s, s]  = float(r)               # ||I_r||_F^2 = r
        for t in range(s + 1, W):
            M          = U_list[s].T @ U_list[t]   # (r, r)
            k          = float(np.sum(M ** 2))
            K[s, t]    = k
            K[t, s]    = k
    return K


def _build_ssr_table_grassmann(K, r, h):
    """
    SSR(i,j) = (j-i+1)·r  −  RectSum(K,i,j) / (j-i+1)

    where  RectSum(K,i,j) = Σ_{s=i}^j Σ_{t=i}^j K[s,t]  is computed in O(1)
    via 2-D prefix sums.  Full table: O(W²).

    SSR = 0 iff all subspaces in [i,j] are identical  (stable regime).
    SSR is maximal when all subspaces are mutually orthogonal.
    """
    W  = K.shape[0]
    PS = np.zeros((W + 1, W + 1))
    PS[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)
    PS_diag    = np.diag(PS)          # PS[k,k] for fast diagonal access

    ssr = np.full((W, W), np.inf)
    for i in range(W - h + 1):
        j_arr = np.arange(i + h - 1, W)
        n_arr = (j_arr - i + 1).astype(float)
        # RectSum(i, j) = PS[j+1,j+1] - PS[i,j+1] - PS[j+1,i] + PS[i,i]
        rs = (PS_diag[j_arr + 1]
              - PS[i, j_arr + 1]
              - PS[j_arr + 1, i]
              + PS[i, i])
        ssr[i, j_arr] = n_arr * r - rs / n_arr
    return ssr


def collect_subspaces_rolling(data, window_size=6, step=1, func=None,
                               start_date='2014-01-01', end_date='2025-07-01'):
    """
    Run rolling windows and return the sequence of loading subspace bases U^(t).

    Like grassman_rolling_window but stores U^(t) ∈ R^{d×r} at every step
    instead of computing distances.  Used as input to bai_perron_grassmann.

    Returns
    -------
    U_list : list of (d, r) orthonormal matrices  (one per window)
    dates  : list of window-start dates
    """
    if func is None:
        func = run_ffm_r1

    U_list        = []
    dates         = []
    current_start = pd.to_datetime(start_date)
    end_date_ts   = pd.to_datetime(end_date)

    while current_start < end_date_ts:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = ((pd.to_datetime(df.index) >= current_start) &
                        (pd.to_datetime(df.index) <= current_end))
                subset[pair] = df.loc[mask]

        if subset:
            try:
                ffm       = func(subset)
                _, U_new  = empilation_loadings(ffm['B_hat'])
                U_list.append(U_new)
                dates.append(current_start.date())
                print(f"  {current_start.date()} → {current_end.date()} "
                      f"| U {U_new.shape}  Nb days={ffm['X'].shape[1]}")
            except Exception as e:
                print(f"  {current_start.date()} skipped: {e}")

        current_start += pd.DateOffset(days=step)

    # K[s,t] = U_s^T U_t requires all U to share the same ambient dimension d = N*M.
    # N can vary (pairs added over time) and M can vary (intraday grid changes).
    # Keep only windows with the most common d to guarantee compatibility.
    if U_list:
        from collections import Counter
        dims         = [U.shape[0] for U in U_list]
        target_d     = Counter(dims).most_common(1)[0][0]
        n_dropped    = sum(d != target_d for d in dims)
        if n_dropped:
            print(f"\n  [collect_subspaces_rolling] dropped {n_dropped}/{len(U_list)} "
                  f"windows with d ≠ {target_d}  (N or M mismatch)")
        keep     = [i for i, d in enumerate(dims) if d == target_d]
        U_list   = [U_list[i] for i in keep]
        dates    = [dates[i]  for i in keep]

    return U_list, dates


def bai_perron_grassmann(U_list, dates=None, max_breaks=5, h_frac=0.15,
                          criterion='bic', verbose=True):
    """
    Bai-Perron structural break test directly on loading subspaces,
    on the Grassmann manifold Gr(r, d).

    Model
    -----
    The subspace U^(t) belongs to regime j for t ∈ [T_{j-1}+1, T_j].
    Within each regime the subspaces cluster around a Fréchet mean P̄_j.

    SSR of segment [i,j] = Fréchet variance of {P^(t)}_{t∈[i,j]}:

        SSR(i,j) = Σ_{t=i}^j ||P^(t) − P̄_{ij}||_F²
                 = (j−i+1)·r − (1/(j−i+1)) · Σ_{s,t∈[i,j]} K[s,t]

    where  K[s,t] = ||U_s^T U_t||_F²  (Gram matrix of projectors, computed once).

    Advantages over bai_perron_on_distances()
    ------------------------------------------
    - No arbitrary reference window
    - Uses the full subspace geometry (not compressed to a scalar)
    - Invariant to factor sign and permutation

    Parameters
    ----------
    U_list     : list of W orthonormal (d, r) matrices from collect_subspaces_rolling
    dates      : list of W date labels (optional)
    max_breaks : max number of breaks to test  (default 5)
    h_frac     : min segment fraction of W     (default 0.15)
    criterion  : 'bic' | 'lwz' | 'supF'
    verbose    : print result table

    Returns
    -------
    dict with: n_breaks, break_idx, break_dates, boundaries,
               segment_cohesion, K, ssr_by_m, bic, lwz, supF_stats
    """
    r  = U_list[0].shape[1]
    W  = len(U_list)
    h  = max(int(np.ceil(h_frac * W)), 2)
    max_breaks = min(max_breaks, W // h - 1)

    if max_breaks < 1:
        raise ValueError(f"W={W} too small for h_frac={h_frac}: no room for a break.")

    if verbose:
        print(f"\n  Building Gram matrix ({W}×{W}, r={r})...")
    K         = _gram_projectors(U_list)
    ssr_table = _build_ssr_table_grassmann(K, r, h)

    ssr_by_m    = [float(ssr_table[0, W - 1])]
    breaks_by_m = [[]]
    for m in range(1, max_breaks + 1):
        ssr_m, brks = _dp_breaks(ssr_table, m)
        ssr_by_m.append(ssr_m)
        breaks_by_m.append(brks)

    # Normalise SSR by r so BIC/LWZ are on a comparable scale to the scalar case
    ssr_norm = [s / r for s in ssr_by_m]

    bic = [W * np.log(s / W) + (k + 1) * np.log(W)
           for k, s in enumerate(ssr_norm)]
    lwz = [W * np.log(s / W) + (k + 1) * 0.299 * (np.log(W) ** 2.1)
           for k, s in enumerate(ssr_norm)]

    ssr0 = ssr_by_m[0]
    supF = []
    for m in range(1, max_breaks + 1):
        ssrm = ssr_by_m[m]
        F    = ((W - (m + 1)) / m) * (ssr0 - ssrm) / ssrm if ssrm > 0 else np.inf
        supF.append(F)

    # Critical values: Bai & Perron (1998) Table 1, p=1, ε=0.15 — used as
    # approximate reference (exact distrib. on Gr(r,d) is non-standard)
    _CV5 = {1: 8.58, 2: 7.22, 3: 5.96, 4: 5.26, 5: 4.84}

    if criterion == 'bic':
        n_breaks = int(np.argmin(bic))
    elif criterion == 'lwz':
        n_breaks = int(np.argmin(lwz))
    elif criterion == 'supF':
        n_breaks = 0
        for idx, F_m in enumerate(supF):
            if F_m > _CV5.get(idx + 1, 4.5):
                n_breaks = idx + 1
    else:
        raise ValueError(f"criterion must be 'bic', 'lwz' or 'supF'")

    opt_breaks = breaks_by_m[n_breaks]
    boundaries = [0] + [b + 1 for b in opt_breaks] + [W]

    # Segment cohesion: mean pairwise similarity within segment, ∈ [0,1]
    # cohesion = 1  ↔  all subspaces in the segment are identical (very stable)
    # cohesion < 1  ↔  subspaces spread around the Fréchet mean
    segment_cohesion = []
    for k in range(n_breaks + 1):
        a, b = boundaries[k], boundaries[k + 1]
        n    = b - a
        rs   = float(np.sum(K[a:b, a:b]))
        segment_cohesion.append(rs / (n ** 2 * r))   # normalised ∈ [0, 1]

    break_dates = None
    if dates is not None and opt_breaks:
        dates_list  = list(dates)
        break_dates = [dates_list[b] for b in opt_breaks]

    if verbose:
        print("\n" + "=" * 70)
        print("  BAI-PERRON ON GRASSMANNIAN  (loading subspaces — no reference)")
        print("=" * 70)
        print(f"  W = {W}   r = {r}   h = {h} ({h_frac:.0%} trimming)"
              f"   criterion = {criterion}")
        print(f"  Selected number of breaks: {n_breaks}\n")
        print(f"  {'m':>3s}  {'SSR_G':>12s}  {'BIC':>10s}  {'LWZ':>10s}  "
              f"{'Sup-F':>9s}  {'CV 5%':>7s}")
        print("  " + "-" * 60)
        for m in range(max_breaks + 1):
            F_str  = f"{supF[m-1]:9.3f}" if m > 0 else "        —"
            cv_str = f"{_CV5.get(m, '—'):>7}" if m > 0 else "       —"
            mark   = "  ◄" if m == n_breaks else ""
            print(f"  {m:>3d}  {ssr_by_m[m]:>12.4f}  {bic[m]:>10.3f}  "
                  f"{lwz[m]:>10.3f}  {F_str}  {cv_str}{mark}")
        if n_breaks > 0:
            print(f"\n  Break points:")
            for k, b_idx in enumerate(opt_breaks):
                b_disp   = break_dates[k] if break_dates else b_idx
                coh_bef  = segment_cohesion[k]
                coh_aft  = segment_cohesion[k + 1]
                print(f"    [{k+1}]  idx={b_idx:4d}  date={b_disp}"
                      f"   cohesion_before={coh_bef:.4f}"
                      f"   cohesion_after={coh_aft:.4f}")
        else:
            print(f"\n  No structural break detected.")
            print(f"  Global subspace cohesion: {segment_cohesion[0]:.4f}")
        print("=" * 70 + "\n")

    return dict(
        n_breaks         = n_breaks,
        break_idx        = opt_breaks,
        break_dates      = break_dates,
        boundaries       = boundaries,
        segment_cohesion = segment_cohesion,
        K                = K,
        ssr_by_m         = ssr_by_m,
        bic              = bic,
        lwz              = lwz,
        supF_stats       = supF,
        criterion        = criterion,
        r                = r,
    )


def plot_bai_perron_grassmann(U_list, dates, bp_result, figsize=(15, 5)):
    """
    Two-panel plot for the Grassmannian Bai-Perron test:
      Left  — Gram matrix K[s,t]/r  (heatmap of pairwise subspace similarity)
      Right — mean similarity over time with segment cohesion and break dates
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={'width_ratios': [1, 1.8]})
    K         = bp_result['K']
    r         = bp_result['r']
    dates_str = [str(d) for d in dates]

    # ── Left: similarity heatmap ─────────────────────────────────────────
    im = ax1.imshow(K / r, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax1, label='K[s,t] / r   (1 = identical subspaces)')

    n_ticks  = min(8, len(dates_str))
    tick_idx = np.linspace(0, len(dates_str) - 1, n_ticks, dtype=int)
    ax1.set_xticks(tick_idx)
    ax1.set_yticks(tick_idx)
    ax1.set_xticklabels([dates_str[i][:7] for i in tick_idx],
                        rotation=45, fontsize=7)
    ax1.set_yticklabels([dates_str[i][:7] for i in tick_idx], fontsize=7)

    for b in bp_result['break_idx']:
        ax1.axhline(b + 0.5, color='red', lw=1.5, linestyle='--')
        ax1.axvline(b + 0.5, color='red', lw=1.5, linestyle='--')
    ax1.set_title('Gram matrix  K[s,t]/r\n(pairwise subspace similarity)', fontsize=9)

    # ── Right: mean similarity over time ─────────────────────────────────
    # For each window t: average similarity to all other windows
    cohesion_t  = K.mean(axis=1) / r
    dates_plot  = pd.to_datetime(dates_str)

    ax2.plot(dates_plot, cohesion_t, color='steelblue', lw=1.2,
             label='Mean similarity to all windows', zorder=2)

    boundaries = bp_result['boundaries']
    seg_coh    = bp_result['segment_cohesion']
    for k in range(bp_result['n_breaks'] + 1):
        seg_d = dates_plot[boundaries[k]:boundaries[k + 1]]
        if len(seg_d) > 0:
            ax2.hlines(seg_coh[k], seg_d[0], seg_d[-1],
                       colors='black', lw=2.2, zorder=3,
                       label='Segment cohesion' if k == 0 else '_nolegend_')

    if bp_result['break_dates']:
        for k, bd in enumerate(bp_result['break_dates']):
            ax2.axvline(pd.to_datetime(str(bd)), color='tomato',
                        lw=1.8, linestyle='--', alpha=0.85, zorder=4,
                        label='Break' if k == 0 else '_nolegend_')

    ax2.set_title(
        f"Subspace cohesion over time — "
        f"{bp_result['n_breaks']} break(s), {bp_result['criterion'].upper()}",
        fontsize=9)
    ax2.set_ylabel('Mean K[t, :] / r')
    ax2.set_xlabel('Date')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(
        f'{FIGURES_ROOT}/STABILITY/bai_perron_grassmann_{bp_result["criterion"]}.png',
        dpi=160, bbox_inches='tight')
    plt.show()


#=================================================================================================================
# PLOTS FUNCTIONS 
#=================================================================================================================
# CUSUM TEST
#=================================================================================================================

def compute_cusum(df, col='distance_grass', date_col='start', confidence=0.95):
    """
    CUSUM test for structural break detection on a Grassmann distance series.

    Statistic :
        S_k = Σ_{t=1}^k (d_t − d̄)
        W_k = S_k / (σ̂ √T)          ← converges to a Brownian bridge under H0

    H0 rejected if  sup_k |W_k| > c_α  (Kolmogorov-Smirnov critical values).

    Note : overlapping windows (step < window_size) induce autocorrelation that 
    makes the KS critical values too liberal — interpret with caution.

    Parameters
    ----------
    df         : DataFrame with at least columns [date_col, col]
    col        : column to test (default 'distance_grass')
    date_col   : date column (default 'start')
    confidence : 0.90 | 0.95 | 0.99

    Returns
    -------
    result   : DataFrame with cusum, cusum_norm, above_cv columns added
    cp_date  : date of the main change-point (argmax |W_k|)
    cv       : critical value used
    """
    d     = df[col].dropna().values
    dates = pd.to_datetime(df.loc[df[col].notna(), date_col].values)
    T     = len(d)

    d_mean     = np.mean(d)
    sigma      = np.std(d, ddof=1)
    cusum      = np.cumsum(d - d_mean)
    cusum_norm = cusum / (sigma * np.sqrt(T))

    cv = {0.90: 1.22, 0.95: 1.36, 0.99: 1.63}[confidence]

    cp_idx  = int(np.argmax(np.abs(cusum_norm)))
    cp_date = dates[cp_idx]
    cp_val  = cusum_norm[cp_idx]

    keep_cols = [date_col, col]
    for extra in ('speed', 'acceleration'):
        if extra in df.columns:
            keep_cols.append(extra)
    result = df[keep_cols].copy()
    result['cusum']      = cusum
    result['cusum_norm'] = cusum_norm
    result['above_cv']   = np.abs(cusum_norm) > cv

    print(f"Change-point principal : {cp_date.date()}  |  "
          f"W_k = {cp_val:.3f}  |  CV({int(confidence*100)}%) = ±{cv}")
    print(f"Significant break  : {np.abs(cp_val) > cv}")

    return result, cp_date, cv

def compute_cusum_ma5(df, col='distance_grass', date_col='start',
                      confidence=0.95):
    """
    CUSUM with an exact MA(5) correction.

    The autocorrelation structure is known a priori: a 6M window with a 1M step
    implies γ(j) = 0 for j ≥ 6 by construction (no shared data).
    We therefore use the kernel truncated at m=5 with uniform weights — more
    parsimonious and more exact than Newey-West with a data-driven bandwidth.

        σ²_LR = γ̂(0) + 2 Σ_{j=1}^{5} γ̂(j)
    """
    mask  = df[col].notna()
    d     = df.loc[mask, col].values
    dates = pd.to_datetime(df.loc[mask, date_col].values)
    T     = len(d)

    d_mean = np.mean(d)
    e      = d - d_mean

    # ── Exact MA(5) long-run variance ───────────────────────────────────────
    m   = 5
    lrv = np.dot(e, e) / T                          # γ̂(0)
    for j in range(1, m + 1):
        gamma_j = np.dot(e[j:], e[:-j]) / T        # γ̂(j), weight = 1
        lrv    += 2 * gamma_j

    if lrv <= 0:
        raise ValueError(f"Negative LRV ({lrv:.4f}) — check the series.")

    sigma_lr = np.sqrt(lrv)

    # ── CUSUM statistic ──────────────────────────────────────────────────────
    cusum      = np.cumsum(e)
    cusum_norm = cusum / (sigma_lr * np.sqrt(T))

    cv      = {0.90: 1.22, 0.95: 1.36, 0.99: 1.63}[confidence]
    cp_idx  = int(np.argmax(np.abs(cusum_norm)))
    cp_date = dates[cp_idx]
    cp_val  = cusum_norm[cp_idx]

    # ── Result ───────────────────────────────────────────────────────────────
    keep_cols = [date_col, col]
    for extra in ('speed', 'acceleration'):
        if extra in df.columns:
            keep_cols.append(extra)
    result = df[keep_cols].copy()
    result['cusum']      = cusum
    result['cusum_norm'] = cusum_norm
    result['above_cv']   = np.abs(cusum_norm) > cv

    print(f"── CUSUM MA(5) exact (m=5 fixe) ──────────────────────────────")
    print(f"σ_LR   = {sigma_lr:.4f}  |  σ_naïf = {np.std(e, ddof=1):.4f}"
          f"  |  ratio = {sigma_lr / np.std(e, ddof=1):.2f}×")
    print(f"Change-point : {cp_date.date()}  |  "
          f"W_k = {cp_val:.3f}  |  CV({int(confidence*100)}%) = ±{cv}")
    print(f"Rupture significative : {np.abs(cp_val) > cv}")
    print(f"──────────────────────────────────────────────────────────────")

    return result, cp_date, cv

def plot_cusum(result, cp_date, cv,
               col='distance_grass', date_col='start',
               window_label=''):
    """
    Two-panel plot:
      top    — distance series (col)
      bottom — normalised CUSUM W_k with critical bands
    """
    import matplotlib.dates as mdates

    dates = pd.to_datetime(result[date_col])

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f'CUSUM Test — Grassmann Distance  {window_label}', fontsize=13)

    ax.plot(dates, result['cusum_norm'], color='darkorange', lw=1.5,
            label='$W_k$')
    ax.axhline( cv, color='red', lw=1.0, ls='--',
                label=f'+{cv}  (CV {int(cv*100-36)}%)')
    ax.axhline(-cv, color='red', lw=1.0, ls='--',
                label=f'-{cv}')
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.axvline(cp_date, color='purple', lw=1.5, ls='-.',
               label=f'CP : {cp_date.date()}')
    ax.fill_between(dates, result['cusum_norm'],  cv,
                    where=result['cusum_norm'] >  cv,
                    alpha=0.25, color='red',  label='Significant ↑')
    ax.fill_between(dates, result['cusum_norm'], -cv,
                    where=result['cusum_norm'] < -cv,
                    alpha=0.25, color='blue', label='Significant ↓')
    ax.set_ylabel('$W_k = S_k\\,/\\,(\\hat{\\sigma}\\sqrt{T})$')
    ax.set_xlabel('Date')
    ax.legend(fontsize=7, loc='upper left', ncol=3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/cusum_{window_label.replace(" ", "_").replace("/", "-")}.png',
                dpi=160, bbox_inches='tight')
    plt.show()


#=================================================================================================================
# MODIFIED MANN-KENDALL TEST (HAMED-RAO 1998)
#=================================================================================================================

def hamed_rao_test(
    df,
    col      = 'distance_grass',
    date_col = 'start',
    alpha    = 0.05,
    verbose  = True,
):
    """
    Modified Mann-Kendall test of Hamed & Rao (1998) to detect a trend
    in an autocorrelated time series — typically the Grassmann distances.

    Unlike the classical MK test, the variance of S is corrected for the
    serial autocorrelation of the ranks, which avoids false positives when
    rolling windows overlap.

    Algorithm
    ----------
    1. S = Σ_{i<j} sgn(x_j − x_i)
    2. Var(S) = [n(n−1)(2n+5) − Σ_p t_p(t_p−1)(2t_p+5)] / 18
    3. Ranks r_i = rank(x_i)
    4. ρ_s(lag) = corr(r_{1:n−lag}, r_{1+lag:n})  for lag = 1..n−1
       Only significant ρ_s are kept: |ρ_s| > z_{α/2}/√n
    5. n_s/n = 1 + (2 / [n(n−1)(2n+5)]) Σ_{lag} (n−l)(n−l−1)(2(n−l)+5) ρ_s(l)
    6. Var*(S) = Var(S) × max(n_s/n, 0)
    7. Z* = (S − sgn(S)) / √Var*(S)   [continuity],  p = 2(1 − Φ(|Z*|))
    8. Sen's slope: β = median[(x_j − x_i)/(j − i)]  for j > i

    Parameters
    ----------
    df       : DataFrame (columns [date_col, col]) or Series
    col      : column to test (ignored if df is a Series)
    date_col : date column of the DataFrame
    alpha    : significance level (default 0.05)
    verbose  : print the results table

    Returns
    -------
    dict : trend, h, p, z, Tau, s, var_s, var_s_corrected, ns_over_n,
           slope, intercept, n, alpha, dates, y, col
    """
    from scipy import stats as sp_stats

    if isinstance(df, pd.Series):
        y     = df.dropna().values
        dates = None
    else:
        mask  = df[col].notna()
        y     = df.loc[mask, col].values
        dates = pd.to_datetime(df.loc[mask, date_col].values) if date_col in df.columns else None

    n = len(y)
    if n < 4:
        raise ValueError(f"Too few observations (n={n}). Minimum 4.")

    # ── S statistic ───────────────────────────────────────────────────────
    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(y[i + 1:] - y[i])))

    # ── Var(S) with tie correction ────────────────────────────────────────
    _, counts        = np.unique(y, return_counts=True)
    ties_corr        = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s            = (n * (n - 1) * (2 * n + 5) - ties_corr) / 18.0

    # ── Hamed-Rao correction: rank autocorrelation ────────────────────────
    ranks  = sp_stats.rankdata(y).astype(float)
    z_crit = sp_stats.norm.ppf(1.0 - alpha / 2) / np.sqrt(n)

    correction = 0.0
    for lag in range(1, n):
        rho = float(np.corrcoef(ranks[:n - lag], ranks[lag:])[0, 1])
        if abs(rho) > z_crit:
            l   = n - lag
            correction += l * (l - 1) * (2 * l + 5) * rho

    ns_over_n       = 1.0 + (2.0 / (n * (n - 1) * (2 * n + 5))) * correction
    var_s_corrected = var_s * max(ns_over_n, 0.0)
    if var_s_corrected <= 0:
        var_s_corrected = var_s

    # ── Z* statistic ──────────────────────────────────────────────────────
    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s_corrected)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s_corrected)
    else:
        z = 0.0

    p   = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    Tau = s / (0.5 * n * (n - 1))

    # ── Sen's slope ───────────────────────────────────────────────────────
    ii, jj    = np.triu_indices(n, k=1)
    all_slopes = (y[jj] - y[ii]) / (jj - ii).astype(float)
    slope      = float(np.median(all_slopes))
    intercept  = float(np.median(y) - slope * np.median(np.arange(n)))

    h     = bool(p < alpha)
    trend = ('increasing' if (h and z > 0) else
             'decreasing' if (h and z < 0) else
             'no trend')

    if verbose:
        sig_str = ('***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'ns')
        print("\n" + "=" * 62)
        print("  MODIFIED MANN-KENDALL — HAMED & RAO (1998)")
        print("=" * 62)
        print(f"  Series         : {col}   n={n}   α={alpha}")
        print(f"  S              = {s:>10d}")
        print(f"  Var(S)         = {var_s:>14.4f}")
        print(f"  n_s / n        = {ns_over_n:>14.4f}  (autocorrelation factor)")
        print(f"  Var*(S)        = {var_s_corrected:>14.4f}")
        print(f"  Z*             = {z:>14.4f}")
        print(f"  p-value        = {p:>14.4f}  {sig_str}")
        print(f"  Kendall's tau  = {Tau:>14.4f}")
        print(f"  Sen's slope    = {slope:>14.6f}  (per observation)")
        print(f"  Trend          : {trend.upper()}")
        print(f"  Reject H₀      : {'Yes' if h else 'No'}  (H₀ = no trend)")
        print("=" * 62 + "\n")

    return dict(
        trend           = trend,
        h               = h,
        p               = p,
        z               = z,
        Tau             = Tau,
        s               = s,
        var_s           = var_s,
        var_s_corrected = var_s_corrected,
        ns_over_n       = ns_over_n,
        slope           = slope,
        intercept       = intercept,
        n               = n,
        alpha           = alpha,
        dates           = dates,
        y               = y,
        col             = col,
    )


def plot_hamed_rao(result, figsize=(13, 5), crisis_dates=None):
    """
    Plot the distance series with Sen's trend line and the result
    of the modified Mann-Kendall test (Hamed-Rao).

    Parameters
    ----------
    result       : dict returned by hamed_rao_test()
    figsize      : figure size
    crisis_dates : dict {label: (start_str, end_str)} — crisis windows to shade
    """
    import matplotlib.dates as mdates

    y         = result['y']
    dates     = result['dates']
    slope     = result['slope']
    intercept = result['intercept']
    trend     = result['trend']
    p         = result['p']
    z         = result['z']
    Tau       = result['Tau']
    col       = result.get('col', 'distance_grass')
    n         = len(y)

    sen_line  = slope * np.arange(n) + intercept
    x_plot    = dates if dates is not None else np.arange(n)

    default_crises = {
        'CHF floor':  ('2015-01-01', '2015-03-01'),
        'Brexit':     ('2016-06-01', '2016-09-01'),
        'TRY crisis': ('2018-07-01', '2019-01-01'),
        'COVID-19':   ('2020-02-01', '2020-06-01'),
        'Ukraine':    ('2022-02-01', '2022-06-01'),
    }
    crises = crisis_dates if crisis_dates is not None else default_crises

    fig, ax = plt.subplots(figsize=figsize)

    if dates is not None:
        for name, (s_, e_) in crises.items():
            ax.axvspan(pd.Timestamp(s_), pd.Timestamp(e_),
                       alpha=0.10, color='grey', zorder=0, label='_nolegend_')

    ax.plot(x_plot, y, color='steelblue', lw=1.0, alpha=0.8, label=col, zorder=2)

    color_sen = ('tomato' if trend == 'increasing' else
                 'seagreen' if trend == 'decreasing' else 'grey')
    ax.plot(x_plot, sen_line, color=color_sen, lw=2.0, linestyle='--',
            label=f"Sen's slope = {slope:+.5f}/obs", zorder=3)

    p_str = f"p={p:.4f}" if p >= 0.001 else "p<0.001"
    sig   = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'ns'
    ax.set_title(
        f"Modified Mann-Kendall (Hamed-Rao 1998)  —  {col}\n"
        f"Trend: {trend.upper()}  |  Z={z:.3f}  |  {p_str} {sig}  |  τ={Tau:.3f}",
        fontsize=10,
    )
    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/hamed_rao_{col}.png', dpi=160, bbox_inches='tight')
    plt.show()


#=================================================================================================================
# ROLLING HAMED-RAO
#=================================================================================================================

def hamed_rao_rolling(
    df,
    col      = 'distance_grass',
    date_col = 'start',
    window   = 36,
    step     = 6,
    alpha    = 0.05,
    min_obs  = 10,
):
    """
    Apply the Hamed-Rao test on a rolling window to see whether and when
    a significant trend appears in the Grassmann distances.

    Parameters
    ----------
    df       : DataFrame with columns [date_col, col]
    col      : column to test
    date_col : date column
    window   : window length in months (default 36 = 3 years)
    step     : step in months (default 6)
    alpha    : significance level
    min_obs  : minimum number of observations to run the test

    Returns
    -------
    DataFrame with columns: window_start, window_end, n, z, p, Tau,
                              slope, ns_over_n, trend, significant
    """
    df       = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df       = df.sort_values(date_col).reset_index(drop=True)

    start_date = df[date_col].min()
    end_date   = df[date_col].max()

    current = start_date
    records = []

    while current + pd.DateOffset(months=window) <= end_date + pd.DateOffset(months=1):
        win_end = current + pd.DateOffset(months=window)
        mask    = (df[date_col] >= current) & (df[date_col] < win_end)
        sub     = df.loc[mask]

        if len(sub) >= min_obs:
            res = hamed_rao_test(sub, col=col, date_col=date_col,
                                 alpha=alpha, verbose=False)
            records.append({
                'window_start' : current,
                'window_end'   : win_end,
                'window_mid'   : current + pd.DateOffset(months=window // 2),
                'n'            : res['n'],
                'z'            : res['z'],
                'p'            : res['p'],
                'Tau'          : res['Tau'],
                'slope'        : res['slope'],
                'ns_over_n'    : res['ns_over_n'],
                'trend'        : res['trend'],
                'significant'  : res['h'],
            })

        current += pd.DateOffset(months=step)

    return pd.DataFrame(records)


def plot_hamed_rao_rolling(
    roll_df,
    dist_df   = None,
    col       = 'distance_grass',
    date_col  = 'start',
    alpha     = 0.05,
    figsize   = (14, 9),
    crisis_dates = None,
    window    = None,
    step      = None,
):
    """
    Three panels:
      1. Grassmann distance series (context)
      2. Rolling Z* with ±1.96 / ±2.58 thresholds, background coloured by trend
      3. Rolling p-value with threshold α

    Parameters
    ----------
    roll_df      : DataFrame returned by hamed_rao_rolling()
    dist_df      : original distance DataFrame (for panel 1), optional
    col          : name of the distance column
    date_col     : date column in dist_df
    alpha        : threshold shown on the p-value panel
    figsize      : figure size
    crisis_dates : dict {label: (start_str, end_str)}
    window, step : parameters used (for the title)
    """
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

    default_crises = {
        'CHF floor':  ('2015-01-01', '2015-03-01'),
        'Brexit':     ('2016-06-01', '2016-09-01'),
        'TRY crisis': ('2018-07-01', '2019-01-01'),
        'COVID-19':   ('2020-02-01', '2020-06-01'),
        'Ukraine':    ('2022-02-01', '2022-06-01'),
    }
    crises = crisis_dates if crisis_dates is not None else default_crises

    n_panels = 3 if dist_df is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 2:
        axes = [None] + list(axes)   # dummy slot for panel 0

    dates_mid = pd.to_datetime(roll_df['window_mid'])
    z_vals    = roll_df['z'].values
    p_vals    = roll_df['p'].values
    trends    = roll_df['trend'].values

    def _shade_crises(ax):
        for name, (s_, e_) in crises.items():
            ax.axvspan(pd.Timestamp(s_), pd.Timestamp(e_),
                       alpha=0.08, color='grey', zorder=0)

    # ── Panel 1: raw series ───────────────────────────────────────────────
    if dist_df is not None:
        ax0 = axes[0]
        _shade_crises(ax0)
        ax0.plot(pd.to_datetime(dist_df[date_col]), dist_df[col],
                 color='steelblue', lw=0.9, alpha=0.85)
        ax0.set_ylabel(col, fontsize=9)
        ax0.set_title(f'Hamed-Rao rolling  —  window={window}M / step={step}M',
                      fontsize=11)
        ax0.grid(True, alpha=0.2)
        for sp in ['top', 'right']: ax0.spines[sp].set_visible(False)

    # ── Panel 2: Z* ───────────────────────────────────────────────────────
    ax1 = axes[1]
    _shade_crises(ax1)

    # background colour: red for a significant upward trend, green for a downward one
    for i, (d, z, tr) in enumerate(zip(dates_mid, z_vals, trends)):
        if tr == 'increasing':
            ax1.axvspan(d - pd.DateOffset(months=1),
                        d + pd.DateOffset(months=1),
                        alpha=0.12, color='tomato', zorder=0)
        elif tr == 'decreasing':
            ax1.axvspan(d - pd.DateOffset(months=1),
                        d + pd.DateOffset(months=1),
                        alpha=0.12, color='seagreen', zorder=0)

    ax1.plot(dates_mid, z_vals, color='#333333', lw=1.5, zorder=3, label='Z*')
    ax1.axhline( 1.96, color='tomato',   lw=1.0, ls='--', alpha=0.8, label='±1.96 (α=5%)')
    ax1.axhline(-1.96, color='tomato',   lw=1.0, ls='--', alpha=0.8)
    ax1.axhline( 2.58, color='darkred',  lw=0.8, ls=':',  alpha=0.7, label='±2.58 (α=1%)')
    ax1.axhline(-2.58, color='darkred',  lw=0.8, ls=':',  alpha=0.7)
    ax1.axhline( 0,    color='black',    lw=0.5, alpha=0.4)
    ax1.set_ylabel('Z*', fontsize=9)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.2)
    for sp in ['top', 'right']: ax1.spines[sp].set_visible(False)

    # ── Panel 3: p-value ──────────────────────────────────────────────────
    ax2 = axes[2]
    _shade_crises(ax2)
    ax2.plot(dates_mid, p_vals, color='#555599', lw=1.5, label='p-value')
    ax2.axhline(alpha, color='tomato', lw=1.2, ls='--',
                label=f'α={alpha}')
    ax2.axhline(0.01,  color='darkred', lw=0.8, ls=':',
                label='α=0.01', alpha=0.7)
    ax2.fill_between(dates_mid, 0, p_vals,
                     where=(p_vals < alpha),
                     color='tomato', alpha=0.15, label='significant')
    ax2.set_ylabel('p-value', fontsize=9)
    ax2.set_xlabel('Date (window mid-point)', fontsize=9)
    ax2.set_ylim(0, min(1.05, p_vals.max() * 1.2))
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    for sp in ['top', 'right']: ax2.spines[sp].set_visible(False)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/hamed_rao_rolling_{col}.png', dpi=160, bbox_inches='tight')
    plt.show()


#=================================================================================================================

def plot_grassman(df_distances, avg_vol, window_size, step, reference=False,
                  show_dynamics=True):

    import os
    os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)

    if show_dynamics:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

    ax1.plot(df_distances['start'], df_distances['distance_grass'], color='tab:blue', label='Grassmann distance')
    ax1.set_ylabel('Grassmann distance', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(avg_vol.index, avg_vol.values, color='tab:orange', alpha=0.7, label='Volatility')
    ax1_twin.set_ylabel('Volatility', color='tab:orange')
    ax1_twin.tick_params(axis='y', labelcolor='tab:orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    if show_dynamics:
        ax2.plot(df_distances['start'], df_distances['speed'], color='tab:orange')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_ylabel('Speed')
        ax2.grid(True, alpha=0.3)

        ax3.plot(df_distances['start'], df_distances['acceleration'], color='tab:red')
        ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax3.set_ylabel('Acceleration')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Date')

    if not reference:
        fig.suptitle('Loadings Subspace stability - Grassmann distance \n '
                     f'Window size: {window_size} months, Step: {step} days', fontsize=13)
    else:
        fig.suptitle('Loadings Subspace stability - Grassmann distance (compared to 2014) \n '
                     f'Window size: {window_size} months, Step: {step} days', fontsize=13)

    plt.tight_layout()
    suffix = 'ref_' if reference else ''
    dyn    = '' if show_dynamics else '_nodyn'
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/grass_dist_{suffix}{window_size}_{step}{dyn}.png', dpi=180)
    plt.show()


def plot_loadings_evolution(vol_matrices, run_ffm, windows, r_plot=2,
                            highlight_pairs=None, add_title=None,
                            use_ref_mean=False):

    if highlight_pairs is None:
        highlight_pairs = []

    palette_18 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#dbdb8d',
    ]
    pair_colors = {p.lower(): palette_18[i % len(palette_18)]
                   for i, p in enumerate(highlight_pairs)}

    n_windows = len(windows)

    tau_labels = ['17h', '00h', '07h', '13h', '17h']
    tau_ticks  = [0, 0.25, 0.45, 0.70, 1.0]

    # ── Reference: mean of the first valid window ────────────────────
    mu_ref_dict = None
    if use_ref_mean:
        for start, end in windows:
            subset_ref = {}
            for pair, df in vol_matrices.items():
                mask = (pd.to_datetime(df.index) >= pd.to_datetime(start)) & \
                       (pd.to_datetime(df.index) <  pd.to_datetime(end))
                if mask.sum() > 10:
                    subset_ref[pair] = df.loc[mask]
            if len(subset_ref) > 0:
                ffm_ref     = run_ffm(subset_ref)
                mu_ref_dict = dict(zip(ffm_ref['pairs'], ffm_ref['mu']))
                break

    # ── Pre-compute the FFM for each window ──────────────────────────
    ffm_cache = []
    for start, end in windows:
        subset = {}
        for pair, df in vol_matrices.items():
            mask = (pd.to_datetime(df.index) >= pd.to_datetime(start)) & \
                   (pd.to_datetime(df.index) <  pd.to_datetime(end))
            if mask.sum() > 10:
                subset[pair] = df.loc[mask]
        if len(subset) == 0:
            ffm_cache.append(None)
        elif use_ref_mean and mu_ref_dict is not None:
            ffm_cache.append(run_ffm(subset, mu_ref=mu_ref_dict))
        else:
            ffm_cache.append(run_ffm(subset))

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1 : Functional loadings b̂_il(τ)
    # ════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(4.5 * n_windows, 3.5 * r_plot + 1.2))
    gs1  = fig1.add_gridspec(r_plot + 1, n_windows,
                             height_ratios=[3.5] * r_plot + [1.0],
                             hspace=0.35, wspace=0.12)

    axes = [[fig1.add_subplot(gs1[row, col])
             for col in range(n_windows)]
            for row in range(r_plot)]

    global_ylo, global_yhi = np.inf, -np.inf

    for col_idx, ((start, end), ffm) in enumerate(zip(windows, ffm_cache)):
        if ffm is None:
            continue

        B_hat = ffm['B_hat']   # (N, k_estimated, M)
        pairs = ffm['pairs']
        M     = B_hat.shape[2]
        tau   = np.linspace(0, 1, M)
        k_est = B_hat.shape[1]

        for row_idx in range(r_plot):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor('#FAFAFA')
            ax.axhline(0, color='#DDD', linewidth=0.8, zorder=0)

            if row_idx >= k_est:
                ax.text(0.5, 0.5, f'factor {row_idx+1}\nnot estimated',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=8, color='#CCC')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
                continue

            for i, pair in enumerate(pairs):
                if pair.lower() not in pair_colors:
                    ax.plot(tau, B_hat[i, row_idx, :],
                            color='#CCCCCC', linewidth=0.7,
                            alpha=0.5, zorder=1)

            for i, pair in enumerate(pairs):
                p = pair.lower()
                if p in pair_colors:
                    y = B_hat[i, row_idx, :]
                    ax.plot(tau, y,
                            color=pair_colors[p], linewidth=1.8,
                            alpha=0.9, zorder=3)

            b_mean = np.mean(B_hat[:, row_idx, :], axis=0)
            ax.plot(tau, b_mean,
                    color='black', linewidth=2.2,
                    linestyle='--', zorder=6,
                    label='Cross-sect. mean')

            # Accumulate global limits over all pairs / factors / windows
            global_ylo = min(global_ylo, B_hat[:, :min(r_plot, k_est), :].min())
            global_yhi = max(global_yhi, B_hat[:, :min(r_plot, k_est), :].max())

            if row_idx == 0:
                ax.set_title(f"{start[:7]} → {end[:7]}",
                             fontsize=9, fontweight='500', pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"Factor {row_idx + 1}", fontsize=9)

            ax.set_xlim(0, 1)
            ax.set_xticks(tau_ticks)
            ax.set_xticklabels(tau_labels, fontsize=7)
            ax.tick_params(axis='y', labelsize=7)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#E0E0E0')
            ax.spines['bottom'].set_color('#E0E0E0')

    # Same scale for all cells (rows × cols)
    if np.isfinite(global_ylo) and np.isfinite(global_yhi):
        margin = (global_yhi - global_ylo) * 0.08
        for row_idx in range(r_plot):
            for col_idx in range(n_windows):
                axes[row_idx][col_idx].set_ylim(global_ylo - margin,
                                                global_yhi + margin)

    ax_legend1 = fig1.add_subplot(gs1[r_plot, :])
    ax_legend1.axis('off')
    handles = [
        plt.Line2D([0], [0], color=pair_colors[p], linewidth=2.5, label=p.upper())
        for p in highlight_pairs if p.lower() in pair_colors
    ]
    handles += [
        plt.Line2D([0], [0], color='#CCCCCC', linewidth=1.5, label='Other pairs'),
        plt.Line2D([0], [0], color='black', linewidth=2.2,
                   linestyle='--', label='Cross-sect. mean'),
    ]
    ax_legend1.legend(handles=handles, loc='center', ncol=6, fontsize=8.5,
                      frameon=True, framealpha=0.9, edgecolor='#DDD',
                      columnspacing=1.0, handlelength=1.8)

    fig1.suptitle(f"Functional loadings b̂_il(τ) — factors 1 to {r_plot}",
                  fontsize=12, fontweight='500', y=1.01)
    if add_title is not None: 
        fig1.savefig(f'{FIGURES_ROOT}/STABILITY/loadings_evolution_{add_title}.png', dpi=160,
                 bbox_inches='tight', facecolor='white')
    if add_title is None: 
        fig1.savefig(f'{FIGURES_ROOT}/STABILITY/loadings_evolution.png', dpi=160,
                 bbox_inches='tight', facecolor='white')
    plt.show()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2 : Common factor scores û_lt
    # ════════════════════════════════════════════════════════════════════
    import matplotlib.dates as mdates

    fig2 = plt.figure(figsize=(4.5 * n_windows, 3.0 * r_plot + 1.2))
    gs2  = fig2.add_gridspec(r_plot, n_windows,
                             hspace=0.45, wspace=0.22)

    axes2 = [[fig2.add_subplot(gs2[row, col])
              for col in range(n_windows)]
             for row in range(r_plot)]

    row_score_ylims = {r: [np.inf, -np.inf] for r in range(r_plot)}

    for col_idx, ((start, end), ffm) in enumerate(zip(windows, ffm_cache)):
        if ffm is None:
            continue

        U_hat = ffm['U_hat']   # (k_estimated, T)
        days  = ffm['days']    # (T,)
        k_est = U_hat.shape[0]
        dates_dt = pd.to_datetime(days)

        for row_idx in range(r_plot):
            ax = axes2[row_idx][col_idx]
            ax.set_facecolor('#FAFAFA')
            ax.axhline(0, color='#DDD', linewidth=0.8, zorder=0)

            if row_idx >= k_est:
                ax.text(0.5, 0.5, f'factor {row_idx+1}\nnot estimated',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=8, color='#CCC')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
                continue

            u = U_hat[row_idx, :]
            ax.fill_between(dates_dt, u, 0,
                            where=(u >= 0), color='#4C9BE8', alpha=0.30, zorder=1)
            ax.fill_between(dates_dt, u, 0,
                            where=(u < 0),  color='#E8754C', alpha=0.30, zorder=1)
            ax.plot(dates_dt, u, color='#1a3a6b', linewidth=0.8, zorder=3)

            row_score_ylims[row_idx][0] = min(row_score_ylims[row_idx][0], u.min())
            row_score_ylims[row_idx][1] = max(row_score_ylims[row_idx][1], u.max())

            if row_idx == 0:
                ax.set_title(f"{start[:7]} → {end[:7]}",
                             fontsize=9, fontweight='500', pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"Factor {row_idx + 1}", fontsize=9)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=7)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#E0E0E0')
            ax.spines['bottom'].set_color('#E0E0E0')

    for row_idx in range(r_plot):
        ylo, yhi = row_score_ylims[row_idx]
        if np.isfinite(ylo) and np.isfinite(yhi):
            margin = (yhi - ylo) * 0.12
            for col_idx in range(n_windows):
                axes2[row_idx][col_idx].set_ylim(ylo - margin, yhi + margin)

    fig2.suptitle(f"Common factor scores û_lt — factors 1 to {r_plot}",
                  fontsize=12, fontweight='500', y=1.01)

    if add_title is not None: 
        fig2.savefig(f'{FIGURES_ROOT}/STABILITY/factors_evolution_{add_title}.png', dpi=160,
                 bbox_inches='tight', facecolor='white')
    if add_title is None: 
        fig2.savefig(f'{FIGURES_ROOT}/STABILITY/factors_evolution.png', dpi=160,
                 bbox_inches='tight', facecolor='white')


    
    plt.show()

def _pair_subspace(B_hat_i):
    """
    B_hat_i : (r, M) — loadings of one pair for the r factors
    Returns U : (M, r) orthonormal basis of Col(B_hat_iᵀ) on Gr(r, M)
    """
    U, _, _ = np.linalg.svd(B_hat_i.T, full_matrices=False)
    return U


def grassman_rolling_by_pair(
        data,
        window_size=6,
        step_months=1,
        func=run_ffm_r1,
        start_date='2014-01-01',
        end_date='2025-12-31',
        ref_start='2014-01-01',
        figsize=None,
):
    """
    Per-pair Grassmann distance to a fixed reference window.

    For each pair i and each rolling window t:
        b̂_i(τ) ∈ R^(r×M)  →  Col(U_i) ∈ Gr(r, M)
        d_G(Col(U_i_ref), Col(U_i_t))  via principal angles

    The reference is the window [ref_start, ref_start + window_size months].
    Same parameters as grassman_rolling_window, plus:
        ref_start : start date of the reference window
        figsize   : figure size (auto if None)

    Returns a DataFrame (index = dates, columns = pairs) of Grassmann distances.
    """
    import matplotlib.dates as mdates

    ref_start_ts = pd.to_datetime(ref_start)
    ref_end_ts   = ref_start_ts + pd.DateOffset(months=window_size)
    end_date_ts  = pd.to_datetime(end_date)

    # ── Reference window ──────────────────────────────────────────────────
    subset_ref = {}
    for pair, df in data.items():
        first_date = pd.to_datetime(df.index).min()
        if first_date < ref_end_ts:
            mask = ((pd.to_datetime(df.index) >= ref_start_ts) &
                    (pd.to_datetime(df.index) <= ref_end_ts))
            subset_ref[pair] = df.loc[mask]

    print(f"Reference: {ref_start_ts.date()} → {ref_end_ts.date()}")
    ffm_ref      = func(subset_ref)
    B_hat_ref    = ffm_ref['B_hat']    # (N_ref, r, M)
    pairs_ref    = ffm_ref['pairs']
    mu_ref_dict  = dict(zip(pairs_ref, ffm_ref['mu']))

    U_ref = {pair: _pair_subspace(B_hat_ref[i])
             for i, pair in enumerate(pairs_ref)}   # {pair: (M, r)}

    # ── Rolling windows (start at the end of the reference window) ───────
    records       = []
    current_start = ref_end_ts

    while current_start < end_date_ts:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = ((pd.to_datetime(df.index) >= current_start) &
                        (pd.to_datetime(df.index) <= current_end))
                subset[pair] = df.loc[mask]

        print(f"Processing → {current_start.date()} to {current_end.date()}")

        if subset:
            try:
                ffm   = func(subset, mu_ref=mu_ref_dict)
                B_hat = ffm['B_hat']   # (N, r, M)
                pairs = ffm['pairs']

                row = {'date': current_start}
                for i, pair in enumerate(pairs):
                    if pair not in U_ref:
                        continue
                    U_new      = _pair_subspace(B_hat[i])   # (M, r)
                    dist, _    = grassmann_distance(U_ref[pair], U_new)
                    row[pair]  = dist

                records.append(row)
            except Exception as e:
                print(f"  Skipped: {e}")

        current_start += pd.DateOffset(months=step_months)

    if not records:
        print("No results.")
        return None

    df_result = pd.DataFrame(records).set_index('date')
    df_result.index = pd.DatetimeIndex(df_result.index)
    pair_cols = list(df_result.columns)
    n_pairs   = len(pair_cols)

    # ── Plot : heatmap seule ──────────────────────────────────────────────
    if figsize is None:
        figsize = (14, max(0.28 * n_pairs, 5))

    fig, ax = plt.subplots(figsize=figsize)

    sorted_pairs = df_result[pair_cols].mean().sort_values().index.tolist()
    Z    = df_result[sorted_pairs].T.values   # (n_pairs, n_windows)
    vmax = np.nanquantile(Z[np.isfinite(Z)], 0.98)

    im = ax.pcolormesh(df_result.index, np.arange(n_pairs), Z,
                       cmap='YlOrRd', vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(im, ax=ax, label='d_G vs reference', pad=0.01)

    fmt_pair = lambda p: f'{p[:3].upper()}/{p[3:].upper()}'
    ax.set_yticks(np.arange(n_pairs) + 0.5)
    ax.set_yticklabels([fmt_pair(p) for p in sorted_pairs], fontsize=7)
    ax.set_ylabel('Pair  (stable → unstable ↑)', fontsize=9)
    ax.set_xlabel('Window start date', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.set_title(
        f'Grassmann distance per pair vs ref {ref_start_ts.strftime("%Y-%m")}\n'
        f'Window: {window_size}m · Step: {step_months}m',
        fontsize=11, fontweight='500'
    )

    plt.tight_layout()
    import os
    os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/grassmann_by_pair_{window_size}_{step_months}.png',
                dpi=160, bbox_inches='tight')
    plt.show()

    return df_result


def plot_loadings_mean_evolution(
        data,
        window_size=6,
        step=1,
        func=run_ffm_r1,
        start_date='2008-01-01',
        end_date='2025-12-31',
        r_plot=2,
        figsize=None,
):
    """
    For each rolling window, compute the cross-sectional mean of the
    loadings  ē_k(τ) = (1/N) Σ_i b̂_ik(τ)  and plot its evolution over time
    as a heatmap  (τ × window start date)  for each factor.

    Same parameters as grassman_rolling_window, plus:
        r_plot   : number of factors to plot
        figsize  : figure size (auto if None)
    """
    import matplotlib.dates as mdates
    from collections import Counter

    results       = []
    current_start = pd.to_datetime(start_date)
    end_date_ts   = pd.to_datetime(end_date)

    while current_start < end_date_ts:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = ((pd.to_datetime(df.index) >= current_start) &
                        (pd.to_datetime(df.index) <= current_end))
                subset[pair] = df.loc[mask]

        print(f"Processing → {current_start.date()} to {current_end.date()}")

        if subset:
            try:
                ffm   = func(subset)
                B_hat = ffm['B_hat']                      # (N, r, M)
                results.append({
                    'date'         : current_start,
                    'mean_loadings': np.mean(B_hat, axis=0),  # (r, M)
                    'k_est'        : B_hat.shape[1],
                    'M'            : B_hat.shape[2],
                })
            except Exception as e:
                print(f"  Skipped: {e}")

        current_start += pd.DateOffset(days=step)

    if not results:
        print("No results to plot.")
        return None, None

    # Keep only the windows with the same number of intraday points M
    target_M = Counter(r['M'] for r in results).most_common(1)[0][0]
    n_dropped = sum(r['M'] != target_M for r in results)
    if n_dropped:
        print(f"  Dropped {n_dropped} windows with M ≠ {target_M}")
    results = [r for r in results if r['M'] == target_M]

    dates             = pd.DatetimeIndex([r['date'] for r in results])
    mean_loadings_arr = np.stack([r['mean_loadings'] for r in results])  # (W, r, M)

    tau        = np.linspace(0, 1, target_M)
    tau_ticks  = [0, 0.25, 0.45, 0.70, 1.0]
    tau_labels = ['17h', '00h', '07h', '13h', '17h']
    n_factors  = min(r_plot, mean_loadings_arr.shape[1])

    if figsize is None:
        figsize = (14, 3.5 * n_factors)

    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
    if n_factors == 1:
        axes = [axes]

    for fac_idx in range(n_factors):
        ax = axes[fac_idx]
        Z  = mean_loadings_arr[:, fac_idx, :].T   # (M, W) — τ en y, temps en x

        vabs = np.nanquantile(np.abs(Z), 0.98)
        im   = ax.pcolormesh(dates, tau, Z,
                             cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                             shading='auto')

        ax.set_yticks(tau_ticks)
        ax.set_yticklabels(tau_labels, fontsize=8)
        ax.set_ylabel(f'Factor {fac_idx + 1}\nτ', fontsize=9)
        plt.colorbar(im, ax=ax, pad=0.01, label='Mean loading')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right', fontsize=8)
    axes[-1].set_xlabel('Window start date', fontsize=9)

    fig.suptitle(
        f'Mean Functional Loading — Window size: {window_size}m · Step: {step}d',
        fontsize=11, fontweight='500'
    )
    plt.tight_layout()

    import os
    save_dir = f'{FIGURES_ROOT}/STABILITY'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/loadings_mean_evo_{window_size}m_{step}d.png',
                dpi=160, bbox_inches='tight')
    plt.show()

    return mean_loadings_arr, dates


def plot_r2_rolling(
        data,
        window_size=6,
        step=30,
        func=run_ffm_r1,
        start_date='2008-01-01',
        end_date='2025-12-31',
        figsize=None,
        r= '1'
):
    """
    For each rolling window, compute the FFM R² per pair and overall:

        R²_i     = ||Chi_hat_i||²_F / ||Y_i||²_F    (per pair)
        R²_total = ||Chi_hat||²_F   / ||Y||²_F       (overall)

    Plot:
        Top    — total R² + selected r̂ (time series)
        Bottom — heatmap of R² per pair (pairs × window date)

    Same parameters as grassman_rolling_window, plus:
        figsize : figure size (auto if None)

    Returns a DataFrame (index = dates, columns = pairs + r2_total + r_hat).
    """
    import matplotlib.dates as mdates

    records       = []
    current_start = pd.to_datetime(start_date)
    end_date_ts   = pd.to_datetime(end_date)

    while current_start < end_date_ts:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = ((pd.to_datetime(df.index) >= current_start) &
                        (pd.to_datetime(df.index) <= current_end))
                subset[pair] = df.loc[mask]

        print(f"Processing → {current_start.date()} to {current_end.date()}")

        if subset:
            try:
                ffm     = func(subset)
                Y       = ffm['Y']        # (N, T, M) centred
                Chi_hat = ffm['Chi_hat']  # (N, T, M) composante commune
                pairs   = ffm['pairs']

                ss_chi   = np.sum(Chi_hat ** 2, axis=(1, 2))          # (N,)
                ss_y     = np.sum(Y ** 2,       axis=(1, 2))          # (N,)
                r2_pair  = np.where(ss_y > 0, ss_chi / ss_y, np.nan) # (N,)
                r2_total = float(np.nansum(ss_chi) / np.nansum(ss_y))

                records.append({
                    'date'    : current_start,
                    'r2_pair' : dict(zip(pairs, r2_pair)),
                    'r2_total': r2_total,
                    'r_hat'   : ffm['r_hat'],
                })
            except Exception as e:
                print(f"  Skipped: {e}")

        current_start += pd.DateOffset(days=step)

    if not records:
        print("No results to plot.")
        return None

    dates  = pd.DatetimeIndex([r['date'] for r in records])
    r2_tot = np.array([r['r2_total'] for r in records])
    r_hats = np.array([r['r_hat']    for r in records])

    # R² matrix per pair — aligned on the union of the pair names
    all_pairs  = sorted({p for r in records for p in r['r2_pair']})
    r2_matrix  = np.full((len(all_pairs), len(records)), np.nan)
    for j, rec in enumerate(records):
        for i, pair in enumerate(all_pairs):
            if pair in rec['r2_pair']:
                r2_matrix[i, j] = rec['r2_pair'][pair]

    # ── Plot ──────────────────────────────────────────────────────────────
    n_pairs = len(all_pairs)
    if figsize is None:
        figsize = (14, 4 + max(0.18 * n_pairs, 3))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [2, max(n_pairs // 6, 3)]},
    )

    # ── Top: total R² + r̂ ───────────────────────────────────────────────
    ax1.plot(dates, r2_tot, color='steelblue', lw=1.8, label='R² total')
    ax1.fill_between(dates, r2_tot, alpha=0.15, color='steelblue')
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('R² total', fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    ax1r = ax1.twinx()
    ax1r.step(dates, r_hats, color='tomato', lw=1.2,
              linestyle=':', alpha=0.8, where='mid', label='r̂')
    ax1r.set_ylabel('r̂', color='tomato', fontsize=9)
    ax1r.tick_params(axis='y', labelcolor='tomato')
    ax1r.set_ylim(0, int(r_hats.max()) + 1.5)
    ax1r.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='lower right', fontsize=8)
    ax1.set_title(
        f'Rolling FFM R² — window {window_size}m, step {step}d',
        fontsize=11, fontweight='500'
    )

    # ── Bottom: heatmap of R² per pair ───────────────────────────────────
    im = ax2.pcolormesh(dates, np.arange(n_pairs), r2_matrix,
                        cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, ax=ax2, label='R²', pad=0.01)

    ax2.set_yticks(np.arange(n_pairs) + 0.5)
    ax2.set_yticklabels(all_pairs, fontsize=7)
    ax2.set_ylabel('Pair', fontsize=9)
    ax2.set_xlabel('Window start date', fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/r2_rolling_{window_size}_{step}_{r}.png',
                dpi=160, bbox_inches='tight')
    plt.show()

    r2_df = pd.DataFrame(r2_matrix.T, index=dates, columns=all_pairs)
    r2_df.insert(0, 'r2_total', r2_tot)
    r2_df.insert(1, 'r_hat',    r_hats)
    return r2_df


def plot_distance_distribution(df, col='distance_grass', label=None, bins=40, figsize=(10, 5), reference = False):
    series = df[col].dropna()
    if not reference: 
        title  = f'Distribution of {col}' + (f' — {label}' if label else '')
    else: 
        title = f'Distribution of {col} (compared to 2014)' + (f' — {label}' if label else '')

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(series, bins=bins, density=True, alpha=0.55, color='steelblue',
            edgecolor='white', label='Histogram')

    kde = stats.gaussian_kde(series)
    xs  = np.linspace(series.min(), series.max(), 300)
    ax.plot(xs, kde(xs), color='steelblue', lw=2, label='KDE')

    mu, sigma = series.mean(), series.std()
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma),
            color='tomato', lw=1.5, linestyle='--', label=f'Normal fit  μ={mu:.3f}, σ={sigma:.3f}')

    p95 = series.quantile(0.95)
    ax.axvline(mu,    color='grey',       lw=1,   linestyle=':')
    ax.axvline(mu - sigma, color='grey',  lw=0.8, linestyle=':')
    ax.axvline(mu + sigma, color='grey',  lw=0.8, linestyle=':')
    ax.axvline(p95,   color='darkorange', lw=1.5, linestyle='--', label=f'p95 = {p95:.3f}')

    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# ADDITIONAL STABILITY / FACTOR ANALYSIS FUNCTIONS
# =============================================================================

def analyze_negative_spikes(U_hat: np.ndarray, dates, N: int = 20,
                             threshold: float = -2.0) -> pd.DataFrame:
    """
    Extract and return the N most negative spikes of factor U_hat below threshold.

    Parameters
    ----------
    U_hat     : array (1, T) or (T,)
    dates     : array-like of datetime (length T)
    N         : number of top spikes to return
    threshold : only keep values below this level
    """
    u = U_hat.flatten()
    neg_idx = np.where(u < threshold)[0]
    neg_vals = u[neg_idx]

    order = np.argsort(neg_vals)
    top_idx = neg_idx[order[:N]]

    df = pd.DataFrame({
        't'    : top_idx,
        'date' : pd.to_datetime(dates)[top_idx],
        'u1t'  : u[top_idx],
    }).sort_values('u1t')

    return df


def regression_grassmann_vol(grassmann_series: pd.Series,
                              vol_series: pd.Series,
                              lags: list = [0, 1, 2, 5]) -> pd.DataFrame:
    """
    Regression: vol_{t+h} = alpha + beta * d_G(t) + eps for h in lags.
    Uses HAC (Newey-West) standard errors. Returns a DataFrame of results and a plot.
    """
    import statsmodels.api as sm
    from scipy import stats

    df = pd.DataFrame({
        'd_G': grassmann_series,
        'vol': vol_series
    }).dropna()

    results = []

    for h in lags:
        y = df['vol'].shift(-h)
        x = df['d_G']

        data = pd.concat([x, y], axis=1).dropna()
        data.columns = ['d_G', 'vol_lead']

        X = sm.add_constant(data['d_G'])
        Y = data['vol_lead']

        model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

        results.append({
            'lag h':       h,
            'alpha':       model.params['const'],
            'beta':        model.params['d_G'],
            'beta_tstat':  model.tvalues['d_G'],
            'beta_pval':   model.pvalues['d_G'],
            'R2':          model.rsquared,
            'N':           int(model.nobs),
            'significant': model.pvalues['d_G'] < 0.05
        })

    df_results = pd.DataFrame(results).set_index('lag h').round(4)

    print("\n=== Regression d_G(t) → vol(t+h) [HAC standard errors] ===\n")
    print(df_results.to_string())
    print()
    for h, row in df_results.iterrows():
        sig = "***" if row['beta_pval'] < 0.01 else ("**" if row['beta_pval'] < 0.05 else "ns")
        print(f"  h={h}: β={row['beta']:.4f} (t={row['beta_tstat']:.2f}, p={row['beta_pval']:.3f}) {sig}  R²={row['R2']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(df['d_G'], df['vol'], alpha=0.3, s=10, color='steelblue')
    x_line = np.linspace(df['d_G'].min(), df['d_G'].max(), 100)
    h0 = df_results.loc[0]
    ax.plot(x_line, h0['alpha'] + h0['beta'] * x_line, 'r-', lw=2)
    ax.set_xlabel('Grassmann distance $d_G(t)$')
    ax.set_ylabel('Mean volatility $\\bar{\\sigma}(t)$')
    ax.set_title(f'Contemporaneous (h=0)  β={h0["beta"]:.4f}  R²={h0["R2"]:.3f}')

    ax = axes[1]
    betas  = df_results['beta']
    pvals  = df_results['beta_pval']
    colors = ['green' if p < 0.05 else 'gray' for p in pvals]
    ax.bar(df_results.index.astype(str), betas, color=colors)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel('Horizon h (days)')
    ax.set_ylabel('β̂')
    ax.set_title('β coefficient by horizon (green = significant at 5%)')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_ROOT}/STABILITY/fig_regression_grassmann.pdf', bbox_inches='tight')
    plt.show()

    return df_results


def loading_heatmap(data: dict, func,
                    ref_start: str = '2014-01-01',
                    end_date: str = '2025-12-31',
                    window_size: int = 6,
                    step_months: int = 1,
                    mode: str = 'ref',
                    crisis_dates: dict = None) -> pd.DataFrame:
    """
    Heatmap of relative L2 loading change per FX pair over time.

    mode='ref'  (default):
        δ_i(t) = ||b_i^(t) - b_i^ref||_L2 / ||b_i^ref||_L2
        Fixed reference = [ref_start, ref_start + window_size months].

    mode='prev':
        δ_i(t) = ||b_i^(t) - b_i^(t-1)||_L2 / ||b_i^(t-1)||_L2
        Each window is compared to the immediately preceding one.
        The reference period serves as the first "previous" window.
        Overlapping windows are kept (standard for rolling analysis).

    Parameters
    ----------
    data        : dict {pair: DataFrame}
    func        : callable — run_ffm variant (e.g. partial(run_ffm, r_forced=1))
    ref_start   : start of reference period
    end_date    : end of rolling window analysis
    window_size : window size in months
    step_months : step size in months (default 1)
    mode        : 'ref' or 'prev'
    crisis_dates: dict {label: (start, end)} for crisis annotation bands
    """
    if mode not in ('ref', 'prev'):
        raise ValueError("mode must be 'ref' or 'prev'")

    ref_end = pd.Timestamp(ref_start) + pd.DateOffset(months=window_size)

    subset_ref = {
        pair: df.loc[
            (pd.to_datetime(df.index) >= pd.Timestamp(ref_start)) &
            (pd.to_datetime(df.index) <= ref_end)
        ]
        for pair, df in data.items()
        if pd.to_datetime(df.index).min() < ref_end
    }

    ffm_ref   = func(subset_ref)
    B_ref     = ffm_ref['B_hat']
    pairs_ref = ffm_ref['pairs']
    b_ref     = B_ref[:, 0, :]
    norm_ref  = np.sqrt(np.mean(b_ref ** 2, axis=1))

    print(f"Reference: {ref_start} → {ref_end.date()} | N={len(pairs_ref)} pairs | mode={mode}")

    # For mode='prev', b_prev/pairs_prev/norm_prev track the last window
    b_prev     = b_ref
    pairs_prev = pairs_ref
    norm_prev  = norm_ref

    results = []
    current = ref_end
    end_dt  = pd.Timestamp(end_date)

    while current < end_dt:
        window_end = current + pd.DateOffset(months=window_size)

        subset = {
            pair: df.loc[
                (pd.to_datetime(df.index) >= current) &
                (pd.to_datetime(df.index) <= window_end)
            ]
            for pair, df in data.items()
            if (current.year != 2012 or pair != 'audjpy') and
               pd.to_datetime(df.index).min() <= current
        }
        subset = {p: d for p, d in subset.items() if len(d) > 10}

        if len(subset) > 0:
            try:
                ffm_w     = func(subset)
                B_new     = ffm_w['B_hat']
                pairs_new = ffm_w['pairs']
                b_new     = B_new[:, 0, :]

                # Choose baseline depending on mode
                if mode == 'ref':
                    b_base     = b_ref
                    pairs_base = pairs_ref
                    norm_base  = norm_ref
                else:  # 'prev'
                    b_base     = b_prev
                    pairs_base = pairs_prev
                    norm_base  = norm_prev

                row = {'date': current.date()}

                for i_new, pair in enumerate(pairs_new):
                    if pair not in pairs_base:
                        continue
                    i_base = list(pairs_base).index(pair)
                    b_n = b_new[i_new].copy()
                    b_b = b_base[i_base]

                    if np.dot(b_n, b_b) < 0:
                        b_n = -b_n

                    l2_diff   = np.sqrt(np.mean((b_n - b_b) ** 2))
                    l2_base   = norm_base[i_base]
                    row[pair] = l2_diff / l2_base if l2_base > 1e-10 else np.nan

                results.append(row)

                if mode == 'prev':
                    b_prev     = b_new
                    pairs_prev = pairs_new
                    norm_prev  = np.sqrt(np.mean(b_new ** 2, axis=1))

            except Exception as e:
                print(f"  Error at {current.date()}: {e}")

        current += pd.DateOffset(months=step_months)

    df_heat = pd.DataFrame(results).set_index('date')
    df_heat.index = pd.to_datetime(df_heat.index)

    pair_order = df_heat.mean().sort_values(ascending=False).index.tolist()
    df_heat    = df_heat[pair_order]

    fig, ax = plt.subplots(figsize=(16, max(6, len(pair_order) * 0.35)))

    if mode == 'ref':
        cbar_label = r'$\delta_i(t) = \|b_i^{(t)} - b_i^{\rm ref}\|_{L^2} / \|b_i^{\rm ref}\|_{L^2}$'
        title      = r'Relative loading change per pair: $\delta_i(t)$ — Factor 1 (ref: ' + ref_start[:7] + ')'
    else:
        cbar_label = r'$\delta_i(t) = \|b_i^{(t)} - b_i^{(t-1)}\|_{L^2} / \|b_i^{(t-1)}\|_{L^2}$'
        title      = r'Loading change vs previous window: $\delta_i(t)$ — Factor 1'

    sns.heatmap(
        df_heat.T, ax=ax, cmap='YlOrRd', linewidths=0,
        xticklabels=False, yticklabels=True,
        cbar_kws={'label': cbar_label}
    )

    n_ticks  = 10
    tick_idx = np.linspace(0, len(df_heat) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx + 0.5)
    ax.set_xticklabels(
        [df_heat.index[i].strftime('%Y-%m') for i in tick_idx],
        rotation=45, ha='right', fontsize=8
    )

    default_crises = {
        'CHF floor': ('2015-01-01', '2015-03-01'),
        'Brexit':    ('2016-06-01', '2017-06-01'),
        'TRY':       ('2018-06-01', '2019-01-01'),
        'COVID':     ('2020-02-01', '2020-07-01'),
        'Ukraine':   ('2022-02-01', '2022-08-01'),
    }
    crises = crisis_dates or default_crises

    for name, (s, e) in crises.items():
        x0 = df_heat.index.searchsorted(pd.Timestamp(s))
        ax.axvline(x0, color='white', lw=1.5, ls='--', alpha=0.7)
        ax.text(x0 + 0.3, -0.5, name, color='white', fontsize=7, rotation=90, va='top')

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylabel('FX pair', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)

    plt.tight_layout()
    fname = f'{FIGURES_ROOT}/STABILITY/heatmap_loading_change_{mode}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

    return df_heat


def plot_eigenvalue_rolling(
        data,
        window_size=6,
        step_months=1,
        func=run_ffm_r1,
        start_date='2008-01-01',
        end_date='2025-12-31',
        figsize=(14, 5),
):
    """
    For each rolling window, extract the first eigenvalue of the normalised
    Gram matrix (λ₁) and the ratio λ₁/λ₂, then plot them side by side.

    Parameters
    ----------
    data        : dict {pair: DataFrame}  — vol_matrices
    window_size : int   — window size in months
    step_months : int   — step in months
    func        : callable — run_ffm_r1 or variant
    start_date  : str
    end_date    : str
    figsize     : tuple

    Returns
    -------
    DataFrame (index = window start dates, columns = lambda1, ratio12)
    """
    import matplotlib.dates as mdates

    records       = []
    current_start = pd.to_datetime(start_date)
    end_date_ts   = pd.to_datetime(end_date)

    while current_start < end_date_ts:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = ((pd.to_datetime(df.index) >= current_start) &
                        (pd.to_datetime(df.index) <= current_end))
                subset[pair] = df.loc[mask]

        print(f"Processing → {current_start.date()} to {current_end.date()}")

        if subset:
            try:
                ffm  = func(subset)
                eigs = ffm['lambda']

                # λ₁ — always available from ffm['lambda'][0]
                lambda1 = float(eigs[0]) if (eigs is not None and len(eigs) >= 1) else np.nan

                # λ₂ — use ffm['lambda'][1] when available (r≥2),
                # otherwise back-compute from ffm['V']:
                #   V[k] = V0 - Σ_{l=1}^{k+1} λ̃_l / T
                #   → V[0]-V[1] = λ̃₂/T = λ̂₂/√T   ⇒  λ̂₂ = (V[0]-V[1])·√T
                if eigs is not None and len(eigs) >= 2:
                    ratio12 = float(eigs[0] / (eigs[1] + 1e-14))
                elif ffm.get('V') is not None and len(ffm['V']) >= 2 and not np.isnan(lambda1):
                    T = ffm['X'].shape[1]
                    dV = float(ffm['V'][0] - ffm['V'][1])
                    lambda2 = dV * np.sqrt(T)
                    ratio12 = lambda1 / (lambda2 + 1e-14) if lambda2 > 0 else np.nan
                else:
                    ratio12 = np.nan

                records.append({
                    'date'    : current_start,
                    'lambda1' : lambda1,
                    'ratio12' : ratio12,
                })
            except Exception as e:
                print(f"  Skipped: {e}")

        current_start += pd.DateOffset(months=step_months)

    if not records:
        print("No results to plot.")
        return None

    df_out  = pd.DataFrame(records).set_index('date')
    dates   = df_out.index
    lambda1 = df_out['lambda1'].values
    ratio12 = df_out['ratio12'].values

    # ── Bar widths: proportional to actual gap between consecutive dates ──
    dates_num = mdates.date2num(dates.to_pydatetime())
    if len(dates_num) > 1:
        gaps   = np.diff(dates_num)
        widths = np.append(gaps, gaps[-1]) * 0.85
    else:
        widths = np.array([28.0])

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    margin = pd.DateOffset(months=1)
    x_min  = dates[0]  - margin
    x_max  = dates[-1] + margin

    for ax in (ax1, ax2):
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        ax.grid(True, alpha=0.25, axis='y')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    ax1.bar(dates, lambda1, width=widths, color='steelblue', alpha=0.8)
    ax1.set_ylabel('λ₁', fontsize=10)
    ax1.set_xlabel('Date', fontsize=9)
    ax1.set_title('First eigenvalue λ₁', fontsize=11)

    ax2.bar(dates, ratio12, width=widths, color='darkorange', alpha=0.8)
    ax2.set_ylabel('λ₁ / λ₂', fontsize=10)
    ax2.set_xlabel('Date', fontsize=9)
    ax2.set_title('Eigenvalue ratio λ₁ / λ₂', fontsize=11)

    fig.suptitle(
        f'Gram matrix eigenvalues — {window_size}m window, {step_months}m step',
        fontsize=12, fontweight='500'
    )
    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(
        f'{FIGURES_ROOT}/STABILITY/eigenvalue_rolling_{window_size}_{step_months}.png',
        dpi=160, bbox_inches='tight'
    )
    plt.show()

    return df_out


def plot_eigensystem_diff_rolling(
        data,
        window_size=6,
        step_months=1,
        func=run_ffm_r1,
        start_date='2008-01-01',
        end_date='2025-12-31',
        figsize=(14, 5),
):
    """
    For each consecutive pair of rolling windows, compute:
      - |Δλ₁| = |λ₁(t) − λ₁(t−1)|              (eigenvalue change)
      - ‖Δv₁‖_L² = ‖v₁(t) − v₁(t−1)‖_L²         (eigenfunction change)

    v₁ is the first loading B_hat[:,0,:] stacked across pairs (N×M → NM)
    and re-normalised to unit Euclidean norm via empilation_loadings (SVD),
    same convention as grassman_rolling_window. With this normalisation
    ‖Δv₁‖² = 2(1−cosθ), directly comparable to the Grassmann distance.
    Sign ambiguity resolved by aligning to previous window.

    Useful to calibrate Δτ / Δv thresholds for the Dette & Kutta test.

    Returns
    -------
    DataFrame (index = window start dates, columns = dlambda1, dv1_l2)
    """
    import matplotlib.dates as mdates

    records       = []
    current_start = pd.to_datetime(start_date)
    end_date_ts   = pd.to_datetime(end_date)

    prev_lambda1 = None
    prev_v1      = None
    prev_M       = None

    while current_start < end_date_ts:
        current_end = current_start + pd.DateOffset(months=window_size)

        subset = {}
        for pair, df in data.items():
            if current_start.year == 2012 and pair == 'audjpy':
                continue
            first_date = pd.to_datetime(df.index).min()
            if first_date < current_end:
                mask = ((pd.to_datetime(df.index) >= current_start) &
                        (pd.to_datetime(df.index) <= current_end))
                subset[pair] = df.loc[mask]

        print(f"Processing → {current_start.date()} to {current_end.date()}")

        dlambda1 = np.nan
        dv1_l2   = np.nan

        if subset:
            try:
                ffm = func(subset)

                # ── Eigenvalue ────────────────────────────────────────────
                eigs    = ffm['lambda']
                lambda1 = float(eigs[0]) if (eigs is not None and len(eigs) >= 1) else np.nan

                # ── Eigenfunction: orthonormal U[:,0] via SVD (same as
                #    grassman_rolling_window → empilation_loadings) ────────
                B = ffm.get('B_hat')   # (N, k, M)
                if B is not None and B.shape[1] >= 1:
                    _, U = empilation_loadings(B)   # U: (N*M, k), columns orthonormal
                    v1   = U[:, 0].astype(float)    # Euclidean unit vector
                    M    = B.shape[2]
                else:
                    v1 = None
                    M  = prev_M

                # ── Differences vs previous window ────────────────────────
                if prev_lambda1 is not None and not np.isnan(lambda1):
                    dlambda1 = abs(lambda1 - prev_lambda1)

                if prev_v1 is not None and v1 is not None and len(v1) == len(prev_v1):
                    # sign alignment; ‖Δv₁‖²_L² = ‖ΔU[:,0]‖²₂ = 2(1−cosθ)
                    v1_aligned = v1 if (v1 @ prev_v1 >= 0) else -v1
                    diff   = v1_aligned - prev_v1
                    dv1_l2 = float(np.sqrt(np.sum(diff ** 2)))

                records.append({
                    'date'     : current_start,
                    'lambda1'  : lambda1,
                    'dlambda1' : dlambda1,
                    'dv1_l2'   : dv1_l2,
                })

                prev_lambda1 = lambda1
                prev_v1      = v1_aligned if prev_v1 is not None else v1
                prev_M       = M

            except Exception as e:
                print(f"  Skipped: {e}")
                prev_lambda1 = None
                prev_v1      = None

        current_start += pd.DateOffset(months=step_months)

    if not records:
        print("No results to plot.")
        return None

    df_out   = pd.DataFrame(records).set_index('date')
    dates    = df_out.index
    dlambda1 = df_out['dlambda1'].values
    dv1_l2   = df_out['dv1_l2'].values

    dates_num = mdates.date2num(dates.to_pydatetime())
    if len(dates_num) > 1:
        gaps   = np.diff(dates_num)
        widths = np.append(gaps, gaps[-1]) * 0.85
    else:
        widths = np.array([28.0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    margin = pd.DateOffset(months=1)
    x_min  = dates[0]  - margin
    x_max  = dates[-1] + margin

    for ax in (ax1, ax2):
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        ax.grid(True, alpha=0.25, axis='y')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    ax1.bar(dates, dlambda1, width=widths, color='steelblue', alpha=0.8)
    ax1.set_ylabel('|Δλ₁|', fontsize=10)
    ax1.set_xlabel('Date', fontsize=9)
    ax1.set_title('|λ₁(t) − λ₁(t−1)|', fontsize=11)

    ax2.bar(dates, dv1_l2, width=widths, color='darkorange', alpha=0.8)
    ax2.set_ylabel('‖Δv₁‖_L²', fontsize=10)
    ax2.set_xlabel('Date', fontsize=9)
    ax2.set_title('‖v₁(t) − v₁(t−1)‖_L²', fontsize=11)

    fig.suptitle(
        f'Eigensystem changes — {window_size}m window, {step_months}m step',
        fontsize=12, fontweight='500'
    )
    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/STABILITY', exist_ok=True)
    plt.savefig(
        f'{FIGURES_ROOT}/STABILITY/eigensystem_diff_rolling_{window_size}_{step_months}.png',
        dpi=160, bbox_inches='tight'
    )
    plt.show()

    return df_out
