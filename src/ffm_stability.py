#Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

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


def align_signs(U_old, U_new):
    """Align the signs of U_new on U_old, column by column."""
    signs = np.sign(np.diag(U_old.T @ U_new))
    signs[signs == 0] = 1
    return U_new * signs

from functools import partial
run_ffm_r1 = partial(run_ffm, r_forced=1, silent= True)


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
    Plot the results of bai_perron_distances_hac:
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
# CUSUM TEST
#=================================================================================================================


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
