"""
=============================================================================
FOCUSED EDA PLOTS — 4 Selected FX Pairs
=============================================================================
Generates 6 figures (P0–P5) for the selected pairs using data as-is,
without applying any internal transform.

The caller is responsible for passing data in the desired state
(raw |r_t|, log-variance, smoothed, etc.) and for labelling it via
the `data_label` parameter, which is used as a filename prefix.

Naming convention (example with data_label='smoothed_log'):
    P0_smoothed_log_mean_curves.png
    P1_smoothed_log_covariance_surfaces.png
    P2_smoothed_log_cross_covariance.png
    P3_smoothed_log_distribution.png
    P4_smoothed_log_lag1_acf.png
    P5_smoothed_log_cross_lag1.png

USAGE:
    from ffm_eda_focused import run_focused_eda

    # vol_ffm_all already contains smoothed log(r_t²+ε) values
    run_focused_eda(vol_ffm_all, selected=['eurusd', 'usdtry'],
                    data_label='smoothed_log')

    # raw |r_t| matrices
    run_focused_eda(vol_matrices, selected=['eurusd', 'usdtry'],
                    data_label='raw_abs')
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import warnings
from paths import fig_dir
warnings.filterwarnings("ignore")

FIGURES_DIR = fig_dir('EDA')

DEFAULT_COLORS = ['#E74C3C', '#2980B9', '#27AE60', '#8E44AD']
CMAP_DIV       = 'RdBu_r'

LABEL_MAP = {
    'audjpy': 'AUD/JPY', 'eurusd': 'EUR/USD',
    'gbpusd': 'GBP/USD', 'usdsgd': 'USD/SGD',
    'usdtry': 'USD/TRY', 'eurjpy': 'EUR/JPY',
}


# =============================================================================
# HELPERS
# =============================================================================

def _align_pairs(vol_matrices, selected):
    """Align selected pairs to a common date/column intersection.
    Returns dict {key: (T, M) ndarray} and tau in [0, 1]."""
    sub = {k: vol_matrices[k] for k in selected if k in vol_matrices}
    if not sub:
        raise KeyError(f"None of {selected} found. "
                       f"Available: {list(vol_matrices.keys())}")

    common_days = None
    common_cols = None
    for df in sub.values():
        d = set(pd.to_datetime(df.index).normalize())
        c = set(df.columns)
        common_days = d if common_days is None else common_days & d
        common_cols = c if common_cols is None else common_cols & c

    common_days = sorted([d for d in common_days
                          if pd.Timestamp(d).dayofweek < 5])
    first_df    = next(iter(sub.values()))
    common_cols = [c for c in first_df.columns if c in common_cols]
    tau = np.linspace(0, 1, len(common_cols))

    out = {}
    for key, df in sub.items():
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        aligned = df2.reindex(index=pd.DatetimeIndex(common_days),
                              columns=common_cols)
        out[key] = aligned.values.astype(np.float64)

    return out, tau


def _corr_surface(Xi, Xj=None):
    """Corr(Xi(τ), Xj(τ')).  If Xj is None: auto-covariance surface."""
    T, M = Xi.shape
    Yi = Xi - np.nanmean(Xi, axis=0, keepdims=True)
    Yj = Yi if Xj is None else (Xj - np.nanmean(Xj, axis=0, keepdims=True))
    Yi_f  = np.nan_to_num(Yi, nan=0.0)
    Yj_f  = np.nan_to_num(Yj, nan=0.0)
    n_obs = (~np.isnan(Yi)).sum(axis=0)
    C     = (Yi_f.T @ Yj_f) / np.maximum(n_obs[:, None] - 1, 1)
    std_i = np.nanstd(Yi, axis=0)
    std_j = np.nanstd(Yj, axis=0)
    denom = np.outer(std_i, std_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        Corr = np.where(denom > 1e-12, C / denom, 0.0)
    return np.clip(Corr, -1, 1)


def _lag1_surface(Xi, Xj=None):
    """Lag-1 surface Corr(X_i[t](τ), X_j[t-1](τ')).
    If Xj is None: auto lag-1 surface."""
    T, M = Xi.shape
    Yi   = Xi - np.nanmean(Xi, axis=0, keepdims=True)
    Yj   = Yi if Xj is None else (Xj - np.nanmean(Xj, axis=0, keepdims=True))
    Yi_f = np.nan_to_num(Yi[1:],  nan=0.0)
    Yj_f = np.nan_to_num(Yj[:-1], nan=0.0)
    C     = (Yi_f.T @ Yj_f) / (T - 2)
    std_i = np.nanstd(Yi[1:],  axis=0)
    std_j = np.nanstd(Yj[:-1], axis=0)
    denom = np.outer(std_i, std_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        Corr = np.where(denom > 1e-12, C / denom, 0.0)
    return np.clip(Corr, -1, 1)


def _heatmap_grid(axes_flat, surfs, titles, xlabel, ylabel, norm):
    """Fill a flat list of axes with heatmap surfaces."""
    for ax, surf, title in zip(axes_flat, surfs, titles):
        im = ax.imshow(surf, origin='lower', cmap=CMAP_DIV, norm=norm,
                       extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)


def _save(fig, output_dir, fname):
    plt.tight_layout()
    fig.savefig(f'{output_dir}/{fname}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# =============================================================================
# FIGURE 0 — MEAN INTRADAY CURVE  μ̂_i(τ)
# =============================================================================

def plot_mean_curves(vol_matrices, selected, output_dir, data_label):
    """Mean intraday profile — raw mean ± 1 std + 80 sample curves."""
    mats, tau = _align_pairs(vol_matrices, selected)

    X_disps, mus, stds = {}, {}, {}
    for key in selected:
        X = mats[key]
        X_disps[key] = X
        mus[key]     = np.nanmean(X, axis=0)
        stds[key]    = np.nanstd(X,  axis=0)

    # Shared y-limits across all 4 pairs
    all_vals = np.concatenate([
        np.concatenate([X_disps[k].ravel(), mus[k] + stds[k], mus[k] - stds[k]])
        for k in selected
    ])
    y_min = np.nanpercentile(all_vals, 1)
    y_max = np.nanpercentile(all_vals, 99)
    margin = (y_max - y_min) * 0.05
    y_lim  = (y_min - margin, y_max + margin)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    np.random.seed(42)
    for idx, key in enumerate(selected):
        ax  = axes.flatten()[idx]
        col = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        lbl = LABEL_MAP.get(key, key.upper())
        T   = X_disps[key].shape[0]

        for i, t in enumerate(np.random.choice(T, min(80, T), replace=False)):
            ax.plot(tau, X_disps[key][t], color='#AAAAAA',
                    linewidth=0.4, alpha=0.35, zorder=1,
                    label='Sample days (80)' if i == 0 else None)

        ax.fill_between(tau, mus[key] - stds[key], mus[key] + stds[key],
                        color=col, alpha=0.20, zorder=2, label='±1 std')
        ax.plot(tau, mus[key] + stds[key], color=col, linewidth=0.8,
                linestyle='--', alpha=0.6, zorder=3)
        ax.plot(tau, mus[key] - stds[key], color=col, linewidth=0.8,
                linestyle='--', alpha=0.6, zorder=3)
        ax.plot(tau, mus[key], color=col, linewidth=2.5, zorder=4,
                label='Mean μ̂_i(τ)')
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel('τ  [0 = 17:00 NY → 1 = 16:55 NY]', fontsize=8)
        ax.set_ylabel(data_label, fontsize=9)
        ax.set_ylim(y_lim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Mean Intraday Volatility Curves  [{data_label}]",
                 fontsize=11, fontweight='bold')
    _save(fig, output_dir, f'P0_{data_label}_mean_curves.png')


# =============================================================================
# FIGURE 1 — INTRADAY COVARIANCE SURFACES  2×2
# =============================================================================

def plot_covariance_surfaces(vol_matrices, selected, output_dir, data_label):
    """Intraday correlation surface Corr(x(τ), x(τ')) — 2×2 grid."""
    mats, tau = _align_pairs(vol_matrices, selected)
    labels = [LABEL_MAP.get(k, k.upper()) for k in selected]

    surfs    = [_corr_surface(mats[k]) for k in selected]
    vmax_raw = max((np.nanmax(np.abs(s)) for s in surfs), default=0.0)
    vmax     = max(min(vmax_raw, 1.0), 1e-6)
    norm     = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    _heatmap_grid(axes.flatten(), surfs, labels, "$\\tau'$", "$\\tau$", norm)
    fig.suptitle(f"Intraday Correlation Matrices  [{data_label}]",
                 fontsize=10, fontweight='bold')
    _save(fig, output_dir, f'P1_{data_label}_covariance_surfaces.png')


# =============================================================================
# FIGURE 2 — CROSS-COVARIANCE SURFACES  2×3
# =============================================================================

def plot_cross_covariance(vol_matrices, selected, output_dir, data_label):
    """Cross-correlation CCov_{ij}(τ,τ') — 2×3 grid (6 pair combos)."""
    mats, tau = _align_pairs(vol_matrices, selected)
    combos = [(selected[i], selected[j])
              for i in range(len(selected))
              for j in range(i + 1, len(selected))]

    surfs  = [_corr_surface(mats[pi], mats[pj]) for pi, pj in combos]
    titles = [f"{LABEL_MAP.get(pi, pi.upper())}  ×  {LABEL_MAP.get(pj, pj.upper())}"
              for pi, pj in combos]
    vmax   = min(max(np.abs(s).max() for s in surfs), 1.0)
    norm   = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, surf, title, (pi, pj) in zip(axes.flatten(), surfs, titles, combos):
        im = ax.imshow(surf, origin='lower', cmap=CMAP_DIV, norm=norm,
                       extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(f"$\\tau'$  [{LABEL_MAP.get(pj, pj.upper())}]", fontsize=9)
        ax.set_ylabel(f"$\\tau$  [{LABEL_MAP.get(pi, pi.upper())}]", fontsize=9)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)

    fig.suptitle(f"Cross-Correlation Matrices  [{data_label}]",
                 fontsize=10, fontweight='bold')
    _save(fig, output_dir, f'P2_{data_label}_cross_covariance.png')


# =============================================================================
# FIGURE 3 — DISTRIBUTION
# =============================================================================

def plot_distribution(vol_matrices, selected, output_dir, data_label):
    """Marginal distribution of pooled values for each pair — 2×2 grid."""
    mats, tau = _align_pairs(vol_matrices, selected)
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for idx, key in enumerate(selected):
        ax  = axes.flatten()[idx]
        col = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        lbl = LABEL_MAP.get(key, key.upper())

        vals = mats[key].ravel()
        vals = vals[np.isfinite(vals)]
        v    = vals[np.random.choice(len(vals), min(5000, len(vals)), replace=False)]

        ax.hist(v, bins=60, density=True, color=col, alpha=0.65, edgecolor='none')
        mu_v, sd_v = np.mean(v), np.std(v)
        xg = np.linspace(v.min(), v.max(), 200)
        ax.plot(xg, stats.norm.pdf(xg, mu_v, sd_v), 'k-',
                linewidth=1.8, label='Normal fit')

        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel(data_label, fontsize=10)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Marginal Distribution  [{data_label}]",
                 fontsize=11, fontweight='bold')
    _save(fig, output_dir, f'P3_{data_label}_distribution.png')


# =============================================================================
# FIGURE 4 — LAG-1 AUTOCORRELATION SURFACES  2×2
# =============================================================================

def plot_lag1_acf(vol_matrices, selected, output_dir, data_label):
    """Lag-1 autocorrelation surface AC1_i(τ,τ') — 2×2 grid."""
    mats, tau = _align_pairs(vol_matrices, selected)
    labels = [LABEL_MAP.get(k, k.upper()) for k in selected]

    surfs    = [_lag1_surface(mats[k]) for k in selected]
    vmax_raw = max((np.nanmax(np.abs(s)) for s in surfs), default=0.0)
    vmax     = max(min(vmax_raw, 1.0), 1e-6)
    norm     = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    _heatmap_grid(axes.flatten(), surfs, labels,
                  "$\\tau'$ yesterday", "$\\tau$ today", norm)
    fig.suptitle(f"Correlation Matrices Lag-1  [{data_label}]",
                 fontsize=10, fontweight='bold')
    _save(fig, output_dir, f'P4_{data_label}_lag1_acf.png')


# =============================================================================
# FIGURE 5 — CROSS LAG-1 SURFACES  2×3
# =============================================================================

def plot_cross_lag1(vol_matrices, selected, output_dir, data_label):
    """Cross lag-1 surface CCor1_{ij}(τ,τ') — 2×3 grid (6 pair combos)."""
    mats, tau = _align_pairs(vol_matrices, selected)
    combos = [(selected[i], selected[j])
              for i in range(len(selected))
              for j in range(i + 1, len(selected))]

    surfs  = [_lag1_surface(mats[pi], mats[pj]) for pi, pj in combos]
    titles = [
        f"{LABEL_MAP.get(pi, pi.upper())} today  ×  "
        f"{LABEL_MAP.get(pj, pj.upper())} lag-1"
        for pi, pj in combos
    ]
    xlabels = [
        f"$\\tau'$  [{LABEL_MAP.get(pj, pj.upper())}] yesterday"
        for _, pj in combos
    ]
    vmax = min(max(np.abs(s).max() for s in surfs), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, surf, title, xl, (pi, pj) in zip(axes.flatten(), surfs, titles, xlabels, combos):
        im = ax.imshow(surf, origin='lower', cmap=CMAP_DIV, norm=norm,
                       extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(f"$\\tau$  [{LABEL_MAP.get(pi, pi.upper())}] today", fontsize=8)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)

    fig.suptitle(f"Cross-Correlation Matrices Lag-1  [{data_label}]",
                 fontsize=10, fontweight='bold')
    _save(fig, output_dir, f'P5_{data_label}_cross_lag1.png')


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_focused_eda(vol_matrices: dict,
                    selected: list = None,
                    output_dir: str = None,
                    data_label: str = 'data'):
    """
    Generate 6 focused EDA figures for the selected pairs.
    Data is used as-is — no internal transform is applied.

    Parameters
    ----------
    vol_matrices : dict {pair_name: DataFrame(days x intervals)}
                   Values in whatever state the caller provides
                   (raw |r_t|, log-variance, smoothed, etc.).
    selected     : list of pair keys (default: ['audjpy','eurusd','gbpusd','usdsgd'])
    output_dir   : output directory (default: src/figures/EDA/)
    data_label   : string used as filename prefix and in plot titles.
                   Describes the state of the input data.
                   Examples: 'raw_abs', 'log_var', 'smoothed_log'

    Output files  (example with data_label='smoothed_log')
    ------------
    P0_smoothed_log_mean_curves.png
    P1_smoothed_log_covariance_surfaces.png
    P2_smoothed_log_cross_covariance.png
    P3_smoothed_log_distribution.png
    P4_smoothed_log_lag1_acf.png
    P5_smoothed_log_cross_lag1.png
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    if selected is None:
        selected = ['audjpy', 'eurusd', 'gbpusd', 'usdsgd']

    norm_map = {k.lower().replace('/', '').replace('_', ''): k
                for k in vol_matrices}
    sel_norm = [k.lower().replace('/', '').replace('_', '')
                for k in selected]
    mapped   = []
    for s in sel_norm:
        if s in norm_map:
            mapped.append(norm_map[s])
        else:
            raise KeyError(f"'{s}' not found. Available: {list(norm_map.keys())}")

    vol_sub = {s: vol_matrices[mapped[i]] for i, s in enumerate(sel_norm)}

    print(f"\n{'='*60}")
    print(f"  FOCUSED EDA — {len(sel_norm)} pairs  |  label: {data_label}")
    print(f"  Pairs  : {sel_norm}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}")

    print(f"  [P0] Mean intraday curves...")
    plot_mean_curves(vol_sub, sel_norm, output_dir, data_label)

    print(f"  [P1] Intraday covariance surfaces...")
    plot_covariance_surfaces(vol_sub, sel_norm, output_dir, data_label)

    print(f"  [P2] Cross-covariance surfaces...")
    plot_cross_covariance(vol_sub, sel_norm, output_dir, data_label)

    print(f"  [P3] Marginal distributions...")
    plot_distribution(vol_sub, sel_norm, output_dir, data_label)

    print(f"  [P4] Lag-1 autocorrelation surfaces...")
    plot_lag1_acf(vol_sub, sel_norm, output_dir, data_label)

    print(f"  [P5] Cross lag-1 surfaces...")
    plot_cross_lag1(vol_sub, sel_norm, output_dir, data_label)

    print(f"\n  ✅  6 figures saved to '{output_dir}/'")
    print(f"      P0–P5_{data_label}_*.png")
    print(f"{'='*60}\n")
