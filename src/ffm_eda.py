"""
Exploratory data analysis helpers for the intraday FX volatility panel.

    missing_by_year_pair        — missing-value rate per year and pair (pair selection)
    get_stable_pairs            — pairs with enough intraday coverage over a period
    plot_abs_returns_over_time  — daily mean of |5-min log-returns| (thesis Figure 2)
    plot_raw_vs_smooth          — raw vs P-spline-smoothed curves on one day (Figure 4)
    plot_mean_curves, plot_distribution, plot_cross_covariance — plotting helpers
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import warnings
from paths import fig_dir, FIGURES_ROOT
warnings.filterwarnings("ignore")

FIGURES_DIR = fig_dir('EDA')


# =============================================================================
# COLOR PALETTE
# =============================================================================

def _make_colors(n=40):
    import matplotlib.cm as cm
    tab20  = [cm.tab20(i)  for i in np.linspace(0, 1, 20, endpoint=False)]
    tab20b = [cm.tab20b(i) for i in np.linspace(0, 1, 20, endpoint=False)]
    palette = tab20 + tab20b
    return ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
            for r, g, b, _ in palette[:n]]

COLORS = _make_colors(40)


# =============================================================================
# 01 — MEAN CURVES
# =============================================================================

def plot_mean_curves(X, tau, pairs, transform, output_dir=f'{FIGURES_ROOT}/EDA'):
    N = len(pairs)
    ncols = 4
    nrows = (N + ncols - 1) // ncols

    is_log = X.min() < -1
    X_disp = np.exp(X / 2) * 1e4 if is_log else X * 1e4
    ylabel = "|r| avg x10^4  [=exp(mu/2)]" if is_log else "|r| x10^4"

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, pair in enumerate(pairs):
        ax  = axes[i]
        mu  = np.nanmean(X_disp[i], axis=0)
        sd  = np.nanstd(X_disp[i], axis=0)
        q10 = np.nanpercentile(X_disp[i], 10, axis=0)
        q90 = np.nanpercentile(X_disp[i], 90, axis=0)

        ax.fill_between(tau, q10, q90, alpha=0.12,
                        color=COLORS[i % len(COLORS)], label='[q10, q90]')
        ax.fill_between(tau, np.maximum(mu - sd, 0), mu + sd,
                        alpha=0.25, color=COLORS[i % len(COLORS)], label='mu +/- 1 std')
        ax.plot(tau, mu, color=COLORS[i % len(COLORS)], linewidth=2, label='mean')

        ax.set_title(pair, fontsize=10, fontweight='bold')
        ax.set_xlabel("tau", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Mean Intraday Volatility Profiles  mu_i(tau) — transform: {transform}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_eda_mean_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_eda_mean_curves.png")


# =============================================================================
# 03 — MARGINAL DISTRIBUTION
# =============================================================================

def plot_distribution(X, tau, pairs, transform, output_dir=f'{FIGURES_ROOT}/EDA'):
    """
    Marginal distribution at representative tau points.
    Shapiro-Wilk test validates Gaussianity — key for the log transform.
    """
    is_log   = X.min() < -1
    selected = pairs[:min(6, len(pairs))]
    idx_sel  = [pairs.index(p) for p in selected]

    tau_idx  = [int(q * X.shape[2]) for q in [0.05, 0.15, 0.35, 0.50, 0.70, 0.90]]
    tau_idx  = [min(t, X.shape[2]-1) for t in tau_idx]

    fig, axes = plt.subplots(len(selected), len(tau_idx),
                             figsize=(18, 3 * len(selected)))

    for row, (ax_row, i) in enumerate(zip(axes, idx_sel)):
        for col, (ax, mi) in enumerate(zip(ax_row, tau_idx)):
            vals = X[i, :, mi]
            vals = vals[np.isfinite(vals)]

            ax.hist(vals, bins=40, density=True,
                    color=COLORS[i % len(COLORS)], alpha=0.7, edgecolor='none')

            if len(vals) > 10:
                mu_v, sd_v = np.mean(vals), np.std(vals)
                xg  = np.linspace(vals.min(), vals.max(), 100)
                ax.plot(xg, stats.norm.pdf(xg, mu_v, sd_v),
                        'k--', linewidth=1.5, alpha=0.8)

                sw_vals = vals[np.random.choice(len(vals),
                               min(200, len(vals)), replace=False)]
                _, pval = stats.shapiro(sw_vals)
                pcolor  = 'green' if pval > 0.05 else 'red'
                ax.set_title(f"tau={tau_idx[col]/X.shape[2]:.2f}\np={pval:.3f}",
                             fontsize=8, color=pcolor)

            if col == 0:
                ax.set_ylabel(selected[row], fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Marginal Distribution by tau — transform: {transform}\n"
                 f"Black dashed = fitted Gaussian | Shapiro-Wilk p-value "
                 f"(green = Gaussian, red = non-Gaussian)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_eda_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_eda_distribution.png")


# =============================================================================
# 06 — CROSS-COVARIANCE BETWEEN PAIRS
# =============================================================================

def plot_cross_covariance(X, tau, pairs, output_dir=f'{FIGURES_ROOT}/EDA'):
    """
    Functional cross-covariance between pairs.
    CCov_{ij}(tau) = Corr(x_i(tau), x_j(tau)) — pointwise correlation by tau.
    Reveals whether pairs are synchronised at specific hours of the day.
    """
    N = len(pairs)

    corr_avg = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                corr_avg[i, j] = 1.0
            else:
                Y_i = X[i] - np.nanmean(X[i], axis=0, keepdims=True)
                Y_j = X[j] - np.nanmean(X[j], axis=0, keepdims=True)
                cc  = np.array([
                    np.corrcoef(Y_i[:, m], Y_j[:, m])[0, 1]
                    if (np.std(Y_i[:, m]) > 1e-10 and np.std(Y_j[:, m]) > 1e-10)
                    else 0.0
                    for m in range(X.shape[2])
                ])
                corr_avg[i, j] = np.nanmean(cc)

    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    dist = 1 - np.abs(corr_avg)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 1)
    dist = (dist + dist.T) / 2
    linkage = hierarchy.linkage(squareform(dist), method='ward')
    order   = hierarchy.leaves_list(linkage)

    corr_ord  = corr_avg[np.ix_(order, order)]
    pairs_ord = [pairs[o] for o in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1)
    im = ax.imshow(corr_ord, cmap='RdYlGn', norm=norm, aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean cross-correlation')
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(pairs_ord, rotation=90, fontsize=7)
    ax.set_yticklabels(pairs_ord, fontsize=7)
    ax.set_title("Cross-correlation matrix\n"
                 "(hierarchical clustering order)")

    ax = axes[1]
    corr_triu = corr_avg.copy()
    np.fill_diagonal(corr_triu, -999)
    flat_idx  = np.argsort(corr_triu.ravel())[-6:]
    pairs_top = [(int(k // N), int(k % N)) for k in flat_idx]
    pairs_top = [(i, j) for i, j in pairs_top if i != j][:4]

    for k, (pi, pj) in enumerate(pairs_top):
        Y_i = X[pi] - np.nanmean(X[pi], axis=0, keepdims=True)
        Y_j = X[pj] - np.nanmean(X[pj], axis=0, keepdims=True)
        cc  = np.array([
            np.corrcoef(Y_i[:, m], Y_j[:, m])[0, 1]
            if (np.std(Y_i[:, m]) > 1e-10 and np.std(Y_j[:, m]) > 1e-10)
            else 0.0
            for m in range(X.shape[2])
        ])
        ax.plot(tau, cc, color=COLORS[k * 3 % len(COLORS)], linewidth=2,
                label=f"{pairs[pi]} / {pairs[pj]}")

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("tau [0=open -> 1=close]")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("Intraday cross-correlation profile\n"
                 "(4 most correlated pairs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.05)

    fig.suptitle("Functional Cross-Covariance Between FX Pairs",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_eda_cross_covariance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_eda_cross_covariance.png")


# =============================================================================
# EDA UTILITY FUNCTIONS
# =============================================================================

def missing_by_year_pair(vol_matrices: dict) -> pd.DataFrame:
    """
    Compute the % of missing values per pair and per year in raw matrices.

    Parameters
    ----------
    vol_matrices : dict {str: DataFrame(days x minutes)}
        Raw matrices before imputation.

    Returns
    -------
    df_pct : DataFrame shape (n_years, n_pairs), values in %.
    """
    records = {}

    for pair, df in vol_matrices.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df[df.index.dayofweek < 5]
        df['_year'] = df.index.year

        for year, grp in df.groupby('_year'):
            if year in (2026, 2000):
                continue
            grp = grp.drop(columns='_year')
            total = grp.size
            n_nan = grp.isna().sum().sum()
            pct   = 100.0 * n_nan / total if total > 0 else np.nan
            records.setdefault(year, {})[pair] = round(pct, 2)

    df_pct = (pd.DataFrame(records).T.sort_index())
    df_pct.index.name   = 'Year'
    df_pct.columns.name = 'Pair'
    df_pct = df_pct[sorted(df_pct.columns)]

    return df_pct


def get_stable_pairs(vol_matrices: dict, start: str, end: str,
                     min_day_coverage: float = 0.8) -> list:
    """
    Return pairs that have sufficient data coverage over [start, end].

    A pair is stable if >= min_day_coverage fraction of its weekdays have
    at least 70% of intraday intervals non-NaN.
    """
    stable = []
    for pair, df in vol_matrices.items():
        mask = (pd.to_datetime(df.index) >= start) & \
               (pd.to_datetime(df.index) <= end)
        df_w = df.loc[mask]
        df_w = df_w[pd.to_datetime(df_w.index).dayofweek < 5]
        frac = (df_w.notna().mean(axis=1) >= 0.7).mean()
        if frac >= min_day_coverage:
            stable.append(pair)
    return stable


def plot_raw_vs_smooth(vol_log_all: dict, vol_ffm_all: dict,
                       day: str, pairs: list) -> None:
    """
    2x2 comparison: raw log-transform vs smoothed for 4 FX pairs on a given day.

    Parameters
    ----------
    vol_log_all : dict {pair: DataFrame(days × minutes)} — raw log(r_t² + ε)
    vol_ffm_all : dict {pair: DataFrame(days × minutes)} — smoothed log-vol (P-spline)
    day         : str  ex. '2019-04-22'
    pairs       : list of 4 str, e.g. ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF']
    """
    if len(pairs) != 4:
        raise ValueError(f"Exactly 4 pairs required, {len(pairs)} given.")

    target = pd.Timestamp(day)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, pair in zip(axes, pairs):
        color_raw    = '#2171b5'
        color_smooth = '#d7191c'

        plotted = False
        for label, vol, color, lw, alpha, zorder in [
            ('Raw',     vol_log_all, color_raw,    0.8, 0.7, 1),
            ('Smooth',  vol_ffm_all, color_smooth, 1.6, 1.0, 2),
        ]:
            if pair not in vol:
                continue
            mat     = vol[pair]
            mat_idx = pd.to_datetime(mat.index).normalize()
            diffs   = np.abs((mat_idx - target).days)
            t       = int(np.argmin(diffs))
            if diffs[t] > 3:
                continue
            row = mat.iloc[t].values.astype(float)
            x   = np.linspace(0, 1, len(row))
            ax.plot(x, row, lw=lw, alpha=alpha, color=color,
                    label=label, zorder=zorder)
            plotted = True

        actual = str(pd.to_datetime(mat.index).normalize()[t])[:10] if plotted else day
        pair_fmt = f'{pair[:3]}/{pair[3:]}' if len(pair) == 6 else pair
        ax.set_title(f'{pair_fmt} — {actual}', fontsize=11, fontweight='bold')
        ax.set_xlabel('τ', fontsize=9)
        ax.set_ylabel('log(r²+ε)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.3, alpha=0.4)
        if plotted:
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(f'Raw vs Smooth — {day}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f'raw_vs_smooth_{day}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"✓ {fname}")
    plt.show()


def plot_abs_returns_over_time(vol_matrices: dict, pairs: list) -> None:
    """
    Single 2x2 plot — daily mean of |r_t| (2014–2025) for 4 pairs.

    Parameters
    ----------
    vol_matrices : dict {pair: DataFrame(days × minutes)} — |r_t|
    pairs        : list of 4 str
    """
    if len(pairs) != 4:
        raise ValueError(f"Exactly 4 pairs required, {len(pairs)} given.")

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()

    for ax, pair in zip(axes, pairs):
        if pair not in vol_matrices:
            print(f"⚠ pair '{pair}' not found")
            ax.set_visible(False)
            continue

        s = np.abs(vol_matrices[pair]).mean(axis=1)
        s.index = pd.to_datetime(s.index).normalize()
        s = s[s.index.dayofweek < 5]

        pair_fmt = f'{pair[:3]}/{pair[3:]}' if len(pair) == 6 else pair

        ax.plot(s.index, s.values, lw=0.5, alpha=0.8, color='steelblue')
        ax.set_title(pair_fmt, fontsize=11, fontweight='bold')
        ax.set_ylabel('mean |r_t|', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.3, alpha=0.4)

    fig.suptitle('Daily mean |r_t| — 2014–2025', fontsize=13, fontweight='bold')
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, 'abs_returns_2014_2025.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"✓ {fname}")
    plt.show()
