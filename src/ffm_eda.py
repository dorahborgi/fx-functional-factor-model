"""
=============================================================================
FDA EXPLORATORY DATA ANALYSIS — Intraday FX Volatility
=============================================================================
Comprehensive exploratory analysis of (day x minute) FX volatility matrices
in the Functional Data Analysis (FDA) framework.

USAGE:
    from ffm_eda import run_eda

    # vol_matrices contains |r_t| (output of intraday_volatility_matrix)
    run_eda(vol_matrices, transform='log')    # log(r_t^2 + eps)  [recommended]
    run_eda(vol_matrices, transform='abs')    # |r_t| raw
    run_eda(vol_matrices, transform='none')   # data as-is

OUTPUT FILES:
    00_eda_missing_values.png    — Missing data heatmap, fill rates, patterns
    01_eda_mean_curves.png       — Mean curves mu_i(tau) with +/- 1 std envelope
    02_eda_sample_curves.png     — Individual daily trajectories (random sample)
    03_eda_distribution.png      — Marginal distribution at selected tau points
    04_eda_covariance_surface.png — Intraday covariance surface Cov(tau, tau')
    05_eda_covariance_lag1.png   — Temporal lag-1 covariance (volatility clustering)
    06_eda_cross_covariance.png  — Cross-covariance between selected pairs
    07_eda_correlation_matrix.png — Cross-pair correlation matrix + dendrogram
    08_eda_variability.png       — Intraday vs inter-day variability decomposition
    09_eda_periodicity.png       — Temporal ACF of daily volatility levels
    10_eda_extremes.png          — Extreme days (most and least volatile)
    11_eda_cross_covariance_surfaces.png — Cross-pair surfaces CCov_ij(tau,tau')
    12_eda_cross_cov_<pair>.png          — Per-pair surfaces vs all others (N files)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# TRANSFORMATION
# =============================================================================

def _apply_transform(vol_matrices: dict, transform: str) -> dict:
    """
    Apply the chosen transformation to the volatility matrices.

    transform : 'log'  -> log(r_t^2 + eps)   [recommended for FDA]
                'abs'  -> |r_t| as-is
                'none' -> raw data unchanged
    """
    result = {}
    for key, df in vol_matrices.items():
        vals = df.values.astype(np.float64)

        if transform == 'log':
            rv2       = vals ** 2
            rv2_clean = np.where(np.isnan(rv2), 0.0, np.maximum(rv2, 0.0))
            pos       = rv2_clean[rv2_clean > 0]
            eps       = np.percentile(pos, 5) if len(pos) > 0 else 1e-12
            out       = np.log(rv2_clean + eps)
        elif transform == 'abs':
            out = np.where(np.isnan(vals), 0.0, np.abs(vals))
        else:
            out = np.where(np.isnan(vals), 0.0, vals)

        result[key] = pd.DataFrame(out, index=df.index, columns=df.columns)
    return result


def _build_panel(vol_matrices: dict) -> tuple:
    """
    Build X (N, T, M) from transformed matrices.
    Uses intersection of dates and columns; removes weekends.
    Returns X, tau, days, pairs, raw_vol_matrices (original, untransformed).
    """
    pairs = list(vol_matrices.keys())
    N     = len(pairs)

    common_days = None
    common_cols = None
    for df in vol_matrices.values():
        d = set(pd.to_datetime(df.index).normalize())
        c = set(df.columns)
        common_days = d if common_days is None else common_days & d
        common_cols = c if common_cols is None else common_cols & c

    common_days = sorted([d for d in common_days
                          if pd.Timestamp(d).dayofweek < 5])
    # Preserve column order from first matrix (respects filter_session overnight)
    first_df    = next(iter(vol_matrices.values()))
    common_cols = [c for c in first_df.columns if c in common_cols]
    T, M = len(common_days), len(common_cols)

    X = np.full((N, T, M), np.nan)
    for i, (pair, df) in enumerate(vol_matrices.items()):
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        aligned = df2.reindex(index=pd.DatetimeIndex(common_days),
                              columns=common_cols)
        X[i] = aligned.values

    tau  = np.linspace(0, 1, M)
    days = pd.DatetimeIndex(common_days)
    return X, tau, days, pairs


# =============================================================================
# 00 — MISSING VALUES ANALYSIS
# =============================================================================

def plot_missing_values(vol_matrices: dict, output_dir: str = f'{FIGURES_ROOT}/EDA'):
    """
    Comprehensive missing value analysis BEFORE any transformation.

    Panels:
    A) Fill rate (% non-NaN) per pair per intraday interval
       -> reveals systematic gaps (night hours, low-liquidity sessions)
    B) Fill rate per pair per day (time series)
       -> reveals calendar effects (holidays, data outages)
    C) Missing value heatmap (pair x day subsample)
       -> visual map of NaN locations
    D) Summary statistics table: total NaN %, worst days, worst intervals
    """
    pairs = list(vol_matrices.keys())
    N     = len(pairs)

    # Align all matrices to common days/cols
    common_days = None
    common_cols = None
    for df in vol_matrices.values():
        d = set(pd.to_datetime(df.index).normalize())
        c = set(df.columns)
        common_days = d if common_days is None else common_days & d
        common_cols = c if common_cols is None else common_cols & c

    common_days = sorted([d for d in common_days
                          if pd.Timestamp(d).dayofweek < 5])
    # Preserve column order from first matrix (respects filter_session overnight)
    first_df    = next(iter(vol_matrices.values()))
    common_cols = [c for c in first_df.columns if c in common_cols]
    T, M = len(common_days), len(common_cols)
    tau  = np.linspace(0, 1, M)

    # Build binary missingness array (N, T, M) : 1=missing, 0=observed
    Miss = np.zeros((N, T, M), dtype=np.float32)
    for i, (pair, df) in enumerate(vol_matrices.items()):
        df2 = df.copy()
        df2.index = pd.to_datetime(df2.index).normalize()
        aligned = df2.reindex(index=pd.DatetimeIndex(common_days),
                              columns=common_cols)
        Miss[i] = aligned.isna().values.astype(np.float32)

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── A. Fill rate by tau (per pair) ────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    fill_by_tau = 1 - Miss.mean(axis=1)   # (N, M)
    for i, pair in enumerate(pairs):
        ax_a.plot(tau, fill_by_tau[i] * 100,
                  color=COLORS[i % len(COLORS)], linewidth=1.5,
                  alpha=0.85, label=pair)
    ax_a.axhline(80, color='red', linestyle='--', linewidth=1.2, label='80% threshold')
    ax_a.set_xlabel("Intraday interval τ [0=open → 1=close]")
    ax_a.set_ylabel("Fill rate (%)")
    ax_a.set_title("A — Fill rate by intraday interval τ\n"
                   "← Drops reveal low-liquidity hours (night, pre-open)")
    ax_a.legend(fontsize=6, ncol=3, loc='lower center')
    ax_a.grid(True, alpha=0.3)
    ax_a.set_ylim(0, 105)

    # ── B. Fill rate by day (cross-sectional average) ─────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    fill_by_day = 1 - Miss.mean(axis=(0, 2))   # (T,) averaged over N and M
    ax_b.fill_between(range(T), fill_by_day * 100,
                      alpha=0.6, color='steelblue')
    ax_b.axhline(80, color='red', linestyle='--', linewidth=1.2)
    ax_b.set_xlabel("Day index t")
    ax_b.set_ylabel("Fill rate (%) — avg over all pairs & intervals")
    ax_b.set_title("B — Cross-sectional fill rate over time\n"
                   "← Dips = holidays, data outages")
    ax_b.grid(True, alpha=0.3)
    ax_b.set_ylim(0, 105)

    # ── C. Missing heatmap per pair (subsample of days) ───────────────────
    ax_c = fig.add_subplot(gs[1, :])
    # Show mean missingness per (pair, day) — averaged over tau
    miss_by_pair_day = Miss.mean(axis=2)   # (N, T)
    # Subsample days for readability
    t_step = max(1, T // 300)
    miss_sub = miss_by_pair_day[:, ::t_step]

    im = ax_c.imshow(miss_sub, aspect='auto', cmap='YlOrRd',
                     vmin=0, vmax=0.5, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.02, pad=0.01)
    cbar.set_label('Missing rate (0=full, 0.5=half missing)')
    ax_c.set_yticks(range(N))
    ax_c.set_yticklabels(pairs, fontsize=8)
    ax_c.set_xlabel(f"Day index (1 tick = {t_step} days)")
    ax_c.set_title("C — Missing data heatmap: pairs × days\n"
                   "← Yellow/red = high missingness (nights, holidays, data gaps)")

    # ── D. Summary table ──────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, :])
    ax_d.axis('off')

    # Compute per-pair stats
    table_data = []
    for i, pair in enumerate(pairs):
        total_miss  = Miss[i].mean() * 100
        worst_tau   = tau[np.argmax(Miss[i].mean(axis=0))]
        worst_day_i = np.argmax(Miss[i].mean(axis=1))
        worst_date  = str(common_days[worst_day_i])[:10]
        # Number of fully missing days
        n_full_miss = (Miss[i].mean(axis=1) > 0.95).sum()
        # Mean fill rate on active intervals (fill > 0)
        active_fill = (1 - Miss[i][:, Miss[i].mean(axis=0) < 0.5]).mean() * 100
        table_data.append([
            pair,
            f"{total_miss:.1f}%",
            f"{active_fill:.1f}%",
            f"τ={worst_tau:.2f}",
            worst_date,
            str(n_full_miss)
        ])

    cols_tbl = ['Pair', 'Overall NaN%', 'Active fill%',
                'Worst τ', 'Worst day', 'Fully missing days']
    tbl = ax_d.table(cellText=table_data, colLabels=cols_tbl,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F8F9FA')
        cell.set_edgecolor('#DEE2E6')

    ax_d.set_title("D — Missing value summary per pair", fontsize=10,
                   fontweight='bold', pad=8)

    fig.suptitle("Missing Value Analysis — Raw Volatility Matrices",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(f'{output_dir}/00_eda_missing_values.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 00_eda_missing_values.png")


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
# 02 — SAMPLE CURVES
# =============================================================================

def plot_sample_curves(X, tau, pairs, days, transform, output_dir=f'{FIGURES_ROOT}/EDA', n_sample=20):
    """
    Individual daily trajectories for ALL pairs.
    Grid layout adapts automatically to the number of pairs.
    """
    N      = len(pairs)
    is_log = X.min() < -1
    X_disp = np.exp(X / 2) * 1e4 if is_log else X * 1e4
    ylabel = "|r| x10^4"

    ncols = 4
    nrows = (N + ncols - 1) // ncols

    np.random.seed(42)
    T     = X.shape[1]
    t_idx = np.random.choice(T, min(n_sample, T), replace=False)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, pair in enumerate(pairs):
        ax = axes[i]
        mu = np.nanmean(X_disp[i], axis=0)
        for t in t_idx:
            ax.plot(tau, X_disp[i, t], color='gray', alpha=0.12, linewidth=0.6)
        ax.plot(tau, mu, color=COLORS[i % len(COLORS)], linewidth=2.2,
                label='mean', zorder=10)
        ax.set_title(pair, fontsize=10, fontweight='bold')
        ax.set_xlabel("tau", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Individual Daily Trajectories — All {N} pairs  "
                 f"({n_sample} random days)\n"
                 f"Grey = individual days | Colour = cross-day mean",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_eda_sample_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_eda_sample_curves.png")


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
# 04 — INTRADAY COVARIANCE SURFACE
# =============================================================================

def plot_covariance_surface(X, tau, pairs, output_dir=f'{FIGURES_ROOT}/EDA'):
    """
    Covariance surface C_i(tau, tau') = Cov(x_i(tau), x_i(tau')).
    Core FDA object — reveals the intraday dependence structure.
    Off-diagonal persistence indicates volatility clustering within the day.
    Shows ALL pairs in an adaptive grid.
    Colormap: white = 0, red = positive, blue = negative.
    """
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.cm as cm

    N     = len(pairs)
    ncols = 4
    nrows = (N + ncols - 1) // ncols

    # Pre-compute all correlation surfaces to get global vmin/vmax
    corr_surfaces = []
    for i in range(N):
        Y_i   = X[i] - np.nanmean(X[i], axis=0, keepdims=True)
        T_i   = Y_i.shape[0]
        C_i   = (Y_i.T @ Y_i) / (T_i - 1)
        std_i = np.sqrt(np.diag(C_i))
        with np.errstate(divide='ignore', invalid='ignore'):
            Corr_i = C_i / np.outer(std_i, std_i)
            Corr_i = np.nan_to_num(Corr_i, nan=0.0)
        corr_surfaces.append(np.clip(Corr_i, -1, 1))

    # Shared colormap: white=0, red=positive, blue=negative
    vmax = max(np.abs(c).max() for c in corr_surfaces)
    vmax = min(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = 'RdBu_r'   # red=positive, blue=negative, white=0

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.2 * nrows))
    axes = axes.flatten()

    for i, pair in enumerate(pairs):
        ax = axes[i]
        im = ax.imshow(corr_surfaces[i], origin='lower', cmap=cmap, norm=norm,
                       extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Corr')
        ax.set_title(pair, fontsize=10, fontweight='bold')
        ax.set_xlabel("tau'", fontsize=8)
        ax.set_ylabel("tau", fontsize=8)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Intraday Covariance Surface  Corr(x(tau), x(tau')) — All pairs\n"
                 "Red = positive correlation | Blue = negative | White = zero\n"
                 "Diagonal = variance | Off-diagonal = intraday temporal dependence",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_eda_covariance_surface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_eda_covariance_surface.png")


# =============================================================================
# 05 — LAG-1 TEMPORAL COVARIANCE
# =============================================================================

def plot_covariance_lag1(X, tau, pairs, output_dir=f'{FIGURES_ROOT}/EDA'):
    """
    Lag-1 autocorrelation: AC1_i(tau) = Corr(x_{it}(tau), x_{i,t-1}(tau)).
    Measures volatility persistence at each intraday interval.
    High AC1 at a given tau = strong day-to-day clustering at that hour.
    """
    N = len(pairs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for i, pair in enumerate(pairs):
        Y_i = X[i] - np.nanmean(X[i], axis=0, keepdims=True)
        ac1 = np.array([
            np.corrcoef(Y_i[1:, m], Y_i[:-1, m])[0, 1]
            if np.std(Y_i[:, m]) > 1e-10 else 0.0
            for m in range(Y_i.shape[1])
        ])
        ax.plot(tau, ac1, color=COLORS[i % len(COLORS)],
                linewidth=1.5, alpha=0.8, label=pair)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("tau [0=open -> 1=close]")
    ax.set_ylabel("Lag-1 Autocorrelation")
    ax.set_title("Lag-1 ACF per intraday interval tau\n"
                 "High values = volatility persistence at that hour")
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ac1_all = []
    for i in range(N):
        Y_i = X[i] - np.nanmean(X[i], axis=0, keepdims=True)
        ac1 = np.array([
            np.corrcoef(Y_i[1:, m], Y_i[:-1, m])[0, 1]
            if np.std(Y_i[:, m]) > 1e-10 else 0.0
            for m in range(Y_i.shape[1])
        ])
        ac1_all.append(ac1)

    bp = ax.boxplot(ac1_all, labels=pairs, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS[:N]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color='red', linewidth=1, linestyle='--')
    ax.set_xlabel("FX Pair")
    ax.set_ylabel("Lag-1 Autocorrelation")
    ax.set_title("Distribution of lag-1 ACF per pair\n"
                 "Median = average volatility persistence")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Lag-1 Temporal Covariance — Intraday Volatility Clustering",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_eda_covariance_lag1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_eda_covariance_lag1.png")


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
# 07 — CROSS-PAIR CORRELATION MATRIX
# =============================================================================

def plot_correlation_matrix(X, tau, pairs, days, output_dir=f'{FIGURES_ROOT}/EDA'):
    """
    Daily level: summarise x_{it}(tau) by its integral (mean over tau)
    and compute cross-pair correlation.
    Reveals the global dependence structure of the FX panel.
    """
    N = len(pairs)
    levels   = np.nanmean(X, axis=2)   # (N, T)
    corr_mat = np.corrcoef(levels)      # (N, N)

    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    dist = 1 - np.abs(corr_mat)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 1)
    dist = (dist + dist.T) / 2
    linkage = hierarchy.linkage(squareform(dist), method='ward')
    order   = hierarchy.leaves_list(linkage)

    corr_ord  = corr_mat[np.ix_(order, order)]
    pairs_ord = [pairs[o] for o in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    im = ax.imshow(corr_ord, cmap='RdYlGn', vmin=-0.5, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Correlation of daily vol levels')
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(pairs_ord, rotation=90, fontsize=8)
    ax.set_yticklabels(pairs_ord, fontsize=8)
    ax.set_title("Cross-pair correlation of daily vol levels\n"
                 "(hierarchical clustering order)")

    ax = axes[1]
    hierarchy.dendrogram(linkage, labels=pairs, orientation='right',
                         leaf_font_size=9, ax=ax,
                         color_threshold=0.7 * max(linkage[:, 2]))
    ax.set_title("Dendrogram — Groups of similar FX pairs")
    ax.set_xlabel("Distance (1 - |correlation|)")
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle("Cross-Pair Correlation Structure — Daily Volatility Levels",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_eda_correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_eda_correlation_matrix.png")


# =============================================================================
# 08 — VARIABILITY DECOMPOSITION
# =============================================================================

def plot_variability(X, tau, pairs, days, output_dir=f'{FIGURES_ROOT}/EDA'):
    """
    Decompose total variance of x_{it}(tau) into:
    - Intraday variance:  Var_tau(x_{it}(tau)) per day t  [shape variability]
    - Inter-day variance: Var_t(x_{it}(tau)) per tau       [level variability]

    In FDA: shape variability captures within-day profile changes;
    level variability captures day-to-day amplitude shifts.
    """
    N, T, M = X.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for i, pair in enumerate(pairs[:6]):
        idx = pairs.index(pair)
        var_intraday = np.nanvar(X[idx], axis=1)
        ax.plot(range(T), var_intraday, color=COLORS[idx % len(COLORS)],
                linewidth=0.8, alpha=0.7, label=pair)
    ax.set_xlabel("Day index t")
    ax.set_ylabel("Var_tau(x_{it}(tau))")
    ax.set_title("Intraday shape variability per day\n"
                 "Peak = day with atypical intraday profile")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, pair in enumerate(pairs[:6]):
        idx = pairs.index(pair)
        var_inter = np.nanvar(X[idx], axis=0)
        ax.plot(tau, var_inter, color=COLORS[idx % len(COLORS)],
                linewidth=1.5, alpha=0.8, label=pair)
    ax.set_xlabel("tau [0=open -> 1=close]")
    ax.set_ylabel("Var_t(x_{it}(tau))")
    ax.set_title("Inter-day level variability by tau\n"
                 "Peak = hour with most day-to-day vol variation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ratios_intra, ratios_inter = [], []
    for i in range(N):
        var_total = np.nanvar(X[i])
        var_intra = np.nanmean(np.nanvar(X[i], axis=1))
        var_inter = np.nanmean(np.nanvar(X[i], axis=0))
        ratios_intra.append(var_intra / (var_total + 1e-30))
        ratios_inter.append(var_inter / (var_total + 1e-30))

    x_pos = np.arange(N)
    ax.bar(x_pos, [r * 100 for r in ratios_intra],
           label='Intraday (shape)', color='steelblue', alpha=0.8)
    ax.bar(x_pos, [r * 100 for r in ratios_inter],
           bottom=[r * 100 for r in ratios_intra],
           label='Inter-day (level)', color='coral', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("% of total variance")
    ax.set_title("Variance decomposition per pair\n"
                 "Blue = shape variability | Red = level variability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    levels      = np.nanmean(X, axis=2)   # (N, T)
    levels_norm = (levels - levels.mean(axis=1, keepdims=True)) / \
                  (levels.std(axis=1, keepdims=True) + 1e-10)
    im = ax.imshow(levels_norm, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    plt.colorbar(im, ax=ax, label='Normalised daily vol level (z-score)')
    ax.set_yticks(range(N))
    ax.set_yticklabels(pairs, fontsize=7)
    ax.set_xlabel("Day index t")
    ax.set_title("Normalised daily volatility level (z-score)\n"
                 "Red = high-vol day | Blue = low-vol day")

    fig.suptitle("Variability Decomposition — Shape vs Level",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_eda_variability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_eda_variability.png")


# =============================================================================
# 09 — PERIODICITY AND TEMPORAL ACF
# =============================================================================

def plot_periodicity(X, tau, pairs, days, output_dir=f'{FIGURES_ROOT}/EDA', max_lag=60):
    """
    Temporal ACF of daily volatility levels per pair.
    Reveals volatility clustering, weekly cycles, and seasonal effects.
    Blue dashed = lag 5 (weekly effect)
    Green dashed = lag 22 (monthly effect)
    """
    N      = len(pairs)
    levels = np.nanmean(X, axis=2)   # (N, T)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        if i >= N:
            break
        series = levels[i]
        series = series[np.isfinite(series)]
        T_s    = len(series)

        lags = list(range(1, min(max_lag + 1, T_s // 2)))
        acf  = [np.corrcoef(series[lag:], series[:-lag])[0, 1] for lag in lags]
        conf = 1.96 / np.sqrt(T_s)

        ax.bar(lags, acf, color=COLORS[i % len(COLORS)], alpha=0.7, width=0.8)
        ax.axhline(conf,  color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(-conf, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        if 5  in lags: ax.axvline(5,  color='blue',  linewidth=1, linestyle=':', alpha=0.7)
        if 22 in lags: ax.axvline(22, color='green', linewidth=1, linestyle=':', alpha=0.7)

        ax.set_title(pair, fontsize=9, fontweight='bold')
        ax.set_xlabel("Lag (days)", fontsize=7)
        ax.set_ylim(-0.3, 0.7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Temporal ACF of Daily Volatility Levels\n"
                 "Red = 95% confidence bands | Blue = lag 5 (weekly) "
                 "| Green = lag 22 (monthly)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_eda_periodicity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_eda_periodicity.png")


# =============================================================================
# 10 — EXTREME DAYS
# =============================================================================

def plot_extremes(X, tau, pairs, days, output_dir=f'{FIGURES_ROOT}/EDA', n_extreme=5):
    """
    Identifies and displays the most and least volatile days.
    Validates that known crises (GFC, COVID, etc.) are correctly captured.
    """
    is_log  = X.min() < -1
    X_disp  = np.exp(X / 2) * 1e4 if is_log else X * 1e4
    ylabel  = "|r| x10^4"

    global_level = np.nanmean(X_disp, axis=(0, 2))   # (T,)
    idx_high     = np.argsort(global_level)[-n_extreme:][::-1]
    idx_low      = np.argsort(global_level)[:n_extreme]

    pair_var  = np.nanvar(X_disp, axis=(1, 2))
    best_pair = int(np.argmax(pair_var))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for ax, (indices, label, cmap) in zip(axes, [
        (idx_high, f'Most volatile days (top {n_extreme})',    'Reds'),
        (idx_low,  f'Calmest days (bottom {n_extreme})',       'Blues'),
    ]):
        colors = plt.cm.get_cmap(cmap)(np.linspace(0.4, 0.9, n_extreme))
        for k, t in enumerate(indices):
            date_str = str(days[t])[:10]
            level    = global_level[t]
            ax.plot(tau, X_disp[best_pair, t], color=colors[k], linewidth=2,
                    label=f"{date_str}  (avg={level:.3f})")

        mu = np.nanmean(X_disp[best_pair], axis=0)
        ax.plot(tau, mu, 'k--', linewidth=1.5, alpha=0.6, label='Overall mean')
        ax.set_title(f"{label} — {pairs[best_pair]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("tau [0=open -> 1=close]")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Extreme Days — Identifying Crisis Episodes and Calm Periods",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_eda_extremes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 10_eda_extremes.png")


# =============================================================================
# 11 — CROSS-PAIR COVARIANCE SURFACES
# =============================================================================

def plot_cross_covariance_surfaces(X, tau, pairs, output_dir=f'{FIGURES_ROOT}/EDA', n_pairs=9):
    """
    Cross-pair covariance surfaces CCov_{ij}(tau, tau').

    For two pairs i and j, the surface at (tau, tau') is:
        CCov_{ij}(tau, tau') = Corr(x_i(tau), x_j(tau'))

    This is the functional generalisation of cross-correlation.
    It captures not only whether two pairs move together (tau = tau'),
    but also whether a move in pair i at hour tau predicts a move
    in pair j at a different hour tau' (lead-lag structure).

    The plot shows the n_pairs most correlated pair combinations,
    selected automatically from the cross-correlation matrix.

    Colormap: white = 0, red = positive, blue = negative (same as plot 04).
    """
    from matplotlib.colors import TwoSlopeNorm

    N, T, M = X.shape

    # Center each series
    Y = X - np.nanmean(X, axis=1, keepdims=True)   # (N, T, M)

    # Find the most correlated pairs (upper triangle, excluding diagonal)
    corr_avg = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            cc = np.array([
                np.corrcoef(Y[i, :, m], Y[j, :, m])[0, 1]
                if (np.std(Y[i, :, m]) > 1e-10 and np.std(Y[j, :, m]) > 1e-10)
                else 0.0
                for m in range(M)
            ])
            corr_avg[i, j] = np.nanmean(cc)
            corr_avg[j, i] = corr_avg[i, j]

    # Select top-n_pairs most correlated combinations (upper triangle only)
    triu_idx = [(i, j) for i in range(N) for j in range(i + 1, N)]
    triu_idx.sort(key=lambda ij: -corr_avg[ij[0], ij[1]])
    selected_pairs = triu_idx[:n_pairs]

    # Compute all surfaces first to get a shared color scale
    surfaces = []
    for (pi, pj) in selected_pairs:
        # CCov(tau, tau') = Corr(x_i(tau), x_j(tau'))
        # = (Y_i^T @ Y_j) / (T-1), then normalise by per-column stds
        C_ij  = (Y[pi].T @ Y[pj]) / (T - 1)   # (M, M)
        std_i = np.nanstd(Y[pi], axis=0)        # (M,)
        std_j = np.nanstd(Y[pj], axis=0)        # (M,)
        denom = np.outer(std_i, std_j)
        with np.errstate(divide='ignore', invalid='ignore'):
            Corr_ij = np.where(denom > 1e-12, C_ij / denom, 0.0)
        surfaces.append(np.clip(Corr_ij, -1, 1))

    # Shared symmetric color scale — white = 0
    vmax = max(np.abs(s).max() for s in surfaces)
    vmax = min(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = 'RdBu_r'

    ncols = 3
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.8 * nrows))
    axes = axes.flatten()

    for k, ((pi, pj), surf) in enumerate(zip(selected_pairs, surfaces)):
        ax  = axes[k]
        im  = ax.imshow(surf, origin='lower', cmap=cmap, norm=norm,
                        extent=[0, 1, 0, 1], aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Cross-corr', fontsize=8)

        rho_mean = corr_avg[pi, pj]
        ax.set_title(f"{pairs[pi]}  ×  {pairs[pj]}\n"
                     f"mean corr = {rho_mean:.2f}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel("tau'  [pair j]", fontsize=8)
        ax.set_ylabel("tau   [pair i]", fontsize=8)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.tick_params(labelsize=7)

        # Mark the diagonal (tau = tau') in black
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)

    for j in range(len(selected_pairs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Cross-Pair Covariance Surfaces  CCov_{{ij}}(tau, tau')\n"
        f"= Corr(x_i(tau), x_j(tau'))  —  top {n_pairs} most correlated pairs\n"
        "Red = positive | Blue = negative | White = zero\n"
        "Diagonal (dashed) = simultaneous correlation (tau = tau')\n"
        "Off-diagonal = lead-lag structure across pairs",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_eda_cross_covariance_surfaces.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 11_eda_cross_covariance_surfaces.png")


# =============================================================================
# 12 — CROSS-COVARIANCE SURFACES PER PAIR (single PNG)
# =============================================================================

def plot_cross_covariance_per_pair(X, tau, pairs, output_dir=f'{FIGURES_ROOT}/EDA'):
    """
    For each pair i, plot the average cross-covariance surface with all
    other pairs j in a single PNG — same layout and colormap as figure 04.

        CCov_i(tau, tau') = mean_{j != i}  Corr(x_i(tau), x_j(tau'))

    This summarises how pair i co-moves with the rest of the FX panel
    across all intraday intervals simultaneously.

    Reading the surface:
    - Diagonal (tau = tau'): average simultaneous cross-correlation
    - Above diagonal: other pairs j lead pair i
    - Below diagonal: pair i leads other pairs j
    - Off-diagonal mass = cross-pair lead-lag structure

    Same colormap as figure 04: white=0, red=positive, blue=negative.
    Shared color scale across all pairs for direct comparability.
    """
    from matplotlib.colors import TwoSlopeNorm

    N, T, M = X.shape
    Y = X - np.nanmean(X, axis=1, keepdims=True)   # (N, T, M) centred

    # Compute average cross-covariance surface for each pair i
    avg_surfaces = []
    for i in range(N):
        stack = []
        for j in range(N):
            if j == i:
                continue
            C_ij  = (Y[i].T @ Y[j]) / (T - 1)
            std_i = np.nanstd(Y[i], axis=0)
            std_j = np.nanstd(Y[j], axis=0)
            denom = np.outer(std_i, std_j)
            with np.errstate(divide='ignore', invalid='ignore'):
                Corr_ij = np.where(denom > 1e-12, C_ij / denom, 0.0)
            stack.append(np.clip(Corr_ij, -1, 1))
        avg_surfaces.append(np.mean(stack, axis=0))   # (M, M)

    # Shared symmetric color scale
    vmax = max(np.abs(s).max() for s in avg_surfaces)
    vmax = min(vmax, 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = 'RdBu_r'

    ncols = 4
    nrows = (N + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.2 * nrows))
    axes = axes.flatten()

    for i, pair in enumerate(pairs):
        ax   = axes[i]
        surf = avg_surfaces[i]
        im   = ax.imshow(surf, origin='lower', cmap=cmap, norm=norm,
                         extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)

        rho_diag = np.nanmean(np.diag(surf))
        ax.set_title(f"{pair}\nmean diag = {rho_diag:.2f}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel("tau'  [other pairs]", fontsize=7)
        ax.set_ylabel("tau   [this pair]",   fontsize=7)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=7)
        # diagonal tau = tau'
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Average Cross-Covariance Surfaces — each pair vs all others\n"
        r"$\overline{\mathrm{CCov}}_i(\tau,\tau') = "
        r"\frac{1}{N-1}\sum_{j\neq i}\,\mathrm{Corr}(x_i(\tau),\,x_j(\tau'))$"
        "\nRed = positive | Blue = negative | White = zero   |   "
        "Dashed diagonal = simultaneous (tau = tau')",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/12_eda_cross_cov_surfaces_per_pair.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 12_eda_cross_cov_surfaces_per_pair.png")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_eda(vol_matrices: dict,
            transform: str  = 'log',
            output_dir: str = None,
            n_sample: int   = 30,
            max_lag: int    = 60) -> dict:
    """
    Run the complete FDA exploratory data analysis.

    Parameters
    ----------
    vol_matrices : dict {str: DataFrame(days x minutes)}
                   values = |r_t| (output of intraday_volatility_matrix)
    transform    : 'log'  -> log(r_t^2 + eps)  [recommended]
                   'abs'  -> |r_t| raw
                   'none' -> data as-is
    output_dir   : output directory for figures
    n_sample     : number of individual trajectories to display
    max_lag      : maximum number of lags for the ACF

    Returns
    -------
    dict with X (panel), tau, days, pairs
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FDA EXPLORATORY DATA ANALYSIS")
    print(f"  Transform : {transform}")
    print(f"  Pairs     : {len(vol_matrices)}")
    print(f"{'='*60}\n")

    # [0] Missing values (on raw data, before transform)
    print("[0] Missing value analysis (raw data)...")
    plot_missing_values(vol_matrices, output_dir)

    # Apply transform
    print("[0] Applying transform...")
    vol_tr = _apply_transform(vol_matrices, transform)

    # Build panel
    print("[0] Building panel...")
    X, tau, days, pairs = _build_panel(vol_tr)
    N, T, M = X.shape
    print(f"    Panel shape : ({N}, {T}, {M})")
    print(f"    Period      : {str(days[0])[:10]} -> {str(days[-1])[:10]}")
    print(f"    Global min  : {X.min():.3f}")
    print(f"    Global max  : {X.max():.3f}")
    print(f"    Global std  : {X.std():.3f}\n")

    print("[1] Mean curves...")
    plot_mean_curves(X, tau, pairs, transform, output_dir)

    print("[2] Sample trajectories...")
    plot_sample_curves(X, tau, pairs, days, transform, output_dir, n_sample)

    print("[3] Marginal distribution...")
    plot_distribution(X, tau, pairs, transform, output_dir)

    print("[4] Intraday covariance surface...")
    plot_covariance_surface(X, tau, pairs, output_dir)

    print("[5] Lag-1 temporal covariance...")
    plot_covariance_lag1(X, tau, pairs, output_dir)

    print("[6] Cross-covariance between pairs...")
    plot_cross_covariance(X, tau, pairs, output_dir)

    print("[7] Cross-pair correlation matrix...")
    plot_correlation_matrix(X, tau, pairs, days, output_dir)

    print("[8] Variability decomposition...")
    plot_variability(X, tau, pairs, days, output_dir)

    print("[9] Temporal ACF (periodicity)...")
    plot_periodicity(X, tau, pairs, days, output_dir, max_lag)

    print("[10] Extreme days...")
    plot_extremes(X, tau, pairs, days, output_dir)

    print("[11] Cross-pair covariance surfaces...")
    plot_cross_covariance_surfaces(X, tau, pairs, output_dir)

    print("[12] Cross-covariance surfaces per pair...")
    plot_cross_covariance_per_pair(X, tau, pairs, output_dir)

    print(f"\n{'='*60}")
    print(f"  EDA complete — 12 figures saved to {output_dir}/")
    print(f"{'='*60}\n")

    return {'X': X, 'tau': tau, 'days': days, 'pairs': pairs}


# =============================================================================
# STANDALONE DEMO
# =============================================================================

if __name__ == "__main__":
    import datetime

    print("Demo mode — generating simulated matrices")
    np.random.seed(42)

    T_DAYS = 500
    M      = 78
    PAIRS  = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'AUD/JPY',
              'EUR/GBP', 'USD/CHF', 'EUR/CHF', 'NZD/USD',
              'USD/CAD', 'CAD/JPY', 'GBP/JPY', 'EUR/AUD']

    dates  = pd.bdate_range('2015-01-01', periods=T_DAYS)
    cols   = pd.date_range('2000-01-01 00:00', periods=M, freq='5min').time
    tau_g  = np.linspace(0, 1, M)
    u_shp  = 1.5*np.exp(-8*tau_g**2) + 1.2*np.exp(-8*(tau_g-1)**2) + 0.4

    f1 = np.zeros(T_DAYS); f2 = np.zeros(T_DAYS)
    for t in range(1, T_DAYS):
        f1[t] = 0.85*f1[t-1] + np.random.normal(0, 0.5)
        f2[t] = 0.60*f2[t-1] + np.random.normal(0, 0.7)

    vol_matrices = {}
    for pair in PAIRS:
        b1 = np.random.uniform(0.5, 1.5) * u_shp / np.linalg.norm(u_shp)
        b2 = np.random.uniform(0.2, 0.8) * np.sin(np.pi * tau_g)
        data = np.zeros((T_DAYS, M))
        for t in range(T_DAYS):
            level  = np.abs(b1 * f1[t] + b2 * f2[t]) + 0.3
            data[t] = np.abs(np.random.normal(0, level * 1e-4, M))
        # Inject realistic NaN patterns
        data[np.random.random(data.shape) < 0.05] = np.nan  # 5% random NaN
        data[:, 65:] = np.nan  # night-like gap at end
        vol_matrices[pair] = pd.DataFrame(data, index=dates, columns=cols)


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

    results = run_eda(vol_matrices, transform='log')


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
