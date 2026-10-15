"""
Forecasting module for the Functional Factor Model.

Forecasts the intraday log-volatility curves by splitting them into the common
component (scalar factor scores) and the idiosyncratic functional component,
then recombining both on the original scale.

Step 1 — Factor scores û_t
    1. Stationarity: ADF, KPSS
    2. Temporal dependence: ACF/PACF, Ljung-Box, GPH estimate of the memory
       parameter d (fractional differencing / ARFIMA if d ∈ (0, 0.5))
    3. Models: random walk, AR(p), ARIMA(p,d,q), ARFIMA (orders by BIC),
       GARCH(p,q) (ARCH test + fit), HAR (daily / weekly / monthly components)

Step 2 — Idiosyncratic curves ξ̂_it(τ)
    1. Functional Ljung-Box (white-noise) test
    2. If not white noise: FAR(p) via FPCA + VAR on the scores, functional
       HAR, operator C_h, or random walk

Step 3 — Reconstruction and evaluation
    1. Forecast curve:  X̂_{i,t+h}(τ) = μ̂_i(τ) + σ · ( Σ_k b̂_ik(τ) û_{k,t+h} + ξ̂_{i,t+h}(τ) )
    2. Functional error metrics:
         MSFE = ∫ (X̂_{i,t+h}(τ) − X_{i,t+h}(τ))² dτ
         MAFE = ∫ |X̂_{i,t+h}(τ) − X_{i,t+h}(τ)| dτ
    3. Model comparison against a RW × RW benchmark, per pair and per horizon
"""

#====================================================================================
# IMPORTS 
#====================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from functools import partial
import contextlib, io

from ffm_fx_real import run_ffm

from statsmodels.tsa.api import VAR

from scipy import stats 

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
#from arch.unitroot import DFGLS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.arima.model import ARFIMA
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from sklearn.decomposition import PCA
import statsmodels.api as sm
from paths import FIGURES_ROOT
#====================================================================================
# Step 0: Input data
#====================================================================================

# results['U_hat'] : T x K 
# results['Xi_hat'] : T x N x M

#====================================================================================
# Step 1: Factors
#====================================================================================
# 1. Stationarity -------------------------------------------------------------------

# 1.1 Augmented Dickey-Fuller (ADF) test:
# tests whether a time series is stationary
# or has a unit root (non-stationary).

def test_adf(series, max_lags=20):
    """
    Stationarity check with the augmented Dickey-Fuller (ADF) test.
    Parameters: 
        factors: T x K array of factor time series
        max_lags: maximum number of lags to consider
    """
    series = np.asarray(series).squeeze()
    
    result = adfuller(series, maxlag=max_lags, autolag='BIC',
                      regression='c', store=False)
    
    adf_stat  = result[0]
    p_value   = float(result[1])
    n_lags    = result[2]
    print("TEST ADF")
    print(f"ADF stat            : {adf_stat:.4f}")
    print(f"p-value             : {p_value:.4f}")
    print(f"Number of lags used : {n_lags}")
    
    print(f"\nConclusion :")
    if p_value < 0.01:
        print(f"  Rejection of H_0 at 1% → STATIONARITY")
    elif p_value < 0.05:
        print(f"  Rejection of H_0 at 5% → STATIONARITY")
    elif p_value < 0.10:
        print(f"  Rejection of H_0 at 10% → MARGINAL STATIONARITY")
    else:
        print(f"  No-rejection H_0 → NO STATIONARITY (unit root)")

    return p_value


# 1.2 KPSS test (Kwiatkowski-Phillips-Schmidt-Shin):
# null hypothesis of stationarity vs
# the alternative of non-stationarity (trend stationary).
def test_kpss(series):

    """
    Stationarity check with the KPSS test.
    Parameters: 
        factors: T x K array of factor time series
        max_lags: maximum number of lags to consider
    """
    series = np.asarray(series).squeeze()
    
    stat, p_value, n_lags, crit_vals = kpss(series, regression='c', nlags='auto')
    
    p_value = float(p_value)
    print("KPSS test")
    print(f"KPSS stat           : {stat:.4f}")
    print(f"p-value             : {p_value:.4f}")
    print(f"Number of lags used : {n_lags}")

    print(f"\nConclusion :")
    if p_value < 0.01:
        print(f"  Rejection of H_0 at 1% → NO STATIONARITY (trend stationary)")
    elif p_value < 0.05:
        print(f"  Rejection of H_0 at 5% → NO STATIONARITY (trend stationary)")
    elif p_value < 0.10:
        print(f"  Rejection of H_0 at 10% → NO STATIONARITY (trend stationary)")
    else:
        print(f"  No-rejection H_0 → STATIONARITY")
    
    return p_value

#1.4 
def test_stationarity (factors, max_lags=20):
    """
    Test the stationarity of a factor time series with the ADF and KPSS tests.
    Parameters: 
        factors: T x K array of factor time series
        max_lags: maximum number of lags to consider
    """
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    
    T, K = factors.shape
    
    for i in range(K):
        series = factors[:, i] 
        
        print(f"{'='*50}")
        print(f"Factor {i+1}")
        print(f"{'='*50}\n")
                
        print(f"Test ADF :")
        p_adf = test_adf(series)
        
        print(f"\nTest KPSS :")
        p_kpss = test_kpss(series)
        
        if p_adf < 0.05 and p_kpss > 0.05:
            print(f"\nConclusion : STATIONARY")
        elif p_adf > 0.05 and p_kpss < 0.05:
            print(f"\nConclusion: NON-STATIONARY (unit root)")
        else:
            print(f"\nConclusion: Contradictory results.")

#2. Time dependence structure: ----------------------------------------------------

#2.1 ACF & PACF
def plot_acf_pacf(series, lags_short=40, lags_long=200):
    from statsmodels.tsa.stattools import acf, pacf
    import os

    series = np.asarray(series)
    if series.ndim == 1:
        series = series[:, np.newaxis]
    T, r = series.shape

    os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)

    BAR_COLOR  = '#2C7BB6'
    BAND_COLOR = '#D7191C'

    def _style_ax(ax, vals, lags, ylabel, title):
        lags_arr   = np.arange(len(vals))
        conf       = 1.96 / np.sqrt(T)

        # Bars
        colors = [BAR_COLOR if abs(v) > conf else '#aaaaaa' for v in vals[1:]]
        ax.bar(lags_arr[1:], vals[1:], color=colors, alpha=0.75, width=0.8, zorder=2)

        # Confidence band
        ax.axhline(0,      color='black',     linewidth=0.8, zorder=3)
        ax.axhline(+conf,  color=BAND_COLOR,  linewidth=0.9, linestyle='--', alpha=0.8)
        ax.axhline(-conf,  color=BAND_COLOR,  linewidth=0.9, linestyle='--', alpha=0.8)
        ax.fill_between([-1, lags + 1], -conf, +conf,
                        color=BAND_COLOR, alpha=0.06, zorder=1)

        ax.set_xlim(-0.5, lags + 0.5)
        ax.set_ylim(min(-0.15, vals[1:].min() - 0.05),
                    max( 0.15, vals[1:].max() + 0.05))
        ax.set_xlabel('Lag (days)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y', zorder=0)
        ax.tick_params(labelsize=8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    for l in range(r):
        s = series[:, l]
        fig, axes = plt.subplots(2, 2, figsize=(14, 7))

        for col, lags in enumerate([lags_short, lags_long]):
            term = f'short  (lags = {lags})' if col == 0 else f'long  (lags = {lags})'

            acf_vals  = acf( s, nlags=lags, fft=True)
            pacf_vals = pacf(s, nlags=lags, method='ywadjusted')

            _style_ax(axes[0, col], acf_vals,  lags,
                      'Autocorrelation',         f'ACF — {term}')
            _style_ax(axes[1, col], pacf_vals, lags,
                      'Partial autocorrelation', f'PACF — {term}')

        fig.suptitle(
            f'Factor {l + 1}  —  ACF & PACF of $\\hat{{u}}_{{{l + 1},t}}$\n'
            f'Dashed lines: 95% confidence band  ($\\pm 1.96/\\sqrt{{T}}$, $T={T}$)',
            fontsize=12, fontweight='bold')
        plt.tight_layout()
        fname = f'{FIGURES_ROOT}/FORECAST/acf_pacf_factor{l + 1}.png'
        plt.savefig(fname, dpi=160, bbox_inches='tight')
        plt.show()
        print(f"  ✓ {fname}")

# 2.2 Ljung-Box test
# tests for autocorrelation up to order r
# H0: no autocorrelation of orders 1 to r
def test_ljung_box(series, lags=[5,10,20,30]):
    """
    Parameters: 
    lags : list of lags at which the statistic is computed
    """
    series = np.asarray(series).squeeze()

    print(f"\nTesting Ljung Box for factor:")
    result = acorr_ljungbox(series,lags=lags,model_df=0,return_df=True)
    print(f"{'Lag':<10} {'Stat Q':<15} {'p-value':<15} {'Conclusion'}")
    for lag, row in result.iterrows():
        stat    = row['lb_stat']
        pval    = row['lb_pvalue']
        conclusion = "✓ White noise" if pval > 0.05 else "✗ Autocorrelation"
        print(f"{lag:<10} {stat:<15.4f} {pval:<15.4f} {conclusion}")
        
    any_rejected = (result['lb_pvalue'] < 0.05).any()
    print(f"\nOverall conclusion:")
    if any_rejected:
        print(f"H0 rejected → residual temporal structure detected")
    else:
        print(f"H0 non rejected → compatible with white noise")

    return result
    
# 2.3 GPH test — estimates d; ARFIMA is justified if d ∈ (0, 0.5) 

def gph_test(series, bandwidth=None):
    
    series = np.asarray(series).squeeze()
    T      = len(series)
    
    if bandwidth is None:
        bandwidth = int(T ** 0.5)
    
    series = series - series.mean()
    
    # Periodogram — alternative normalisation
    j           = np.arange(1, bandwidth + 1)
    freq        = 2 * np.pi * j / T
    ft          = np.fft.rfft(series)
    periodogram = (np.abs(ft[1:bandwidth+1]) ** 2) / T   # ← without 2π
    
    ln_freq  = np.log(freq)
    ln_perio = np.log(periodogram)
    
    # Regression — centring ln_freq improves numerical stability
    ln_freq_centered = ln_freq - ln_freq.mean()
    X    = np.column_stack([np.ones(bandwidth), ln_freq_centered])
    beta = np.linalg.lstsq(X, ln_perio, rcond=None)[0]
    
    d_hat = -beta[1] / 2
    
    residuals = ln_perio - X @ beta
    s2        = np.sum(residuals**2) / (bandwidth - 2)
    se        = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]) / 2
    
    t_stat  = d_hat / se
    p_value = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(t_stat)))
    
    print(f"d̂          : {d_hat:.4f}")
    print(f"95% CI     : [{d_hat - 1.96*se:.4f}, {d_hat + 1.96*se:.4f}]")
    print(f"t-stat     : {t_stat:.4f}")
    print(f"p-value    : {p_value:.4f}")

    print("Conclusion:")
    if p_value >= 0.05:
        print("→ d not significant — ARMA is enough")
    elif d_hat < 0:
        print("→ Negative memory — check the series")
    elif d_hat < 0.1:
        print("→ Very weak long memory — AR/HAR are enough")
    elif d_hat < 0.5:
        print("→ Stationary long memory — ARFIMA(p,d,q) applicable")
    elif d_hat < 1.0:
        print("→ Non-stationary long memory — Wiener-type model recommended")
    else:
        print("→ d ≥ 1.0 — difference the series")
    
    return {'d_hat': d_hat, 'se': se, 't_stat': t_stat, 'p_value': p_value}
    
def fractional_diff(series, d, threshold=1e-4):
    """
    Fractional differencing of order d.
    Jensen & Nielsen (2014) — rolling-window implementation.
    
    Parameters
    ----------
    series    : np.array 1D
    d         : float — differencing order (e.g. 0.639)
    threshold : float — weight truncation (numerical stability)
    """
    # Weights w_k = -Γ(k-d) / (Γ(-d)·Γ(k+1))
    weights = [1.0]
    k = 1
    while abs(weights[-1]) > threshold:
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
        k += 1
    weights = np.array(weights)
    
    T      = len(series)
    n_w    = len(weights)
    result = np.full(T, np.nan)
    
    for t in range(n_w - 1, T):
        result[t] = np.dot(weights, series[t:t - n_w:-1] if t >= n_w 
                           else series[t::-1])
    
    # Drop the leading NaN
    result = result[~np.isnan(result)]
    
    return result

#3 Model estimation ----------------------------------------------------------------

#3.1 Random walk
# 3.2 Parametric models:
    # 3.2.1 AR(p) (p selected by BIC)

def select_p_ar(series, max_p=20):
    """
    Select p 
    """
    series = np.asarray(series).squeeze()
    bic = {}
    for p in range(1, max_p +1):
        model = AutoReg(series, lags=p, trend='c')
        result = model.fit()
        bic[p] = result.bic

    best_p   = min(bic, key=bic.get)
    best_bic = bic[best_p]
    print(f"Optimal p : {best_p}  (BIC = {best_bic:.2f})")
    return best_p

def fit_ar(series, p_max=20):

    series = np.asarray(series).squeeze()
    p = select_p_ar(series, p_max)

    model  = AutoReg(series, lags=p, trend='c')
    result = model.fit()
    
    print(f"Constant  : {result.params[0]:.4f}")
    for i in range(1, p + 1):
        print(f"φ_{i}       : {result.params[i]:.4f}  (p={result.pvalues[i]:.3f})")
    print(f"\nAIC : {result.aic:.2f}")
    print(f"BIC : {result.bic:.2f}")
    
    return result, p

def forecast_ar(series, h, p_max, n_display=200, plot=True):
    print(f"{'─'*40}")
    print('Forecast AR(p): ')
    print(f"{'─'*40}")
    
    series = np.asarray(series).squeeze()
    T      = len(series)
    
    result, p = fit_ar(series, p_max)
    forecasts = result.forecast(steps=h)
    
    se_forecast = np.array([
        np.sqrt(result.sigma2) * np.sqrt(i + 1)
        for i in range(h)
    ])

    ci_low  = forecasts - 1.96 * se_forecast
    ci_high = forecasts + 1.96 * se_forecast
    
    t_insample = np.arange(T)
    t_forecast = np.arange(T, T + h)
    
    if plot:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(t_insample[-n_display:], series[-n_display:],
                color='steelblue', linewidth=0.8, label='Observed')
        ax.plot([t_insample[-1], t_forecast[0]],
                [series[-1], forecasts[0]],
                color='red', linewidth=1.5, linestyle='--')
        ax.plot(t_forecast, forecasts,
                color='red', linewidth=1.5, linestyle='--',
                marker='o', markersize=4 if h <= 30 else 2,
                label=f'Forecast AR({p})')
        



        ax.fill_between(t_forecast, ci_low, ci_high,
                        alpha=0.2, color='red', label='IC 95%')
        ax.axvline(T - 1, color='black', linestyle=':',
                   linewidth=1, label='Start forecast')
        ax.set_title(f"AR({p}) — Series and forecast (h={h})")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
        plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_ar_p{p}_h{h}.png', dpi=160, bbox_inches='tight')
        plt.show()
    
    return result, forecasts, ci_low, ci_high

# 3.2.2 ARIMA(p,d,q) (orders selected by BIC), to be used on the differenced factors

def select_param_arima(series,d=5, p_max=5, q_max=3):
    """
    d is determined by the GPH test.
    Select the optimal (p, 0, q) by BIC.
    """
    series = np.asarray(series).squeeze()
    best_bic = np.inf
    best_order = (1, 0, 0)

    for p in range(0, p_max + 1):
        for q in range(0, q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(series, order=(p, d, q), trend='c').fit()
                marker = ' ← best' if res.bic < best_bic else ''
                print(f"ARIMA({p},{d},{q})     {res.aic:>10.2f} {res.bic:>10.2f}{marker}")
                if res.bic < best_bic:
                    best_bic   = res.bic
                    best_order = (p, d, q)
            except Exception:
                continue

    print(f"Optimal order : ARIMA{best_order}  (BIC = {best_bic:.2f})")
    return best_order

def fit_arima(series,d=5, p_max =5, q_max = 3 ):
    series = np.asarray(series).squeeze()
    params = select_param_arima(series,d,  p_max, q_max)
    result = ARIMA(series, order = params, trend='c').fit()

    print(f"\nARIMA{params} — Results")
    print(f"{'─'*55}")
    print(f"{'Parameter':<18} {'Coef':>9} {'SE':>9} {'t-stat':>9} {'p-val':>9}  Sig")
    print(f"{'─'*55}")

    for name, coef, se, pval in zip(
        result.param_names, result.params,
        result.bse,         result.pvalues
    ):
        sig = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else ''
        print(f"{name:<18} {coef:>9.4f} {se:>9.4f} {coef/se:>9.3f} {pval:>9.4f}  {sig}")

    print(f"{'─'*55}")
    print(f"AIC : {result.aic:.2f}   |   BIC : {result.bic:.2f}")

    lb = acorr_ljungbox(result.resid, lags=[10, 20], return_df=True)
    print(f"\nLjung-Box on residuals:")
    for lag, row in lb.iterrows():
        ok = '✓ OK' if row['lb_pvalue'] > 0.05 else '✗ residual autocorrelation'
        print(f"  lag {lag:2d} : p = {row['lb_pvalue']:.4f}  {ok}")

    return result, params

def forecast_arima(series,h=1,d=0.5,  p_max=5,q_max=3, n_display=100,  plot=True):
    print(f"{'─'*40}")
    print('Forecast ARIMA(p,d,q): ')
    print(f"{'─'*40}")

    series = np.asarray(series).squeeze()
    T= len(series)

    result, order = fit_arima(series,d, p_max=p_max, q_max=q_max)

    fc_obj    = result.get_forecast(steps=h)
    forecasts = np.asarray(fc_obj.predicted_mean)
    ci        = fc_obj.conf_int(alpha=0.05)
    ci_low    = ci[:, 0]
    ci_high   = ci[:, 1]
    
    if plot: 
        t_in  = np.arange(T)
        t_out = np.arange(T, T + h)

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(t_in[-n_display:], series[-n_display:],
            color='steelblue', linewidth=0.8, label='Observed')

        ax.plot([t_in[-1], t_out[0]],
            [series[-1], forecasts[0]],
            color='crimson', linewidth=1.5, linestyle='--')
        
        ax.plot(t_out, forecasts,
            color='crimson', linewidth=2, linestyle='--',
            marker='o', markersize=4 if h <= 30 else 2,
            label=f'ARIMA{order} forecast (h={h})')



        ax.fill_between(t_out, ci_low, ci_high,
                    alpha=0.2, color='crimson', label='IC 95%')

        ax.axvline(T - 1, color='black', linestyle=':',
               linewidth=1.2)

        ax.set_title(f"ARIMA{order} — Forecast h={h} step")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
        plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_arima_h{h}.png', dpi=160, bbox_inches='tight')
        plt.show()

    return result, forecasts, ci_low, ci_high

# 3.2.3 ARFIMA(p,d,q) (orders selected by BIC)
# equivalent to fitting an ARIMA on the fractionally differenced series

#3.3 GARCH 

def test_arch(series):
    series = np.asarray(series).squeeze()
    resid  = series - series.mean()
    lm_stat, lm_p,_ , _ = het_arch(resid, nlags=10)
    print(f"Test ARCH:")
    print(f"LM stat : {lm_stat:.4f}")
    print(f"p-value : {lm_p:.4f}")

    if lm_p < 0.05:
        print("✓ ARCH effects decteted → GARCH justified")
    else:
        print("✗ No ARCH detected → GARCH useless")

def fit_garch(series, mean_model= 'AR', p=1, q=0):
    """
    Fit GARCH (1,1) + test Ljung Box 
    """
    series = np.asarray(series).squeeze()
    scale  = 100
    s      = series * scale

    if mean_model == 'AR': 
        if p==0: raise ValueError("Error: p=0 ")
        model  = arch_model(s,mean='AR',lags=p,vol='GARCH',p=1, q=1,dist='normal')
        mean_str = f"AR({p})"

    elif mean_model == 'ARMA':
        if p==0 and q==0: raise ValueError("Error: p=0 & q=0 ")
        model  = arch_model(s,mean='ARMAX',lags=p,vol='GARCH',p=1, q=1,dist='normal')
        mean_str = f"ARMA({p}, {q})"

    elif mean_model == 'Constant':
        model  = arch_model(s,mean='Constant',lags=p,vol='GARCH',p=1, q=1,dist='normal')
        mean_str = "Constant"

    result = model.fit(disp='off')

    print(f"\n{mean_str}-GARCH(1,1) — Results")
    print(f"{'─'*55}")
    print(result.summary().tables[1].as_text())

    alpha = result.params.get('alpha[1]', 0)
    beta  = result.params.get('beta[1]',  0)
    print(f"\n GARCH persistance(α+β) : {alpha+beta:.4f}")
    if alpha + beta > 0.95:
        print("→ Highly persistent variance (long-lasting regimes)")
    elif alpha + beta > 0.85:
        print("→ Moderate persistence")
    else:
        print("→ Fast reversion to the mean variance")

    std_resid = result.std_resid
    test_ljung_box(std_resid)

    return result

def forecast_garch(series,mean_model = 'AR',  p=1, q=0,  h=5, plot=True, n_display=100, i=1):
    series = np.asarray(series).squeeze()
    T      = len(series)
    scale =100

    result = fit_garch(series, mean_model, p, q)

    fc      = result.forecast(horizon=h, reindex=False)
    f_mean  = fc.mean.values[-1, :]     / scale
    f_var   = fc.variance.values[-1, :] / scale**2
    f_std   = np.sqrt(f_var)
    ci_low  = f_mean - 1.96 * f_std
    ci_high = f_mean + 1.96 * f_std

    cond_vol = result.conditional_volatility / scale # historical conditional volatility

    if plot: 
        t_in  = np.arange(T)
        t_out = np.arange(T, T + h)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Axis 1 — series + forecast + dynamic CI
        axes[0].plot(t_in[-n_display:], series[-n_display:],
                 color='steelblue', linewidth=0.8,
                 label='Observed')
        axes[0].plot([t_in[-1], t_out[0]],
                 [series[-1], f_mean[0]],
                 color='purple', linewidth=1.5, linestyle='--')
        axes[0].plot(t_out, f_mean,
                 color='purple', linewidth=2, linestyle='--',
                 marker='o', markersize=5,
                 label=f'AR({p})-GARCH(1,1) forecast')
        axes[0].fill_between(t_out, ci_low, ci_high,
                         alpha=0.25, color='purple',
                         label='IC 95% dynamique')
        axes[0].axvline(T - 1, color='black', linestyle=':',
                    linewidth=1, label='Forecast start')
        axes[0].set_title(f"AR({p})-GARCH(1,1) — Forecast h={h}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Axis 2 — historical conditional volatility
        axes[1].plot(t_in[-n_display:], cond_vol[-n_display:],
                 color='darkorange', linewidth=0.8)
        axes[1].set_title(f"Conditional volatility σ_t — Factor {i}")
        axes[1].set_xlabel("Time")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
        plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_garch_p{p}_h{h}_factor{i}.png', dpi=160, bbox_inches='tight')
        plt.show()

    # ── Numerical summary ─────────────────────────────────
    
    if mean_model =='AR': print(f"\nForecast AR({p})-GARCH(1,1) — h={h}")
    if mean_model =='ARMA': print(f"\nForecast ARMA({p},{q})-GARCH(1,1) — h={h}")
    print(f"{'─'*50}")
    print(f"{'Horizon':<10} {'Mean':>10} {'σ_t':>10} "
          f"{'IC lower':>10} {'IC upper':>10}")
    print(f"{'─'*50}")
    for i in range(h):
        print(f"{i+1:<10} {f_mean[i]:>10.4f} {f_std[i]:>10.4f} "
              f"{ci_low[i]:>10.4f} {ci_high[i]:>10.4f}")

    return result, f_mean, f_std, (ci_low, ci_high)

# 3.4 Semi-parametric models:
    # - HAR (Heterogeneous Autoregressive): HAR(3) with daily, weekly and monthly components

def har_regression(series, summary =False):
    series = series.squeeze()
    factor = pd.DataFrame(series)
    factor.columns = ['factor']

    df = pd.DataFrame({'daily': series})
    df['weekly']= factor['factor'].rolling(5).mean()
    df['monthly']= factor['factor'].rolling(22).mean()

    df=df.dropna()

    y = df['daily'].shift(-1).dropna()
    X = df[['daily', 'weekly', 'monthly']].iloc[:-1]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    if summary: 
        print(model.summary())


    print("\nSignificance (5% level):")
    for var, pval in model.pvalues.items():
        if pval <0.05: print(f"Variable '{var}': significant ")
        else: print(f"Variable '{var}':  not significant ")
        #print(var)
        #print(pval)

    return model

def forecast_har_direct_h(series, h = 1, summary=False):
    """
    Forecast HAR direct for different horizons 
    Different model for each h 
    
    """
    series    = np.asarray(series).squeeze()

    df = pd.DataFrame({'u': series})

    df['daily'] = df['u']
    df['weekly'] = df['u'].rolling(5).mean()
    df['monthly'] = df['u'].rolling(22).mean()

    df[f'y_h{h}'] = df['u'].shift(-h)

    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.dropna()

    X = df[['daily', 'weekly', 'monthly']]
    y = df[f'y_h{h}']

    X_train = sm.add_constant(X)
    model   = sm.OLS(y, X_train).fit()


    if summary:
        print(model.summary())

    last     = df[['daily', 'weekly', 'monthly']].iloc[-1]
    X_last   = np.array([1.0,
                         last['daily'],
                         last['weekly'],
                         last['monthly']]).reshape(1, -1) 

    pred = model.predict(X_last)[0]

    sigma2 = np.sum(model.resid**2) / (len(y) - 4)
    se_h   = np.sqrt(sigma2)
    ci_low  = pred - 1.96 * se_h
    ci_high = pred + 1.96 * se_h

    return model, pred, (ci_low, ci_high)

def plot_forecast_har(series, horizon = [1,2,3], n_display=100,  summary=False, plot= False):
    series = np.asarray(series).squeeze()
    T      = len(series)

    fc_har  = []
    ci_low  = []
    ci_high = []

    for h in horizon:
        model, pred, ci = forecast_har_direct_h(series, h=h, summary=summary)
        print(f"h={h:<3} : f̂ = {pred:.4f}  IC = [{ci[0]:.4f}, {ci[1]:.4f}]")
        fc_har.append(pred)
        ci_low.append(ci[0])
        ci_high.append(ci[1])

    fc_har  = np.array(fc_har)
    ci_low  = np.array(ci_low)
    ci_high = np.array(ci_high)

    if plot: 
        t_in  = np.arange(T)
        t_out = np.array([T + h - 1 for h in horizon])

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(t_in[-n_display:], series[-n_display:],
            color='steelblue', linewidth=0.8,
            label='Observed')

        ax.plot([t_in[-1], t_out[0]],
            [series[-1], fc_har[0]],
            color='darkorange', linewidth=1, linestyle='--', alpha=0.5)

        ax.plot(t_out, fc_har,
            color='darkorange', linewidth=2, linestyle='--',
            marker='o', markersize=5,
            label='HAR direct forecast')

        ax.fill_between(t_out, ci_low, ci_high,
                    alpha=0.2, color='darkorange',
                    label='IC 95%')


        ax.axvline(T - 1, color='black', linestyle=':',
               linewidth=1, label='Start forecast')

        ax.set_title(f"HAR direct — Forecast horizons {horizon}")
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
        plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_har_h{"_".join(str(x) for x in horizon)}.png', dpi=160, bbox_inches='tight')
        plt.show()

    return fc_har, ci_low, ci_high


def forecast_factor_rw(series, horizon=[1, 2, 3, 4, 5], alpha=0.05,
                       n_display=100, plot=True):
    """
    Random-walk benchmark forecast for a factor series.

    Forecast: Û_{T+h} = U_T  for all h  (last observed value)

    Confidence interval: U_T ± z_{α/2} · σ_Δ · √h
    where σ_Δ = std(ΔU_t) is the innovation standard deviation estimated
    from first differences. The CI widens with √h as expected for a random walk.

    Parameters
    ----------
    series    : array-like (T,) — factor time series (e.g. U_hat[0, :])
    horizon   : list of int    — forecast horizons
    alpha     : float          — significance level (default 0.05 → 95% CI)
    n_display : int            — number of past observations shown in the plot
    plot      : bool           — whether to show the plot

    Returns
    -------
    fc_rw   : np.ndarray (len(horizon),) — point forecasts (constant = last value)
    ci_low  : np.ndarray (len(horizon),) — lower CI bound
    ci_high : np.ndarray (len(horizon),) — upper CI bound
    """
    series  = np.asarray(series).squeeze()
    T       = len(series)
    z       = stats.norm.ppf(1 - alpha / 2)
    last    = series[-1]
    sigma_d = np.std(np.diff(series), ddof=1)

    fc_rw   = np.full(len(horizon), last)
    ci_low  = np.array([last - z * sigma_d * np.sqrt(h) for h in horizon])
    ci_high = np.array([last + z * sigma_d * np.sqrt(h) for h in horizon])

    for h, fc, cl, ch in zip(horizon, fc_rw, ci_low, ci_high):
        print(f"h={h:<3} : f̂ = {fc:.4f}  IC = [{cl:.4f}, {ch:.4f}]")

    if plot:
        t_in  = np.arange(T)
        t_out = np.array([T + h - 1 for h in horizon])

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(t_in[-n_display:], series[-n_display:],
                color='steelblue', linewidth=0.8, label='Observed')

        ax.plot([t_in[-1], t_out[0]], [last, last],
                color='tomato', linewidth=1, linestyle='--', alpha=0.5)

        ax.plot(t_out, fc_rw,
                color='tomato', linewidth=2, linestyle='--',
                marker='o', markersize=5, label='Random walk forecast')

        ax.fill_between(t_out, ci_low, ci_high,
                        alpha=0.2, color='tomato', label=f'IC {int((1-alpha)*100)}%')

        ax.axvline(T - 1, color='black', linestyle=':', linewidth=1,
                   label='Start forecast')

        ax.set_title(f'Random walk — Forecast horizons {horizon}')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
        plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_factor_rw_h{"_".join(str(x) for x in horizon)}.png', dpi=160, bbox_inches='tight')
        plt.show()

    return fc_rw, ci_low, ci_high





#====================================================================================
# Step 2: Idiosyncratic component
#====================================================================================

# 1. Functional Ljung-Box test

def test_ljung_box_func(Xi_hat, pairs_name,lags = [5,10], n_pc =5):
    """
    Idea: project Xi_hat curves on the first n_pc principal functional components

    H0: Xi_hat functional white noise
    H1: residual temporal structure
    """
    N,T,n_tau = Xi_hat.shape

    print("Functional Ljung-Box test:")

    rejected = []

    for n in range(N):
        curves = Xi_hat[n, :, :]

        # centre the curves
        mean_curve = curves.mean(axis=0)
        centered   = curves - mean_curve

        #U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        pca = PCA(n_components=n_pc)
        scores = pca.fit_transform(centered)

        #scores = U[:, :n_pc] * S[:n_pc]

        var_exp = pca.explained_variance_ratio_ 
        print(f"{'─'*55}")
        print(f'Ljung-Box test for pair {pairs_name[n]}:')
        print(f"Explained variance by {n_pc} CP : " f"{var_exp.sum()*100:.1f}%")


        for k in range(n_pc):
            score_k = scores[:, k]
            print(f"Direction {k}:")
            lb = test_ljung_box(score_k, lags)
            p_vals = lb['lb_pvalue'].values

            if any(p < 0.05 for p in p_vals): 
                rejected.append(pairs_name[n])

    print("Ljung-Box rejected for:")
    rejected = list(set(rejected))
    if len(rejected) ==0 : print("--")
    else: 
        print(rejected)
        print(f'Number of fx rejected: {len(rejected)}')

    return rejected

# 2. Modelling -> FAR(p)

def select_p_far(Xi_hat, pairs_name, p_max = 5, n_pc = 5 ):
    """
    Select params p for each pair

    PCA => scores scalaires 
    Select p by BIC for each score 
    """
    print('Order selection FAR(p) for each pair')
    N, T, n_tau = Xi_hat.shape
    orders      = {}

    for n in range(N):
        curves     = Xi_hat[n, :, :]
        mean_curve = curves.mean(axis=0)
        centered   = curves - mean_curve

        pca = PCA(n_components=n_pc)
        scores = pca.fit_transform(centered)

        best_p_per_cp = []

        print(f"{'─'*55}")
        print(f'Pair:{pairs_name[n]}: ')
        print(f"{'─'*55}")

        d_hat_per_pair=[]

        for k in range(n_pc):
            print(f'PC: {k}:')
            score_k  = scores[:, k]
            best_bic = np.inf
            best_p   = 1
            t_gph = gph_test(score_k)
            d_hat_per_pair.append(t_gph['d_hat'])

            for p in range(1, p_max + 1):
                try:
                    res = AutoReg(score_k, lags=p, trend='c').fit()
                    if res.bic < best_bic:
                        best_bic = res.bic
                        best_p   = p
                except Exception:
                    continue

            best_p_per_cp.append(best_p)
        far_order = max(best_p_per_cp)
        orders[pairs_name[n]] = {
            'far_order'     : far_order,
            'p_per_cp'      : best_p_per_cp, 
            'd_hat'         : d_hat_per_pair
        }

        print(f"{pairs_name[n]:<12} : p per PC = {best_p_per_cp}"
              f"  → FAR({far_order})")

    return orders



###
#FAR
#

def select_var_order_fpca(Xi_hat, n_components=5, p_max=10, criterion="bic"):
    """
    Xi_hat: (T, M) for one asset OR (N, T, M) handled outside
    returns: optimal p
    """

    T, M = Xi_hat.shape

    # 1. FPCA (PCA on discretized functions)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xi_hat)  # (T, K)

    # 2. VAR order selection
    model = VAR(scores)

    order_results = model.select_order(maxlags=p_max)

    if criterion == "bic":
        return order_results.selected_orders["bic"]
    elif criterion == "aic":
        return order_results.selected_orders["aic"]
    elif criterion == "hqic":
        return order_results.selected_orders["hqic"]
    else:
        raise ValueError("criterion must be bic, aic or hqic")
    
def forecast_functional_VAR(Xi_hat, h=1, n_components=5, p=None, p_max=10):
    """
    Xi_hat: (T, M) for ONE asset
    returns: predicted function at horizon h (shape M,)
    """

    T, M = Xi_hat.shape

    # 1. FPCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xi_hat)      # (T, K)
    loadings = pca.components_              # (K, M)
    mean_func = Xi_hat.mean(axis=0)

    # 2. choose p if not given
    if p is None:
        model_tmp = VAR(scores)
        p = model_tmp.select_order(maxlags=min(p_max, T//10)).selected_orders["bic"]

        if p is None or p < 1:
            p = 1

    # 3. fit VAR
    model = VAR(scores)
    results = model.fit(p)

    # 4. forecast scores
    forecast_scores = results.forecast(scores[-p:], steps=h)[-1]  # (K,)

    # 5. reconstruct function
    forecast_func = mean_func + forecast_scores @ loadings

    return forecast_func, p

def forecast_all_assets_FAR(Xi_hat,pairs,  h=1, K=5, p=None, p_max=10):
    """
    Xi_hat: (N, T, M)
    returns: (N, M)
    """

    N, T, M = Xi_hat.shape
    forecasts = np.zeros((N, M))

    for i in range(N):
        forecasts[i],p = forecast_functional_VAR(
            Xi_hat[i],
            h=h,
            n_components=K,
            p=p,
            p_max=p_max
        )
        print(f'{pairs[i]}: {p} ')

    return forecasts



#====================================================================================
# Step 3: Comparison & Evaluation
#====================================================================================


def reconstruct_forecast(fc_factor, fc_idio, ffm):
    """
    Reconstruct x̂_{i,T+h}(τ) for a single horizon h.

    Parameters
    ----------
    fc_factor : float or np.array (r,) — factor forecast(s) at T+h
    fc_idio   : np.array (N, M)        — idiosyncratic forecast at T+h
                                         (output of predict_operator_h for a given h)
    ffm       : dict                   — output of run_ffm

    Returns
    -------
    X_hat_log : np.array (N, M) — log(r²+ε)
    X_hat_abs : np.array (N, M) — |r_t| = exp(x̂/2)
    """
    B_hat     = ffm['B_hat']          # (N, r, M)
    sigma     = ffm['sigma']
    mu        = ffm['mu']             # (N, M)
    N, r, M   = B_hat.shape
    fc_factor = np.atleast_1d(fc_factor)   # (r,)

    X_hat = np.zeros((N, M))
    for i in range(N):
        common = np.zeros(M)
        for l in range(r):
            common += fc_factor[l] * B_hat[i, l, :]
        X_hat[i] = (common + fc_idio[i]) * sigma + mu[i]

    return X_hat, np.exp(X_hat / 2)   # (N, M), (N, M)


# =============================================================================
# ADDITIONAL FORECASTING FUNCTIONS
# =============================================================================

def select_garch_order(resid: np.ndarray, p_max: int = 3, q_max: int = 3) -> tuple:
    """Select optimal GARCH(p, q) order by BIC."""
    from arch import arch_model as _arch_model
    resid     = np.asarray(resid).squeeze() * 100
    best_bic  = np.inf
    best_order = (1, 1)

    print("GARCH order selection by BIC")
    print(f"{'─'*40}")
    print(f"{'Order':<15} {'BIC':>12}")
    print(f"{'─'*40}")

    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            try:
                model  = _arch_model(resid, mean='Zero', vol='GARCH', p=p, q=q)
                result = model.fit(disp='off')
                marker = ' ← best' if result.bic < best_bic else ''
                print(f"GARCH({p},{q})      {result.bic:>12.2f}{marker}")
                if result.bic < best_bic:
                    best_bic   = result.bic
                    best_order = (p, q)
            except Exception:
                continue

    print(f"{'─'*40}")
    print(f"Optimal order: GARCH{best_order}  (BIC = {best_bic:.2f})")
    return best_order


def select_garch_variant(resid: np.ndarray) -> tuple:
    """Compare GARCH, EGARCH, GJR-GARCH variants by BIC (normal vs Student-t)."""
    from arch import arch_model as _arch_model
    resid   = np.asarray(resid).squeeze() * 100
    results = {}

    specs = [
        ('GARCH(1,1) Normal',  'GARCH',  1, 1, 'normal'),
        ('GARCH(1,1) Student', 'GARCH',  1, 1, 't'),
        ('EGARCH(1,1)',        'EGARCH', 1, 1, 'normal'),
        ('GJR-GARCH(1,1)',     'GARCH',  1, 1, 'normal'),
        ('GARCH(2,1)',         'GARCH',  2, 1, 'normal'),
        ('GARCH(1,2)',         'GARCH',  1, 2, 'normal'),
    ]

    print("GARCH variant comparison by BIC")
    print(f"{'─'*45}")
    print(f"{'Model':<25} {'AIC':>10} {'BIC':>10}")
    print(f"{'─'*45}")

    best_bic  = np.inf
    best_name = ''

    for name, vol, p, q, dist in specs:
        try:
            o = 1 if 'GJR' in name else 0
            model  = _arch_model(resid, mean='Zero', vol=vol, p=p, q=q, o=o, dist=dist)
            result = model.fit(disp='off')
            marker = ' ← best' if result.bic < best_bic else ''
            print(f"{name:<25} {result.aic:>10.2f} {result.bic:>10.2f}{marker}")
            results[name] = result
            if result.bic < best_bic:
                best_bic  = result.bic
                best_name = name
        except Exception:
            continue

    print(f"{'─'*45}")
    print(f"Best model: {best_name}")
    return results, best_name


def plot_all_forecasts(series: np.ndarray, forecasts_dict: dict,
                       ci_dict: dict = None, h: int = 5,
                       n_display: int = 200, dates=None):
    """
    Plot observed series + multiple forecast models side by side.

    Parameters
    ----------
    forecasts_dict : dict — {'model_name': np.array (h,)}
    ci_dict        : dict — {'model_name': (ci_low, ci_high)} — optional
    dates          : array-like of datetime-like, length T — if provided, x-axis shows real dates
    """
    import pandas as pd
    import matplotlib.dates as mdates

    series = np.asarray(series).squeeze()
    T      = len(series)

    if dates is not None:
        dates = pd.DatetimeIndex(dates)
        future_dates = pd.bdate_range(start=dates[-1], periods=h + 1, freq='B')[1:]
        x_in  = dates[-n_display:]
        x_out = future_dates[:h]
        x_last = dates[-1]
        x_connector_start = dates[-1]
    else:
        x_in  = np.arange(T)[-n_display:]
        x_out = np.arange(T, T + h)
        x_last = T - 1
        x_connector_start = T - 1

    colors = {
        'Random Walk': 'gray',
        'AR'         : 'green',
        'ARIMA'      : 'crimson',
        'HAR'        : 'darkorange',
        'ARFIMA'     : 'purple',
        'Wiener'     : 'brown',
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x_in, series[-n_display:],
            color='steelblue', linewidth=0.8, label='Observed', zorder=5)


    for name, fc in forecasts_dict.items():
        fc = np.asarray(fc).squeeze()[:h]
        col = colors.get(name, 'black')

        ax.plot([x_connector_start, x_out[0]], [series[-1], fc[0]],
            color=col, linestyle='--', alpha=0.7)

        ax.plot(x_out, fc,
            color=col,
            linestyle='--',
            marker='o',
            label=name)
        if ci_dict is not None and name in ci_dict:
            ci_low  = np.asarray(ci_dict[name][0])[:h]
            ci_high = np.asarray(ci_dict[name][1])[:h]
            if h == 1:
                x = x_out[0]
                if dates is not None:
                    tick_w = pd.Timedelta(days=max(1, n_display // 80))
                else:
                    tick_w = n_display * 0.012
                ax.vlines(x, ci_low[0], ci_high[0], color=col, alpha=0.7, linewidth=1.5)
                ax.hlines([ci_low[0], ci_high[0]], x - tick_w, x + tick_w,
                          color=col, alpha=0.9, linewidth=1.5, label=f'{name} IC 95%')
            else:
                ax.fill_between(x_out, ci_low, ci_high,
                                alpha=0.15, color=col, label=f'{name} IC 95%')

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=30, ha='right')
        ax.set_xlabel("Date")
    else:
        ax.set_xlabel("Time")

    ax.set_title(f"Forecast comparison — h={h} steps")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_all_models_h{h}.png', dpi=160, bbox_inches='tight')
    plt.show()


def predict_operator_h(Xi_hat: np.ndarray, sigma: np.ndarray,
                       pairs_name: list, h: int = 1, K: int = 5) -> np.ndarray:
    """
    Optimal FPCA predictor Ĉ_h Ĉ_0^{-1} at horizon h for idiosyncratic components.

    Parameters
    ----------
    Xi_hat     : np.ndarray (N, T, n_tau)
    sigma      : normalization scale
    pairs_name : list of pair names (N,)
    h          : forecast horizon
    K          : number of FPCA components

    Returns
    -------
    forecasts : np.ndarray (N, n_tau)
    """
    N, T, n_tau = Xi_hat.shape
    Xi_hat_c    = Xi_hat.copy() / sigma
    forecasts   = np.zeros((N, n_tau))

    for i in range(N):
        curves   = Xi_hat_c[i, :, :]
        mean_c   = curves.mean(axis=0)
        centered = curves - mean_c

        pca      = PCA(n_components=K)
        scores   = pca.fit_transform(centered)
        loadings = pca.components_

        C0    = (scores[:-h].T @ scores[:-h]) / (T - h)
        Ch    = (scores[h:].T  @ scores[:-h]) / (T - h)
        Psi_h = Ch @ np.linalg.inv(C0)

        forecasts[i, :] = Psi_h @ scores[-1] @ loadings + mean_c

    return forecasts


def forecast_idio_rw(
    Xi_hat: np.ndarray,
    h: int = 1,
    alpha: float = 0.05,
) -> tuple:
    """
    Random-walk benchmark forecast for idiosyncratic components.

    Forecast: ξ̂_i(T+h, τ) = ξ_i(T, τ)  for all pairs i and intraday points τ

    Confidence interval (pointwise in τ):
        ξ_i(T, τ) ± z_{α/2} · σ_Δ_i(τ) · √h
    where σ_Δ_i(τ) = std(ξ_i(t,τ) − ξ_i(t−1,τ)) is the innovation std
    estimated from first differences at each τ.

    Parameters
    ----------
    Xi_hat : np.ndarray (N, T, n_tau)
    h      : forecast horizon
    alpha  : significance level (default 0.05 → 95% CI)

    Returns
    -------
    forecasts : np.ndarray (N, n_tau) — last observed curve per pair
    ci_low    : np.ndarray (N, n_tau) — lower CI
    ci_high   : np.ndarray (N, n_tau) — upper CI
    """
    N, T, n_tau = Xi_hat.shape
    z           = stats.norm.ppf(1 - alpha / 2)

    forecasts = Xi_hat[:, -1, :].copy()                      # (N, n_tau)

    delta     = np.diff(Xi_hat, axis=1)                      # (N, T-1, n_tau)
    sigma_d   = delta.std(axis=1, ddof=1)                    # (N, n_tau)

    ci_low  = forecasts - z * sigma_d * np.sqrt(h)
    ci_high = forecasts + z * sigma_d * np.sqrt(h)

    return forecasts, ci_low, ci_high


def forecast_idio_har(
    Xi_hat: np.ndarray,
    pairs_name: list,
    h: int = 1,
    K: int = 5,
    alpha: float = 0.05,
    center_window: int = 252,
    fit_window: int = 756,
) -> tuple:
    """
    HAR forecast of idiosyncratic components via FPCA score reduction.

    For each pair i:
      1. PCA loadings are fitted globally (all T days) for stable intraday shapes.
      2. Local mean is computed on the last center_window days to anchor the
         forecast to the recent regime (avoids flat predictions from global mean).
      3. Scores are projected globally then HAR is fitted on the last fit_window
         days only — shorter window → coefficients calibrated to recent persistence
         (β_d+β_w+β_m closer to 1) → less mean-reversion → less flat predictions.
      4. CI via pointwise variance propagation through the FPCA basis.

    Parameters
    ----------
    Xi_hat        : np.ndarray (N, T, n_tau)
    pairs_name    : list of pair names (N,)
    h             : forecast horizon (days ahead)
    K             : number of FPCA components
    alpha         : significance level for CI (default 0.05 → 95% CI)
    center_window : days for local mean centering (default 252 ≈ 1 year)
    fit_window    : days of recent history used to fit each HAR model
                    (default 756 ≈ 3 years). Shorter → more persistent coefficients.

    Returns
    -------
    forecasts : np.ndarray (N, n_tau)
    ci_low    : np.ndarray (N, n_tau)
    ci_high   : np.ndarray (N, n_tau)
    """
    N, T, n_tau = Xi_hat.shape
    z           = stats.norm.ppf(1 - alpha / 2)

    forecasts = np.zeros((N, n_tau))
    ci_low    = np.zeros((N, n_tau))
    ci_high   = np.zeros((N, n_tau))

    for i in range(N):
        curves = Xi_hat[i, :, :]   # (T, n_tau)

        # ── Step 1 : global PCA for stable loadings ────────────────────────
        global_mean = curves.mean(axis=0)
        pca         = PCA(n_components=K)
        pca.fit(curves - global_mean)
        loadings = pca.components_               # (K, n_tau)

        # ── Step 2 : local mean — anchor to recent regime ─────────────────
        w      = min(center_window, T) if center_window is not None else T
        mean_c = curves[-w:].mean(axis=0)

        # ── Step 3 : project ALL curves onto global loadings ───────────────
        scores = (curves - mean_c) @ loadings.T  # (T, K)

        # ── Step 4 : HAR fitted on recent fit_window only ─────────────────
        fw         = min(fit_window, T) if fit_window is not None else T
        scores_fit = scores[-fw:]                # (fw, K)

        fc_scores  = np.zeros(K)
        var_scores = np.zeros(K)

        for k in range(K):
            s     = pd.Series(scores[:, k])       # full series for rolling lags
            s_fit = pd.Series(scores_fit[:, k])   # recent window for fitting

            # Rolling lags computed on the full series, then sliced to fit_window
            s_d = s
            s_w = s.rolling(5).mean()
            s_m = s.rolling(22).mean()
            s_y = s.shift(-h)

            df = pd.DataFrame({'d': s_d, 'w': s_w, 'm': s_m, 'y': s_y})
            df = df.dropna().iloc[-fw:]           # restrict to fit_window

            X_mat = sm.add_constant(df[['d', 'w', 'm']])
            model = sm.OLS(df['y'], X_mat).fit()

            # ── Predict at T ───────────────────────────────────────────────
            x_new = np.array([[1.0,
                                s.iloc[-1],
                                s.rolling(5).mean().iloc[-1],
                                s.rolling(22).mean().iloc[-1]]])

            fc_scores[k] = model.predict(x_new)[0]

            # ── Prediction variance ────────────────────────────────────────
            X_arr    = X_mat.values
            XtX_inv  = np.linalg.inv(X_arr.T @ X_arr)
            sigma2_k = np.sum(model.resid ** 2) / (len(df) - 4)
            var_scores[k] = sigma2_k * (1.0 + float(x_new @ XtX_inv @ x_new.T))

        # ── Reconstruct forecast curve ─────────────────────────────────────
        forecasts[i, :] = fc_scores @ loadings + mean_c

        # ── Propagate uncertainty pointwise ───────────────────────────────
        var_curve = (loadings ** 2).T @ var_scores
        std_curve = np.sqrt(var_curve)

        ci_low[i, :]  = forecasts[i, :] - z * std_curve
        ci_high[i, :] = forecasts[i, :] + z * std_curve

    return forecasts, ci_low, ci_high


def plot_forecast_idiosyncratic(Xi_hat: np.ndarray, pairs_name: list,
                                h: int = 1, K: int = 5,
                                n_last: int = 5,
                                forecasts: np.ndarray = None,
                                title: str = None) -> np.ndarray:
    """
    Plot idiosyncratic forecasts for all pairs on a shared scale.

    Parameters
    ----------
    Xi_hat    : np.ndarray (N, T, n_tau) — idiosyncratic components
    pairs_name: list of pair names
    h         : horizon (used for label and internal computation if forecasts=None)
    K         : number of FPCA components (used only if forecasts=None)
    n_last    : number of recent observed curves to overlay
    forecasts : np.ndarray (N, n_tau), optional — pre-computed forecast curves.
                If None, forecasts are computed internally via the FPCA operator.
    title     : optional figure title override

    Returns
    -------
    np.ndarray (N, n_tau) — forecast curves used in the plot
    """
    N, T, n_tau = Xi_hat.shape
    tau         = np.linspace(0, 1, n_tau)
    n_cols      = 4
    n_rows      = int(np.ceil(N / n_cols))

    all_last   = []
    all_pred   = []
    all_mean   = []
    all_recent = []

    for i in range(N):
        curves = Xi_hat[i, :, :]
        mean_c = curves.mean(axis=0)

        if forecasts is not None:
            pred_curve = forecasts[i, :]
        else:
            centered = curves - mean_c
            pca      = PCA(n_components=K)
            scores   = pca.fit_transform(centered)
            loadings = pca.components_
            C0       = (scores[:-h].T @ scores[:-h]) / (T - h)
            Ch       = (scores[h:].T  @ scores[:-h]) / (T - h)
            Psi_h    = Ch @ np.linalg.inv(C0)
            pred_curve = (Psi_h @ scores[-1, :]) @ loadings + mean_c

        all_last.append(curves[-1, :])
        all_pred.append(pred_curve)
        all_mean.append(mean_c)
        all_recent.append(curves[-n_last:, :])

    all_pred_arr = np.stack(all_pred)   # (N, n_tau)

    all_values = np.concatenate(
        all_last + all_pred + all_mean + [r.flatten() for r in all_recent]
    )
    y_min = all_values.min() - (all_values.max() - all_values.min()) * 0.05
    y_max = all_values.max() + (all_values.max() - all_values.min()) * 0.05

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    if title is None:
        if forecasts is not None:
            title = f'Idiosyncratic forecast (external)  h={h}'
        else:
            title = (f'Idiosyncratic forecast — '
                     f'$\\hat{{\\mathcal{{C}}}}_h\\hat{{\\mathcal{{C}}}}_0^{{-1}}$  h={h}, K={K}')
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for i in range(N):
        ax     = axes[i]
        recent = all_recent[i]

        ax.plot(tau, all_mean[i], color='gray', linewidth=1, linestyle=':', label='Mean ε̄(τ)')

        for j in range(n_last):
            a     = 0.2 + 0.6 * (j / (n_last - 1))
            label = 'Last obs.' if j == n_last - 1 else None
            ax.plot(tau, recent[j, :], color='steelblue', linewidth=0.8,
                    linestyle='--', alpha=a, label=label)

        ax.plot(tau, all_pred[i], color='darkorange', linewidth=1.5,
                linestyle='--', label=f'Forecast h={h}')

        ax.set_ylim(y_min, y_max)
        ax.set_title(pairs_name[i].upper(), fontsize=9)
        ax.set_xlabel('τ', fontsize=8)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(N, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_idiosyncratic_h{h}_K{K}.png', dpi=160, bbox_inches='tight')
    plt.show()

    return all_pred_arr


# =============================================================================
# PLOTS
# =============================================================================

def plot_forecast_vs_observed(
    target_date,
    ffm: dict,
    X_abs: np.ndarray,
    vol_matrices_2026: dict,
    vol_ffm_2026: dict = None,
    h: int = 1,
    n_cols: int = 4,
):
    """
    Plot forecast vs raw observed vs smoothed observed for all FFM pairs.

    Parameters
    ----------
    target_date       : str or pd.Timestamp — date to display (e.g. '2026-01-02')
    ffm               : dict — output of run_ffm (keys: 'pairs', 'tau', 'mu')
    X_abs             : np.ndarray (N, M) — forecast in |r| space (exp(x/2))
    vol_matrices_2026 : dict {pair: DataFrame} — raw 2026 log-return matrices
    vol_ffm_2026      : dict {pair: DataFrame}, optional — smoothed log-space 2026 data
    h                 : int — forecast horizon label
    n_cols            : int — number of subplot columns
    """
    target_date = pd.Timestamp(target_date)
    pairs = ffm['pairs']
    tau   = ffm['tau']
    N     = len(pairs)

    n_rows = int(np.ceil(N / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten()

    for i, pair in enumerate(pairs):
        ax = axes_flat[i]

        # ── Raw observed ──────────────────────────────────────────────────
        row_true = None
        if pair in vol_matrices_2026:
            df_raw = vol_matrices_2026[pair]
            mask   = pd.to_datetime(df_raw.index).normalize() == target_date
            if mask.any():
                row_true = np.abs(df_raw.loc[mask].values[0])

        # ── Smoothed observed (log-space → |r|) ───────────────────────────
        row_smooth = None
        if vol_ffm_2026 is not None and pair in vol_ffm_2026:
            df_sm   = vol_ffm_2026[pair]
            mask_sm = pd.to_datetime(df_sm.index).normalize() == target_date
            if mask_sm.any():
                row_smooth = np.exp(df_sm.loc[mask_sm].values[0] / 2)

        # ── Training mean ─────────────────────────────────────────────────
        mean_abs = np.exp(ffm['mu'][i] / 2)

        # ── Plot ──────────────────────────────────────────────────────────
        ax.plot(tau, mean_abs, color='gray', linewidth=1.0,
                linestyle=':', alpha=0.6, label='μ̂_i (train mean)')

        if row_true is not None:
            ax.plot(tau, row_true, color='#2980B9', linewidth=1.0,
                    linestyle='-', alpha=0.5, label='Raw observed')
        else:
            ax.text(0.5, 0.5, f'{target_date.date()}\nabsent',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=7, color='gray')

        if row_smooth is not None:
            ax.plot(tau, row_smooth, color='#27AE60', linewidth=1.5,
                    linestyle='--', alpha=0.9, label='Smoothed observed')

        ax.plot(tau, X_abs[i, :], color='#E74C3C', linewidth=1.8,
                label=f'Forecast h={h}')

        ax.set_title(pair, fontsize=9, fontweight='bold')
        ax.set_xlabel('τ', fontsize=7)
        ax.set_ylabel('|r_t|', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(N, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'Forecast vs Observed — {target_date.date()}  (h={h})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/forecast_vs_observed_{target_date.date()}_h{h}.png', dpi=160, bbox_inches='tight')
    plt.show()


# =============================================================================
# EVALUATION
# =============================================================================

def _mafe(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Absolute Functional Error: (1/M) Σ_τ |a(τ) - b(τ)|"""
    diff = a - b
    return float(np.nanmean(np.abs(diff)))


def _msfe(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Functional Error: (1/M) Σ_τ (a(τ) - b(τ))²"""
    diff = a - b
    return float(np.nanmean(diff ** 2))


def decompose_forecast_error(
    target_date,
    ffm: dict,
    X_log: np.ndarray,
    vol_matrices_2026: dict,
    vol_ffm_2026: dict,
    exclude_pairs: list = None,
) -> pd.DataFrame:
    """
    Decompose the total forecast error into two additive terms for each pair.

    All quantities are kept in log-variance space (log r²), consistent with
    the functional factor model estimation.

        total error   = model error + smoothing error

    where:
        model error     : forecast vs smoothed observed  (captures model quality)
        smoothing error : smoothed observed vs raw       (captures smoothing approximation)
        total error     : forecast vs raw observed

    Both MAFE and MSFE are computed as integrals over τ (approximated by the
    mean over the M intraday time points).

    Parameters
    ----------
    target_date       : str or pd.Timestamp
    ffm               : dict — run_ffm output (keys: 'pairs', 'tau', 'mu')
    X_log             : np.ndarray (N, M) — forecast curves in log-variance space
                        (first output of reconstruct_forecast)
    vol_matrices_2026 : dict {pair: DataFrame} — raw 5-min log-return matrices
    vol_ffm_2026      : dict {pair: DataFrame} — smoothed log-variance matrices
    exclude_pairs     : list of pair names to exclude (e.g. ['usdcnh'])

    Returns
    -------
    pd.DataFrame with columns:
        pair,
        MAFE_model, MSFE_model,       # forecast vs smoothed (log-variance)
        MAFE_smooth, MSFE_smooth,     # smoothed vs raw      (log-variance)
        MAFE_total, MSFE_total        # forecast vs raw      (log-variance)
    """
    target_date   = pd.Timestamp(target_date)
    pairs         = ffm['pairs']
    _exclude      = set(exclude_pairs) if exclude_pairs else set()
    rows          = []

    for i, pair in enumerate(pairs):
        if pair in _exclude:
            continue
        fc = X_log[i, :]

        # ── Raw observed in log-variance space: 2·log|r| ─────────────────
        f_raw = None
        if pair in vol_matrices_2026:
            df_raw = vol_matrices_2026[pair]
            mask   = pd.to_datetime(df_raw.index).normalize() == target_date
            if mask.any():
                raw_vals = df_raw.loc[mask].values[0].astype(float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    f_raw = np.where(
                        np.isnan(raw_vals) | (raw_vals == 0),
                        np.nan,
                        2.0 * np.log(np.abs(raw_vals)),
                    )

        # ── Smoothed observed (already log-variance) ──────────────────────
        f_smooth = None
        if pair in vol_ffm_2026:
            df_sm   = vol_ffm_2026[pair]
            mask_sm = pd.to_datetime(df_sm.index).normalize() == target_date
            if mask_sm.any():
                f_smooth = df_sm.loc[mask_sm].values[0]

        if f_raw is None or f_smooth is None:
            rows.append({'pair': pair,
                         'MAFE_model': np.nan, 'MSFE_model': np.nan,
                         'MAFE_smooth': np.nan, 'MSFE_smooth': np.nan,
                         'MAFE_total': np.nan,  'MSFE_total': np.nan})
            continue

        rows.append({
            'pair'        : pair,
            'MAFE_model'  : _mafe(fc, f_smooth),
            'MSFE_model'  : _msfe(fc, f_smooth),
            'MAFE_smooth' : _mafe(f_smooth, f_raw),
            'MSFE_smooth' : _msfe(f_smooth, f_raw),
            'MAFE_total'  : _mafe(fc, f_raw),
            'MSFE_total'  : _msfe(fc, f_raw),
        })

    df = pd.DataFrame(rows).set_index('pair')
    return df


def print_error_decomposition(df_errors: pd.DataFrame) -> None:
    """Print a formatted error decomposition table."""
    cols = ['MAFE_model', 'MAFE_smooth', 'MAFE_total',
            'MSFE_model', 'MSFE_smooth', 'MSFE_total']
    print(f"\n{'Pair':<14}", end='')
    for c in cols:
        print(f"  {c:>13}", end='')
    print()
    print('─' * (14 + 15 * len(cols)))
    for pair, row in df_errors.iterrows():
        print(f"{pair:<14}", end='')
        for c in cols:
            v = row[c]
            print(f"  {v:>13.4e}" if not np.isnan(v) else f"  {'—':>13}", end='')
        print()
    print('─' * (14 + 15 * len(cols)))
    mean_row = df_errors[cols].mean()
    print(f"{'MEAN':<14}", end='')
    for c in cols:
        print(f"  {mean_row[c]:>13.4e}", end='')
    print()


def plot_error_decomposition(
    df_errors: pd.DataFrame,
    metric: str = 'MAFE',
    title_suffix: str = '',
) -> None:
    """
    Stacked bar chart: model error + smoothing error per pair.

    Parameters
    ----------
    df_errors     : output of decompose_forecast_error
    metric        : 'MAFE' or 'MSFE'
    title_suffix  : appended to the figure title (e.g. the date)
    """
    if metric not in ('MAFE', 'MSFE'):
        raise ValueError("metric must be 'MAFE' or 'MSFE'")

    col_model  = f'{metric}_model'
    col_smooth = f'{metric}_smooth'
    col_total  = f'{metric}_total'

    df = df_errors[[col_model, col_smooth, col_total]].dropna()
    pairs = df.index.tolist()
    x     = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 0.6), 5))

    bars_model  = ax.bar(x, df[col_model],  label='Model error (forecast vs smoothed)',
                         color='#E74C3C', alpha=0.85)
    bars_smooth = ax.bar(x, df[col_smooth], bottom=df[col_model],
                         label='Smoothing error (smoothed vs raw)',
                         color='#3498DB', alpha=0.85)

    # Total as scatter
    ax.scatter(x, df[col_total], color='black', zorder=5, s=30,
               label=f'{metric} total (forecast vs raw)')

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'Forecast error decomposition — {metric}'
                 + (f'  |  {title_suffix}' if title_suffix else ''),
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    _suffix = f'_{title_suffix.replace(" ", "_")}' if title_suffix else ''
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/error_decomposition_{metric}{_suffix}.png', dpi=160, bbox_inches='tight')
    plt.show()


def compare_models_error(
    target_dates,
    models: dict,
    vol_matrices_2026: dict,
    vol_ffm_2026: dict,
    metric: str = 'both',
    exclude_pairs: list = None,
) -> pd.DataFrame:
    """
    Compare forecast errors (vs raw observed) across multiple FFM models and dates.

    The error is the *total* functional error: forecast vs raw observed,
    averaged over all target dates provided.

    Parameters
    ----------
    target_dates      : str, pd.Timestamp, or list thereof
    models            : dict {model_name: (ffm, X_log)}
                        ffm    — run_ffm output
                        X_log  — np.ndarray (N, M) forecast in log-variance space
                                 (first output of reconstruct_forecast)
    vol_matrices_2026 : dict {pair: DataFrame} — raw 5-min log-return matrices
    vol_ffm_2026      : dict {pair: DataFrame} — smoothed log-variance matrices
    metric            : 'MAFE', 'MSFE', or 'both'
    exclude_pairs     : list of pair names to exclude (e.g. ['usdcnh'])

    Returns
    -------
    pd.DataFrame  index=pair, columns=['{model}_MAFE', '{model}_MSFE', ...]
                  values are means over target_dates
    """
    if isinstance(target_dates, (str, pd.Timestamp)):
        target_dates = [target_dates]
    target_dates = [pd.Timestamp(d) for d in target_dates]

    # Accumulate errors per model per pair over dates
    # structure: {model_name: {pair: {'MAFE': [], 'MSFE': []}}}
    accum = {name: {} for name in models}

    for date in target_dates:
        for name, (ffm, X_log) in models.items():
            df_err = decompose_forecast_error(
                date, ffm, X_log, vol_matrices_2026, vol_ffm_2026,
                exclude_pairs=exclude_pairs,
            )
            for pair, row in df_err.iterrows():
                if pair not in accum[name]:
                    accum[name][pair] = {'MAFE': [], 'MSFE': []}
                if not np.isnan(row['MAFE_total']):
                    accum[name][pair]['MAFE'].append(row['MAFE_total'])
                    accum[name][pair]['MSFE'].append(row['MSFE_total'])

    # Build summary DataFrame
    all_pairs = sorted({p for name in accum for p in accum[name]})
    records   = []

    for pair in all_pairs:
        row = {'pair': pair}
        for name in models:
            vals = accum[name].get(pair, {'MAFE': [], 'MSFE': []})
            row[f'{name}_MAFE'] = float(np.mean(vals['MAFE'])) if vals['MAFE'] else np.nan
            row[f'{name}_MSFE'] = float(np.mean(vals['MSFE'])) if vals['MSFE'] else np.nan
        records.append(row)

    df = pd.DataFrame(records).set_index('pair')
    return df


def print_models_comparison(df_compare: pd.DataFrame) -> None:
    """Print a formatted comparison table across models."""
    cols = df_compare.columns.tolist()
    width = 14
    print(f"\n{'Pair':<{width}}", end='')
    for c in cols:
        print(f"  {c:>{width}}", end='')
    print()
    print('─' * (width + (width + 2) * len(cols)))
    for pair, row in df_compare.iterrows():
        print(f"{pair:<{width}}", end='')
        for c in cols:
            v = row[c]
            print(f"  {v:>{width}.4e}" if not np.isnan(v) else f"  {'—':>{width}}", end='')
        print()
    print('─' * (width + (width + 2) * len(cols)))
    print(f"{'MEAN':<{width}}", end='')
    for c in cols:
        print(f"  {df_compare[c].mean():>{width}.4e}", end='')
    print()


def plot_models_comparison(
    df_compare: pd.DataFrame,
    metric: str = 'MAFE',
    title_suffix: str = '',
) -> None:
    """
    Grouped bar chart comparing MAFE or MSFE across models, per pair.

    Parameters
    ----------
    df_compare    : output of compare_models_error
    metric        : 'MAFE' or 'MSFE'
    title_suffix  : e.g. the target date or date range
    """
    cols = [c for c in df_compare.columns if c.endswith(f'_{metric}')]
    if not cols:
        raise ValueError(f"No columns ending with '_{metric}' in df_compare")

    model_names = [c.replace(f'_{metric}', '') for c in cols]
    df          = df_compare[cols].dropna(how='all')
    pairs       = df.index.tolist()
    n_pairs     = len(pairs)
    n_models    = len(cols)

    x      = np.arange(n_pairs)
    width  = 0.8 / n_models
    colors = ['#E74C3C', '#2980B9', '#27AE60', '#8E44AD', '#F39C12']

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_pairs * 0.7), 5))

    for ax_idx, (ax, met) in enumerate(zip(axes, [metric, metric])):
        met_cols = [c for c in df_compare.columns if c.endswith(f'_{met}')]
        if ax_idx == 1:
            # right panel: relative difference vs first model
            base_col = met_cols[0]
            ax.set_title(f'Relative gain vs {model_names[0]}  ({met})',
                         fontsize=11, fontweight='bold')
            for j, (col, name) in enumerate(zip(met_cols[1:], model_names[1:]), 1):
                rel = (df_compare[col] - df_compare[base_col]) / df_compare[base_col]
                rel = rel.reindex(pairs)
                ax.bar(x + (j - 1) * width, rel, width,
                       label=name, color=colors[j % len(colors)], alpha=0.85)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_ylabel('Relative difference', fontsize=9)
        else:
            # left panel: absolute values
            ax.set_title(f'Forecast error comparison — {met}'
                         + (f'  |  {title_suffix}' if title_suffix else ''),
                         fontsize=11, fontweight='bold')
            for j, (col, name) in enumerate(zip(met_cols, model_names)):
                ax.bar(x + j * width, df_compare[col].reindex(pairs),
                       width, label=name,
                       color=colors[j % len(colors)], alpha=0.85)
            ax.set_ylabel(met, fontsize=9)

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    _suffix = f'_{title_suffix.replace(" ", "_")}' if title_suffix else ''
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/models_comparison_{metric}{_suffix}.png', dpi=160, bbox_inches='tight')
    plt.show()


def make_benchmark_forecast(
    target_date,
    ffm: dict,
    vol_ffm_all: dict,
    vol_ffm_2026: dict = None,
    benchmark: str = 'naive',
) -> np.ndarray:
    """
    Build a benchmark forecast X_abs array (N, M) for a given target_date.

    The returned array has the same layout as the X_abs produced by
    reconstruct_forecast — it can be passed directly into compare_models_error
    alongside the FFM model.

    Parameters
    ----------
    target_date   : str or pd.Timestamp — date being forecast
    ffm           : run_ffm output — provides the pairs ordering and tau grid
    vol_ffm_all   : dict {pair: DataFrame} — smoothed training data (log-variance)
    vol_ffm_2026  : dict {pair: DataFrame} — smoothed 2026 data (log-variance), optional
    benchmark     : 'naive'     — previous trading day's smoothed curve
                    'hist_mean' — mean curve over the full training period

    Returns
    -------
    np.ndarray (N, M) — benchmark forecast in log-variance space
    """
    if benchmark not in ('naive', 'hist_mean'):
        raise ValueError("benchmark must be 'naive' or 'hist_mean'")

    target_date = pd.Timestamp(target_date)
    pairs   = ffm['pairs']
    M       = len(ffm['tau'])
    X_bench = np.full((len(pairs), M), np.nan)

    for i, pair in enumerate(pairs):
        if benchmark == 'naive':
            # Merge training + 2026 data, take the last day before target_date
            frames = []
            if pair in vol_ffm_all:
                frames.append(vol_ffm_all[pair])
            if vol_ffm_2026 and pair in vol_ffm_2026:
                frames.append(vol_ffm_2026[pair])
            if not frames:
                continue
            df = pd.concat(frames).sort_index()
            df.index = pd.to_datetime(df.index).normalize()
            past = df[df.index < target_date]
            if past.empty:
                continue
            X_bench[i, :] = past.iloc[-1].values   # already log-variance

        else:  # hist_mean
            if pair not in vol_ffm_all:
                continue
            df_train = vol_ffm_all[pair].copy()
            X_bench[i, :] = df_train.values.mean(axis=0)   # already log-variance

    return X_bench


def run_model_comparison(
    target_dates,
    ffm: dict,
    vol_matrices_2026: dict,
    vol_ffm_2026: dict,
    horizons: list = [1, 2],
    models_factor: list = ['RW', 'AR', 'HAR', 'ARIMA'],
    models_idio: list = ['RW', 'FAR', 'HAR'],
    exclude_pairs: list = None,
    d_arima: float = 1,
    p_max_ar: int = 10,
    K: int = 5,
) -> pd.DataFrame:
    """
    Run all (factor model × idio model × horizon) combinations and return a
    comparison table of MAFE / MSFE averaged across pairs.

    Factor models : 'RW'    — random walk (forecast_factor_rw)
                    'AR'    — AR(p) selected by BIC  (forecast_ar)
                    'HAR'   — HAR direct-h           (forecast_har_direct_h)
                    'ARIMA' — ARIMA(p,d,q) by BIC    (forecast_arima)

    Idio models   : 'RW'    — random walk            (forecast_idio_rw)
                    'FAR'   — FPCA operator Ĉ_hĈ_0⁻¹ (predict_operator_h)
                    'HAR'   — HAR on FPCA scores     (forecast_idio_har)

    Parameters
    ----------
    target_dates     : list of str/Timestamp, one per horizon — the observed date
                       to evaluate against for each h (e.g. ['2026-01-02', '2026-01-03']
                       for horizons=[1, 2])
    ffm              : run_ffm output
    vol_matrices_2026: dict {pair: DataFrame} — raw 2026 log-return matrices
    vol_ffm_2026     : dict {pair: DataFrame} — smoothed 2026 matrices
    horizons         : list of forecast horizons to evaluate
    models_factor    : subset of ['RW', 'AR', 'HAR', 'ARIMA']
    models_idio      : subset of ['RW', 'FAR', 'HAR']
    exclude_pairs    : pairs to drop from evaluation (e.g. ['usdcnh'])
    d_arima          : d parameter for ARIMA (default 1)
    p_max_ar         : max AR order for BIC selection
    K                : FPCA components for FAR and HAR idio models

    Returns
    -------
    pd.DataFrame — MultiIndex columns (horizon, metric), rows (factor, idio)
    """
    silence      = contextlib.redirect_stdout(io.StringIO())
    U_hat        = ffm['U_hat'][0, :]
    target_dates = list(target_dates)
    if len(target_dates) != len(horizons):
        raise ValueError("target_dates must have one entry per horizon")

    records = []

    for h, target_date in zip(horizons, target_dates):
        print(f'\n── h = {h}  →  target {target_date} ────────────────')

        # ── Factor forecasts ──────────────────────────────────────────────
        fc_factor = {}

        with silence:
            if 'RW' in models_factor:
                fc_rw, _, _ = forecast_factor_rw(U_hat, horizon=[h], plot=False)
                fc_factor['RW'] = float(fc_rw[0])

            if 'AR' in models_factor:
                _, fc_ar, _, _ = forecast_ar(U_hat, h=h, p_max=p_max_ar, plot=False)
                fc_factor['AR'] = float(fc_ar[-1])

            if 'HAR' in models_factor:
                _, fc_har, _ = forecast_har_direct_h(U_hat, h=h)
                fc_factor['HAR'] = float(fc_har)

            if 'ARIMA' in models_factor:
                _, fc_arima, _, _ = forecast_arima(U_hat, h=h, d=d_arima, plot=False)
                fc_factor['ARIMA'] = float(fc_arima[-1])

        # ── Idio forecasts ────────────────────────────────────────────────
        fc_idio = {}

        with silence:
            if 'RW' in models_idio:
                fc_idio['RW'], _, _ = forecast_idio_rw(ffm['Xi_hat'], h=h)

            if 'FAR' in models_idio:
                fc_idio['FAR'] = forecast_all_assets_FAR(
                    ffm['Xi_hat'],ffm['pairs'],  h=h, K=K
                )

            if 'HAR' in models_idio:
                fc_idio['HAR'], _, _ = forecast_idio_har(
                    ffm['Xi_hat'], ffm['pairs'], h=h, K=K
                )

        # ── Evaluate all combinations ─────────────────────────────────────
        for mf in models_factor:
            for mi in models_idio:
                if mf not in fc_factor or mi not in fc_idio:
                    continue

                X_log, _ = reconstruct_forecast(fc_factor[mf], fc_idio[mi], ffm)

                df_err = decompose_forecast_error(
                    target_date       = target_date,
                    ffm               = ffm,
                    X_log             = X_log,
                    vol_matrices_2026 = vol_matrices_2026,
                    vol_ffm_2026      = vol_ffm_2026,
                    exclude_pairs     = exclude_pairs,
                )

                records.append({
                    'h'           : h,
                    'Factor'      : mf,
                    'Idio'        : mi,
                    'MAFE_raw'    : df_err['MAFE_total'].mean(),
                    'MAFE_smooth' : df_err['MAFE_model'].mean(),
                    'MSFE_raw'    : df_err['MSFE_total'].mean(),
                    'MSFE_smooth' : df_err['MSFE_model'].mean(),
                })

                print(f"  {mf:<6} × {mi:<4}  "
                      f"MAFE_raw={records[-1]['MAFE_raw']:.4e}  "
                      f"MSFE_raw={records[-1]['MSFE_raw']:.4e}")

    # ── Build table ───────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.set_index(['Factor', 'Idio', 'h']).unstack('h')
    df.columns = pd.MultiIndex.from_tuples(
        [(metric, f'h={h}') for metric, h in df.columns],
        names=['metric', 'horizon']
    )
    df = df.sort_index()
    return df


def print_model_comparison(df: pd.DataFrame) -> None:
    """Print run_model_comparison output as a formatted table."""
    metrics  = df.columns.get_level_values('metric').unique()
    horizons = df.columns.get_level_values('horizon').unique()

    header = f"{'Factor':<8} {'Idio':<5}"
    for h in horizons:
        for m in metrics:
            header += f"  {m}/{h}".rjust(16)
    print(header)
    print('─' * len(header))

    for (factor, idio), row in df.iterrows():
        line = f"{factor:<8} {idio:<5}"
        for h in horizons:
            for m in metrics:
                v = row[(m, h)]
                line += f"  {v:>14.4e}"
        print(line)


_METRIC_LABELS = {
    'MAFE_raw'    : 'MAFE — total',
    'MAFE_smooth' : 'MAFE — model',
    'MSFE_raw'    : 'MSFE — total',
    'MSFE_smooth' : 'MSFE — model',
}


def plot_model_comparison(df: pd.DataFrame, period=None, h=None) -> None:
    """
    Heatmap grid from run_model_comparison output.

    When a single horizon is selected (h=1), the 4 metrics are arranged in a
    2×2 grid.  When multiple horizons are shown, the layout is
    rows=metrics × cols=horizons.

    Values are % error vs RW×RW baseline:
    positive (red) = worse than RW×RW, negative (green) = better.

    Parameters
    ----------
    h : int or str, optional
        If provided, only that horizon is shown (e.g. h=1 or h='h=1').
        If None, all horizons in the DataFrame are shown.
    """
    metrics  = ['MAFE_raw', 'MAFE_smooth', 'MSFE_raw', 'MSFE_smooth']
    all_h    = df.columns.get_level_values('horizon').unique().tolist()
    h_key    = f'h={h}' if isinstance(h, int) else h
    horizons = [h_key] if h_key is not None else all_h

    factors = df.index.get_level_values('Factor').unique().tolist()
    idios   = df.index.get_level_values('Idio').unique().tolist()

    single_h = len(horizons) == 1
    if single_h:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = len(metrics), len(horizons)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    title = 'Model comparison — % error vs RW×RW baseline'
    if not single_h:
        pass
    else:
        title += f'  ({horizons[0]})'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for idx_m, metric in enumerate(metrics):
        for idx_h, hh in enumerate(horizons):
            if single_h:
                r, c = divmod(idx_m, 2)
            else:
                r, c = idx_m, idx_h
            ax = axes[r][c]

            try:
                rw_rw = df.loc[('RW', 'RW'), (metric, hh)]
            except KeyError:
                rw_rw = np.nan

            mat = np.full((len(factors), len(idios)), np.nan)
            for fi, fac in enumerate(factors):
                for ii, idio in enumerate(idios):
                    try:
                        v = df.loc[(fac, idio), (metric, hh)]
                        if not np.isnan(rw_rw) and rw_rw != 0:
                            mat[fi, ii] = (v - rw_rw) / rw_rw * 100
                    except KeyError:
                        pass

            abs_max = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1
            im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                           vmin=-abs_max, vmax=abs_max)

            for fi in range(len(factors)):
                for ii in range(len(idios)):
                    v = mat[fi, ii]
                    if np.isnan(v):
                        continue
                    norm  = (v + abs_max) / (2 * abs_max + 1e-12)
                    color = 'white' if norm > 0.8 or norm < 0.2 else 'black'
                    sign  = '+' if v > 0 else ''
                    ax.text(ii, fi, f'{sign}{v:.1f}%', ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')

            ax.set_xticks(range(len(idios)))
            ax.set_xticklabels(idios, fontsize=8)
            ax.set_yticks(range(len(factors)))
            ax.set_yticklabels(factors, fontsize=8)

            panel_title = _METRIC_LABELS.get(metric, metric)
            if not single_h:
                panel_title = f'{hh} — {panel_title}'
            ax.set_title(panel_title, fontsize=9, fontweight='bold')
            ax.set_xlabel('Idiosyncratic model', fontsize=8)
            ax.set_ylabel('Factor model', fontsize=8)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('% vs RW×RW', fontsize=7)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    fname = f'model_comparison_heatmap_{period}.png' if period else 'model_comparison_heatmap.png'
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/{fname}', dpi=160, bbox_inches='tight')
    plt.show()


def decompose_forecast_error_raw(
    target_date,
    ffm: dict,
    X_log: np.ndarray,
    vol_matrices_2026: dict,
    exclude_pairs: list = None,
) -> pd.DataFrame:
    """
    Forecast error vs raw observed data only — for evaluation data that has
    not been smoothed (no ``vol_ffm_2026`` available), so the model/smoothing
    decomposition of ``decompose_forecast_error`` does not apply.

    Parameters
    ----------
    target_date       : str or pd.Timestamp
    ffm               : run_ffm output (keys: 'pairs', 'tau')
    X_log             : np.ndarray (N, M) — forecast curves in log-variance space
                        (first output of reconstruct_forecast)
    vol_matrices_2026 : dict {pair: DataFrame} — raw 5-min log-return matrices
    exclude_pairs     : list of pair names to exclude (e.g. ['usdcnh'])

    Returns
    -------
    pd.DataFrame with columns: MAFE_raw, MSFE_raw  (forecast vs raw, log-variance)
    """
    target_date = pd.Timestamp(target_date)
    pairs        = ffm['pairs']
    _exclude     = set(exclude_pairs) if exclude_pairs else set()
    rows         = []

    for i, pair in enumerate(pairs):
        if pair in _exclude:
            continue
        fc = X_log[i, :]

        f_raw = None
        if pair in vol_matrices_2026:
            df_raw = vol_matrices_2026[pair]
            mask   = pd.to_datetime(df_raw.index).normalize() == target_date
            if mask.any():
                raw_vals = df_raw.loc[mask].values[0].astype(float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    f_raw = np.where(
                        np.isnan(raw_vals) | (raw_vals == 0),
                        np.nan,
                        2.0 * np.log(np.abs(raw_vals)),
                    )

        if f_raw is None:
            rows.append({'pair': pair, 'MAFE_raw': np.nan, 'MSFE_raw': np.nan})
            continue

        rows.append({
            'pair'     : pair,
            'MAFE_raw' : _mafe(fc, f_raw),
            'MSFE_raw' : _msfe(fc, f_raw),
        })

    return pd.DataFrame(rows).set_index('pair')


def run_model_comparison_raw(
    target_dates,
    ffm: dict,
    vol_matrices_2026: dict,
    horizons: list = [1, 2],
    models_factor: list = ['RW', 'AR', 'HAR', 'ARIMA'],
    models_idio: list = ['RW', 'FAR', 'HAR'],
    exclude_pairs: list = None,
    d_arima: float = 1,
    p_max_ar: int = 10,
    K: int = 5,
) -> pd.DataFrame:
    """
    Like ``run_model_comparison``, but for unsmoothed evaluation data: only
    MAFE/MSFE forecast vs raw observed are computed (no model/smoothing
    decomposition, since there is no smoothed series to decompose against).

    See ``run_model_comparison`` for the model definitions and parameters.

    Returns
    -------
    pd.DataFrame — MultiIndex columns (metric, horizon) with metric in
    {'MAFE', 'MSFE'}, rows (Factor, Idio)
    """
    silence      = contextlib.redirect_stdout(io.StringIO())
    U_hat        = ffm['U_hat'][0, :]
    target_dates = list(target_dates)
    if len(target_dates) != len(horizons):
        raise ValueError("target_dates must have one entry per horizon")

    records = []

    for h, target_date in zip(horizons, target_dates):
        print(f'\n── h = {h}  →  target {target_date} ────────────────')

        # ── Factor forecasts ──────────────────────────────────────────────
        fc_factor = {}

        with silence:
            if 'RW' in models_factor:
                fc_rw, _, _ = forecast_factor_rw(U_hat, horizon=[h], plot=False)
                fc_factor['RW'] = float(fc_rw[0])

            if 'AR' in models_factor:
                _, fc_ar, _, _ = forecast_ar(U_hat, h=h, p_max=p_max_ar, plot=False)
                fc_factor['AR'] = float(fc_ar[-1])

            if 'HAR' in models_factor:
                _, fc_har, _ = forecast_har_direct_h(U_hat, h=h)
                fc_factor['HAR'] = float(fc_har)

            if 'ARIMA' in models_factor:
                _, fc_arima, _, _ = forecast_arima(U_hat, h=h, d=d_arima, plot=False)
                fc_factor['ARIMA'] = float(fc_arima[-1])

        # ── Idio forecasts ────────────────────────────────────────────────
        fc_idio = {}

        with silence:
            if 'RW' in models_idio:
                fc_idio['RW'], _, _ = forecast_idio_rw(ffm['Xi_hat'], h=h)

            if 'FAR' in models_idio:
                fc_idio['FAR'] = forecast_all_assets_FAR(
                    ffm['Xi_hat'], ffm['pairs'], h=h, K=K
                )

            if 'HAR' in models_idio:
                fc_idio['HAR'], _, _ = forecast_idio_har(
                    ffm['Xi_hat'], ffm['pairs'], h=h, K=K
                )

        # ── Evaluate all combinations ─────────────────────────────────────
        for mf in models_factor:
            for mi in models_idio:
                if mf not in fc_factor or mi not in fc_idio:
                    continue

                X_log, _ = reconstruct_forecast(fc_factor[mf], fc_idio[mi], ffm)

                df_err = decompose_forecast_error_raw(
                    target_date       = target_date,
                    ffm               = ffm,
                    X_log             = X_log,
                    vol_matrices_2026 = vol_matrices_2026,
                    exclude_pairs     = exclude_pairs,
                )

                records.append({
                    'h'      : h,
                    'Factor' : mf,
                    'Idio'   : mi,
                    'MAFE'   : df_err['MAFE_raw'].mean(),
                    'MSFE'   : df_err['MSFE_raw'].mean(),
                })

                print(f"  {mf:<6} × {mi:<4}  "
                      f"MAFE={records[-1]['MAFE']:.4e}  "
                      f"MSFE={records[-1]['MSFE']:.4e}")

    # ── Build table ───────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.set_index(['Factor', 'Idio', 'h']).unstack('h')
    df.columns = pd.MultiIndex.from_tuples(
        [(metric, f'h={h}') for metric, h in df.columns],
        names=['metric', 'horizon']
    )
    df = df.sort_index()
    return df


_METRIC_LABELS_RAW = {
    'MAFE' : 'MAFE — forecast vs raw',
    'MSFE' : 'MSFE — forecast vs raw',
}


def plot_model_comparison_raw(df: pd.DataFrame, period=None, h=None) -> None:
    """
    Heatmap grid from ``run_model_comparison_raw`` output (unsmoothed data).

    Same convention as ``plot_model_comparison``: values are % error vs
    RW×RW baseline (negative/green = better, positive/red = worse). Only
    two metrics (MAFE, MSFE) are shown since no smoothing decomposition is
    available — rows=metrics, cols=horizons.

    Parameters
    ----------
    h : int or str, optional
        If provided, only that horizon is shown (e.g. h=1 or h='h=1').
        If None, all horizons in the DataFrame are shown.
    """
    metrics  = ['MAFE', 'MSFE']
    all_h    = df.columns.get_level_values('horizon').unique().tolist()
    h_key    = f'h={h}' if isinstance(h, int) else h
    horizons = [h_key] if h_key is not None else all_h

    factors = df.index.get_level_values('Factor').unique().tolist()
    idios   = df.index.get_level_values('Idio').unique().tolist()

    n_rows, n_cols = len(metrics), len(horizons)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    title = 'Model comparison (raw, unsmoothed) — % error vs RW×RW baseline'
    if n_cols == 1:
        title += f'  ({horizons[0]})'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for idx_m, metric in enumerate(metrics):
        for idx_h, hh in enumerate(horizons):
            r, c = idx_m, idx_h
            ax = axes[r][c]

            try:
                rw_rw = df.loc[('RW', 'RW'), (metric, hh)]
            except KeyError:
                rw_rw = np.nan

            mat = np.full((len(factors), len(idios)), np.nan)
            for fi, fac in enumerate(factors):
                for ii, idio in enumerate(idios):
                    try:
                        v = df.loc[(fac, idio), (metric, hh)]
                        if not np.isnan(rw_rw) and rw_rw != 0:
                            mat[fi, ii] = (v - rw_rw) / rw_rw * 100
                    except KeyError:
                        pass

            abs_max = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1
            im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                           vmin=-abs_max, vmax=abs_max)

            for fi in range(len(factors)):
                for ii in range(len(idios)):
                    v = mat[fi, ii]
                    if np.isnan(v):
                        continue
                    norm  = (v + abs_max) / (2 * abs_max + 1e-12)
                    color = 'white' if norm > 0.8 or norm < 0.2 else 'black'
                    sign  = '+' if v > 0 else ''
                    ax.text(ii, fi, f'{sign}{v:.1f}%', ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')

            ax.set_xticks(range(len(idios)))
            ax.set_xticklabels(idios, fontsize=8)
            ax.set_yticks(range(len(factors)))
            ax.set_yticklabels(factors, fontsize=8)

            panel_title = _METRIC_LABELS_RAW.get(metric, metric)
            if n_cols > 1:
                panel_title = f'{hh} — {panel_title}'
            ax.set_title(panel_title, fontsize=9, fontweight='bold')
            ax.set_xlabel('Idiosyncratic model', fontsize=8)
            ax.set_ylabel('Factor model', fontsize=8)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('% vs RW×RW', fontsize=7)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    fname = f'model_comparison_raw_heatmap_{period}.png' if period else 'model_comparison_raw_heatmap.png'
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/{fname}', dpi=160, bbox_inches='tight')
    plt.show()


def plot_comparison_periods(
    df_full: pd.DataFrame,
    df_s2:   pd.DataFrame,
    label_full: str = 'Full 2024–25',
    label_s2:   str = '2025 S2',
    fname: str = 'model_comparison_periods.png',
    h=None,
) -> None:
    """
    Side-by-side heatmap comparing two training periods against a common baseline.

    Layout : rows = metrics (MAFE_raw, MAFE_smooth, MSFE_raw, MSFE_smooth)
             cols = horizons × 2  (label_full | label_s2 for each h)
    Baseline : RW×RW from df_full (same denominator for both periods so values
               are directly comparable).
    Convention: positive (red) = error larger than RW×RW (worse),
               negative (green) = error smaller than RW×RW (better).

    Parameters
    ----------
    h : int or str, optional
        If provided, only the column for that horizon is shown (e.g. h=1 or h='h=1').
        If None, all horizons in the DataFrame are shown.
    """
    import os
    metrics  = ['MAFE_raw', 'MAFE_smooth', 'MSFE_raw', 'MSFE_smooth']
    all_h    = df_full.columns.get_level_values('horizon').unique().tolist()
    h_key    = f'h={h}' if isinstance(h, int) else h
    horizons = [h_key] if h_key is not None else all_h
    factors  = df_full.index.get_level_values('Factor').unique().tolist()
    idios    = df_full.index.get_level_values('Idio').unique().tolist()

    n_rows = len(metrics)
    n_cols = len(horizons) * 2   # two panels per horizon

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f'Model comparison — % error vs RW×RW ({label_full} baseline)\n'
        f'Left: {label_full}   |   Right: {label_s2}   —   red = worse than RW×RW, green = better',
        fontsize=12, fontweight='bold', y=1.01,
    )

    def _build_mat(df, metric, h, rw_rw):
        mat = np.full((len(factors), len(idios)), np.nan)
        for fi, fac in enumerate(factors):
            for ii, idio in enumerate(idios):
                try:
                    v = df.loc[(fac, idio), (metric, h)]
                    if not np.isnan(rw_rw) and rw_rw != 0:
                        mat[fi, ii] = (v - rw_rw) / rw_rw * 100
                except KeyError:
                    pass
        return mat

    def _annotate(ax, mat, abs_max):
        for fi in range(len(factors)):
            for ii in range(len(idios)):
                v = mat[fi, ii]
                if np.isnan(v):
                    continue
                norm  = (v + abs_max) / (2 * abs_max + 1e-12)
                color = 'white' if norm > 0.8 or norm < 0.2 else 'black'
                sign  = '+' if v > 0 else ''
                ax.text(ii, fi, f'{sign}{v:.1f}%', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    for r, metric in enumerate(metrics):
        for hi, h in enumerate(horizons):
            col_full = hi * 2
            col_s2   = hi * 2 + 1

            # Common RW×RW baseline from df_full
            try:
                rw_rw = df_full.loc[('RW', 'RW'), (metric, h)]
            except KeyError:
                rw_rw = np.nan

            mat_full = _build_mat(df_full, metric, h, rw_rw)
            mat_s2   = _build_mat(df_s2,   metric, h, rw_rw)

            # Shared symmetric color scale across both panels
            abs_max = max(
                np.nanmax(np.abs(mat_full)) if not np.all(np.isnan(mat_full)) else 0,
                np.nanmax(np.abs(mat_s2))   if not np.all(np.isnan(mat_s2))   else 0,
            ) or 1

            for ax, mat, lbl in [
                (axes[r][col_full], mat_full, label_full),
                (axes[r][col_s2],   mat_s2,   label_s2),
            ]:
                im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                               vmin=-abs_max, vmax=abs_max)
                _annotate(ax, mat, abs_max)

                ax.set_xticks(range(len(idios)))
                ax.set_xticklabels(idios, fontsize=8)
                ax.set_yticks(range(len(factors)))
                ax.set_yticklabels(factors, fontsize=8)

                if r == 0:
                    ax.set_title(f'{h}\n{lbl}', fontsize=9, fontweight='bold')
                if r == n_rows - 1:
                    ax.set_xlabel('Idio model', fontsize=8)

            axes[r][col_full].set_ylabel(metric, fontsize=9, fontweight='bold')

            cbar = plt.colorbar(im, ax=axes[r][col_s2], fraction=0.046, pad=0.04)
            cbar.set_label('% error vs RW×RW\n(+ = worse, − = better)', fontsize=7)

    plt.tight_layout()
    os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/{fname}', dpi=160, bbox_inches='tight')
    plt.show()


def plot_comparison_raw_vs_smooth(
    df_raw: pd.DataFrame,
    df_smooth: pd.DataFrame,
    label_raw: str = 'FFM on raw data',
    label_smooth: str = 'FFM on smoothed data',
    h=None,
    fname: str = 'model_comparison_raw_vs_smooth.png',
) -> None:
    """
    Side-by-side heatmap comparing two FFM pipelines, both evaluated against
    the same raw observed series: one estimated directly on raw (unsmoothed)
    data (``run_model_comparison_raw`` output) vs one estimated on smoothed
    data (``run_model_comparison`` output — only its ``*_raw`` columns are
    used, since that's the metric comparable to df_raw).

    Layout : rows = metrics (MAFE, MSFE)
             cols = horizons × 2  (label_raw | label_smooth for each h)
    Baseline : a single common reference — RW×RW from df_raw, i.e. the naive
    "raw vs raw" forecast (RW factor + RW idio, FFM estimated on raw data,
    evaluated against raw observed). Both panels are normalised by this same
    denominator (and share a symmetric color scale), so the two pipelines
    are directly comparable — using each pipeline's own RW×RW would put them
    on different absolute scales and make cross-panel comparison meaningless.
    Convention: positive (red) = worse than the raw RW×RW baseline,
               negative (green) = better.

    Parameters
    ----------
    df_raw    : output of run_model_comparison_raw    (metrics: MAFE, MSFE)
    df_smooth : output of run_model_comparison         (metrics: MAFE_raw,
                MAFE_smooth, MSFE_raw, MSFE_smooth)
    h : int or str, optional
        If provided, only that horizon's column pair is shown. If None, all
        horizons present in df_raw are shown.
    """
    metrics_raw    = ['MAFE', 'MSFE']           # df_raw columns
    metrics_smooth = ['MAFE_raw', 'MSFE_raw']   # matching df_smooth columns

    all_h    = df_raw.columns.get_level_values('horizon').unique().tolist()
    h_key    = f'h={h}' if isinstance(h, int) else h
    horizons = [h_key] if h_key is not None else all_h

    factors = df_raw.index.get_level_values('Factor').unique().tolist()
    idios   = df_raw.index.get_level_values('Idio').unique().tolist()

    n_rows = len(metrics_raw)
    n_cols = len(horizons) * 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f'Model comparison — % error vs RW×RW (raw vs raw baseline)\n'
        f'Left: {label_raw}   |   Right: {label_smooth}   —   '
        f'red = worse than RW×RW, green = better',
        fontsize=12, fontweight='bold', y=1.01,
    )

    def _build_mat(df, metric, hh, rw_rw):
        mat = np.full((len(factors), len(idios)), np.nan)
        for fi, fac in enumerate(factors):
            for ii, idio in enumerate(idios):
                try:
                    v = df.loc[(fac, idio), (metric, hh)]
                    if not np.isnan(rw_rw) and rw_rw != 0:
                        mat[fi, ii] = (v - rw_rw) / rw_rw * 100
                except KeyError:
                    pass
        return mat

    def _annotate(ax, mat, abs_max):
        for fi in range(len(factors)):
            for ii in range(len(idios)):
                v = mat[fi, ii]
                if np.isnan(v):
                    continue
                norm  = (v + abs_max) / (2 * abs_max + 1e-12)
                color = 'white' if norm > 0.8 or norm < 0.2 else 'black'
                sign  = '+' if v > 0 else ''
                ax.text(ii, fi, f'{sign}{v:.1f}%', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    for r, (m_raw, m_smooth) in enumerate(zip(metrics_raw, metrics_smooth)):
        for hi, hh in enumerate(horizons):
            col_raw    = hi * 2
            col_smooth = hi * 2 + 1

            # Single common baseline: RW×RW from the raw pipeline (raw vs raw)
            try:
                rw_rw = df_raw.loc[('RW', 'RW'), (m_raw, hh)]
            except KeyError:
                rw_rw = np.nan

            mat_raw    = _build_mat(df_raw,    m_raw,    hh, rw_rw)
            mat_smooth = _build_mat(df_smooth, m_smooth, hh, rw_rw)

            # Shared symmetric color scale across both panels
            abs_max = max(
                np.nanmax(np.abs(mat_raw))    if not np.all(np.isnan(mat_raw))    else 0,
                np.nanmax(np.abs(mat_smooth)) if not np.all(np.isnan(mat_smooth)) else 0,
            ) or 1

            for ax, mat, lbl in [
                (axes[r][col_raw],    mat_raw,    label_raw),
                (axes[r][col_smooth], mat_smooth, label_smooth),
            ]:
                im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                               vmin=-abs_max, vmax=abs_max)
                _annotate(ax, mat, abs_max)

                ax.set_xticks(range(len(idios)))
                ax.set_xticklabels(idios, fontsize=8)
                ax.set_yticks(range(len(factors)))
                ax.set_yticklabels(factors, fontsize=8)

                if r == 0:
                    ax.set_title(f'{hh}\n{lbl}', fontsize=9, fontweight='bold')
                if r == n_rows - 1:
                    ax.set_xlabel('Idio model', fontsize=8)

            axes[r][col_raw].set_ylabel(m_raw, fontsize=9, fontweight='bold')

            cbar = plt.colorbar(im, ax=axes[r][col_smooth], fraction=0.046, pad=0.04)
            cbar.set_label('% vs raw RW×RW\n(+ = worse, − = better)', fontsize=7)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/{fname}', dpi=160, bbox_inches='tight')
    plt.show()


# ── Per-pair model comparison ─────────────────────────────────────────────────

def run_model_comparison_by_pair(
    target_dates,
    ffm: dict,
    vol_matrices_2026: dict,
    vol_ffm_2026: dict,
    horizons: list = [1, 2],
    models_factor: list = ['RW', 'AR', 'HAR', 'ARIMA'],
    models_idio: list = ['RW', 'FAR', 'HAR'],
    exclude_pairs: list = None,
    d_arima: float = 1,
    p_max_ar: int = 10,
    K: int = 5,
) -> pd.DataFrame:
    """
    Same as run_model_comparison but retains per-pair errors instead of averaging.

    Returns
    -------
    pd.DataFrame — MultiIndex rows (pair, Factor, Idio), MultiIndex columns (metric, horizon)
    """
    silence      = contextlib.redirect_stdout(io.StringIO())
    U_hat        = ffm['U_hat'][0, :]
    target_dates = list(target_dates)
    if len(target_dates) != len(horizons):
        raise ValueError("target_dates must have one entry per horizon")

    records = []

    for h, target_date in zip(horizons, target_dates):
        print(f'\n── h = {h}  →  target {target_date} ────────────────')

        fc_factor = {}
        with silence:
            if 'RW' in models_factor:
                fc_rw, _, _ = forecast_factor_rw(U_hat, horizon=[h], plot=False)
                fc_factor['RW'] = float(fc_rw[0])
            if 'AR' in models_factor:
                _, fc_ar, _, _ = forecast_ar(U_hat, h=h, p_max=p_max_ar, plot=False)
                fc_factor['AR'] = float(fc_ar[-1])
            if 'HAR' in models_factor:
                _, fc_har, _ = forecast_har_direct_h(U_hat, h=h)
                fc_factor['HAR'] = float(fc_har)
            if 'ARIMA' in models_factor:
                _, fc_arima, _, _ = forecast_arima(U_hat, h=h, d=d_arima, plot=False)
                fc_factor['ARIMA'] = float(fc_arima[-1])

        fc_idio = {}
        with silence:
            if 'RW' in models_idio:
                fc_idio['RW'], _, _ = forecast_idio_rw(ffm['Xi_hat'], h=h)
            if 'FAR' in models_idio:
                fc_idio['FAR'] = forecast_all_assets_FAR(
                    ffm['Xi_hat'],ffm['pairs'],  h=h, K=K
                )
            if 'HAR' in models_idio:
                fc_idio['HAR'], _, _ = forecast_idio_har(
                    ffm['Xi_hat'], ffm['pairs'], h=h, K=K
                )

        for mf in models_factor:
            for mi in models_idio:
                if mf not in fc_factor or mi not in fc_idio:
                    continue

                X_log, _ = reconstruct_forecast(fc_factor[mf], fc_idio[mi], ffm)

                df_err = decompose_forecast_error(
                    target_date       = target_date,
                    ffm               = ffm,
                    X_log             = X_log,
                    vol_matrices_2026 = vol_matrices_2026,
                    vol_ffm_2026      = vol_ffm_2026,
                    exclude_pairs     = exclude_pairs,
                )

                for pair, row in df_err.iterrows():
                    records.append({
                        'h'           : h,
                        'pair'        : pair,
                        'Factor'      : mf,
                        'Idio'        : mi,
                        'MAFE_raw'    : row['MAFE_total'],
                        'MAFE_smooth' : row['MAFE_model'],
                        'MSFE_raw'    : row['MSFE_total'],
                        'MSFE_smooth' : row['MSFE_model'],
                    })

    df = pd.DataFrame(records)
    df = df.set_index(['pair', 'Factor', 'Idio', 'h']).unstack('h')
    df.columns = pd.MultiIndex.from_tuples(
        [(metric, f'h={h}') for metric, h in df.columns],
        names=['metric', 'horizon']
    )
    df = df.sort_index()
    return df


def print_model_comparison_by_pair(df: pd.DataFrame) -> None:
    """
    Print run_model_comparison_by_pair output as one formatted block per pair.
    """
    pairs    = df.index.get_level_values('pair').unique()
    metrics  = df.columns.get_level_values('metric').unique()
    horizons = df.columns.get_level_values('horizon').unique()

    header = f"{'Factor':<8} {'Idio':<5}"
    for h in horizons:
        for m in metrics:
            header += f"  {m}/{h}".rjust(16)
    sep = '─' * len(header)

    for pair in pairs:
        sub = df.xs(pair, level='pair')
        print(f'\n{"═" * len(header)}')
        print(f'  {pair.upper()}')
        print(f'{"═" * len(header)}')
        print(header)
        print(sep)
        for (factor, idio), row in sub.iterrows():
            line = f"{factor:<8} {idio:<5}"
            for h in horizons:
                for m in metrics:
                    v = row[(m, h)]
                    line += f"  {v:>14.4e}" if not np.isnan(v) else f"  {'—':>14}"
            print(line)


def plot_model_comparison_by_pair(df: pd.DataFrame, period, h=None) -> None:
    """
    For each (metric, horizon): heatmap with pairs on y-axis and
    Factor×Idio model combinations on x-axis.
    Values are % error vs RW×RW baseline per pair:
    positive (red) = error larger than RW×RW (worse), negative (green) = error smaller (better).

    Parameters
    ----------
    h : int or str, optional
        If provided, only the column for that horizon is shown (e.g. h=1 or h='h=1').
        If None, all horizons in the DataFrame are shown.
    """
    metrics  = df.columns.get_level_values('metric').unique().tolist()
    all_h    = df.columns.get_level_values('horizon').unique().tolist()
    h_key    = f'h={h}' if isinstance(h, int) else h
    horizons = [h_key] if h_key is not None else all_h
    pairs    = df.index.get_level_values('pair').unique().tolist()
    fi_map   = df.index.droplevel('pair').unique().tolist()
    combos   = [f"{f}×{i}" for f, i in fi_map]

    n_rows = len(metrics)
    n_cols = len(horizons)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 1.1 * len(combos)) * n_cols,
                 max(3, 0.45 * len(pairs)) * n_rows),
        squeeze=False,
    )
    fig.suptitle('Per-pair model comparison — % error vs RW×RW baseline\n'
                 '(red = worse than RW×RW, green = better)',
                 fontsize=13, fontweight='bold', y=1.01)

    for r, metric in enumerate(metrics):
        for c, h in enumerate(horizons):
            ax = axes[r][c]

            # Build % improvement matrix: rows=pairs, cols=model combos
            mat = np.full((len(pairs), len(fi_map)), np.nan)
            for pi, pair in enumerate(pairs):
                # Get RW×RW baseline for this pair
                try:
                    rw_rw = df.loc[(pair, 'RW', 'RW'), (metric, h)]
                except KeyError:
                    rw_rw = np.nan

                for ci, (fac, idio) in enumerate(fi_map):
                    try:
                        v = df.loc[(pair, fac, idio), (metric, h)]
                        if not np.isnan(rw_rw) and rw_rw != 0:
                            mat[pi, ci] = (v - rw_rw) / rw_rw * 100
                    except KeyError:
                        pass

            abs_max = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1
            im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                           vmin=-abs_max, vmax=abs_max)

            for pi in range(len(pairs)):
                for ci in range(len(fi_map)):
                    v = mat[pi, ci]
                    if np.isnan(v):
                        continue
                    norm = (v + abs_max) / (2 * abs_max + 1e-12)
                    color = 'white' if norm > 0.8 or norm < 0.2 else 'black'
                    sign = '+' if v > 0 else ''
                    ax.text(ci, pi, f'{sign}{v:.1f}%', ha='center', va='center',
                            fontsize=6.5, color=color, fontweight='bold')

            ax.set_xticks(range(len(fi_map)))
            ax.set_xticklabels(combos, fontsize=7, rotation=35, ha='right')
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels(pairs, fontsize=8)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('% error vs RW×RW\n(+ = worse, − = better)', fontsize=7)

            if r == 0:
                ax.set_title(h, fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(metric, fontsize=9, fontweight='bold')
            if r == n_rows - 1:
                ax.set_xlabel('Factor × Idio model', fontsize=8)

    plt.tight_layout()
    import os; os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/model_comparison_by_pair_heatmap_{period}.png', dpi=160, bbox_inches='tight')
    plt.show()


def plot_model_bar_by_pair(
    df: pd.DataFrame,
    factor: str = 'AR',
    idio: str = 'HAR',
    period=None,
    h=None,
    vmax: float | None = None,
) -> None:
    """
    2×2 grid of bar charts (one per metric) showing % error vs RW×RW baseline
    for a single Factor×Idio model, across all FX pairs.
    Green = better than baseline, red = worse.

    Parameters
    ----------
    factor, idio : str
        Model combination to display (default: AR×HAR).
    h : int or str, optional
        Horizon to show (e.g. h=1). Defaults to first available.
    vmax : float, optional
        Symmetric y-axis limit ±vmax. If None, auto-scales per subplot.
    """
    import os

    metrics = ['MAFE_raw', 'MAFE_smooth', 'MSFE_raw', 'MSFE_smooth']
    all_h   = df.columns.get_level_values('horizon').unique().tolist()
    h_key   = f'h={h}' if isinstance(h, int) else h
    horizon = h_key if h_key is not None else all_h[0]

    pairs = df.index.get_level_values('pair').unique().tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=False)
    fig.suptitle(
        f'{factor}×{idio} — % error vs RW×RW baseline  ({horizon})',
        fontsize=13, fontweight='bold',
    )

    for idx_m, metric in enumerate(metrics):
        r, c = divmod(idx_m, 2)
        ax = axes[r][c]

        vals = []
        for pair in pairs:
            try:
                rw_rw = df.loc[(pair, 'RW', 'RW'), (metric, horizon)]
                v     = df.loc[(pair, factor, idio), (metric, horizon)]
                pct   = (v - rw_rw) / rw_rw * 100 if (not np.isnan(rw_rw) and rw_rw != 0) else np.nan
            except KeyError:
                pct = np.nan
            vals.append(pct)

        colors = ['#27AE60' if (not np.isnan(v) and v < 0) else '#C0392B' for v in vals]
        x = np.arange(len(pairs))

        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
        ax.axhline(0, color='black', linewidth=0.9, linestyle='--', zorder=4)

        if vmax is not None:
            ax.set_ylim(-vmax, vmax)

        ax.set_xticks(x)
        ax.set_xticklabels([_pair_label(p) for p in pairs], rotation=90, ha='center', fontsize=6.5)
        ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=10, fontweight='bold')
        ax.set_ylabel('% vs RW×RW', fontsize=8)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    fname = f'model_bar_by_pair_{factor}x{idio}_{period or ""}_{horizon}.png'
    plt.savefig(f'{FIGURES_ROOT}/FORECAST/{fname}', dpi=160, bbox_inches='tight')
    plt.show()


def plot_model_comparison_per_pair_bar(
    df: pd.DataFrame,
    period=None,
    h=None,
    vmax: float | None = None,
) -> None:
    """
    One figure per FX pair: 2×2 grid of bar charts, one per metric.
    Each bar = a Factor×Idio combination, y-axis = % error vs RW×RW baseline.
    Green = better than baseline, red = worse.

    Parameters
    ----------
    h : int or str, optional
        Horizon to show (e.g. h=1). Defaults to first available.
    vmax : float, optional
        Fix symmetric y-axis limits to ±vmax across all subplots (better
        comparability across pairs). If None, each subplot auto-scales.
    """
    import os

    metrics  = ['MAFE_raw', 'MAFE_smooth', 'MSFE_raw', 'MSFE_smooth']
    all_h    = df.columns.get_level_values('horizon').unique().tolist()
    h_key    = f'h={h}' if isinstance(h, int) else h
    horizon  = h_key if h_key is not None else all_h[0]

    pairs  = df.index.get_level_values('pair').unique().tolist()
    fi_map = df.index.droplevel('pair').unique().tolist()
    combos = [f"{f}×{i}" for f, i in fi_map]

    os.makedirs(f'{FIGURES_ROOT}/FORECAST/per_pair', exist_ok=True)

    for pair in pairs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
        fig.suptitle(
            f'{pair} — % error vs RW×RW baseline  ({horizon})',
            fontsize=13, fontweight='bold',
        )

        for idx_m, metric in enumerate(metrics):
            r, c = divmod(idx_m, 2)
            ax = axes[r][c]

            try:
                rw_rw = df.loc[(pair, 'RW', 'RW'), (metric, horizon)]
            except KeyError:
                rw_rw = np.nan

            vals, labels = [], []
            for (fac, idio), combo in zip(fi_map, combos):
                if fac == 'RW' and idio == 'RW':
                    continue
                try:
                    v = df.loc[(pair, fac, idio), (metric, horizon)]
                    pct = (v - rw_rw) / rw_rw * 100 if (not np.isnan(rw_rw) and rw_rw != 0) else np.nan
                except KeyError:
                    pct = np.nan
                vals.append(pct)
                labels.append(combo)

            colors = ['#27AE60' if (not np.isnan(v) and v < 0) else '#C0392B' for v in vals]
            x = np.arange(len(labels))

            bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
            ax.axhline(0, color='black', linewidth=0.9, linestyle='--', zorder=4)

            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                sign = '+' if v > 0 else ''
                offset = 0.4 if v >= 0 else -0.4
                va = 'bottom' if v >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                        f'{sign}{v:.1f}%', ha='center', va=va,
                        fontsize=7, fontweight='bold', color='#2c3e50')

            if vmax is not None:
                ax.set_ylim(-vmax, vmax)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
            ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=10, fontweight='bold')
            ax.set_ylabel('% vs RW×RW', fontsize=8)
            ax.grid(axis='y', alpha=0.3, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        fname = f'per_pair_{pair}_{period or ""}_{horizon}.png'
        plt.savefig(f'{FIGURES_ROOT}/FORECAST/per_pair/{fname}', dpi=160, bbox_inches='tight')
        plt.show()


# =============================================================================
# THESIS PLOTS
# =============================================================================

_THESIS_COLORS = [
    '#2471A3',  # steel blue
    '#E67E22',  # orange
    '#27AE60',  # green
    '#8E44AD',  # purple
    '#C0392B',  # red
    '#16A085',  # teal
]



def _pair_label(pair: str) -> str:
    """'eurusd' → 'EUR/USD'"""
    p = pair.upper()
    return p[:3] + '/' + p[3:]


def plot_idio_forecast_comparison(
    ffm: dict,
    forecasts_dict: dict,
    pairs_to_plot: list = None,
    n_last_obs: int = 10,
    h: int = 1,
    figsize: tuple = None,
    save_path: str = None,
) -> None:
    """
    Compare idiosyncratic component forecasts across models for selected pairs.

    Plots the h-step-ahead idiosyncratic forecast curves from multiple models on
    the same panel, together with the last ``n_last_obs`` observed curves (light
    grey) and 95 % pointwise confidence bands. The y-axis is shared across all
    panels to facilitate cross-pair comparison.

    Parameters
    ----------
    ffm : dict
        Output of ``run_ffm``. Must contain ``'Xi_hat'`` (N, T, n_tau),
        ``'pairs'`` and ``'tau'``.
    forecasts_dict : dict
        ``{label: (fc, ci_low, ci_high)}`` where ``fc`` is (N, n_tau) and
        ``ci_low`` / ``ci_high`` are (N, n_tau) or ``None``.
    pairs_to_plot : list[str], optional
        Pairs to display.  Defaults to
        ``['eurusd', 'audjpy', 'eurchf', 'usdtry']``.
    n_last_obs : int
        Number of last observed curves overlaid in grey.  Default 10.
    h : int
        Forecast horizon label.  Default 1.
    figsize : tuple, optional
        Figure size.  Defaults to ``(13, 10)``.
    save_path : str, optional
        File path for the saved figure.  Defaults to
        ``f'{FIGURES_ROOT}/FORECAST/idio_forecast_comparison_h{h}.pdf'``.
    """
    if pairs_to_plot is None:
        pairs_to_plot = ['eurusd', 'audjpy', 'eurchf', 'usdtry']

    Xi_hat     = ffm['Xi_hat']   # (N, T, n_tau)
    pairs_name = [p.lower() for p in ffm['pairs']]
    tau        = ffm['tau']

    pair_idx = {}
    for p in pairs_to_plot:
        pl = p.lower()
        if pl not in pairs_name:
            raise ValueError(f"Pair '{p}' not found in ffm['pairs'].")
        pair_idx[p] = pairs_name.index(pl)

    # ── Shared y-axis limits across all four panels ───────────────────────────
    all_vals = []
    for p, idx in pair_idx.items():
        all_vals.append(Xi_hat[idx, -n_last_obs:, :].ravel())
        for _, (fc, ci_l, ci_h) in forecasts_dict.items():
            all_vals.append(fc[idx].ravel())
            if ci_l is not None:
                all_vals.append(ci_l[idx].ravel())
            if ci_h is not None:
                all_vals.append(ci_h[idx].ravel())
    cat    = np.concatenate(all_vals)
    spread = cat.max() - cat.min()
    ymin   = cat.min() - 0.07 * spread
    ymax   = cat.max() + 0.07 * spread

    # ── Layout ────────────────────────────────────────────────────────────────
    ncols  = 2
    nrows  = int(np.ceil(len(pairs_to_plot) / ncols))
    fw, fh = figsize if figsize else (13, 5.5 * nrows)
    colors = _THESIS_COLORS[:len(forecasts_dict)]

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh),
                             sharey=True, sharex=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx_p, pair in enumerate(pairs_to_plot):
        r, c = divmod(idx_p, ncols)
        ax   = axes[r, c]
        i    = pair_idx[pair]

        # ── Last n_last_obs observed curves ───────────────────────────────
        recent = Xi_hat[i, -n_last_obs:, :]
        for j in range(n_last_obs):
            alpha_j = 0.12 + 0.35 * (j / max(n_last_obs - 1, 1))
            ax.plot(tau, recent[j], color='#AAAAAA', linewidth=0.8,
                    alpha=alpha_j,
                    label=rf'Last {n_last_obs} observations' if j == 0 else None,
                    zorder=1)

        # ── Forecast curves + 95 % CI ─────────────────────────────────────
        for (label, (fc, ci_l, ci_h)), color in zip(forecasts_dict.items(), colors):
            ax.plot(tau, fc[i], color=color, linewidth=2.0,
                    label=label, zorder=3)
            if ci_l is not None and ci_h is not None:
                ax.fill_between(tau, ci_l[i], ci_h[i],
                                color=color, alpha=0.13, zorder=2)

        ax.set_title(_pair_label(pair), fontsize=13, fontweight='bold', pad=7)
        ax.set_ylim(ymin, ymax)
        if r == nrows - 1:
            ax.set_xlabel(r'Intraday time $\tau$', fontsize=10)
        if c == 0:
            ax.set_ylabel(r'$\hat{\xi}_i(\tau)$', fontsize=12)
        ax.grid(True, alpha=0.20, linestyle='--', linewidth=0.6)
        if idx_p == 0:
            ax.legend(fontsize=9, framealpha=0.92,
                      loc='upper right', edgecolor='#CCCCCC')

    # ── Hide unused axes ──────────────────────────────────────────────────
    for j in range(len(pairs_to_plot), nrows * ncols):
        axes.flat[j].set_visible(False)

    has_ci  = any(v[1] is not None for v in forecasts_dict.values())
    ci_note = r' — shaded bands: 95\% pointwise CI' if has_ci else ''
    fig.suptitle(
        rf'Idiosyncratic Component — $h={h}$ Forecast Comparison'
        + ci_note,
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    import os
    os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    path = save_path or f'{FIGURES_ROOT}/FORECAST/idio_forecast_comparison_h{h}.pdf'
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.show()


def plot_reconstructed_forecast(
    ffm: dict,
    models_dict: dict,
    target_date,
    vol_ffm_2026: dict = None,
    vol_matrices_2026: dict = None,
    pairs_to_plot: list = None,
    h: int = 1,
    figsize: tuple = None,
    save_path: str = None,
) -> None:
    """
    Plot reconstructed total volatility forecasts against observed data (log-space).

    For each model combination in ``models_dict``, reconstructs the full
    log-variance curve  x̂_{i,T+h}(τ) = μ̂_i + σ̂_i [B̂_i f̂_{T+h} + ξ̂_{i,T+h}]
    using ``reconstruct_forecast``, then overlays the smoothed and raw observed
    curves for the target date.  Everything is kept in log-space (no exponential
    transform), making the comparison directly interpretable in terms of the
    model's functional decomposition.

    Parameters
    ----------
    ffm : dict
        Output of ``run_ffm``.  Must contain ``'B_hat'``, ``'sigma'``, ``'mu'``,
        ``'pairs'``, ``'tau'``.
    models_dict : dict
        ``{label: (fc_factor, fc_idio)}`` where ``fc_factor`` is a scalar or
        array (r,) — factor forecast at horizon h — and ``fc_idio`` is
        (N, n_tau) — pre-computed idiosyncratic forecast.
    target_date : str or pd.Timestamp
        Date of the observation to compare against (e.g. ``'2026-01-02'``).
    vol_ffm_2026 : dict, optional
        ``{pair: DataFrame}`` — smoothed log-variance curves (output of
        ``smoothing_for_ffm``).  Index contains timestamps; values are
        log(r²) per intraday interval.  Pass ``None`` to skip the smoothed
        observed series entirely (e.g. when working without smoothing).
    vol_matrices_2026 : dict, optional
        ``{pair: DataFrame}`` — raw 5-min log-return matrices.
        Index contains 5-min timestamps; each row is one trading day.
        Pass ``None`` to skip the raw observed series.
    pairs_to_plot : list[str], optional
        Pairs to display.  Defaults to
        ``['eurusd', 'audjpy', 'eurchf', 'usdtry']``.
    h : int
        Forecast horizon label used in titles and file name.  Default 1.
    figsize : tuple, optional
        Figure size.  Defaults to ``(13, 10)``.
    save_path : str, optional
        File path for the saved figure.  Defaults to
        ``f'{FIGURES_ROOT}/FORECAST/reconstructed_forecast_{date}_h{h}.pdf'``.
    """
    if pairs_to_plot is None:
        pairs_to_plot = ['eurusd', 'audjpy', 'eurchf', 'usdtry']

    target_date = pd.Timestamp(target_date)
    pairs_name  = [p.lower() for p in ffm['pairs']]
    tau         = ffm['tau']

    # ── Index requested pairs ─────────────────────────────────────────────────
    pair_idx = {}
    for p in pairs_to_plot:
        pl = p.lower()
        if pl not in pairs_name:
            raise ValueError(f"Pair '{p}' not found in ffm['pairs'].")
        pair_idx[p] = pairs_name.index(pl)

    # ── Pre-compute reconstructed log-curves for each model ───────────────────
    # reconstruct_forecast returns (X_hat_log (N,M), X_hat_abs (N,M))
    reconstructed = {}
    for label, (fc_factor, fc_idio) in models_dict.items():
        X_log, _ = reconstruct_forecast(fc_factor, fc_idio, ffm)
        reconstructed[label] = X_log   # (N, n_tau)

    # ── Layout ────────────────────────────────────────────────────────────────
    ncols  = 2
    nrows  = int(np.ceil(len(pairs_to_plot) / ncols))
    fw, fh = figsize if figsize else (13, 5.5 * nrows)
    colors = _THESIS_COLORS[:len(models_dict)]

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx_p, pair in enumerate(pairs_to_plot):
        r, c = divmod(idx_p, ncols)
        ax   = axes[r, c]
        i    = pair_idx[pair]
        pl   = pair.lower()

        # ── Last 7 observed smoothed curves (log-space) ───────────────────
        if vol_ffm_2026 is not None and pl in vol_ffm_2026:
            df_sm_all = vol_ffm_2026[pl].copy()
            df_sm_all.index = pd.to_datetime(df_sm_all.index).normalize()
            past_curves = df_sm_all[df_sm_all.index < target_date].iloc[-7:]
            n_past = len(past_curves)
            for j, (_, row) in enumerate(past_curves.iterrows()):
                alpha_j = 0.12 + 0.35 * (j / max(n_past - 1, 1))
                ax.plot(tau, row.values, color='#AAAAAA', linewidth=0.8,
                        alpha=alpha_j,
                        label='Last 7 obs.' if j == 0 else None,
                        zorder=1)

        # ── Raw observed (log-space: 2·log|r| ) ────────────────────────────
        if vol_matrices_2026 is not None and pl in vol_matrices_2026:
            df_raw = vol_matrices_2026[pl]
            mask   = pd.to_datetime(df_raw.index).normalize() == target_date
            if mask.any():
                raw_vals = df_raw.loc[mask].values[0].astype(float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    raw_log = np.where(
                        np.isnan(raw_vals) | (raw_vals == 0),
                        np.nan,
                        2.0 * np.log(np.abs(raw_vals)),
                    )
                ax.plot(tau, raw_log, color='#85C1E9', linewidth=1.0,
                        alpha=0.55, linestyle='-',
                        label='Raw observed', zorder=2)

        # ── Smoothed observed (log-space: already log(r²)) ─────────────────
        if vol_ffm_2026 is not None and pl in vol_ffm_2026:
            df_sm    = vol_ffm_2026[pl]
            mask_sm  = pd.to_datetime(df_sm.index).normalize() == target_date
            if mask_sm.any():
                smooth_log = df_sm.loc[mask_sm].values[0]
                ax.plot(tau, smooth_log, color='#1E8449', linewidth=2.0,
                        linestyle='-', alpha=0.9,
                        label='Smoothed observed', zorder=3)

        # ── Reconstructed forecasts (RW baseline drawn behind the rest) ────
        for idx_m, ((label, X_log), color) in enumerate(zip(reconstructed.items(), colors)):
            z = 4 if 'RW' in label.upper() else 5 + idx_m
            ax.plot(tau, X_log[i], color=color, linewidth=1.8,
                    linestyle='--', label=label, zorder=z)

        ax.set_title(_pair_label(pair), fontsize=13, fontweight='bold', pad=7)
        if r == nrows - 1:
            ax.set_xlabel(r'Intraday time $\tau$', fontsize=10)
        if c == 0:
            ax.set_ylabel(r'$\log r^2(\tau)$', fontsize=12)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.20, linestyle='--', linewidth=0.6)
        if idx_p == 0:
            ax.legend(fontsize=9, framealpha=0.92,
                      loc='upper right', edgecolor='#CCCCCC')

    # ── Hide unused axes ──────────────────────────────────────────────────
    for j in range(len(pairs_to_plot), nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.suptitle(
        rf'Reconstructed $h={h}$ Forecast vs Observed — {target_date.date()}',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    import os
    os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    path = (save_path
            or f'{FIGURES_ROOT}/FORECAST/reconstructed_forecast_{target_date.date()}_h{h}.pdf')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.show()


# =============================================================================
# DEGRADATION UNDER A FROZEN LOADING DIRECTION (Chapter 5 rolling grid)
# =============================================================================

def forecast_degradation_frozen_direction(
    data: dict,
    window_size: int = 6,
    step: int = 30,
    start_date: str = '2014-01-01',
    end_date: str = '2025-07-01',
    r_forced: int = 1,
) -> pd.DataFrame:
    """
    One-step degradation of the forecast when the leading loading
    direction is frozen at the January–June 2014 reference window
    instead of being re-estimated on each window.

    Uses the same rolling grid as ffm_stability.grassman_rolling_window_ref
    (window_size months, step days, reference window =
    [start_date, start_date + window_size]). For every subsequent window w:
      - 'fresh'  : leading direction B_hat estimated on w itself; score =
                   the window's own factor û_{1,t} (U_hat).
      - 'frozen' : leading direction of the reference window; score
                   obtained by projecting w's data onto that fixed
                   direction: ŝ_t = ⟨Y_t, B_ref⟩ / ⟨B_ref, B_ref⟩
                   (reduces exactly to û_{1,t} when B_ref = B_hat).
    In both cases the score is carried one step forward as a random walk;
    the mean curve, scale and idiosyncratic forecast are shared between
    the two forecasts — only the loading direction differs. MSFE is
    evaluated, for both forecasts alike, against the smoothed observed
    curve on the first day outside the window.

    Parameters
    ----------
    data        : dict {pair: DataFrame(days × M)} — smoothed log-variance
                  curves (e.g. vol_ffm_all)
    window_size : window length in months
    step        : grid step in days
    start_date  : start of the reference window
    end_date    : end of the grid
    r_forced    : number of factors (the "leading" direction → 1)

    Returns
    -------
    pd.DataFrame with columns: start, end, target_date, n_pairs,
    MSFE_fresh, MSFE_frozen, delta (= MSFE_frozen - MSFE_fresh).
    """
    run_ffm_r1 = partial(run_ffm, r_forced=r_forced, silent=True)

    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)

    # ── Reference window [start_date, start_date+window_size] → frozen direction ──
    ref_end    = start_ts + pd.DateOffset(months=window_size)
    subset_ref = {}
    for pair, df in data.items():
        first_date = pd.to_datetime(df.index).min()
        if first_date < ref_end:
            mask = (pd.to_datetime(df.index) >= start_ts) & \
                   (pd.to_datetime(df.index) <= ref_end)
            subset_ref[pair] = df.loc[mask]

    ffm_ref    = run_ffm_r1(subset_ref)
    B_ref_dict  = dict(zip(ffm_ref['pairs'], ffm_ref['B_hat'][:, 0, :]))   # {pair: (M,)}
    mu_ref_dict = dict(zip(ffm_ref['pairs'], ffm_ref['mu']))
    print(f"Reference window (frozen direction): {start_ts.date()} → {ref_end.date()} "
          f"| N={len(subset_ref)} pairs")

    # ── Master trading-day calendar, to locate "the first day outside the window" ──
    all_days = pd.DatetimeIndex(
        sorted(set().union(*[pd.to_datetime(df.index) for df in data.values()]))
    )

    results = []
    current_start = ref_end

    while current_start < end_ts:
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

        future_days = all_days[all_days > current_end]
        if len(subset) == 0 or len(future_days) == 0:
            current_start += pd.DateOffset(days=step)
            continue
        target_date = future_days.min()

        ffm_w   = run_ffm_r1(subset, mu_ref=mu_ref_dict)
        pairs_w = [p for p in ffm_w['pairs'] if p in B_ref_dict]
        if len(pairs_w) == 0:
            current_start += pd.DateOffset(days=step)
            continue
        idx_w = [ffm_w['pairs'].index(p) for p in pairs_w]

        B_ref = np.stack([B_ref_dict[p] for p in pairs_w], axis=0)   # (n, M)
        Y_w   = ffm_w['Y'][idx_w]                                    # (n, T, M)

        score_fresh  = ffm_w['U_hat'][0, :]
        score_frozen = np.einsum('itm,im->t', Y_w, B_ref) / np.sum(B_ref ** 2)

        fc_fresh,  _, _ = forecast_factor_rw(score_fresh,  horizon=[1], plot=False)
        fc_frozen, _, _ = forecast_factor_rw(score_frozen, horizon=[1], plot=False)
        fc_idio,   _, _ = forecast_idio_rw(ffm_w['Xi_hat'], h=1)

        X_fresh_log, _ = reconstruct_forecast(fc_fresh[0], fc_idio, ffm_w)

        ffm_frozen = dict(ffm_w)
        ffm_frozen['B_hat'] = ffm_w['B_hat'].copy()
        ffm_frozen['B_hat'][idx_w, 0, :] = B_ref
        X_frozen_log, _ = reconstruct_forecast(fc_frozen[0], fc_idio, ffm_frozen)

        msfe_fresh_pairs, msfe_frozen_pairs = [], []
        for i, pair in zip(idx_w, pairs_w):
            df_pair = data[pair]
            mask_t  = pd.to_datetime(df_pair.index) == target_date
            if not mask_t.any():
                continue
            actual = df_pair.loc[mask_t].values[0]
            msfe_fresh_pairs.append(np.mean((X_fresh_log[i]  - actual) ** 2))
            msfe_frozen_pairs.append(np.mean((X_frozen_log[i] - actual) ** 2))

        if len(msfe_fresh_pairs) > 0:
            msfe_fresh  = float(np.mean(msfe_fresh_pairs))
            msfe_frozen = float(np.mean(msfe_frozen_pairs))
            results.append({
                'start'       : current_start.date(),
                'end'         : current_end.date(),
                'target_date' : target_date.date(),
                'n_pairs'     : len(msfe_fresh_pairs),
                'MSFE_fresh'  : msfe_fresh,
                'MSFE_frozen' : msfe_frozen,
                'delta'       : msfe_frozen - msfe_fresh,
            })
            print(f"{current_start.date()} → {current_end.date()} | "
                  f"target={target_date.date()} | "
                  f"MSFE_fresh={msfe_fresh:.4e}  MSFE_frozen={msfe_frozen:.4e}  "
                  f"Δ={msfe_frozen - msfe_fresh:+.4e}")

        current_start += pd.DateOffset(days=step)

    return pd.DataFrame(results)


def plot_forecast_degradation(
    df: pd.DataFrame,
    window_size: int = 6,
    step: int = 30,
    figsize: tuple = (12, 6),
    save_path: str = None,
) -> None:
    """
    Plot the one-step degradation Δ_w = MSFE_frozen(w) − MSFE_fresh(w)
    across the rolling grid, from forecast_degradation_frozen_direction.
    """
    starts = pd.to_datetime(df['start'])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )

    ax1.plot(starts, df['MSFE_fresh'],  color='tab:blue',
             label=r"$\hat{b}_w(\tau)$ — up-to-date direction")
    ax1.plot(starts, df['MSFE_frozen'], color='tab:red',
             label=r"$\hat{b}_{\mathrm{ref}}(\tau)$ — reference direction (2014)")
    ax1.set_ylabel(r'One-step MSFE ($\log r^2$)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    colors = np.where(df['delta'] >= 0, 'tab:red', 'tab:green')
    ax2.bar(starts, df['delta'], width=step * 0.8, color=colors)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_ylabel(r'$\Delta_w = \mathrm{MSFE}_{\mathrm{ref}} - \mathrm{MSFE}_w$')
    ax2.set_xlabel('Window start')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        r'Forecast degradation under a frozen direction — $\hat{b}_{\mathrm{ref}}$ vs $\hat{b}_w$'
        + f'\nWindow: {window_size} months, step: {step} days', fontsize=13,
    )
    plt.tight_layout()

    import os
    os.makedirs(f'{FIGURES_ROOT}/FORECAST', exist_ok=True)
    plt.savefig(
        save_path or f'{FIGURES_ROOT}/FORECAST/degradation_frozen_vs_fresh_{window_size}_{step}.png',
        dpi=180, bbox_inches='tight',
    )
    plt.show()

