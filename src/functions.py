import os
import zipfile
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# ─────────────────────────────────────────── DATA PROCESSING ───────────────────────────────────────────

def treatment_data(rate_path: str):
    """
    Load and concatenate the HistData CSV files of one currency pair (e.g. 'EURUSD').
    - Extract every ZIP archive found in the folder
    - Concatenate all CSV files matching DAT_MT_<PAIR>_M1_*.csv
    - Return a pandas DataFrame indexed by datetime
    """
    
    # 1. Extract ZIP archives in rate_path
    for file in os.listdir(rate_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(rate_path, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(rate_path)

    # 2. Collect all CSV files
    pattern = os.path.join(rate_path, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No CSV file found in {rate_path}")
        return pd.DataFrame()

    # 3. Read and concatenate the CSV files
    dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            names=["date", "time", "open", "high", "low", "close", "volume"]
        )
        dfs.append(df)

    fx = pd.concat(dfs, ignore_index=True)

    # 4. Parse datetime and sort
    fx["datetime"] = pd.to_datetime(fx["date"] + " " + fx["time"], infer_datetime_format=True)
    fx = fx.sort_values("datetime")

    # 5. Drop exact duplicates on date + time (keep the first occurrence)
    fx = fx.drop_duplicates(subset=["date", "time"], keep="first")

    # 6. Use datetime as index
    fx = fx.set_index("datetime")
    return fx


def intraday_to_daily_matrix(df):
    """
    Reshape an intraday DataFrame into a (day × time) matrix.
    
    df must contain the columns: 'date', 'time', 'close'
    - date : format 'YYYY-MM-DD'
    - time : format 'HH:MM:SS' or 'HH:MM'
    """
    # Combine date and time into a datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    # Extract the date and the time of day
    df['day'] = df['datetime'].dt.date
    df['hour_minute'] = df['datetime'].dt.strftime('%H:%M')
    
    # Pivot: rows = days, columns = times, values = close
    df_pivot = df.pivot(index='day', columns='hour_minute', values='close')
    
    # Sort columns chronologically
    df_pivot = df_pivot.sort_index(axis=1)
    
    return df_pivot

def intraday_volatility_matrix(df):
    """
    Compute the (day × time) matrix of intraday volatility.
    
    Returns |r_{t,k}| = |log(P_{5k}) - log(P_{5(k-1)})| for each
    5-min interval k of day t.
    
    NaN on the first interval of each day (overnight return excluded).
    """

    data = df.copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data['log_price'] = np.log(data['close'])

    # Subsample to 5 min
    data_5min = data[data.index.minute % 5 == 0].copy()

    # 5-min log-returns (diff(1) on data already at 5 min)
    data_5min['returns'] = data_5min['log_price'].diff(1)

    # Drop the overnight return (first bar of each day)
    first_bar = data_5min.groupby(data_5min.index.date).head(1).index
    data_5min.loc[first_bar, 'returns'] = np.nan

    data_5min['returns_abs'] = data_5min['returns'].abs()

    # Day × time matrix
    data_5min['day']  = data_5min.index.date
    data_5min['time'] = data_5min.index.time

    matrix = data_5min.pivot(index='day', columns='time',
                              values='returns_abs')
    return matrix

# ─────────────────────────────────────────── COMPUTATIONS ──────────────────────────────────────────────

def compute_log_return(fx_matrix):
    """
    Compute log returns for a day × minute matrix.

    Parameters
    ----------
    fx_matrix : pandas.DataFrame
        Index = days, columns = minutes (HH:MM), values = exchange rates

    Returns
    -------
    pandas.DataFrame
        Log return matrix of the same shape as fx_matrix
    """
    log_return_matrix = np.log(fx_matrix / fx_matrix.shift(axis=1))
    log_return_matrix.index = pd.to_datetime(log_return_matrix.index)
    log_return_matrix = log_return_matrix.sort_index()
    return log_return_matrix

def mean_curve_period(log_ret_matrix: pd.DataFrame, start_date, end_date) -> pd.Series:
    """
    Compute the mean curve over a given period.
    
    Parameters
    ----------
    log_ret_matrix : pd.DataFrame
        Day × minute DataFrame of log-returns.
        The index must contain dates (datetime or convertible).
    start_date : str or pd.Timestamp
        Start date (inclusive)
    end_date : str or pd.Timestamp
        End date (inclusive)
        
    Returns
    -------
    pd.Series
        Mean curve (average log-return) for each minute over the period.
    """
    
    # Make sure the index is a DatetimeIndex
    if not pd.api.types.is_datetime64_any_dtype(log_ret_matrix.index):
        log_ret_matrix.index = pd.to_datetime(log_ret_matrix.index)
    
    # Select the period
    selected = log_ret_matrix.loc[start_date:end_date]
    
    # Check that at least one day is selected
    if selected.empty:
        raise ValueError("No day in the selected period")
    
    # Mean curve (average over days)
    mean_curve = selected.mean(axis=0)
    
    return mean_curve

def covariance_function(log_returns_matrix):
    """
    log_returns_matrix : array (n_days, n_minutes)
    returns : covariance matrix (n_minutes, n_minutes)
    """

    X = log_returns_matrix.copy()
    
    # Mean curve (ignore NaN)
    mean_curve = np.nanmean(X, axis=0)

    X_centered = X - mean_curve
    
    # Replace NaN with 0 after centering (important)
    X_centered = np.nan_to_num(X_centered)
    
    n = X_centered.shape[0]
    
    # Covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
    
    return cov_matrix

def lag1_covariance(data):
    """
    Compute the lag-1 covariance matrix of the columns of a data matrix.
    
    Parameters
    ----------
    data : np.ndarray
        T x N matrix (T observations, N variables)
    
    Returns
    -------
    cov_lag1 : np.ndarray
        N x N lag-1 covariance matrix
        cov_lag1[i,j] = Cov(X_i(t), X_j(t+1))
    """
    data = np.asarray(data)
    
    # Drop the last row for t and the first row for t+1
    X_t   = data[:-1, :]  # 0 to T-2
    X_tp1 = data[1:, :]   # 1 to T-1
    
    # Center the data
    X_t_centered   = X_t - np.nanmean(X_t, axis=0)
    X_tp1_centered = X_tp1 - np.nanmean(X_tp1, axis=0)

    X_t_centered   = np.nan_to_num(X_t_centered)
    X_tp1_centered = np.nan_to_num(X_tp1_centered)
    
    # Covariance lag-1
    cov_lag1 = (X_t_centered.T @ X_tp1_centered) / (X_t_centered.shape[0] - 1)
    
    return cov_lag1

def cross_covariance_function(log_returns_matrix_1, log_returns_matrix_2):
    """
    log_returns_matrix_1 : array (n_days, n_minutes)  ex: EUR/USD
    log_returns_matrix_2 : array (n_days, n_minutes)  ex: USD/JPY
    
    returns : cross-covariance matrix (n_minutes, n_minutes)
    """

    X = log_returns_matrix_1.copy()
    Y = log_returns_matrix_2.copy()

    # Mean curves
    mean_X = np.nanmean(X, axis=0)
    mean_Y = np.nanmean(Y, axis=0)

    # Center
    X_centered = X - mean_X
    Y_centered = Y - mean_Y

    # Replace NaN
    X_centered = np.nan_to_num(X_centered)
    Y_centered = np.nan_to_num(Y_centered)

    n = X_centered.shape[0]

    # Cross covariance
    cross_cov_matrix = (X_centered.T @ Y_centered) / (n - 1)

    return cross_cov_matrix

def cross_covariance_lag1(data1, data2):
    """
    Compute the lag-1 cross-covariance matrix between two matrices.

    Cov(X_i(t), Y_j(t+1))

    Parameters
    ----------
    data1 : np.ndarray
        T x N matrix (pair 1)

    data2 : np.ndarray
        T x N matrix (pair 2)

    Returns
    -------
    cross_cov : np.ndarray
        N x N matrix
    """

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # alignement temporel
    X_t = data1[:-1, :]
    Y_tp1 = data2[1:, :]

    # centrer
    X_centered = X_t - np.nanmean(X_t, axis=0)
    Y_centered = Y_tp1 - np.nanmean(Y_tp1, axis=0)

    X_centered = np.nan_to_num(X_centered)
    Y_centered = np.nan_to_num(Y_centered)

    # cross covariance lag 1
    cross_cov = (X_centered.T @ Y_centered) / (X_centered.shape[0] - 1)

    return cross_cov


# ─────────────────────────────────────────── PLOTS ─────────────────────────────────────────────────────
def plot_distribution(log_returns, title):

    log_returns_values = log_returns.values.flatten()

    log_returns_values = log_returns_values[~np.isnan(log_returns_values)]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(log_returns_values, bins=500, density=True, 
            alpha=0.5, color='steelblue', label='Empiric')

    kde = stats.gaussian_kde(log_returns_values)
    x = np.linspace(log_returns_values.min(), log_returns_values.max(), 1000)
    ax.plot(x, kde(x), 'b-', linewidth=2, label='KDE')

    mu, sigma = log_returns_values.mean(), log_returns_values.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 
            'r--', linewidth=2, label='Theorical Normal')

    stats_str = f'μ={mu:.5f}, σ={sigma:.5f}, skew={stats.skew(log_returns_values):.2f}, kurt={stats.kurtosis(log_returns_values):.2f}'

    ax.set_title(f'{title} | {stats_str}')

    ax.legend()
    return fig, ax

def meancurve_plot_comparison(log_returns, title='Global Mean Intraday Curve with ±2 Std Band'):

    logreturns_mc_monthly = log_returns.groupby(log_returns.index.to_period("M")).mean()

    global_mean = np.nanmean(logreturns_mc_monthly.values, axis=0)
    global_std  = np.nanstd(logreturns_mc_monthly.values, axis=0)

    upper_band = global_mean + 2 * global_std
    lower_band = global_mean - 2 * global_std

    mean_df = pd.DataFrame(logreturns_mc_monthly.values, columns=logreturns_mc_monthly.columns)

    plt.figure(figsize=(10,5))

    for i in range(len(mean_df)):
        plt.plot(mean_df.iloc[i], color='gray', alpha=0.3)

    plt.plot(global_mean, linewidth=3)

    plt.fill_between(range(len(global_mean)),lower_band,upper_band,alpha=0.3)

    plt.xlabel("Time of Day ")
    plt.ylabel("Mean log return")
    plt.title(title)
    n_points = len(global_mean)
    ticks_per_hour = n_points // 24
    hour_ticks = np.arange(0, n_points, ticks_per_hour)
    hour_labels = [f"{h}:00" for h in range(24)]
    plt.xticks(hour_ticks, hour_labels, rotation=90)
    plt.show()

def plot_heatmap(df_matrix,figsize=(10, 5),cmap="viridis",vmin=None,vmax=None,title="Intraday Heatmap",ytick_step=30, robust=True, start_date=None, end_date=None, drop_weekends=False):
    """
    Display a heatmap for a day × minute matrix.

    Parameters
    ----------
    df_matrix : pandas.DataFrame
        Index = days, columns = minutes (HH:MM), values = exchange rates
    figsize : tuple
        Size of the figure
    cmap : str
        Colormap
    vmin : float or None
        Minimum value for color scaling
    vmax : float or None
        Maximum value for color scaling
    title : str
        Title of the heatmap
    ytick_step : int
        Show one day label every ytick_step days
    start_date : str or None
        Start date for filtering (inclusive), format 'YYYY-MM-DD'
    end_date : str or None
        End date for filtering (inclusive), format 'YYYY-MM-DD'
    drop_weekends : bool
        If True, remove Saturday and Sunday rows from the heatmap
    """

    if drop_weekends:
        idx_dt = pd.to_datetime(df_matrix.index)
        df_matrix = df_matrix[idx_dt.dayofweek < 5]

    if start_date is not None or end_date is not None:
        idx_dt = pd.to_datetime(df_matrix.index)
        mask = pd.Series(True, index=df_matrix.index)
        if start_date is not None:
            mask &= idx_dt >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= idx_dt <= pd.Timestamp(end_date)
        df_matrix = df_matrix[mask]

    values = df_matrix.values.flatten()
    values = values[~np.isnan(values)]

    if vmin is None or vmax is None:
        if robust:
            vmin = np.quantile(values, 0.01)
            vmax = np.quantile(values, 0.99)
        else:
            vmin = np.min(values)
            vmax = np.max(values)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            "label": "Exchange Rate",
            "shrink": 0.6
        },
        xticklabels=60,
        yticklabels=False  # disable automatic labels
    )

    # Manually set y-ticks
    if ytick_step is not None and ytick_step > 0:
        yticks = np.arange(0, len(df_matrix.index), ytick_step)
        ax.set_yticks(yticks)
        labels = pd.to_datetime(df_matrix.index[::ytick_step]).strftime('%Y-%m-%d')
        ax.set_yticklabels(labels, fontsize=7, rotation=0)

    plt.xlabel("Minute of the Day (HH:MM)")
    plt.ylabel("Day")
    plt.title(title)

    plt.tight_layout()
    return fig,ax

def plot_heatmap_grid(pairs, titles, start_date=None, end_date=None, ytick_step=90,
                      cmap="viridis", robust=True, figsize=(18, 12)):
    """
    Plot 4 heatmaps in a 2x2 grid with a single shared colorbar.

    Parameters
    ----------
    pairs : list of pd.DataFrame
        4 day×minute matrices (one per pair).
    titles : list of str
        4 subplot titles.
    """
    assert len(pairs) == 4 and len(titles) == 4

    # Filter dates and compute global vmin/vmax
    filtered = []
    for df in pairs:
        if start_date is not None or end_date is not None:
            idx_dt = pd.to_datetime(df.index)
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= idx_dt >= pd.Timestamp(start_date)
            if end_date is not None:
                mask &= idx_dt <= pd.Timestamp(end_date)
            df = df[mask]
        filtered.append(df)

    all_values = np.concatenate([df.values.flatten() for df in filtered])
    all_values = all_values[~np.isnan(all_values)]
    if robust:
        vmin = np.quantile(all_values, 0.01)
        vmax = np.quantile(all_values, 0.99)
    else:
        vmin, vmax = np.min(all_values), np.max(all_values)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, df, title in zip(axes, filtered, titles):
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=60,
            yticklabels=False,
        )
        if ytick_step and ytick_step > 0:
            yticks = np.arange(0, len(df.index), ytick_step)
            ax.set_yticks(yticks)
            labels = pd.to_datetime(df.index[::ytick_step]).strftime('%Y-%m-%d')
            ax.set_yticklabels(labels, fontsize=7, rotation=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Minute of the Day (HH:MM)", fontsize=8)
        ax.set_ylabel("Day", fontsize=8)

    # Leave space on the right for the colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)

    # Dedicated colorbar axis: [left, bottom, width, height] in figure coords
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.65])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Volatility", fontsize=10)

    fig.suptitle("Intraday Volatility (2014–2025)", fontsize=14)
    return fig, axes

def heatmap_covariance_month(log_returns, title):
    months = sorted(log_returns.index.month.unique())
    cov_matrices_monthly = []

    for month in months:
        monthly_data = log_returns[log_returns.index.month == month]
        cov_matrix = covariance_function(monthly_data.values)
        cov_matrices_monthly.append(cov_matrix)

    all_values = np.concatenate([cov.flatten() for cov in cov_matrices_monthly])
    vmin = np.quantile(all_values, 0.02)
    vmax = np.quantile(all_values, 0.98)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()


    for i, (month, cov_matrix) in enumerate(zip(months, cov_matrices_monthly)):
        im = axes[i].imshow(
            cov_matrix,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"
        )
        axes[i].set_title(f"Month {month}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Time")

        num_minutes = cov_matrices_monthly[0].shape[0]
        interval = max(1, num_minutes // 24)  # automatique
        hour_ticks = np.arange(0, num_minutes, interval)
        hour_labels = [f"{h:02d}:00" for h in range(len(hour_ticks))]

        axes[i].set_xticks(hour_ticks)
        axes[i].set_xticklabels(hour_labels, rotation=90)
        axes[i].set_yticks(hour_ticks)
        axes[i].set_yticklabels(hour_labels)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Covariance")

    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def heatmap_covariance_lag1_month(log_returns, title):
    months = sorted(log_returns.index.month.unique())
    cov_matrices_monthly = []

    for month in months:
        monthly_data = log_returns[log_returns.index.month == month]
        cov_matrix = lag1_covariance(monthly_data.values)
        cov_matrices_monthly.append(cov_matrix)

    cov_matrix_plot = cov_matrices_monthly

    all_values = np.concatenate([cov.flatten() for cov in cov_matrix_plot])
    vmin = np.quantile(all_values, 0.02)
    vmax = np.quantile(all_values, 0.98)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    num_minutes = cov_matrix_plot[0].shape[0]
    hour_ticks = np.arange(0, num_minutes, 60)
    hour_labels = [f"{h:02d}:00" for h in range(0, 24)]

    for i, (month, cov_matrix) in enumerate(zip(months, cov_matrix_plot)):
        im = axes[i].imshow(
            cov_matrix,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"
        )
        axes[i].set_title(f"Month {month}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Time")

        num_minutes = cov_matrices_monthly[0].shape[0]
        interval = max(1, num_minutes // 24)  # automatique
        hour_ticks = np.arange(0, num_minutes, interval)
        hour_labels = [f"{h:02d}:00" for h in range(len(hour_ticks))]

        axes[i].set_xticks(hour_ticks)
        axes[i].set_xticklabels(hour_labels, rotation=90)
        axes[i].set_yticks(hour_ticks)
        axes[i].set_yticklabels(hour_labels)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Covariance")

    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def heatmap_cross_covariance_month(log_returns_1, log_returns_2, year, title):

    data1 = log_returns_1[log_returns_1.index.year == year]
    data2 = log_returns_2[log_returns_2.index.year == year]

    months = sorted(data1.index.month.unique())
    cross_cov_matrices = []

    for month in months:

        monthly_data1 = data1[data1.index.month == month]
        monthly_data2 = data2[data2.index.month == month]

        cross_cov = cross_covariance_function(
            monthly_data1.values,
            monthly_data2.values
        )

        cross_cov_matrices.append(cross_cov)

    all_values = np.concatenate([cov.flatten() for cov in cross_cov_matrices])
    vmin = np.quantile(all_values, 0.02)
    vmax = np.quantile(all_values, 0.98)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, (month, cov_matrix) in enumerate(zip(months, cross_cov_matrices)):

        im = axes[i].imshow(
            cov_matrix,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"
        )

        axes[i].set_title(f"Month {month}")
        axes[i].set_xlabel("Pair 1 Time")
        axes[i].set_ylabel("Pair 2 Time")

        num_minutes = cov_matrix.shape[0]
        interval = max(1, num_minutes // 24)

        hour_ticks = np.arange(0, num_minutes, interval)
        hour_labels = [f"{h:02d}:00" for h in range(len(hour_ticks))]

        axes[i].set_xticks(hour_ticks)
        axes[i].set_xticklabels(hour_labels, rotation=90)

        axes[i].set_yticks(hour_ticks)
        axes[i].set_yticklabels(hour_labels)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cross Covariance")

    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def heatmap_cross_cov_lag1_month(log_returns_1, log_returns_2, year, title):

    data1 = log_returns_1[log_returns_1.index.year == year]
    data2 = log_returns_2[log_returns_2.index.year == year]

    months = sorted(data1.index.month.unique())
    cross_cov_matrices = []

    for month in months:

        monthly_data1 = data1[data1.index.month == month]
        monthly_data2 = data2[data2.index.month == month]

        cross_cov = cross_covariance_lag1(
            monthly_data1.values,
            monthly_data2.values
        )

        cross_cov_matrices.append(cross_cov)

    all_values = np.concatenate([cov.flatten() for cov in cross_cov_matrices])
    vmin = np.quantile(all_values, 0.02)
    vmax = np.quantile(all_values, 0.98)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, (month, cov_matrix) in enumerate(zip(months, cross_cov_matrices)):

        im = axes[i].imshow(
            cov_matrix,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"
        )

        axes[i].set_title(f"Month {month}")
        axes[i].set_xlabel("Pair 1 Time (t)")
        axes[i].set_ylabel("Pair 2 Time (t+1)")

        num_minutes = cov_matrix.shape[0]
        interval = max(1, num_minutes // 24)

        hour_ticks = np.arange(0, num_minutes, interval)
        hour_labels = [f"{h:02d}:00" for h in range(len(hour_ticks))]

        axes[i].set_xticks(hour_ticks)
        axes[i].set_xticklabels(hour_labels, rotation=90)

        axes[i].set_yticks(hour_ticks)
        axes[i].set_yticklabels(hour_labels)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cross Covariance Lag 1")

    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

def yearly_covariance_difference_matrix(df, lag1=False):
    years = df.index.year.unique()
    years = np.sort(years)
    n_years = len(years)
    
    cov_matrices = []
    for y in years:
        yearly_data = df[df.index.year == y].values
        if lag1:
            cov = lag1_covariance(yearly_data)
        else:
            cov = covariance_function(yearly_data)
        cov_matrices.append(cov)
    
    frob_diff_matrix = np.zeros((n_years, n_years))
    for i in range(n_years):
        for j in range(n_years):
            diff = cov_matrices[i] - cov_matrices[j]
            frob_diff_matrix[i, j] = np.linalg.norm(diff, ord='fro')
    
    return frob_diff_matrix, years


def _avg_cov_semester(vol_matrices, year, sem, lag1):
    """
    For one semester (year, sem), compute the average M×M covariance kernel
    across all pairs that have enough data.

    C_s = (1 / N_valid) * Σ_i  cov_i(s)

    This is the M-direction dual of the T×T Gram matrix F.
    """
    cov_sum = None
    n_valid = 0
    for df in vol_matrices.values():
        idx = pd.to_datetime(df.index)
        mask = (idx.year == year) & (idx.month <= 6 if sem == 1 else idx.month >= 7)
        data = df.loc[mask].values
        if data.shape[0] < 2:
            continue
        c = lag1_covariance(data) if lag1 else covariance_function(data)
        cov_sum = c if cov_sum is None else cov_sum + c
        n_valid += 1
    if n_valid == 0:
        return None
    return cov_sum / n_valid


def semester_covariance_difference_matrix(vol_matrices, lag1=False):
    """
    Pairwise Frobenius norm differences of the average M×M covariance kernel,
    computed semester by semester across ALL pairs in vol_matrices.

    For each semester s:
        C_s = (1/N_s) * Σ_i  cov_i(s)     (M×M, averaged over N_s valid pairs)

    Then:
        frob_diff[s, s'] = ||C_s - C_{s'}||_F

    Parameters
    ----------
    vol_matrices : dict  {pair_name: DataFrame with DatetimeIndex}
                   All DataFrames must share the same intraday column structure.
    lag1         : if True, use lag-1 covariance instead of standard covariance

    Returns
    -------
    frob_diff_matrix : ndarray (n_semesters, n_semesters)
    labels           : list of strings, e.g. ['S1 2003', 'S2 2003', ...]
    """
    all_dates = pd.to_datetime(
        np.concatenate([pd.to_datetime(df.index).values for df in vol_matrices.values()])
    )
    semester_keys = sorted(set((d.year, 1 if d.month <= 6 else 2) for d in all_dates))

    cov_matrices = []
    labels = []
    for year, sem in semester_keys:
        c = _avg_cov_semester(vol_matrices, year, sem, lag1)
        if c is not None:
            cov_matrices.append(c)
            labels.append(f"S{sem} {year}")

    n = len(cov_matrices)
    frob_diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            frob_diff_matrix[i, j] = np.linalg.norm(
                cov_matrices[i] - cov_matrices[j], ord='fro'
            )

    return frob_diff_matrix, labels


def plot_semester_covariance_heatmaps(vol_matrices, figsize=(18, 8), cmap='viridis'):
    """
    Side-by-side heatmaps of Frobenius norm differences between semester
    average covariance kernels, computed across ALL pairs in vol_matrices.

    Left  : standard covariance    C_s = (1/N_s) Σ_i cov_i(s)
    Right : lag-1 covariance       C_s = (1/N_s) Σ_i cov_lag1_i(s)

    Parameters
    ----------
    vol_matrices : dict  {pair_name: DataFrame with DatetimeIndex}
    figsize      : figure size
    cmap         : colormap passed to seaborn heatmap
    """
    frob,      labels = semester_covariance_difference_matrix(vol_matrices, lag1=False)
    frob_lag1, _      = semester_covariance_difference_matrix(vol_matrices, lag1=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, matrix, title in zip(
        axes,
        [frob, frob_lag1],
        ['Frobenius Norm of Covariance Differences',
         'Frobenius Norm of Covariance Differences lag1'],
    ):
        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cmap=cmap,
            cbar_kws={'label': 'Frobenius norm difference'},
        )
        ax.set_title(title)
        step = max(1, len(labels) // 20)
        for i, tick in enumerate(ax.get_xticklabels()):
            tick.set_visible(i % step == 0)
            tick.set_rotation(90)
        for i, tick in enumerate(ax.get_yticklabels()):
            tick.set_visible(i % step == 0)
            tick.set_rotation(0)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# EDA — Low-rank structure: top eigenvalues of the Gram matrix F
# ─────────────────────────────────────────────────────────────────────────────

def gram_eigenvalues(vol_matrices, n_components=5, start_date=None, end_date=None,
                     min_coverage=0.8):
    """
    Build the Gram matrix F from vol_matrices and return the top eigenvalues
    with their cumulative variance proportions.

    F_{st} = (1/NM) * Σ_i Σ_m  Y_{i,s,m} * Y_{i,t,m}   (T×T)

    The proportion of variance explained by factor k is  λ_k / trace(F).

    Parameters
    ----------
    vol_matrices  : dict {pair: DataFrame (T×M)}
    n_components  : number of top eigenvalues to return (default 5)
    start_date    : optional filter 'YYYY-MM-DD'
    end_date      : optional filter 'YYYY-MM-DD'
    min_coverage  : minimum fraction of pairs present on a given day

    Returns
    -------
    df : pd.DataFrame with columns ['eigenvalue', 'var_explained', 'cum_var_explained']
         index = factor number (1-based)
    """
    pairs = list(vol_matrices.keys())

    # Filter dates
    filtered = {}
    for p, df in vol_matrices.items():
        if start_date or end_date:
            idx = pd.to_datetime(df.index)
            mask = pd.Series(True, index=df.index)
            if start_date:
                mask &= idx >= pd.Timestamp(start_date)
            if end_date:
                mask &= idx <= pd.Timestamp(end_date)
            df = df[mask]
        filtered[p] = df

    # Common days with enough pair coverage
    all_days = sorted(set.union(*[set(df.index) for df in filtered.values()]))
    common_days = [
        d for d in all_days
        if sum(d in filtered[p].index for p in pairs) / len(pairs) >= min_coverage
    ]

    N = len(pairs)
    M = filtered[pairs[0]].shape[1]
    T = len(common_days)

    # Build centered panel Y (N, T, M)
    Y = np.zeros((N, T, M))
    for i, p in enumerate(pairs):
        df = filtered[p]
        for t, day in enumerate(common_days):
            if day in df.index:
                Y[i, t, :] = df.loc[day].values
            else:
                Y[i, t, :] = np.nan
    mean = np.nanmean(Y, axis=1, keepdims=True)
    Y = np.nan_to_num(Y - mean)

    # Gram matrix F (T×T)
    F = np.einsum('ism,itm->st', Y, Y) / (N * M)
    F = (F + F.T) / 2

    # Top eigenvalues (scipy eigh is stable for symmetric matrices)
    from scipy.linalg import eigh
    k = min(n_components, T)
    eigenvalues, _ = eigh(F, subset_by_index=[T - k, T - 1])
    eigenvalues = eigenvalues[::-1]          # descending
    eigenvalues = np.maximum(eigenvalues, 0) # numerical safety

    total_var = np.trace(F)
    var_exp   = eigenvalues / total_var
    cum_var   = np.cumsum(var_exp)

    df_out = pd.DataFrame({
        'eigenvalue':       eigenvalues,
        'var_explained':    var_exp,
        'cum_var_explained': cum_var,
    }, index=pd.Index(range(1, k + 1), name='factor'))

    return df_out


def print_gram_eigenvalues(vol_matrices, n_components=5, **kwargs):
    """
    Print a readable table of the top eigenvalues of F and their variance shares.
    """
    df = gram_eigenvalues(vol_matrices, n_components=n_components, **kwargs)
    print(f"{'Factor':>8}  {'Eigenvalue':>12}  {'Var. expl.':>12}  {'Cum. var.':>12}")
    print("  " + "-" * 50)
    for k, row in df.iterrows():
        print(f"  {k:>6}    {row['eigenvalue']:>11.4f}    "
              f"{row['var_explained']:>10.1%}    {row['cum_var_explained']:>10.1%}")
    print(f"\n  → Top {n_components} factors explain "
          f"{df['cum_var_explained'].iloc[-1]:.1%} of total panel variance.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EDA — MxM covariance / correlation heatmaps (one per FX pair)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mxm_heatmaps(
    vol_matrices,
    pairs,
    correlation=False,
    lag1=False,
    start_date=None,
    end_date=None,
    cmap="RdBu_r",
    figsize=(20, 16),
    save_dir=None,
):
    """
    Plot a 2x2 grid of M×M covariance (or correlation) matrices, one per FX pair.

    Parameters
    ----------
    vol_matrices : dict {pair_name: DataFrame (T×M)}
    pairs        : list of 4 pair names (keys of vol_matrices)
    correlation  : if True, normalise to correlation matrix (values in [-1,1])
    lag1         : if True, compute lag-1 cross-covariance C[m,m'] = Cov(v_m(t), v_{m'}(t+1))
    start_date   : optional filter 'YYYY-MM-DD'
    end_date     : optional filter 'YYYY-MM-DD'
    cmap         : colormap (default RdBu_r, good for correlation)
    figsize      : figure size
    save_dir     : directory to save figures (e.g. 'src/figures/EDA')
    """
    assert len(pairs) == 4, "Pass exactly 4 pairs."

    matrices = {}
    for p in pairs:
        df = vol_matrices[p]
        if start_date is not None or end_date is not None:
            idx = pd.to_datetime(df.index)
            mask = pd.Series(True, index=df.index)
            if start_date:
                mask &= idx >= pd.Timestamp(start_date)
            if end_date:
                mask &= idx <= pd.Timestamp(end_date)
            df = df[mask]
        data = df.values
        C = lag1_covariance(data) if lag1 else covariance_function(data)
        if correlation:
            # Symmetrise first (lag-1 is not symmetric)
            C_sym = (C + C.T) / 2 if lag1 else C
            d = np.sqrt(np.diag(covariance_function(data)))
            with np.errstate(invalid="ignore"):
                C = C / np.outer(d, d)
            np.fill_diagonal(C, 1.0)
        matrices[p] = C

    # Shared color scale
    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    if correlation:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = np.quantile(all_vals, 0.01)
        vmax = np.quantile(all_vals, 0.99)
        # symmetric scale for lag-1 (can be negative)
        if lag1:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound

    # Build time labels from the columns of the first available pair
    sample_df = vol_matrices[pairs[0]]
    time_labels = list(sample_df.columns)          # e.g. ["17:00", "17:05", ...]
    n_cols = len(time_labels)
    # Show one label per hour: every 12 ticks for 5-min bars, every 60 for 1-min bars
    bar_minutes = 5 if n_cols <= 300 else 1
    tick_step = max(1, 60 // bar_minutes)
    tick_pos   = np.arange(0, n_cols, tick_step)
    tick_labels = [time_labels[i] for i in tick_pos]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, p in zip(axes, pairs):
        M = matrices[p]

        sns.heatmap(
            M,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.invert_yaxis()
        ax.set_xticks(tick_pos + 0.5)
        ax.set_yticks(tick_pos + 0.5)
        ax.set_xticklabels(tick_labels, fontsize=11, rotation=90)
        ax.set_yticklabels(tick_labels, fontsize=11, rotation=0)
        ax.set_title(p.upper(), fontsize=16, fontweight="bold", pad=8)
        ax.set_xlabel("Time of day", fontsize=13)
        ax.set_ylabel("Time of day", fontsize=13)

    fig.subplots_adjust(left=0.09, right=0.88, top=0.93, bottom=0.09,
                        hspace=0.22, wspace=0.22)
    cbar_ax = fig.add_axes([0.90, 0.09, 0.018, 0.82])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Correlation" if correlation else "Covariance", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    kind = "correlation" if correlation else "covariance"
    lag_tag = "_lag1" if lag1 else ""
    title = f"Intraday M×M {'Correlation' if correlation else 'Covariance'} Matrix per Pair"
    if lag1:
        title += " (Lag-1)"
    fig.suptitle(title, fontsize=17)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"mxm_{kind}{lag_tag}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# EDA — PxP cross-pair covariance / correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_pxp_heatmap(
    vol_matrices,
    correlation=True,
    start_date=None,
    end_date=None,
    cmap="RdBu_r",
    figsize=(10, 8),
    save_dir=None,
):
    """
    P×P cross-pair covariance (or correlation) matrix.

    For each pair of pairs (p, q), compute:
        C[p, q] = mean over minutes m of  Cov_t( v_p(t,m), v_q(t,m) )

    i.e. average the scalar covariance across the M intraday minutes.
    This collapses the minute dimension and yields a P×P matrix.

    Parameters
    ----------
    vol_matrices : dict {pair_name: DataFrame (T×M)}
    correlation  : if True, normalise to [-1,1]
    start_date   : optional filter
    end_date     : optional filter
    cmap         : colormap
    figsize      : figure size
    save_dir     : directory to save the figure
    """
    pairs = list(vol_matrices.keys())
    P = len(pairs)

    # Filter and align on common days
    filtered = {}
    for p, df in vol_matrices.items():
        if start_date is not None or end_date is not None:
            idx = pd.to_datetime(df.index)
            mask = pd.Series(True, index=df.index)
            if start_date:
                mask &= idx >= pd.Timestamp(start_date)
            if end_date:
                mask &= idx <= pd.Timestamp(end_date)
            df = df[mask]
        filtered[p] = df

    common_idx = filtered[pairs[0]].index
    for p in pairs[1:]:
        common_idx = common_idx.intersection(filtered[p].index)

    # Build P×P matrix: mean over minutes of Cov_t(v_p(t,m), v_q(t,m))
    C = np.zeros((P, P))
    for i, pi in enumerate(pairs):
        Xi = filtered[pi].loc[common_idx].values          # T×M
        Xi = Xi - np.nanmean(Xi, axis=0)                  # centre
        Xi = np.nan_to_num(Xi)
        for j, pj in enumerate(pairs):
            Xj = filtered[pj].loc[common_idx].values
            Xj = Xj - np.nanmean(Xj, axis=0)
            Xj = np.nan_to_num(Xj)
            T = Xi.shape[0]
            # scalar covariance per minute then average
            cov_per_min = np.sum(Xi * Xj, axis=0) / (T - 1)   # shape (M,)
            C[i, j] = np.mean(cov_per_min)

    if correlation:
        std = np.sqrt(np.diag(C))
        with np.errstate(invalid="ignore"):
            C = C / np.outer(std, std)
        np.fill_diagonal(C, 1.0)

    labels = [p.upper() for p in pairs]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        C,
        ax=ax,
        cmap=cmap,
        vmin=-1 if correlation else None,
        vmax=1 if correlation else None,
        annot=False,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Correlation" if correlation else "Covariance", "shrink": 0.8},
    )
    kind = "Correlation" if correlation else "Covariance"
    ax.set_title(f"P×P Cross-Pair {kind} (averaged over intraday minutes)", fontsize=13)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"pxp_{'correlation' if correlation else 'covariance'}.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)