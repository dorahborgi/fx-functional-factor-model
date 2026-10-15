"""
Log transformation for the FFM — works with |r_t| or r_t².

NaN values are preserved as-is throughout the transformation.
ε and frac_zero are computed on non-NaN values only.
"""

import numpy as np
import pandas as pd


def log_transform(vol_matrices: dict, eps_pct: float = 5.0,
                  eps_dict: dict = None) -> dict:
    """
    Apply log(x + ε) pair by pair.

    Works with |r_t| or r_t².

    Parameters
    ----------
    eps_dict : dict {pair: float}, optional
        Fixed epsilon per pair (estimated on the training data).
        If provided, eps_pct is ignored for the pairs present in eps_dict.

    NaN BEHAVIOUR:
    - NaN stay NaN in the output.
    - ε and frac_zero are computed on non-NaN values only.
    """
    vol_log = {}
    src = "fixed" if eps_dict is not None else f"pct={eps_pct}%"
    print(f"  {'Pair':15s}  {'ε (' + src + ')':>16s}  {'min log':>10s}  "
          f"{'max log':>10s}  {'std':>8s}  {'NaN':>8s}  {'inf':>8s}")
    print("  " + "-" * 90)

    for key, df in vol_matrices.items():
        vol = df.values.astype(np.float64)

        nan_mask = np.isnan(vol)
        non_nan  = vol[~nan_mask]

        pos = non_nan[non_nan > 0]
        if len(pos) == 0:
            print(f"  {key:15s}  ⚠ no positive value — pair skipped")
            continue

        if eps_dict is not None and key in eps_dict:
            eps = float(eps_dict[key])
        else:
            frac_zero = (non_nan <= 0).mean()
            pct_used  = max(eps_pct, 10.0) if frac_zero > 0.20 else eps_pct
            eps       = np.percentile(pos, pct_used)

        log_vals = np.where(nan_mask, np.nan, np.log(np.maximum(vol, 0.0) + eps))

        n_nan = nan_mask.sum()
        n_inf = np.isinf(log_vals).sum()

        vol_log[key] = pd.DataFrame(log_vals, index=df.index, columns=df.columns)

        frac_zero = (non_nan <= 0).mean()
        sparse_flag = f" ⚠sparse({frac_zero:.0%})" if frac_zero > 0.20 else ""
        print(f"  {key:15s}  {eps:>18.3e}  {np.nanmin(log_vals):>10.3f}  "
              f"{np.nanmax(log_vals):>10.3f}  {np.nanstd(log_vals):>8.3f}  "
              f"{n_nan:>8d}  {n_inf:>8d}{sparse_flag}")

    return vol_log


def extract_epsilon(vol_sq: dict, eps_pct: float = 5.0) -> dict:
    """
    Compute the epsilon used in log_transform for each pair.

    Parameters
    ----------
    vol_sq : dict {pair: DataFrame} — r_t² matrices (already squared)

    Returns
    -------
    eps_dict : dict {pair: float}
    """
    eps_dict = {}
    for key, df in vol_sq.items():
        vol     = df.values.astype(np.float64)
        non_nan = vol[~np.isnan(vol)]
        pos     = non_nan[non_nan > 0]
        if len(pos) == 0:
            continue
        frac_zero = (non_nan <= 0).mean()
        pct_used  = max(eps_pct, 10.0) if frac_zero > 0.20 else eps_pct
        eps_dict[key] = float(np.percentile(pos, pct_used))
    return eps_dict
