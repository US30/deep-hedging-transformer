"""
Data loaders for calibration and real-data OOS testing.
Uses yfinance for SPX and FRED for VIX term structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


def load_spx(start: str = "2000-01-01", end: Optional[str] = None) -> pd.Series:
    """SPX daily closes via yfinance. Returns log returns."""
    import yfinance as yf
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return spx.dropna()


def load_vix_term(start: str = "2005-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """
    Download VIX indices for term structure calibration.
    Tickers: VIX9D, VIX, VIX3M, VIX6M, VIX1Y
    Approximate maturities: 9/252, 1/12, 3/12, 6/12, 1.0 years.
    """
    import yfinance as yf
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    tickers = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]
    maturities = [9 / 252, 1 / 12, 3 / 12, 6 / 12]
    dfs = []
    for t in tickers:
        try:
            s = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)["Close"]
            s.name = t.replace("^", "")
            dfs.append(s)
        except Exception:
            pass
    df = pd.concat(dfs, axis=1).dropna()
    df.columns = [c.replace("^", "") for c in df.columns]
    df.attrs["maturities"] = maturities[: len(df.columns)]
    return df


def compute_realized_vol(spx: pd.Series, window: int = 21) -> pd.Series:
    """Rolling realised vol (annualised) from log returns."""
    log_ret = np.log(spx / spx.shift(1)).dropna()
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv.dropna()


def realized_roughness(spx: pd.Series, q_vals=(1, 2, 3), lags=None) -> dict:
    """
    Estimate Hurst exponent H from structure function scaling.
    m(q, Δ) = E[|log_rv(t+Δ) - log_rv(t)|^q] ~ C_q * Δ^{q*H}
    log m(q, Δ) vs log Δ slope = q*H  →  H = slope / q.
    """
    log_rv = np.log(compute_realized_vol(spx) + 1e-9).values
    if lags is None:
        lags = [1, 2, 5, 10, 21, 42]
    results = {}
    for q in q_vals:
        slopes = []
        for lag in lags:
            diffs = np.abs(log_rv[lag:] - log_rv[:-lag]) ** q
            slopes.append((lag, np.mean(diffs)))
        x = np.log([s[0] for s in slopes])
        y = np.log([s[1] for s in slopes])
        coef = np.polyfit(x, y, 1)
        H_est = coef[0] / q
        results[q] = {"H_estimate": H_est, "slope": coef[0]}
    return results
