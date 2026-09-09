"""Indicator library. Every function returns values aligned to the bar they are
computed ON (using data up to and including that bar). Strategies must shift or
use them only for the NEXT bar's decision; the engine then fills at the open of
the bar after that."""
import numpy as np
import pandas as pd


def ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def sma(s, n):   return s.rolling(n).mean()


def atr(df, n=14):
    h, l, c = df.high, df.low, df.close.shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(df, n=14):
    up = df.high.diff(); dn = -df.low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, mdi


def bbands(s, n=20, k=2.0):
    m = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
    return m, m + k * sd, m - k * sd, sd


def keltner(df, n=20, k=1.5):
    m = ema(df.close, n); a = atr(df, n)
    return m, m + k * a, m - k * a


def zscore(s, n):
    return (s - s.rolling(n).mean()) / s.rolling(n).std(ddof=0)


def realized_vol(s, n):
    return np.log(s).diff().rolling(n).std(ddof=0)


def donchian(df, n):
    return df.high.rolling(n).max(), df.low.rolling(n).min()


def hurst(s, n=200, lags=(2, 4, 8, 16, 32)):
    """Rolling Hurst exponent proxy via variance of lagged differences."""
    lp = np.log(s)
    out = pd.Series(index=s.index, dtype=float)
    L = np.log(np.array(lags))
    mats = [lp.diff(k).rolling(n).std(ddof=0) for k in lags]
    M = np.vstack([m.to_numpy() for m in mats])
    with np.errstate(all="ignore"):
        Y = np.log(M)
        Lc = L - L.mean()
        slope = (Lc[:, None] * (Y - np.nanmean(Y, axis=0))).sum(axis=0) / (Lc ** 2).sum()
    out[:] = slope
    return out


def choppiness(df, n=14):
    a = atr(df, 1).rolling(n).sum()
    rng = df.high.rolling(n).max() - df.low.rolling(n).min()
    return 100 * np.log10(a / rng.replace(0, np.nan)) / np.log10(n)


def session_flags(idx):
    """UTC hour based session masks for the gold market."""
    h = idx.hour
    return {
        "asia": (h >= 0) & (h < 7),
        "london": (h >= 7) & (h < 12),
        "overlap": (h >= 12) & (h < 17),   # London/NY overlap - most range
        "ny_pm": (h >= 17) & (h < 21),
    }
