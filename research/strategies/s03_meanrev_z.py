"""S3 - Volatility-conditioned mean reversion.

Rationale: gold is a "safe haven" with strong short-horizon mean reversion in
calm regimes but trends in stressed regimes. We fade stretched moves (z-score of
close vs a moving average) ONLY when the realised-vol regime is quiet and the
market is non-trending (low ADX / high choppiness), which is where reversion
dominates. Target is a return to the mean, stop is volatility-scaled.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, z_n=48, z_in=2.0, sl_atr=2.5, atr_n=24, tp_r=1.0, adx_max=25,
           adx_n=14, max_bars=48, vol_n=100, vol_pct_max=0.7, rsi_n=0, rsi_lo=10, rsi_hi=90):
    z = ind.zscore(df.close, z_n)
    a = ind.atr(df, atr_n)
    adxv, _, _ = ind.adx(df, adx_n)
    rv = ind.realized_vol(df.close, 24)
    rank = rv.rolling(vol_n * 5, min_periods=vol_n).rank(pct=True)
    ok = (adxv < adx_max) & (rank < vol_pct_max)
    long_s = ok & (z < -z_in)
    short_s = ok & (z > z_in)
    if rsi_n:
        r = ind.rsi(df.close, rsi_n)
        long_s &= r < rsi_lo
        short_s &= r > rsi_hi
    sig = pd.Series(0.0, index=df.index)
    sig[long_s] = 1.0
    sig[short_s] = -1.0
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = tp_r * sl_atr * a
    out["max_bars"] = max_bars
    return out
