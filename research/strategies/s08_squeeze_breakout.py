"""S8 - Volatility-squeeze expansion breakout.

Rationale: volatility is the most persistent and most forecastable property of
price. Periods of unusually compressed range (Bollinger width in the bottom
decile of its own history, the 'squeeze') are followed by range expansion. We do
not predict direction; we buy the break of the compressed range in whichever
direction it resolves, with the opposite side of the compression box as the stop.
Because the box is tight, the stop is tight, giving a large reward-to-risk if the
expansion runs - the pattern's whole economic logic.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, bb_n=20, sq_lookback=120, sq_pct=0.20, sl_atr=1.5, atr_n=14,
           tp_r=3.0, trail_atr=0.0, max_bars=0, box_stop=True, trend_n=0):
    m, hi, lo, sd = ind.bbands(df.close, bb_n, 2.0)
    a = ind.atr(df, atr_n)
    width = (hi - lo) / m
    sq = width.rolling(sq_lookback).rank(pct=True)
    boxh = df.high.rolling(bb_n).max()
    boxl = df.low.rolling(bb_n).min()
    was_sq = sq.shift(1) < sq_pct
    up = was_sq & (df.close > boxh.shift(1))
    dn = was_sq & (df.close < boxl.shift(1))
    sig = pd.Series(0.0, index=df.index)
    sig[up] = 1.0
    sig[dn] = -1.0
    if trend_n:
        e = ind.ema(df.close, trend_n)
        sig[(sig > 0) & (df.close < e)] = 0.0
        sig[(sig < 0) & (df.close > e)] = 0.0
    if box_stop:
        dist = pd.Series(np.where(sig > 0, df.close - boxl.shift(1),
                         np.where(sig < 0, boxh.shift(1) - df.close, np.nan)), index=df.index)
        dist = dist.clip(lower=(0.5 * a))
    else:
        dist = sl_atr * a
    out = pd.DataFrame({"sig": sig, "sl_dist": dist})
    out["tp_dist"] = tp_r * dist if tp_r else np.nan
    if trail_atr:
        out["trail"] = trail_atr * a
    if max_bars:
        out["max_bars"] = max_bars
    return out
