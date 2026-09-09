"""S9 - Liquidity-sweep (stop-run) reversal.

Rationale: resting stop orders cluster just beyond the previous session's high
and low. A bar that trades THROUGH that level but closes back INSIDE the prior
range is evidence the move was a liquidity grab rather than a genuine breakout -
the market-microstructure version of a failed breakout. We fade it, stopping out
just beyond the sweep extreme (a tight, well-defined invalidation) and targeting
the opposite side of the prior range.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, ref_n=24, sl_pad_atr=0.25, atr_n=24, tp_mode="range", tp_r=2.0,
           max_bars=48, min_sweep_atr=0.0, trend_filter=0):
    a = ind.atr(df, atr_n)
    ph = df.high.rolling(ref_n).max().shift(1)
    pl = df.low.rolling(ref_n).min().shift(1)
    swept_hi = (df.high > ph) & (df.close < ph)
    swept_lo = (df.low < pl) & (df.close > pl)
    if min_sweep_atr:
        swept_hi &= (df.high - ph) > min_sweep_atr * a
        swept_lo &= (pl - df.low) > min_sweep_atr * a
    sig = pd.Series(0.0, index=df.index)
    sig[swept_hi] = -1.0
    sig[swept_lo] = 1.0
    if trend_filter:
        e = ind.ema(df.close, trend_filter)
        sig[(sig > 0) & (df.close < e)] = 0.0
        sig[(sig < 0) & (df.close > e)] = 0.0
    dist = pd.Series(np.where(sig < 0, df.high - df.close + sl_pad_atr * a,
                     np.where(sig > 0, df.close - df.low + sl_pad_atr * a, np.nan)),
                     index=df.index).clip(lower=0.3 * a)
    out = pd.DataFrame({"sig": sig, "sl_dist": dist})
    if tp_mode == "range":
        tp = pd.Series(np.where(sig < 0, df.close - pl, np.where(sig > 0, ph - df.close, np.nan)), index=df.index)
        out["tp_dist"] = tp.clip(lower=0.5 * a)
    else:
        out["tp_dist"] = tp_r * dist
    out["max_bars"] = max_bars
    return out
