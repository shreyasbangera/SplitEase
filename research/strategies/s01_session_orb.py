"""S1 - Asian-range / session opening-range breakout for gold.

Concept (from the intraday-session literature: 60-70% of gold's daily range forms
in the London-NY overlap, while the Asian session is quiet and range-bound):
build a reference range from the quiet Asian hours, then trade the first decisive
break of that range during the liquid London / NY hours, in the direction of the
break, with a volatility-scaled stop and a session-end time exit.

All quantities are computed from COMPLETED bars only; the engine fills at the
open of the following bar.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, range_end_h=6, trade_start_h=7, trade_end_h=16, exit_h=20,
           buf_atr=0.10, sl_atr=1.0, tp_r=2.0, atr_n=24, min_range_atr=0.0,
           max_range_atr=99.0, trend_n=0, one_per_day=True, trail_atr=0.0,
           be_at_r=0.0):
    idx = df.index
    a = ind.atr(df, atr_n)
    day = idx.normalize()
    hour = idx.hour

    in_range = hour <= range_end_h
    hi = df.high.where(in_range)
    lo = df.low.where(in_range)
    # running high/low of the Asian window, forward-filled through the day
    rh = hi.groupby(day).cummax().groupby(day).ffill()
    rl = lo.groupby(day).cummin().groupby(day).ffill()
    rh = pd.Series(rh.to_numpy(), index=idx).groupby(day).ffill()
    rl = pd.Series(rl.to_numpy(), index=idx).groupby(day).ffill()

    width = (rh - rl) / a
    tradable = (hour >= trade_start_h) & (hour <= trade_end_h) & \
               (width >= min_range_atr) & (width <= max_range_atr) & a.notna()

    buf = buf_atr * a
    long_sig = tradable & (df.close > rh + buf)
    short_sig = tradable & (df.close < rl - buf)

    if trend_n:
        e = ind.ema(df.close, trend_n)
        long_sig &= df.close > e
        short_sig &= df.close < e

    sig = pd.Series(0.0, index=idx)
    sig[long_sig] = 1.0
    sig[short_sig] = -1.0

    if one_per_day:                      # only the first break of each day
        fired = (sig != 0)
        first = fired & (fired.groupby(day).cumsum() == 1)
        sig = sig.where(first, 0.0)

    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = tp_r * sl_atr * a if tp_r else np.nan
    out["exit"] = (hour >= exit_h).astype(float)
    if trail_atr:
        out["trail"] = trail_atr * a
    if be_at_r:
        out["be_at_r"] = be_at_r
    return out
