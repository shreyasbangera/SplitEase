"""S5 - Session drift harvesting.

Rationale: measured on 21 years of hourly gold, the Asian hours carry a
statistically significant positive drift (+2.1bp at 01:00 UTC, t=6.2) while the
London/NY afternoon hours carry negative drift (-1.1bp at 16:00 UTC, t=-2.3).
This mirrors the published intraday-seasonality result for gold. The strategy is
a pure calendar rule: hold long across the Asian window, hold short across the
US afternoon window, flat otherwise, with a protective volatility stop.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, long_start=0, long_end=5, short_start=14, short_end=17,
           sl_atr=4.0, atr_n=24, do_short=True, trend_n=0):
    idx = df.index
    hour = idx.hour
    a = ind.atr(df, atr_n)
    sig = pd.Series(0.0, index=idx)
    sig[hour == long_start] = 1.0
    if do_short:
        sig[hour == short_start] = -1.0
    if trend_n:
        e = ind.ema(df.close, trend_n)
        sig[(sig > 0) & (df.close < e)] = 0.0
        sig[(sig < 0) & (df.close > e)] = 0.0
    ex = ((hour >= long_end) & (hour < short_start)) | (hour >= short_end)
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = np.nan
    out["exit"] = ex.astype(float)
    return out
