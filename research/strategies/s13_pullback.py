"""S13 - Buy-the-dip inside an established uptrend (pullback entry).

Rationale: breakout entries (S11) buy strength; the opposite family buys the
short-term washout WITHIN a longer uptrend. The two have very different trade
timing on the same instrument, so if both carry an edge they diversify each
other. Entry: short-term oversold (RSI(2) or a close below the lower Keltner
band) while price is above its long moving average. Exit: mean reversion back
to the moving average, a time stop, or a volatility stop.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, ma_n=200, rsi_n=2, rsi_thr=10, use_keltner=False, kelt_k=2.0,
           sl_atr=3.0, atr_n=14, hold_bars=12, tp_r=1.5, exit_ma=20):
    c = df.close
    a = ind.atr(df, atr_n)
    ma = ind.sma(c, ma_n)
    r = ind.rsi(c, rsi_n)
    trig = r < rsi_thr
    if use_keltner:
        _, _, kl = ind.keltner(df, 20, kelt_k)
        trig |= c < kl
    sig = pd.Series(0.0, index=df.index)
    sig[trig & (c > ma)] = 1.0
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = tp_r * sl_atr * a if tp_r else np.nan
    out["max_bars"] = hold_bars
    return out
