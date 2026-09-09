"""S12 - Bear-regime short module.

Rationale: the daily edge scan on BTC showed that when BOTH the 20-day and the
250-day return are negative, the next 10-20 days carry a large NEGATIVE excess
return (+296bp to +513bp of excess to the short side, t~2.1-2.3). That is the
crypto analogue of the equity bear-market drift: liquidation cascades cluster in
established downtrends. Trading only that state gives a module whose returns are
by construction negatively correlated with the long-only momentum module.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, fast=20, slow=250, trig_n=10, sl_atr=3.0, atr_n=14,
           hold_bars=20, trail_atr=0.0, require_break=True):
    c = df.close
    a = ind.atr(df, atr_n)
    bear = (c.pct_change(fast) < 0) & (c.pct_change(slow) < 0)
    _, ll = ind.donchian(df, trig_n)
    trig = (c < ll.shift(1)) if require_break else pd.Series(True, index=df.index)
    sig = pd.Series(0.0, index=df.index)
    sig[bear & trig] = -1.0
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = np.nan
    out["max_bars"] = hold_bars
    if trail_atr:
        out["trail"] = trail_atr * a
    return out
