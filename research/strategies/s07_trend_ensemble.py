"""S7 - Multi-lookback trend ensemble (CTA core).

Rationale: single-lookback trend systems are fragile to the choice of lookback.
The standard institutional fix is to average the sign of several lookbacks and
trade the consensus. Entry requires agreement of k of n lookbacks; the position
is exited when consensus decays or a volatility trailing stop is hit.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, looks=(20, 60, 120), k=3, sl_atr=3.0, atr_n=14, trail_atr=4.0,
           long_only=False, vol_filter=0.0, exit_k=1):
    a = ind.atr(df, atr_n)
    votes = sum(np.sign(df.close / df.close.shift(L) - 1) for L in looks)
    n = len(looks)
    sig = pd.Series(0.0, index=df.index)
    sig[votes >= k] = 1.0
    sig[votes <= -k] = -1.0
    if long_only:
        sig[sig < 0] = 0.0
    if vol_filter:
        rv = ind.realized_vol(df.close, 20)
        rank = rv.rolling(500, min_periods=150).rank(pct=True)
        sig[rank > vol_filter] = 0.0
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = np.nan
    if trail_atr:
        out["trail"] = trail_atr * a
    out["exit"] = (votes.abs() < exit_k).astype(float)
    return out
