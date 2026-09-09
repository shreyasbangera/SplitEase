"""S6 - Time-series momentum (Moskowitz, Ooi & Pedersen 2012) with vol scaling.

Sign of the trailing k-period return predicts the next period's return across
asset classes including gold. Daily bars, monthly-ish lookbacks, position held
until the sign flips. Stops are wide (pure trend exposure) and position size is
inverse-volatility scaled, which is the standard CTA construction.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, look=60, sl_atr=6.0, atr_n=20, confirm=0, long_only=False):
    a = ind.atr(df, atr_n)
    mom = df.close / df.close.shift(look) - 1
    s = np.sign(mom)
    if confirm:
        s2 = np.sign(df.close / df.close.shift(confirm) - 1)
        s = s.where(s == s2, 0.0)
    if long_only:
        s = s.clip(lower=0)
    out = pd.DataFrame({"sig": s.fillna(0.0), "sl_dist": sl_atr * a})
    out["tp_dist"] = np.nan
    out["exit"] = (s.fillna(0.0) == 0).astype(float)   # flat signal => close
    return out
