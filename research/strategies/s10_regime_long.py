"""S10 - Regime-filtered leveraged exposure ("participate in the trend, sit out
the drawdowns").

Rationale: both gold and BTC have a large positive unconditional drift, and the
single most robust filter in the literature (Faber, "A Quantitative Approach to
Tactical Asset Allocation") is a long-term moving-average regime switch: hold the
asset while it is above its long moving average, hold cash when below. That does
not add return per unit of exposure, it removes the left tail - which is exactly
what a drawdown-constrained mandate needs, and it is what allows leverage to be
applied on top. A volatility brake cuts exposure when realised vol is extreme.

Entries and exits are decided on completed daily bars and executed next open.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, ma_n=200, sl_atr=8.0, atr_n=20, vol_n=20, vol_cap_pct=0.0,
           allow_short=False, confirm_n=0, trail_atr=0.0):
    c = df.close
    a = ind.atr(df, atr_n)
    ma = ind.sma(c, ma_n)
    long_ok = c > ma
    if confirm_n:
        long_ok &= c > c.shift(confirm_n)
    sig = pd.Series(0.0, index=df.index)
    sig[long_ok] = 1.0
    if allow_short:
        short_ok = c < ma
        if confirm_n:
            short_ok &= c < c.shift(confirm_n)
        sig[short_ok] = -1.0
    if vol_cap_pct:
        rv = ind.realized_vol(c, vol_n)
        rank = rv.rolling(500, min_periods=150).rank(pct=True)
        sig[rank > vol_cap_pct] = 0.0
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = np.nan
    out["exit"] = (sig == 0).astype(float)
    if trail_atr:
        out["trail"] = trail_atr * a
    return out
