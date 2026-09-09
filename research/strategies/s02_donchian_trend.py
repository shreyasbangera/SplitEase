"""S2 - Classic time-series trend following (Donchian channel breakout).

Rationale: trend following is the single most-replicated anomaly in commodity
futures (Moskowitz/Ooi/Pedersen time-series momentum; the original Turtle rules).
Gold is a core CTA market. Entry on an N-bar channel break, exit on an opposite
M-bar channel break or an ATR trailing stop. Wide, volatility-scaled stops keep
transaction cost small relative to the risk taken (cost/R well under 10%).
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, entry_n=55, exit_n=20, sl_atr=3.0, atr_n=20, trail_atr=0.0,
           tp_r=0.0, trend_filter=0, long_only=False, vol_target=False):
    hh, ll = ind.donchian(df, entry_n)
    a = ind.atr(df, atr_n)
    # break of the channel formed by the PREVIOUS entry_n bars (exclude current bar)
    up = df.close > hh.shift(1)
    dn = df.close < ll.shift(1)
    sig = pd.Series(0.0, index=df.index)
    sig[up] = 1.0
    sig[dn] = -1.0
    if long_only:
        sig[sig < 0] = 0.0
    if trend_filter:
        e = ind.ema(df.close, trend_filter)
        sig[(sig > 0) & (df.close < e)] = 0.0
        sig[(sig < 0) & (df.close > e)] = 0.0
    xh, xl = ind.donchian(df, exit_n)
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = tp_r * sl_atr * a if tp_r else np.nan
    if trail_atr:
        out["trail"] = trail_atr * a
    # opposite channel exit handled by flip signals; explicit exit column unused
    return out
