"""S4 - Market intraday momentum (Gao, Han, Li & Zhou, JFE 2018).

The published result: the FIRST half-hour return of the day predicts the LAST
half-hour return, in the same direction, across many markets. Implemented here
for gold on hourly bars: take the return of the first hour of the active session
(and optionally the overnight gap), and hold the last k hours of the US session
in that direction. Position is opened at the open of the entry hour and closed
at the session close - a pure intraday holding period, no overnight risk.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, first_h=13, entry_h=19, exit_h=21, sl_atr=3.0, atr_n=24,
           min_move_atr=0.0, use_overnight=False, invert=False):
    idx = df.index
    day = idx.normalize()
    a = ind.atr(df, atr_n)
    hour = idx.hour
    # return of the "first" (signal) hour, known at its close
    r_first = (df.close / df.open - 1).where(hour == first_h)
    r_first = pd.Series(r_first.to_numpy(), index=idx).groupby(day).ffill()
    if use_overnight:
        prev_close = df.close.where(hour == 21).groupby(day).ffill().groupby(day).last()
    move_atr = (df.close - df.open).where(hour == first_h).abs()
    move_atr = pd.Series((move_atr / a).to_numpy(), index=idx).groupby(day).ffill()
    d = np.sign(r_first).fillna(0)
    if invert:
        d = -d
    fire = (hour == entry_h - 1) & (move_atr >= min_move_atr) & d.ne(0)
    sig = pd.Series(0.0, index=idx)
    sig[fire] = d[fire]
    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = np.nan
    out["exit"] = (hour >= exit_h).astype(float)
    return out
