"""S11 - Trend-gated momentum pulse (4h bars).

Built from the measured edge table rather than from a chart pattern. On 4h BTC
data, three triggers each show positive EXCESS forward return over a 12-24h
horizon, all of them momentum-continuation and all of them only while price is
above its long moving average:

    trigger A  20-bar Donchian high break        +38bp excess / 12h  (t~2.8)
    trigger B  RSI(2) > 95  (short-term thrust)  +37bp excess / 12h  (t~3.1)
    trigger C  break out of a volatility squeeze +14bp excess /  4h  (t~2.4)

Down-breaks show NEGATIVE excess for shorts (they mean-revert), so the strategy
is long/flat by default. The position is a fixed-horizon "pulse": hold for a set
number of bars unless a volatility stop or trailing stop ends it sooner.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/user/SplitEase/research/engine")
import indicators as ind


def signal(df, ma_n=200, don_n=20, rsi_thr=95, use_donch=True, use_rsi=True,
           use_squeeze=True, sq_n=120, sq_pct=0.2, hold_bars=6, sl_atr=2.5,
           atr_n=14, trail_atr=0.0, tp_r=0.0, allow_short=False, be_at_r=0.0):
    c = df.close
    a = ind.atr(df, atr_n)
    ma = ind.sma(c, ma_n)
    hh, ll = ind.donchian(df, don_n)
    r2 = ind.rsi(c, 2)
    bm, bh, bl, sd = ind.bbands(c, 20, 2.0)
    sq = (sd / c).rolling(sq_n).rank(pct=True)

    trig_up = pd.Series(False, index=df.index)
    if use_donch:
        trig_up |= c > hh.shift(1)
    if use_rsi:
        trig_up |= r2 > rsi_thr
    if use_squeeze:
        trig_up |= (sq.shift(1) < sq_pct) & (c > hh.shift(1))
    long_s = trig_up & (c > ma)

    sig = pd.Series(0.0, index=df.index)
    sig[long_s] = 1.0
    if allow_short:
        trig_dn = pd.Series(False, index=df.index)
        if use_donch:
            trig_dn |= c < ll.shift(1)
        if use_rsi:
            trig_dn |= r2 < (100 - rsi_thr)
        sig[trig_dn & (c < ma)] = -1.0

    out = pd.DataFrame({"sig": sig, "sl_dist": sl_atr * a})
    out["tp_dist"] = tp_r * sl_atr * a if tp_r else np.nan
    out["max_bars"] = hold_bars
    if trail_atr:
        out["trail"] = trail_atr * a
    if be_at_r:
        out["be_at_r"] = be_at_r
    return out
