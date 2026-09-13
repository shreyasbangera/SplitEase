"""
S171 - V7's SELECTION machinery on the breakout, in the structure it works in.

WHAT S170 GOT WRONG
--------------------
S170 fed the breakout signal into V7's wrapper by copying the whole wrapper -
including its 12-hour decision grid and its fixed stop/target/hold exit. The raw
book came out at Sharpe -0.08, against 1.08 for the same signal on daily bars
with an ATR trailing exit. The signal did not survive the translation.

That is a real finding rather than a bug: the breakout's edge lives in LETTING A
MOVE RUN, and a fixed take-profit at 2-3x the stop cuts exactly the tail it is
trying to capture. V7's crowding signal is a mean-reverting fade, which wants the
opposite exit. The exit structure is part of the strategy, not packaging.

So the wrapper is split into its two parts, and only the transferable one is used:

    NOT TRANSFERABLE   the 12h grid and the stop/target/hold exit - these belong
                       to a fade, not a breakout
    TRANSFERABLE       the CONFIGURATION GRID, the quarterly reselection on
                       trailing Calmar, and the top-k blend

So: keep the breakout exactly as it works - daily bars, range-expansion entry,
ATR trailing stop - build a grid over ITS OWN parameters, and select quarterly
the way V7 does.

    grid    5 lookbacks x 4 quantiles x 3 trail widths x 4 trend gates = 240
    rank    trailing 12-month Calmar, refit quarterly, causal
    hold    top k at equal weight, k swept

The honest comparison is not the raw single config. It is the GRID AVERAGE -
holding all 240 equally - because S146 found averaging across parameters to be
the one combination that does not dilute, and because if selection cannot beat
"hold everything" then selection is worth nothing. That is the test S166 applied
to the allocation, and the allocation failed it.
"""
import sys, os, json; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s135_more as S135
from strategies.s96_rank import at_gate, stats_of

LOOKBACK, RESELECT = 12, 3
NS = (34, 55, 89, 144, 233)
QS = (0.80, 0.85, 0.90, 0.95)
KS = (2.0, 3.0, 4.0)
GATES = (0, 50, 100, 200)
GRID = [(n, q, k, g) for n in NS for q in QS for k in KS for g in GATES]
FEE_BPS, SLIP_BPS = 5.0, 3.0


def base():
    px, hi, lo, qv, tbq, fd = S135.base()
    px.index = pd.to_datetime(px.index)
    if px.index.tz:
        px.index = px.index.tz_localize(None)
    for d in (hi, lo, fd):
        d.index = px.index
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    return px, hi, lo, fd, atr


def positions(px, hi, lo, atr, N, q, atr_k, gate_span):
    """Range-expansion entry, ATR trailing exit, optional trend gate on shorts.

    Everything is lagged: the threshold uses bars strictly before the one it
    judges, and the resulting position is shifted a further bar before it earns.
    """
    rng = (hi - lo) / px.replace(0, np.nan)
    thr = rng.rolling(N, min_periods=max(N // 2, 20)).quantile(q).shift(1)
    up = ((rng > thr) & (px > px.shift(1))).fillna(False).to_numpy()
    dn = ((rng > thr) & (px < px.shift(1))).fillna(False).to_numpy()
    if gate_span:
        trend = (px > px.ewm(span=gate_span, adjust=False).mean()
                 ).shift(1).fillna(False).to_numpy()
        dn = dn & ~trend                      # no shorts above the trend line
    c = px.to_numpy(float); a = atr.to_numpy(float)
    s = np.zeros(len(c)); cur = 0.0; peak = 0.0
    for i in range(len(c)):
        if not np.isfinite(a[i]) or not np.isfinite(c[i]):
            s[i] = cur; continue
        if cur == 0.0:
            if up[i]:   cur, peak = 1.0, c[i]
            elif dn[i]: cur, peak = -1.0, c[i]
        elif cur > 0:
            peak = max(peak, c[i])
            if c[i] < peak - atr_k * a[i]: cur = 0.0
        else:
            peak = min(peak, c[i])
            if c[i] > peak + atr_k * a[i]: cur = 0.0
        s[i] = cur
    return pd.Series(s, index=px.index).shift(1).fillna(0.0)


def book(px, pos, fd, target_vol=0.30, vol_win=60, max_lev=3.0):
    """Risk-sized daily returns, net of fees, slippage and funding."""
    r = px.pct_change().fillna(0.0)
    rv = r.rolling(vol_win, min_periods=40).std() * np.sqrt(365.25)
    w = (pos * (target_vol / rv.replace(0, np.nan))).clip(-max_lev, max_lev).fillna(0.0)
    turn = w.diff().abs().fillna(w.abs())
    cost = turn * (FEE_BPS + SLIP_BPS) / 1e4
    carry = w * fd.reindex_like(w).fillna(0.0)
    return w * r - cost - carry


def calmar(a):
    a = np.asarray(a, float)
    if len(a) < 60 or np.allclose(a, 0):
        return -9e9
    eq = np.cumprod(1.0 + a)
    if not np.isfinite(eq[-1]) or eq[-1] <= 0:
        return -9e9
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    if dd > -1e-6:
        return -9e9
    return (eq[-1] ** (365.25 / len(a)) - 1) / abs(dd)


def gate_of(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return np.nan, np.nan
    g = at_gate(a, lo=1e-4, hi=80.0, iters=80)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan, np.nan
    return g["cagr"] * 100, g["scale"]


if __name__ == "__main__":
    px, hi, lo, fd, atr = base()
    print("S171 - V7's selection machinery on the breakout, daily + ATR trail\n")
    print(f"BTC daily {px.index.min().date()} -> {px.index.max().date()}, "
          f"{len(px)} bars")
    print(f"{len(GRID)} configurations, {LOOKBACK}m lookback, {RESELECT}m "
          f"reselection\n")

    cache = "/home/user/quant/results/s171_streams.pkl"
    if os.path.exists(cache):
        ST = pd.read_pickle(cache)
    else:
        ST = {}
        for i, (n, q, k, gsp) in enumerate(GRID):
            pos = positions(px, hi, lo, atr, n, q, k, gsp)
            ST[(n, q, k, gsp)] = book(px, pos, fd)
            if (i + 1) % 60 == 0:
                print(f"   built {i+1}/{len(GRID)} configurations", flush=True)
        pd.to_pickle(ST, cache)
    B = pd.DataFrame(ST)
    print(f"built {B.shape[1]} return streams\n")

    print("BASELINE - hold everything, no selection at all")
    avg = B.mean(axis=1)
    a = avg.to_numpy(float); s = stats_of(a); g, sc = gate_of(a)
    print(f"   grid average (all {B.shape[1]}): Sharpe {s['sharpe']:.2f}, "
          + (f"{g:.1f}% at a 20% drawdown" if np.isfinite(g) else "gate n/a"))

    idx = B.index
    t0 = idx.min() + pd.DateOffset(months=LOOKBACK)
    edges = pd.date_range(t0, idx.max(), freq=f"{RESELECT}MS")
    print(f"\nQUARTERLY SELECTION on trailing {LOOKBACK}m Calmar, "
          f"{len(edges)} refits")
    print(f"   {'top k held':>12}{'Shp':>8}{'CAGR':>9}{'maxDD':>8}{'at -20%':>10}"
          f"{'baseline':>11}{'lift':>10}")
    results = {}
    for k in (1, 3, 5, 8, 12, 20, 40):
        pieces = []
        for e in edges:
            tr = B[(idx >= e - pd.DateOffset(months=LOOKBACK)) & (idx < e)]
            fw = B[(idx >= e) & (idx < e + pd.DateOffset(months=RESELECT))]
            if len(tr) < 250 or not len(fw):
                continue
            sc_ = tr.apply(lambda c: calmar(c.to_numpy(float)))
            top = sc_.sort_values(ascending=False).index[:k]
            pieces.append(fw[list(top)].mean(axis=1))
        if not pieces:
            continue
        ser = pd.concat(pieces)
        a = ser.to_numpy(float); st = stats_of(a); gg, scl = gate_of(a)
        base_g, _ = gate_of(avg.reindex(ser.index).to_numpy(float))
        results[k] = (ser, gg, scl)
        print(f"   {k:>12}{st['sharpe']:>8.2f}{st['cagr']*100:>8.1f}%"
              f"{st['dd']*100:>7.1f}%"
              + (f"{gg:>9.1f}%" if np.isfinite(gg) else f"{'n/a':>10}")
              + (f"{base_g:>10.1f}%" if np.isfinite(base_g) else f"{'n/a':>11}")
              + (f"{gg-base_g:>+9.1f}pp" if np.isfinite(gg) and np.isfinite(base_g)
                 else f"{'n/a':>10}"))

    if results:
        bk = max(results, key=lambda k: results[k][1]
                 if np.isfinite(results[k][1]) else -1)
        ser, gg, scl = results[bk]
        scaled = ser * scl
        s = stats_of(scaled.to_numpy(float))
        print(f"\n{'='*72}\nBEST: top-{bk} blend, {gg:.1f}% at a 20% drawdown")
        print(f"{'='*72}")
        print(f"   Sharpe {s['sharpe']:.2f}   window {ser.index.min().date()} -> "
              f"{ser.index.max().date()} ({len(ser)/365.25:.1f} years)")
        for y, v in scaled.groupby(scaled.index.year):
            eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
            print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
                  f"{dd*100:>6.1f}%   ({len(v)} days)")
        pd.to_pickle(ser, "/home/user/quant/results/s171_best.pkl")
    print("\ndone: grid selection on the breakout")
