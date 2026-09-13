"""
S170 - V7's machinery, applied to price signals instead of crowding.

THE OBSERVATION THIS IS BUILT ON
---------------------------------
V7's edge is not its signal. The raw crowding composite is Sharpe 1.32 (S118).
Wrapped in V7's machinery it becomes 2.13 - a 60% lift from the WRAPPER, not the
idea. The wrapper is:

    a grid of 200 configurations   5 conviction exponents x 2 stop distances
                                   x 2 reward:risk targets x 2 holding caps
                                   x 5 trend gates
    quarterly reselection          rank on trailing 12-month Calmar, refit every
                                   3 months, using only data available then
    a top-3 blend                  hold the best three at a third of the risk each

That wrapper has never been pointed at anything except the crowding composite.
The breakout family - the best price-based signal in this study at Sharpe 1.08,
and only +0.139 correlated to V7 - has only ever been run as a plain always-on
book with an ATR trailing stop.

So this file feeds the breakout signal into V7's wrapper unchanged. If the lift
transfers, Sharpe 1.08 becomes something near 1.7, and the target of 150% at a
20% drawdown is in reach from a book that shares no data with the live one. If it
does not transfer, the lift was specific to the crowding signal and that is worth
knowing too.

WHAT IS HELD FIXED, SO THE COMPARISON MEANS SOMETHING
------------------------------------------------------
The grid, the lookback, the reselection frequency, the blend width and the risk
budget are all V7's, copied rather than re-tuned. The ONLY thing changed is the
signal being fed in. Anything else and this would be a new strategy that happens
to resemble V7, rather than a measurement of what the wrapper is worth.

The selection is causal - each quarter ranks on the preceding twelve months and
nothing later - but the GRID was specified by copying V7's, which was itself
built with the sample in view. That inherited bias is stated, not removed.
"""
import sys, os, json; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s45_single as S
from research.harness import backtest, OOS_END
from strategies.s69_calsel import daily, calmar_of
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"
LOOKBACK, RESELECT = 12, 3
EXPS = (1.0, 1.5, 2.0, 2.5, 3.0)
STOPS = (2.5, 3.0)
RRS = (2.0, 3.0)
HOLDS = (14, 21)
GATES = [(0, ""), (100, "s"), (150, "s"), (200, "s"), (300, "s")]
GRID = [(p, st, rr, h, sp, md) for p in EXPS for st in STOPS for rr in RRS
        for h in HOLDS for (sp, md) in GATES]

_G = {}


def grid12():
    """V7's 12h decision grid, with true 12h high/low added for the breakout."""
    if "g" in _G:
        return _G["g"]
    g = S.grid(S.FULL_START).copy()
    b = pd.read_parquet(f"{D}/fut_1h.parquet")
    b["dt"] = pd.to_datetime(b.dt, utc=True)
    b = b.set_index("dt")
    r = b.resample("12h")
    hi = r["high"].max(); lo = r["low"].min()
    g["high"] = hi.reindex(g.dt).to_numpy()
    g["low"] = lo.reindex(g.dt).to_numpy()
    _G["g"] = g
    return g


def rex_signal(g, N, q, cap=2.0):
    """Continuous range-expansion breakout on the 12h grid.

    Magnitude is how far the bar's range exceeds its own trailing quantile, so
    the conviction exponent in the grid has something to act on. Everything is
    shifted: the threshold uses bars strictly before the one being judged.
    """
    c = g.close.astype(float)
    rng = (g.high.astype(float) - g.low.astype(float)) / c.replace(0, np.nan)
    thr = rng.rolling(N, min_periods=max(N // 2, 20)).quantile(q).shift(1)
    excess = (rng / thr.replace(0, np.nan) - 1.0).clip(lower=0.0, upper=cap)
    direction = np.sign(c.diff())
    v = (direction * excess).where(rng > thr, 0.0)
    return np.nan_to_num(v.to_numpy(float))


def don_signal(g, N, cap=2.0):
    """Donchian channel breakout, magnitude = how far beyond the channel."""
    c = g.close.astype(float)
    up = g.high.astype(float).rolling(N).max().shift(1)
    dn = g.low.astype(float).rolling(N).min().shift(1)
    a = g.atr14.astype(float).replace(0, np.nan)
    v = np.where(c > up, ((c - up) / a).clip(0, cap),
                 np.where(c < dn, -((dn - c) / a).clip(0, cap), 0.0))
    return np.nan_to_num(v)


def trend_of(g, span):
    c = g.close.astype(float)
    return (c > c.ewm(span=span, adjust=False).mean()).to_numpy()


def shape(v, p, nz=None):
    """V7's conviction transform: raise |v| to p, rescale so mean |u| is
    unchanged, cap at 3."""
    nz = np.abs(v) > 0 if nz is None else nz
    if nz.sum() == 0:
        return v
    bm = float(np.abs(v[nz]).mean())
    u = np.sign(v) * np.abs(v) ** p
    u = u * (bm / max(np.abs(u[nz]).mean(), 1e-12))
    return np.sign(u) * np.minimum(np.abs(u), 3.0)


def sim(g, v, cfg, start, end, risk):
    p, stp, rr, hold, sp, md = cfg
    u0 = shape(v, p)
    u = u0.copy()
    if sp:
        up = trend_of(g, sp)
        block = np.zeros(len(u), bool)
        if "s" in md:
            block |= (u < 0) & up
        if "l" in md:
            block |= (u > 0) & ~up
        u = np.where(block, 0.0, u)
    a = g.atr14.to_numpy(float)
    arr = dict(entry=np.nan_to_num(u), stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u0) <= 0.0).astype(float))
    return backtest(g, arr, "12h", start=start, end=end, risk=risk,
                    max_lev=10.0, max_bars_h=hold * 24)


def rankings(g, v, cache):
    """Quarterly top-config ranking on trailing 12-month Calmar. Causal."""
    if os.path.exists(cache):
        return [(s, e, [tuple(c) for c in cs]) for s, e, cs in json.load(open(cache))]
    t0 = pd.Timestamp(S.FULL_START, tz="UTC")
    t = t0 + pd.DateOffset(months=LOOKBACK)
    end = pd.Timestamp(OOS_END, tz="UTC")
    out = []
    while t < end:
        te = min(t + pd.DateOffset(months=RESELECT), end)
        tr0 = t - pd.DateOffset(months=LOOKBACK)
        scored = []
        for cfg in GRID:
            m = sim(g, v, cfg, str(tr0.date()), str(t.date()), 0.05)
            scored.append((calmar_of(m), cfg))
        scored.sort(key=lambda x: -x[0])
        out.append((str(t.date()), str(te.date()), [c for _, c in scored[:8]]))
        t = te
    json.dump(out, open(cache, "w"))
    return [(s, e, [tuple(c) for c in cs]) for s, e, cs in out]


def blend(g, v, R, k, risk):
    segs = []
    for s, e, cfgs in R:
        rs = []
        for cfg in cfgs[:k]:
            m = sim(g, v, cfg, s, e, risk / k)
            rs.append(daily(m))
        segs.append(pd.DataFrame({j: r for j, r in enumerate(rs)})
                    .fillna(0.0).sum(axis=1))
    return pd.concat(segs)


def gate_of(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return np.nan, np.nan
    gg = at_gate(a, lo=1e-4, hi=80.0, iters=80)
    if not np.isfinite(gg["dd"]) or abs(gg["dd"] + 0.20) > tol:
        return np.nan, np.nan
    return gg["cagr"] * 100, gg["scale"]


if __name__ == "__main__":
    g = grid12()
    print("S170 - V7's machinery on price signals\n")
    print(f"12h grid {g.dt.min().date()} -> {g.dt.max().date()}, {len(g)} bars")
    print(f"grid of {len(GRID)} configurations, {LOOKBACK}m lookback, "
          f"{RESELECT}m reselection, top-3 blend  (all copied from V7)\n")

    FAM = {
        "rex144": rex_signal(g, 144, 0.95),
        "rex89": rex_signal(g, 89, 0.95),
        "rex_q80": rex_signal(g, 144, 0.80),
        "don55": don_signal(g, 55),
    }
    os.makedirs("/home/user/quant/results/s170", exist_ok=True)
    print(f"{'signal':>10}{'raw Shp':>10}{'raw gate':>11}"
          f"{'V7-wrapped Shp':>17}{'wrapped gate':>15}{'lift':>8}")
    out = {}
    for name, v in FAM.items():
        nz = np.abs(v) > 0
        if nz.sum() < 200:
            print(f"{name:>10}   too few signals ({int(nz.sum())})"); continue
        # raw: the plain always-on book, one config, no grid
        mraw = sim(g, v, (1.0, 3.0, 2.0, 21, 0, ""), S.FULL_START, OOS_END, 0.05)
        araw = daily(mraw).to_numpy(float)
        sraw = stats_of(araw); graw, _ = gate_of(araw)

        R = rankings(g, v, f"/home/user/quant/results/s170/{name}.json")
        br = blend(g, v, R, 3, 0.144)
        br = pd.Series(np.asarray(br, float), index=pd.to_datetime(br.index))
        br.index = br.index.tz_localize(None) if br.index.tz else br.index
        br = br.groupby(br.index.normalize()).sum()
        a = br.to_numpy(float)
        st = stats_of(a); gw, sc = gate_of(a)
        out[name] = br
        print(f"{name:>10}{sraw['sharpe']:>10.2f}"
              + (f"{graw:>10.1f}%" if np.isfinite(graw) else f"{'n/a':>11}")
              + f"{st['sharpe']:>17.2f}"
              + (f"{gw:>14.1f}%" if np.isfinite(gw) else f"{'n/a':>15}")
              + f"{st['sharpe']/max(sraw['sharpe'],1e-9):>8.2f}x")

    if out:
        pd.to_pickle(out, "/home/user/quant/results/s170_books.pkl")
        print("\nBEST WRAPPED BOOK, year by year")
        best = max(out, key=lambda k: (gate_of(out[k].to_numpy(float))[0]
                                       if np.isfinite(gate_of(out[k].to_numpy(float))[0])
                                       else -1))
        b = out[best]; gw, sc = gate_of(b.to_numpy(float))
        scaled = b * sc
        s = stats_of(scaled.to_numpy(float))
        print(f"   {best}: Sharpe {s['sharpe']:.2f}, {gw:.1f}% at a 20% drawdown")
        for y, v2 in scaled.groupby(scaled.index.year):
            eq = (1 + v2).cumprod(); dd = (eq / eq.cummax() - 1).min()
            print(f"      {y}: {((1+v2).prod()-1)*100:>+9.1f}%   in-year DD "
                  f"{dd*100:>6.1f}%   ({len(v2)} days)")
    print("\ndone: wrapped price signals")
