"""
S172 - Holding V7 to the standard everything else was held to.

WHY THIS IS OVERDUE
--------------------
Every strategy in S147-S171 was measured against a surrogate control, a
point-in-time universe, a delisting haircut and an out-of-sample split. V7 - the
one with live capital on it - has never been put through any of that, and it is
the book the whole study has been benchmarking against.

Its quarterly configuration selection IS causal: each quarter ranks on the
preceding twelve months and nothing later, and the cached rankings confirm 18
quarters from 2022-03. That part is clean. What is NOT clean is everything the
selection was handed:

    THE GRID        5 exponents x 2 stops x 2 targets x 2 holds x 5 gates. Those
                    ranges were arrived at across S45-S87 with the full sample
                    in view.
    THE SIGNALS     LONG = [flow, cmpx, btcdom, fundz, posn], annotated in the
                    source with the span "2021-03 -> 2026-08" - the whole sample.
    THE ARCHITECTURE 12-month lookback, 3-month reselection, top-3 blend.

None of those can be un-chosen. But each can be PERTURBED, and that measures the
same thing from the other side: if a design choice was fitted, disturbing it
should hurt disproportionately. If V7 survives having its grid halved at random,
a signal removed, and its architecture varied, then the specific choices were not
carrying the result.

FOUR TESTS
----------
  1  RANDOM GRID HALVES. Run the machinery on 200 random halves of the config
     grid. If the full grid's result sits at the top of that distribution, the
     grid was fitted. If it sits in the middle, it was not.
  2  LEAVE ONE SIGNAL OUT. Rebuild the composite five times, dropping each
     signal in turn. A book that collapses when any one is removed is a book
     whose signal set was chosen by looking.
  3  ARCHITECTURE SWEEP. k, lookback and reselection frequency varied around
     V7's (3, 12, 3). A sharp peak at exactly V7's values is a fitted peak.
  4  DECAY. Rolling 12-month return through the live period. S119 found the
     crowding edge falling from 21.9% in 2021 to 0.8% by 2025, and V7 is built
     on that family.

Config streams are computed ONCE over the full sample and then sliced, rather
than re-run per window. That is also closer to what a live book does - it does
not reset its equity every quarter.
"""
import sys, os; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s45_single as S
import strategies.s46_net as S46
from research.harness import backtest, OOS_END
from strategies.s69_calsel import daily
from strategies.s96_rank import at_gate, stats_of

LOOKBACK, RESELECT, TOPK = 12, 3, 3
EXPS = (1.0, 1.5, 2.0, 2.5, 3.0)
STOPS = (2.5, 3.0)
RRS = (2.0, 3.0)
HOLDS = (14, 21)
GATES = [(0, ""), (100, "s"), (150, "s"), (200, "s"), (300, "s")]
GRID = [(p, st, rr, h, sp, md) for p in EXPS for st in STOPS for rr in RRS
        for h in HOLDS for (sp, md) in GATES]
RISK = 0.144
RNG = np.random.default_rng(17)
_G = {}


def grid_and_trend():
    if "g" not in _G:
        g = S.grid(S.FULL_START)
        c = pd.Series(g.close.to_numpy(float))
        _G["g"] = g
        _G["trend"] = {sp: (c > c.ewm(span=sp, adjust=False).mean()).to_numpy()
                       for sp, _ in GATES if sp}
    return _G["g"], _G["trend"]


def shape(v, p, cap=3.0):
    nz = np.abs(v) > 0
    if nz.sum() == 0:
        return v
    bm = float(np.abs(v[nz]).mean())
    u = np.sign(v) * np.abs(v) ** p
    u = u * (bm / max(np.abs(u[nz]).mean(), 1e-12))
    return np.sign(u) * np.minimum(np.abs(u), cap)


def streams(signals, tag, risk=RISK / TOPK):
    """One daily return series per configuration, full sample, cached."""
    cache = f"/home/user/quant/results/s172_{tag}.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    g, TR = grid_and_trend()
    v = S.composite(g, signals)
    a = g.atr14.to_numpy(float)
    out = {}
    for i, (p, stp, rr, hold, sp, md) in enumerate(GRID):
        u0 = np.nan_to_num(shape(v, p))
        u = u0.copy()
        if sp:
            block = (u < 0) & TR[sp] if "s" in md else np.zeros(len(u), bool)
            u = np.where(block, 0.0, u)
        arr = dict(entry=u, stop=stp * a, tp=stp * rr * a,
                   exit=(np.abs(u0) <= 0.0).astype(float))
        m = backtest(g, arr, "12h", start=S.FULL_START, end=OOS_END, risk=risk,
                     max_lev=10.0, max_bars_h=hold * 24)
        out[(p, stp, rr, hold, sp, md)] = daily(m)
        if (i + 1) % 50 == 0:
            print(f"      {tag}: {i+1}/{len(GRID)}", flush=True)
    B = pd.DataFrame(out)
    B.index = pd.to_datetime(B.index)
    if B.index.tz:
        B.index = B.index.tz_localize(None)
    pd.to_pickle(B, cache)
    return B


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


def wf(B, k=TOPK, lookback=LOOKBACK, reselect=RESELECT, cols=None):
    """Quarterly top-k on trailing Calmar, causal, over the given columns."""
    Bc = B[list(cols)] if cols is not None else B
    idx = Bc.index
    t0 = idx.min() + pd.DateOffset(months=lookback)
    edges = pd.date_range(t0, idx.max(), freq=f"{reselect}MS")
    pieces = []
    for e in edges:
        tr = Bc[(idx >= e - pd.DateOffset(months=lookback)) & (idx < e)]
        fw = Bc[(idx >= e) & (idx < e + pd.DateOffset(months=reselect))]
        if len(tr) < min(200, lookback * 25) or not len(fw):
            continue
        sc = tr.apply(lambda c: calmar(c.to_numpy(float)))
        top = sc.sort_values(ascending=False).index[:k]
        pieces.append(fw[list(top)].sum(axis=1))
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def gate_of(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return np.nan
    g = at_gate(a, lo=1e-4, hi=80.0, iters=80)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


if __name__ == "__main__":
    print("S172 - stress-testing V7\n")
    print(f"grid {len(GRID)} configs, architecture (top-{TOPK}, {LOOKBACK}m "
          f"lookback, {RESELECT}m reselect), signals {S46.LONG}\n")
    B = streams(S46.LONG, "full")
    base = wf(B)
    bg = gate_of(base.to_numpy(float))
    bs = stats_of(base.to_numpy(float))
    print(f"REPRODUCED V7: Sharpe {bs['sharpe']:.2f}, {bg:.1f}% at a 20% "
          f"drawdown, {base.index.min().date()} -> {base.index.max().date()}")
    print(f"   (s87's published figure is 178.8%; this rebuild slices full-sample"
          f"\n    streams instead of re-running each window, so it will differ "
          f"slightly)\n")

    # ---- 1. random grid halves -------------------------------------------
    print("1. RANDOM GRID HALVES - is the result carried by the specific grid?")
    cols = list(B.columns); half = len(cols) // 2
    vals = []
    for _ in range(200):
        sub = list(RNG.choice(len(cols), half, replace=False))
        g_ = gate_of(wf(B, cols=[cols[i] for i in sub]).to_numpy(float))
        if np.isfinite(g_):
            vals.append(g_)
    vals = np.array(vals)
    pct = float((vals < bg).mean() * 100)
    print(f"   {len(vals)} random halves: median {np.median(vals):.1f}%, "
          f"5th {np.percentile(vals,5):.1f}%, 95th {np.percentile(vals,95):.1f}%")
    print(f"   the full grid sits at the {pct:.0f}th percentile of that "
          f"distribution")
    print(f"   -> {'FITTED: the specific grid is doing the work' if pct > 90 else 'the grid is not carrying the result'}\n")

    # ---- 2. leave one signal out -----------------------------------------
    print("2. LEAVE ONE SIGNAL OUT - was the five-signal set chosen by looking?")
    print(f"   {'signals used':>34}{'Shp':>7}{'at -20%':>10}{'vs full':>10}")
    print(f"   {'all five (V7)':>34}{bs['sharpe']:>7.2f}{bg:>9.1f}%{'—':>10}")
    for drop in S46.LONG:
        sub = [s for s in S46.LONG if s != drop]
        try:
            Bs = streams(sub, "drop_" + drop)
            r = wf(Bs)
            g_ = gate_of(r.to_numpy(float)); s_ = stats_of(r.to_numpy(float))
            print(f"   {'without ' + drop:>34}{s_['sharpe']:>7.2f}"
                  + (f"{g_:>9.1f}%" if np.isfinite(g_) else f"{'n/a':>10}")
                  + (f"{g_-bg:>+9.1f}pp" if np.isfinite(g_) else f"{'n/a':>10}"))
        except Exception as e:
            print(f"   {'without ' + drop:>34}   failed: {e}")

    # ---- 3. architecture sweep -------------------------------------------
    print("\n3. ARCHITECTURE SWEEP - is (top-3, 12m, 3m) a peak or a plateau?")
    print(f"   {'top k':>8}{'lookback':>10}{'reselect':>10}{'Shp':>7}{'at -20%':>10}")
    for k in (1, 3, 5, 8, 12):
        r = wf(B, k=k)
        g_ = gate_of(r.to_numpy(float)); s_ = stats_of(r.to_numpy(float))
        print(f"   {k:>8}{LOOKBACK:>10}{RESELECT:>10}{s_['sharpe']:>7.2f}"
              + (f"{g_:>9.1f}%" if np.isfinite(g_) else f"{'n/a':>10}"))
    for lb in (6, 18, 24):
        r = wf(B, lookback=lb)
        if not len(r):
            print(f"   {TOPK:>8}{lb:>10}{RESELECT:>10}      no usable windows")
            continue
        g_ = gate_of(r.to_numpy(float)); s_ = stats_of(r.to_numpy(float))
        print(f"   {TOPK:>8}{lb:>10}{RESELECT:>10}{s_['sharpe']:>7.2f}"
              + (f"{g_:>9.1f}%" if np.isfinite(g_) else f"{'n/a':>10}"))
    for rs in (1, 6):
        r = wf(B, reselect=rs)
        if not len(r):
            print(f"   {TOPK:>8}{LOOKBACK:>10}{rs:>10}      no usable windows")
            continue
        g_ = gate_of(r.to_numpy(float)); s_ = stats_of(r.to_numpy(float))
        print(f"   {TOPK:>8}{LOOKBACK:>10}{rs:>10}{s_['sharpe']:>7.2f}"
              + (f"{g_:>9.1f}%" if np.isfinite(g_) else f"{'n/a':>10}"))

    # ---- 4. decay ---------------------------------------------------------
    print("\n4. DECAY - rolling 12-month return through the live period")
    eq = (1 + base).cumprod()
    roll = eq / eq.shift(365) - 1
    roll = roll.dropna()
    print(f"   {'window ending':>16}{'trailing 12m':>14}")
    for d in pd.date_range(roll.index.min(), roll.index.max(), freq="6MS"):
        near = roll.index[np.abs((roll.index - d).days).argmin()]
        print(f"   {str(near.date()):>16}{roll.loc[near]*100:>13.1f}%")
    print(f"\n   best 12m {roll.max()*100:.1f}%, worst {roll.min()*100:.1f}%, "
          f"median {roll.median()*100:.1f}%")
    h = len(roll) // 2
    print(f"   first half median {roll.iloc[:h].median()*100:.1f}%, "
          f"second half median {roll.iloc[h:].median()*100:.1f}%")
    print("\nYEAR BY YEAR (unscaled, as the streams run)")
    for y, v in base.groupby(base.index.year):
        e2 = (1 + v).cumprod(); dd = (e2 / e2.cummax() - 1).min()
        print(f"   {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")
    print("\ndone: V7 stress test")
