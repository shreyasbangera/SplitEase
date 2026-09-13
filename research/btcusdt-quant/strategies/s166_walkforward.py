"""
S166 - Walk-forward combination selection. The number that is actually defensible.

WHY THIS FILE EXISTS
---------------------
S165 searched 1,786 subset-weight combinations over 4.5 years and the winner
reached 250.0% at a realised 20% drawdown. It also measured what that search
costs: a combination selected on the first half retained 55% of its figure on the
second. So roughly HALF of any headline produced by that search is the search
itself, and continuing to search until something crosses 300% would be
manufacturing a number rather than finding one.

The correct way to use a selector is to never let it see the data it is judged
on. So the allocation is refit on a schedule, using only data available at the
time, and the out-of-sample quarters are stitched into one stream:

    every RESELECT months, rank every subset-weight combination on the trailing
    LOOKBACK months, take the best, hold it for the next RESELECT months, and
    record only what it earned in that forward window.

That is the same architecture V7 itself uses for its own configuration
selection, so it is consistent with the thing being allocated.

THREE BASELINES, because a walk-forward selector is worth nothing unless it
beats not selecting at all - and S106b already showed informed weighting losing
to equal weights by 8 standard deviations in this study:

    V7 alone                       no allocation decision at all
    60% V7 + equal rest            one round number, nothing fitted
    walk-forward selection         the selector, judged out of sample

WHAT REMAINS IN-SAMPLE EVEN HERE, STATED PLAINLY
-------------------------------------------------
The ALLOCATION is walk-forward. The SLEEVES are not. V7's 200-config grid, the
crowding feature set and its signs, and the rex parameters were all built with
the full sample in view. This measures the honesty of the combination step only,
and the sleeve-construction bias sits underneath every figure below. It cannot
be removed without rebuilding the sleeves from scratch on a truncated sample,
which is a different and much larger piece of work.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, itertools, json

import strategies.s162_maxprofit as MP
import strategies.s165_search as SR
import strategies.s158_final as F8
from strategies.s96_rank import at_gate, stats_of

LOOKBACK, RESELECT = 12, 3          # months
MIN_DAYS = 250


def calmar(a):
    a = np.asarray(a, float)
    if len(a) < 60 or np.allclose(a, 0):
        return -9e9
    eq = np.cumprod(1.0 + a)
    if not np.isfinite(eq[-1]) or eq[-1] <= 0:
        return -9e9
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    yrs = len(a) / 365.25
    cg = eq[-1] ** (1 / yrs) - 1
    return cg / abs(dd) if dd < -1e-6 else -9e9


def walk(Z, cols, v7="V7", lookback=LOOKBACK, reselect=RESELECT):
    """Refit the allocation every `reselect` months on the trailing `lookback`
    months, and keep only what the chosen allocation earns FORWARD."""
    others = [c for c in cols if c != v7]
    combos = []
    for r in range(0, len(others) + 1):
        for cb in itertools.combinations(others, r):
            for wv in (SR.WEIGHTS if cb else (1.0,)):
                combos.append((wv, cb))
    idx = Z.index
    start = idx.min() + pd.DateOffset(months=lookback)
    edges = pd.date_range(start, idx.max(), freq=f"{reselect}MS")
    pieces, picks = [], []
    for e in edges:
        tr = Z[(idx >= e - pd.DateOffset(months=lookback)) & (idx < e)]
        fw = Z[(idx >= e) & (idx < e + pd.DateOffset(months=reselect))]
        if len(tr) < MIN_DAYS or not len(fw):
            continue
        best, bs = None, -9e9
        for wv, cb in combos:
            rest = (1 - wv) / len(cb) if cb else 0.0
            s = tr[v7] * wv + (tr[list(cb)].sum(axis=1) * rest if cb else 0.0)
            c = calmar(s.to_numpy(float))
            if c > bs:
                bs, best = c, (wv, cb)
        wv, cb = best
        rest = (1 - wv) / len(cb) if cb else 0.0
        pieces.append(fw[v7] * wv + (fw[list(cb)].sum(axis=1) * rest if cb else 0.0))
        picks.append((str(e.date()), wv, cb))
    return (pd.concat(pieces) if pieces else pd.Series(dtype=float)), picks


def rep(a, tag, w=38):
    a_ = np.asarray(a, float)
    st = stats_of(a_); rg, sc = MP.realised_gate(a_); hg, _ = F8.honest_gate(a_)
    print(f"   {tag:>{w}}{st['sharpe']:>7.2f}"
          + (f"{rg:>10.1f}%" if np.isfinite(rg) else f"{'n/a':>11}")
          + (f"{hg:>9.1f}%" if np.isfinite(hg) else f"{'n/a':>10}")
          + f"{st['dd']*100:>8.1f}%")
    return rg, sc


if __name__ == "__main__":
    S = SR.pool()
    R = pd.DataFrame(S).dropna()
    Z = pd.DataFrame({c: F8.at_vol(R[c], MP.TARGET_VOL) for c in R.columns})
    cols = list(Z.columns)
    others = [c for c in cols if c != "V7"]
    print("S166 - walk-forward allocation\n")
    print(f"pool {len(cols)} sleeves, {R.index.min().date()} -> "
          f"{R.index.max().date()} ({len(R)/365.25:.1f} years)")
    print(f"refit every {RESELECT} months on the trailing {LOOKBACK} months\n")

    wf, picks = walk(Z, cols)
    print(f"{'book':>38}{'Shp':>7}{'realised':>11}{'honest':>10}{'maxDD':>8}")
    rep(Z["V7"], "V7 alone (no allocation decision)")
    rep(Z["V7"] * 0.6 + Z[others].sum(axis=1) * (0.4 / len(others)),
        "60% V7 + equal rest (nothing fitted)")
    rep(Z["V7"] * 0.7 + Z[others].sum(axis=1) * (0.3 / len(others)),
        "70% V7 + equal rest (nothing fitted)")
    if len(wf):
        wfs = Z.loc[wf.index]
        rep(wfs["V7"], "  (V7 alone over the same window)")
        rg, sc = rep(wf, "WALK-FORWARD selection (out of sample)")
    else:
        print("   walk-forward produced no windows."); sys.exit(0)

    print(f"\nwalk-forward covers {wf.index.min().date()} -> "
          f"{wf.index.max().date()}, {len(wf)} days "
          f"({len(wf)/365.25:.1f} years), {len(picks)} refits")
    print("\n   what the selector picked, quarter by quarter:")
    for d, wv, cb in picks:
        print(f"      {d}  {wv*100:>3.0f}% V7 + "
              f"{'+'.join(cb) if cb else '(nothing)'}")

    scaled = wf * sc
    s = stats_of(scaled.to_numpy(float))
    print(f"\n{'='*78}\nTHE DEFENSIBLE NUMBER (allocation never saw its own test data)")
    print(f"{'='*78}")
    print(f"   Sharpe {s['sharpe']:.2f}   CAGR {s['cagr']*100:.1f}%   "
          f"maxDD {s['dd']*100:.1f}%   scale x{sc:.2f}")
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")
    print(f"\n   target 300%: "
          + ("REACHED" if s["cagr"] * 100 >= 300
             else f"{300/max(s['cagr']*100,1e-9):.2f}x short"))
    print(f"   S165 in-sample headline was 250.0%; this is the same machinery")
    print(f"   with the selector blindfolded.")
    json.dump({"cagr": float(s["cagr"] * 100), "sharpe": float(s["sharpe"]),
               "dd": float(s["dd"] * 100), "days": int(len(wf)),
               "refits": len(picks)},
              open("/home/user/quant/results/s166_walkforward.json", "w"), indent=1)
    print("\ndone: walk-forward")
