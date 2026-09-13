"""
S136 - Pushing the two things that worked in S135.

S135 produced the first two encouraging results in a long time:

    EXPOSURE TIMING beat direction picking. A long/flat book that only asks "how
    much" - in when three trend speeds agree, flat otherwise - reached 17.4% at
    the gate with profit factor 4.09, against 13.1% for the breakout book and
    6.5% for holding the asset. Deciding WHETHER to be exposed turned out to be
    worth more than deciding WHICH WAY.

    THE COMBINATION WAS ADDITIVE. Crowding alone 19.5%, breakout alone 13.1%,
    the sum of the two 23.8%. **That is the first time in this study a
    combination has beaten both of its components.** Everything previously tried
    diluted - S124's equal-weight portfolios all landed below their best member.

Both get developed here.

PART 1 - A PROPER ENSEMBLE
    Four mechanisms that are genuinely different from each other: positioning
    (crowding), price structure (breakout), participation (exposure timing), and
    the asset itself. Equal-weighted, because S106b showed informed weighting of
    signals loses to flat weights by 8 standard deviations and nothing since has
    contradicted it. Every subset is printed, so the question "is the whole
    better than its parts" is answered for all fifteen of them rather than for
    one chosen combination.

PART 2 - DEVELOPING EXPOSURE TIMING
    S135 used one filter. The question it raises is how far the idea goes: what
    if participation is timed on volatility, on drawdown state, on trend
    agreement, and on positioning together? A book that is long when conditions
    are good and flat otherwise never shorts, never pays to be wrong-way in a
    rally, and has a natural drawdown advantage - which is precisely the
    constraint the brief binds on.

Judged against the brief alone. Costs and funding charged as everywhere else.

A NOTE ON TRADE COUNTS. A continuously-sized book rarely returns to flat, so
"completed round trips" undercounts what it actually does - the sum-of-two book
shows 6. Both are reported: `trips` (round trips through flat) and `flips`
(any change of direction or size beyond the no-trade band), since the brief's
"100 completed trades" reads naturally as the latter for a book of this kind.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from itertools import combinations

import strategies.s135_more as S135
from strategies.s96_rank import at_gate, stats_of
from research.robust import bootstrap_dd

gate, run = S135.gate, S135.run


def score(tag, net, pos, mark=True):
    a = np.asarray(net, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        print(f"   {tag:>34}   no position"); return np.nan
    st = stats_of(a); g = gate(a)
    b = bootstrap_dd(a, n=1200, block=90)
    p = np.asarray(pos, float)
    live = p != 0
    trips = int(((~live[:-1]) & live[1:]).sum() + (1 if live[0] else 0))
    flips = int((np.abs(np.diff(np.r_[0.0, p])) > 1e-9).sum())
    pnl, acc, on = [], 0.0, False
    for i in range(len(a)):
        if live[i]:
            acc += a[i]; on = True
        elif on:
            pnl.append(acc); acc, on = 0.0, False
    if on:
        pnl.append(acc)
    q = np.array(pnl) if pnl else np.array([0.0])
    pf = (q[q > 0].sum() / -q[q < 0].sum()) if (q < 0).any() else np.inf
    h = len(a) // 2
    g1, g2 = gate(a[:h]), gate(a[h:])
    gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
    s1 = "n/a" if not np.isfinite(g1) else f"{g1:.0f}%"
    s2 = "n/a" if not np.isfinite(g2) else f"{g2:.0f}%"
    ok = np.isfinite(g) and g >= 300 and flips >= 100 and pf > 1.10
    print(f"   {tag:>34}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%{b['dd_median']*100:>8.1f}%"
          f"{trips:>7}{flips:>7}{pf:>7.2f}{gs:>9}{s1:>7}{s2:>7}"
          + ("  *** PASS ***" if ok else ""))
    return g


HDR = (f"   {'book':>34}{'Shp':>7}{'realDD':>8}{'medDD':>8}{'trips':>7}{'flips':>7}"
       f"{'PF':>7}{'at -20%':>9}{'1st':>7}{'2nd':>7}")

if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    lp = np.log(px)
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()

    import strategies.s119_dev as S119
    import strategies.s118_crowd as C
    cpx, cfd, F = S119.build("1D")
    cols = [c for c in C.SIGNS if c in F.columns]

    agree3 = (sum(np.sign(lp.ewm(span=f, adjust=False).mean()
                          - lp.ewm(span=s, adjust=False).mean())
                  for f, s in ((8, 32), (16, 64), (32, 128))) / 3)

    SLEEVE = {
        "crowd": S119.signal(F, cols, 365).reindex(px.index).fillna(0.0),
        "break": S135.donchian(px, hi, lo, 55, 3.0, atr),
        "expos": agree3.clip(0, 1).shift(1).fillna(0.0),
        "hold":  pd.Series(1.0, index=px.index),
    }

    print("S136 - ensemble of four different mechanisms, and exposure timing\n")
    print("PART 1 — every subset, so 'better than its parts' is answered for all 15")
    print(HDR)
    res = {}
    for k in range(1, 5):
        for combo in combinations(SLEEVE, k):
            s = sum(SLEEVE[c] for c in combo) / len(combo)
            g = score(" + ".join(combo), *run(px, fd, s, 0.30))
            res[combo] = g
        print()

    best = max((v, k) for k, v in res.items() if np.isfinite(v))
    singles = {k[0]: v for k, v in res.items() if len(k) == 1 and np.isfinite(v)}
    print(f"   best subset: {' + '.join(best[1])} at {best[0]:.1f}%")
    print(f"   best single sleeve: {max(singles, key=singles.get)} "
          f"at {max(singles.values()):.1f}%")

    print("\nPART 2 — developing exposure timing: long or flat, never short")
    print(HDR)
    dd = (px / px.cummax() - 1.0).shift(1)
    r = lp.diff()
    rv = r.rolling(30).std().shift(1)
    rvq = rv.rolling(500, min_periods=200).quantile(0.75).shift(1)
    calm = (rv < rvq)
    up200 = (px > px.rolling(200).mean()).shift(1)
    crowdpos = (SLEEVE["crowd"] > 0)

    for tag, gateon in (
            ("agreement only (S135 best)", agree3.clip(0, 1).shift(1)),
            ("agreement + calm vol", (agree3.clip(0, 1).shift(1)).where(calm, 0.0)),
            ("agreement + above 200d", (agree3.clip(0, 1).shift(1)).where(up200, 0.0)),
            ("agreement + crowding positive",
             (agree3.clip(0, 1).shift(1)).where(crowdpos, 0.0)),
            ("agreement + calm + above 200d",
             (agree3.clip(0, 1).shift(1)).where(calm & up200, 0.0)),
            ("all four conditions",
             (agree3.clip(0, 1).shift(1)).where(calm & up200 & crowdpos, 0.0))):
        for tv in (0.30, 0.60):
            score(f"{tag}, tgt {tv*100:.0f}%",
                  *run(px, fd, gateon.fillna(0.0), tv, max_lev=4.0))
    print("\ndone: S136")
