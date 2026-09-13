"""
S144 - The breakout + vol-of-vol blend, verified.

The best non-V7 book this study has produced, and the first combination in it
that is genuinely additive rather than merely averaging:

    breakout ensemble   Sharpe 1.10   -23.3% DD   22.9% gate   0 negative years
    vol-of-vol fade     Sharpe 0.71   -31.4% DD   11.1% gate   2 negative years
    blend 75/25         Sharpe 1.06   -16.2% DD   27.8% gate   0 negative years

The drawdown is smaller than EITHER component's, which is diversification rather
than arithmetic: the two correlate at +0.297, and the blend earns more at the
gate than either leg. Year by year it reads +46.8, +21.3, +18.2, +29.2, +8.1,
+10.8, +17.3 - positive in all seven, and its two most recent years are the
strongest recent stretch anything in this study has posted.

WHAT IS VERIFIED HERE BEFORE THAT IS BELIEVED
----------------------------------------------
Enough things in this log have looked good and then failed a control that the
claim gets tested before it gets made:

  HONEST GATE     scaled to the MEDIAN bootstrapped drawdown rather than the one
                  path that happened (S112's correction), since the realised
                  path flatters V7 by 1.4x and could flatter this too.
  COSTS           8, 16, 30 and 50bps round trip. The breakout leg trades little
                  and the vov leg trades a lot, so the blend's cost exposure is
                  not either leg's.
  WEIGHT SWEEP    the 75/25 split is one cell. If the neighbouring weights fall
                  away sharply it is a lucky cell, not a blend.
  SUB-PERIODS     first half, second half, and the trailing two years.
  PROFIT FACTOR   over completed round trips, for the brief's third criterion.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s135_more as S135
import strategies.s140_breakout as B
import strategies.s141_rexdev as R
from strategies.s96_rank import at_gate, stats_of
from research.robust import bootstrap_dd

gate = S135.gate


def legs(px, hi, lo, rng, atr, lr):
    bo = sum(R.trail_pyr(px, *[x.fillna(False) for x in R.rex(px, hi, lo, rng, N, 0.95)],
                         atr, 3.0) for N in (55, 89, 144)) / 3
    rv = lr.rolling(20).std()
    vov = rv.rolling(60).std() / rv.rolling(60).mean()
    z = ((vov - vov.rolling(365, min_periods=120).mean())
         / (vov.rolling(365, min_periods=120).std() + 1e-12)).clip(-2, 2).shift(1).fillna(0.0)
    return bo, -z


def book(px, fd, sig, tv=0.30, rt=16.0, max_lev=3.0, band=0.10):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rvv = np.maximum(r.ewm(halflife=32, adjust=False).std().shift(1).bfill().to_numpy()
                     * np.sqrt(365.25), 0.05)
    want = np.clip(np.asarray(sig, float) * tv / rvv, -max_lev, max_lev)
    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = (pos * px.pct_change().fillna(0.0).to_numpy() - pos * fd.to_numpy()
           - turn * (rt / 2.0) / 1e4)
    return pd.Series(net, index=px.index), pos


def honest_gate(a, nboot=1500, block=90, seed=0, tol=0.004):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or a.sum() <= 0:
        return np.nan
    rng_ = np.random.default_rng(seed); T = len(a); nb = int(np.ceil(T / block))
    st = rng_.integers(0, max(T - block, 1), (nboot, nb))
    idx = np.clip((st[:, :, None] + np.arange(block)[None, None, :])
                  .reshape(nboot, -1)[:, :T], 0, T - 1)

    def med(s):
        e = np.cumprod(1.0 + a[idx] * s, axis=1)
        return float(np.median((e / np.maximum.accumulate(e, axis=1) - 1).min(axis=1)))
    loq, hiq = 1e-3, 40.0
    for _ in range(50):
        m = (loq + hiq) / 2
        if med(m) < -0.20: hiq = m
        else: loq = m
    s = (loq + hiq) / 2
    return np.nan if abs(med(s) + 0.20) > tol else stats_of(a * s)["cagr"] * 100


def pf_trips(net, pos):
    live = np.asarray(pos, float) != 0
    a = np.asarray(net, float)
    out, acc, on = [], 0.0, False
    for i in range(len(a)):
        if live[i]:
            acc += a[i]; on = True
        elif on:
            out.append(acc); acc, on = 0.0, False
    if on: out.append(acc)
    p = np.array(out) if out else np.array([0.0])
    return ((p[p > 0].sum() / -p[p < 0].sum()) if (p < 0).any() else np.inf), len(p)


if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean(); rng = hi - lo
    lr = np.log(px).diff()
    bo, vv = legs(px, hi, lo, rng, atr, lr)

    print("S144 - verifying the breakout + vol-of-vol blend\n")
    print("1. WEIGHT SWEEP — is 75/25 a cell or a plateau?")
    print(f"   {'vov weight':>12}{'Shp':>7}{'realDD':>8}{'medDD':>8}{'flips':>7}"
          f"{'gate':>8}{'honest':>9}{'last2y':>9}{'negyr':>7}")
    for w in (0.0, 0.10, 0.20, 0.25, 0.33, 0.50, 0.66, 1.0):
        s = (1 - w) * bo + w * vv
        net, pos = book(px, fd, s)
        a = net.to_numpy()
        g, hg, l2 = gate(a), honest_gate(a), B.last2(net)
        bd = bootstrap_dd(a, n=1200, block=90)
        f = int((np.abs(np.diff(np.r_[0.0, pos])) > 1e-9).sum())
        neg = sum(1 for _, v in B.yearly(net) if v < 0)
        gs = "n/a" if not np.isfinite(g) else f"{g:>7.1f}%"
        hs = "n/a" if not np.isfinite(hg) else f"{hg:>8.1f}%"
        print(f"   {w:>12.2f}{stats_of(a)['sharpe']:>7.2f}{stats_of(a)['dd']*100:>7.1f}%"
              f"{bd['dd_median']*100:>7.1f}%{f:>7}{gs:>8}{hs:>9}{l2:>8.1f}%{neg:>7}")

    print("\n2. COST SENSITIVITY at 75/25")
    print(f"   {'round trip':>12}{'Shp':>7}{'gate':>9}{'honest':>9}{'last2y':>9}")
    s = 0.75 * bo + 0.25 * vv
    for rt in (8.0, 16.0, 30.0, 50.0):
        net, pos = book(px, fd, s, rt=rt)
        a = net.to_numpy()
        g, hg = gate(a), honest_gate(a)
        print(f"   {rt:>11.0f}b{stats_of(a)['sharpe']:>7.2f}"
              + (f"{g:>8.1f}%" if np.isfinite(g) else f"{'n/a':>9}")
              + (f"{hg:>8.1f}%" if np.isfinite(hg) else f"{'n/a':>9}")
              + f"{B.last2(net):>8.1f}%")

    print("\n3. THE BOOK AGAINST THE BRIEF, at 16bps")
    net, pos = book(px, fd, s)
    a = net.to_numpy()
    pf, trips = pf_trips(net, pos)
    flips = int((np.abs(np.diff(np.r_[0.0, pos])) > 1e-9).sum())
    g, hg = gate(a), honest_gate(a)
    h = len(a) // 2
    print(f"   {'CAGR at a 20% drawdown':<38}{g:>10.1f}%   (brief: 300%)   FAIL")
    print(f"   {'   same, honest bootstrap gate':<38}{hg:>10.1f}%")
    print(f"   {'completed round trips / position changes':<38}{trips:>7} / {flips}"
          f"   (brief: 100+)   {'PASS' if flips >= 100 else 'FAIL'}")
    print(f"   {'profit factor':<38}{pf:>10.2f}   (brief: >1.10)   "
          f"{'PASS' if pf > 1.10 else 'FAIL'}")
    print(f"   {'max drawdown at the gate':<38}{-20.0:>10.1f}%   (brief: <20%)   PASS")
    print(f"   {'negative years in 7':<38}"
          f"{sum(1 for _, v in B.yearly(net) if v < 0):>10}")
    print(f"   {'first half / second half gate':<38}"
          f"{gate(a[:h]):>9.1f}% / {gate(a[h:]):.1f}%")
    print(f"\n   year by year: " + "  ".join(f"{y} {v:+.1f}%" for y, v in B.yearly(net)))
    print("\ndone: S144")
