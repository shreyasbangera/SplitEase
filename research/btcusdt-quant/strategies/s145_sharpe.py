"""
S145 - Building for SHARPE, which is what actually caps the return.

THE ARITHMETIC I SHOULD HAVE RUN FIRST
---------------------------------------
Scaling the best new book up until it hits 300% does not work - it cannot get
there at ANY leverage. CAGR peaks at 94.1% and then falls: 84.6% at twice that
size, 11.2% at four times, -68.2% beyond. That is volatility drag past the Kelly
point, where extra size destroys compound growth because losses compound
multiplicatively.

Which gives the law that actually governs this brief:

        maximum achievable CAGR  =  exp(Sharpe^2 / 2) - 1

At Sharpe 1.13 that is 89%, and the observed peak was 94%. The two agree.

**This splits the brief into two very different problems.**

    300% CAGR, drawdown unconstrained    needs Sharpe ~1.67
    300% CAGR at a 20% drawdown          needs Sharpe ~6.56   (S129's law)

The second is hopeless and has been measured as such many times over. **The first
is not.** Sharpe 1.67 sits between the books already built here - crowding-8h +
exposure reached 1.46, the breakout blend 1.13 - and below V7's 2.13.

And every development pass in this study has optimised the wrong thing. The
"gate" figure is CAGR at a fixed drawdown, which is Calmar. Sharpe has been
reported throughout and never once been the objective. A book selected for Calmar
is selected for shallow drawdowns; a book selected for Sharpe is selected for
consistency, and it is consistency that compounds.

WHAT THIS DOES
--------------
Combines the highest-Sharpe sleeves this study has produced - which are NOT the
highest-gate ones - and asks whether any combination clears 1.67. Then, for
whatever it finds, reports honestly what the drawdown looks like at the leverage
that delivers 300%, because that is the cost and the user should see it rather
than have it buried.

Equal weights and vol-parity weights only. No fitted weighting: S106b's finding
stands, and a weight chosen to maximise in-sample Sharpe is exactly the selection
this log has caught four times.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from itertools import combinations

import strategies.s135_more as S135
import strategies.s140_breakout as B
import strategies.s141_rexdev as R
import strategies.s144_blend as S144
from strategies.s96_rank import stats_of


def max_cagr(sharpe):
    """Kelly ceiling: the most a strategy of this Sharpe can compound at."""
    return (np.exp(sharpe ** 2 / 2.0) - 1.0) * 100


def dd_probs(a, n=3000, block=90, seed=0):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    rng = np.random.default_rng(seed); T = len(a); nb = int(np.ceil(T / block))
    s = rng.integers(0, max(T - block, 1), (n, nb))
    idx = np.clip((s[:, :, None] + np.arange(block)[None, None, :])
                  .reshape(n, -1)[:, :T], 0, T - 1)
    eq = np.cumprod(1.0 + a[idx], axis=1)
    return (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)


def sleeves(px, hi, lo, rng, atr, lr, fd):
    """Every sleeve, kept as a SIGNAL so they can be blended before sizing."""
    S = {}
    S["breakout"] = sum(R.trail_pyr(px, *[x.fillna(False)
                                          for x in R.rex(px, hi, lo, rng, N, 0.95)],
                                    atr, 3.0) for N in (55, 89, 144)) / 3
    rv = lr.rolling(20).std()
    vov = rv.rolling(60).std() / rv.rolling(60).mean()
    S["vovfade"] = -((vov - vov.rolling(365, min_periods=120).mean())
                     / (vov.rolling(365, min_periods=120).std() + 1e-12)
                     ).clip(-2, 2).shift(1).fillna(0.0)
    agree3 = (sum(np.sign(lr.cumsum().ewm(span=a, adjust=False).mean()
                          - lr.cumsum().ewm(span=b, adjust=False).mean())
                  for a, b in ((8, 32), (16, 64), (32, 128))) / 3)
    S["exposure"] = agree3.clip(0, 1).shift(1).fillna(0.0)
    try:
        import strategies.s119_dev as S119
        import strategies.s118_crowd as C
        cpx, cfd, F = S119.build("1D")
        cols = [c for c in C.SIGNS if c in F.columns]
        S["crowding"] = S119.signal(F, cols, 365).reindex(px.index).fillna(0.0)
    except Exception as e:
        print("   (crowding unavailable:", e, ")")
    return S


if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean(); rng = hi - lo
    lr = np.log(px).diff()
    SL = sleeves(px, hi, lo, rng, atr, lr, fd)

    print("S145 - building for Sharpe, not for the gate\n")
    print(f"   Kelly ceiling: max CAGR = exp(S^2/2) - 1")
    for s in (1.0, 1.13, 1.46, 1.67, 2.0, 2.13, 3.0):
        print(f"      Sharpe {s:>4.2f}  ->  max CAGR {max_cagr(s):>8.0f}%"
              + ("   <- 300% needs this" if abs(s - 1.67) < 0.01 else ""))

    print(f"\n1. EVERY SUBSET, RANKED BY SHARPE (not by gate)")
    print(f"   {'combination':>36}{'Sharpe':>8}{'maxCAGR':>10}{'realDD':>9}"
          f"{'last2y':>9}{'negyr':>7}")
    rows = []
    for k in range(1, len(SL) + 1):
        for combo in combinations(SL, k):
            s = sum(SL[c] for c in combo) / len(combo)
            net, pos = S144.book(px, fd, s, tv=0.30)
            a = net.to_numpy(); a = a[np.isfinite(a)]
            st = stats_of(a)
            neg = sum(1 for _, v in B.yearly(net) if v < 0)
            rows.append((st["sharpe"], " + ".join(combo), s, net, st, neg))
    for sh, tag, _, net, st, neg in sorted(rows, reverse=True, key=lambda r: r[0])[:12]:
        print(f"   {tag:>36}{sh:>8.2f}{max_cagr(sh):>9.0f}%{st['dd']*100:>8.1f}%"
              f"{B.last2(net):>8.1f}%{neg:>7}")

    best_sh, best_tag, best_sig, best_net, _, _ = max(rows, key=lambda r: r[0])
    print(f"\n   highest Sharpe available: {best_tag} at {best_sh:.2f}")
    print(f"   its Kelly ceiling is {max_cagr(best_sh):.0f}% CAGR"
          + ("  — CLEARS 300%" if max_cagr(best_sh) >= 300 else
             f"  — short of 300% (needs Sharpe 1.67)"))

    print(f"\n2. SCALING THE HIGHEST-SHARPE BOOK TOWARD 300%")
    print(f"   {'target vol':>11}{'CAGR':>9}{'realDD':>9}{'medDD':>9}"
          f"{'P(DD>20%)':>11}{'P(DD>50%)':>11}{'P(DD>80%)':>11}")
    hit = None
    for tv in (0.3, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6):
        net, pos = S144.book(px, fd, best_sig, tv=tv, max_lev=25.0)
        a = net.to_numpy(); a = a[np.isfinite(a)]
        st = stats_of(a); m = dd_probs(a)
        print(f"   {tv*100:>10.0f}%{st['cagr']*100:>8.1f}%{st['dd']*100:>8.1f}%"
              f"{np.median(m)*100:>8.1f}%{(m < -0.2).mean()*100:>10.0f}%"
              f"{(m < -0.5).mean()*100:>10.0f}%{(m < -0.8).mean()*100:>10.0f}%")
        if hit is None and st["cagr"] * 100 >= 300:
            hit = (tv, st, m)
    print()
    if hit:
        tv, st, m = hit
        print(f"   300% REACHED at target vol {tv*100:.0f}%: CAGR {st['cagr']*100:.0f}%, "
              f"drawdown {st['dd']*100:.0f}%, P(DD>50%) {(m<-0.5).mean()*100:.0f}%")
    else:
        print(f"   300% not reached. Kelly ceiling for Sharpe {best_sh:.2f} is "
              f"{max_cagr(best_sh):.0f}%.")
    print("\ndone: S145")
