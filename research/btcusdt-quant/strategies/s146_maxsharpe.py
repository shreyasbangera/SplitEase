"""
S146 - Maximising Sharpe, which is the only lever that moves the answer.

THE POSITION
------------
Relaxing the drawdown limit from 20% to 25% buys a uniform +27-30%, taking the
best book from 38.7% to 50.0%. It does not change the problem, because:

    300% at a 25% drawdown        needs Sharpe 5.69
    300% at ANY drawdown (Kelly)  needs Sharpe 1.67
    best available here                  Sharpe 1.41

The drawdown limit is not the binding constraint. **Sharpe is.** At 1.41 the
Kelly ceiling is 171%, so this book cannot compound at 300% even with no
drawdown limit whatsoever. Every remaining unit of effort therefore goes at
Sharpe and nothing else.

WHAT RAISES SHARPE, IN ORDER OF WHAT IS LEFT
---------------------------------------------
Sharpe of a combination is s x sqrt(N / (1 + (N-1) rho)). Only three terms, and
only two of them are still reachable:

    MORE SLEEVES (N)     every sleeve variant this study has produced, not just
                         the four canonical ones - different clocks, different
                         channel families, different standardisation windows.
                         Sixteen candidates rather than four.
    LOWER CORRELATION    the full correlation matrix, so combinations can be
                         chosen for being different rather than for being good.
    HIGHER s             exhausted. Every data class on the disk and every
                         reachable external source has been opened.

Equal weights throughout. A weighting fitted to maximise in-sample Sharpe is
precisely the selection this log has caught four times (S106b, S109, S118b,
S131), and S124 already measured what it buys with perfect hindsight: 4%.

Reported at a 25% drawdown, per the revised constraint, with the honest
bootstrap gate beside it.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from itertools import combinations

import strategies.s135_more as S135
import strategies.s140_breakout as B
import strategies.s141_rexdev as R
import strategies.s144_blend as S144
from strategies.s96_rank import at_gate, stats_of

DD = -0.25


def gate_at(a, target=DD, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or a.sum() <= 0:
        return np.nan
    g = at_gate(a, target=target, lo=1e-3, hi=40.0, iters=60)
    return np.nan if (not np.isfinite(g["dd"]) or abs(g["dd"] - target) > tol) \
        else g["cagr"] * 100


def max_cagr(s):
    return (np.exp(s ** 2 / 2.0) - 1.0) * 100


def all_sleeves(px, hi, lo, rng, atr, lr):
    """Sixteen candidates: every variant this study produced, not just the four."""
    S = {}
    for N in (55, 89, 144):
        S[f"rex{N}"] = R.trail_pyr(px, *[x.fillna(False)
                                         for x in R.rex(px, hi, lo, rng, N, 0.95)],
                                   atr, 3.0)
    for N in (55, 89):
        up = hi.rolling(N).max().shift(1); dn = lo.rolling(N).min().shift(1)
        S[f"don{N}"] = R.trail_pyr(px, (px > up).fillna(False),
                                   (px < dn).fillna(False), atr, 3.0)
        e = px.ewm(span=N, adjust=False).mean().shift(1)
        S[f"kel{N}"] = R.trail_pyr(px, (px > e + 2 * atr.shift(1)).fillna(False),
                                   (px < e - 2 * atr.shift(1)).fillna(False), atr, 3.0)
    for rw, vw in ((20, 60), (20, 40), (10, 60)):
        rv = lr.rolling(rw).std()
        vov = rv.rolling(vw).std() / rv.rolling(vw).mean()
        S[f"vov{rw}_{vw}"] = -((vov - vov.rolling(365, min_periods=120).mean())
                               / (vov.rolling(365, min_periods=120).std() + 1e-12)
                               ).clip(-2, 2).shift(1).fillna(0.0)
    lp = lr.cumsum()
    S["expos"] = (sum(np.sign(lp.ewm(span=a, adjust=False).mean()
                              - lp.ewm(span=b, adjust=False).mean())
                      for a, b in ((8, 32), (16, 64), (32, 128))) / 3
                  ).clip(0, 1).shift(1).fillna(0.0)
    S["trend"] = (sum(np.sign(lp.ewm(span=a, adjust=False).mean()
                              - lp.ewm(span=b, adjust=False).mean())
                      for a, b in ((8, 32), (16, 64), (32, 128))) / 3
                  ).shift(1).fillna(0.0)
    try:
        import strategies.s119_dev as S119
        import strategies.s118_crowd as C
        cpx, cfd, F = S119.build("1D")
        cols = [c for c in C.SIGNS if c in F.columns]
        for w in (180, 365):
            S[f"crowd{w}"] = S119.signal(F, cols, w).reindex(px.index).fillna(0.0)
    except Exception as e:
        print("   (crowding unavailable:", e, ")")
    return S


if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean(); rng = hi - lo
    lr = np.log(px).diff()
    SL = all_sleeves(px, hi, lo, rng, atr, lr)

    print(f"S146 - maximising Sharpe. {len(SL)} sleeve candidates.\n")
    nets = {}
    print(f"   {'sleeve':>14}{'Sharpe':>8}{'@25%DD':>9}{'last2y':>9}")
    for k, s in SL.items():
        net, _ = S144.book(px, fd, s, tv=0.30)
        nets[k] = net
        a = net.to_numpy()
        g = gate_at(a)
        print(f"   {k:>14}{stats_of(a)['sharpe']:>8.2f}"
              + (f"{g:>8.1f}%" if np.isfinite(g) else f"{'n/a':>9}")
              + f"{B.last2(net):>8.1f}%")

    R_ = pd.DataFrame({k: v for k, v in nets.items()}).dropna()
    C_ = R_.corr()
    iu = np.triu_indices(len(C_), 1)
    print(f"\n   pairwise correlation: mean {C_.to_numpy()[iu].mean():+.3f}, "
          f"min {C_.to_numpy()[iu].min():+.3f}, max {C_.to_numpy()[iu].max():+.3f}")

    print(f"\n   best combinations by SHARPE (equal weight, all sizes searched)")
    print(f"   {'combination':>46}{'Sharpe':>8}{'maxCAGR':>10}{'@25%DD':>9}"
          f"{'last2y':>9}{'negyr':>7}")
    rows = []
    keys = list(SL)
    for k in range(1, 6):
        for c in combinations(keys, k):
            s = sum(SL[x] for x in c) / len(c)
            net, _ = S144.book(px, fd, s, tv=0.30)
            a = net.to_numpy(); a = a[np.isfinite(a)]
            if not len(a) or np.allclose(a, 0):
                continue
            sh = stats_of(a)["sharpe"]
            rows.append((sh, c, net))
    rows.sort(reverse=True, key=lambda r: r[0])
    for sh, c, net in rows[:10]:
        a = net.to_numpy()
        g = gate_at(a)
        neg = sum(1 for _, v in B.yearly(net) if v < 0)
        print(f"   {' + '.join(c):>46}{sh:>8.2f}{max_cagr(sh):>9.0f}%"
              + (f"{g:>8.1f}%" if np.isfinite(g) else f"{'n/a':>9}")
              + f"{B.last2(net):>8.1f}%{neg:>7}")

    top = rows[0]
    print(f"\n   HIGHEST SHARPE FOUND: {top[0]:.2f}  ({' + '.join(top[1])})")
    print(f"   Kelly ceiling {max_cagr(top[0]):.0f}% CAGR;  "
          f"300% needs Sharpe 1.67;  300% at a 25% drawdown needs 5.69")
    print("\ndone: S146")
