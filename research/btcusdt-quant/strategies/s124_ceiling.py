"""
S124 - The portfolio ceiling. What is the best ANY combination of these can do?

Enough one-lever-at-a-time. This computes the bound.

THE ARITHMETIC THAT GOVERNS EVERYTHING
--------------------------------------
At the gate a book is scaled until its drawdown sits on 20%, so what the brief
asks for is a Calmar of 300/20 = **15**. Empirically, across every book in this
study, Calmar at the gate tracks the square of Sharpe:

    V7          Sharpe 2.13   honest gate 129.4%   ->  Calmar 6.5
    crowding    Sharpe 1.32   honest gate  29.2%   ->  Calmar 1.5
    trend       Sharpe 0.70   honest gate   8.3%   ->  Calmar 0.4

which is Calmar ~ 1.4 x Sharpe^2. Calmar 15 therefore needs **Sharpe ~3.2**, not
some unimaginable number - about 50% more than the best single book here.

The only way to raise Sharpe by 50% without a better signal is to combine
imperfectly correlated streams. For N streams of Sharpe s at average pairwise
correlation rho, the combination reaches s x sqrt(N / (1 + (N-1) rho)). That
formula is the whole question, and every term in it is now measurable.

WHAT THIS COMPUTES, AND WHY THE ORACLE IS THE POINT
---------------------------------------------------
Three combinations of every stream this study has produced:

    EQUAL WEIGHT    the honest one. S106b showed informed weighting of V7's
                    signals loses to equal weights by 8 standard deviations, so
                    this is what an out-of-sample portfolio actually gets.

    ORACLE          maximum-Sharpe weights computed on the FULL SAMPLE with
                    perfect hindsight. **Not attainable and not proposed.** It
                    is an upper bound: no causal weighting scheme can beat the
                    weights chosen by looking at the answer.

    ORACLE, LONG-ONLY WEIGHTS   the same, without shorting a stream, since
                    shorting a strategy is rarely operationally real.

If the ORACLE is below 300%, then no combination of these streams reaches the
brief - not a better weighting rule, not a smarter allocator, not more tuning.
That converts "I could not find it" into "it is not there", which is a different
and far more useful statement.

Correlations are computed on the DATE INTERSECTION, which S116 had to correct
this log for: comparing series positionally when they start in different years
drove every correlation to ~0.00 and made unrelated sleeves look orthogonal.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of


def streams():
    """Every standalone return stream this study has produced, on a date index."""
    S = {}

    import strategies.s117_mf as S117
    px, fd = S117.data()
    net, _, _ = S117.run(px, fd, "longshort", 0.40)
    S["trend"] = net
    S["hold"] = pd.Series(px.pct_change().fillna(0.0).to_numpy() - fd.to_numpy(),
                          index=px.index)

    import strategies.s119_dev as S119
    import strategies.s118_crowd as C
    cpx, cfd, F = S119.build("1D")
    cols = [c for c in C.SIGNS if c in F.columns]
    sig = S119.signal(F, cols, 365)
    cnet, _, _ = S119.run(cpx, cfd, sig, 0.20, 0.30, 32, 365.25)
    S["crowding"] = cnet

    import strategies.s120c_cascade as X
    o = X.bars(60)
    xnet, _ = X.book(o, 4.0, 4, 10.0)
    S["cascade"] = X.to_daily(xnet)

    import strategies.s110_meta as M
    import strategies.s87_combined as S87
    M.use_clock()
    R = S87.rankings()
    vr, _ = S87.blend(R, 3, 0.144)
    vr = pd.Series(np.asarray(vr, float))
    # V7's returns come back without a date index; rebuild it from the panel it
    # was generated on, anchored at the end so the last observation lines up.
    end = cpx.index.max()
    vr.index = pd.date_range(end=end, periods=len(vr), freq="1D", tz=cpx.index.tz)
    S["V7"] = vr

    out = {}
    for k, v in S.items():
        v = pd.Series(np.asarray(v, float), index=pd.to_datetime(v.index))
        v.index = v.index.tz_localize(None) if v.index.tz else v.index
        out[k] = v.groupby(v.index.normalize()).sum()
    return out


TARGET_VOL = 0.30          # annualised, the scale every stream is put on


def gate_of(a, tol=0.01):
    """Sharpe and CAGR at a 20% drawdown, REFUSING to report a gate it did not
    actually reach.

    `at_gate` bisects a scale over [0.2, 6.0] by default. Handed a series whose
    natural scale lies outside that bracket it saturates at an endpoint, and on
    a unit-variance series it overflows outright and returns dd = nan - which the
    bisection then compares against, silently. The first version of this file
    normalised streams with R/R.std() and printed a 2182.9% oracle and a
    311.6% equal-weight book off exactly that, a combination reading HIGHER
    than its own best constituent at a LOWER Sharpe. Checking that the achieved
    drawdown is really -20% is the guard that catches it in one line.
    """
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return 0.0, np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] - (-0.20)) > tol:
        return stats_of(a)["sharpe"], np.nan       # gate not reached: no number
    return stats_of(a)["sharpe"], g["cagr"] * 100


def at_vol(s, target=TARGET_VOL):
    """Put a stream on a common annualised volatility, not on unit variance."""
    sd = float(np.std(np.asarray(s, float))) * np.sqrt(365.25)
    return s * (target / sd) if sd > 0 else s * 0.0


def maxsharpe(R, long_only=False):
    """Full-sample maximum-Sharpe weights. Hindsight, by construction."""
    mu = R.mean().to_numpy()
    C = R.cov().to_numpy()
    C = C + np.eye(len(C)) * 1e-12
    w = np.linalg.solve(C, mu)
    if long_only:
        # project onto the simplex by dropping negatives and re-solving
        for _ in range(len(w)):
            if (w >= -1e-12).all():
                break
            keep = w > 0
            if keep.sum() == 0:
                w = np.ones(len(w)); break
            w2 = np.zeros(len(w))
            w2[keep] = np.linalg.solve(C[np.ix_(keep, keep)], mu[keep])
            w = w2
        w = np.clip(w, 0, None)
    s = np.abs(w).sum()
    return w / s if s > 0 else w


if __name__ == "__main__":
    S = streams()
    print("S124 - the portfolio ceiling\n")
    print(f"{'stream':>12}{'from':>13}{'to':>13}{'days':>7}{'Sharpe':>8}{'gate':>9}")
    for k, v in S.items():
        sh, g = gate_of(v)
        print(f"{k:>12}{str(v.index.min().date()):>13}"
              f"{str(v.index.max().date()):>13}{len(v):>7}{sh:>8.2f}"
              + ("     n/a" if not np.isfinite(g) else f"{g:>8.1f}%"))

    R = pd.DataFrame(S).dropna()
    print(f"\ndate INTERSECTION of all streams: {R.index.min().date()} -> "
          f"{R.index.max().date()}, {len(R)} days ({len(R)/365.25:.1f} years)")

    print("\ncorrelation on the intersection:")
    Cm = R.corr()
    print("            " + "".join(f"{c:>10}" for c in Cm.columns))
    for i, row in Cm.iterrows():
        print(f"{i:>12}" + "".join(f"{row[c]:>10.3f}" for c in Cm.columns))

    print("\nre-measured on the intersection alone (like-for-like):")
    print(f"{'stream':>12}{'Sharpe':>8}{'gate':>9}")
    for c in R.columns:
        sh, g = gate_of(R[c])
        print(f"{c:>12}{sh:>8.2f}" + ("     n/a" if not np.isfinite(g) else f"{g:>8.1f}%"))

    # --- vol parity at a REALISTIC scale, so no stream dominates by size and
    #     the combined series stays inside the range the gate can bisect ---
    Z = pd.DataFrame({c: at_vol(R[c]) for c in R.columns})
    print(f"\nevery stream rescaled to {TARGET_VOL*100:.0f}% annualised vol "
          f"before combining. gate figures are refused (shown as 'n/a') unless "
          f"the\nbisection actually reaches a -20% drawdown.")

    print(f"\n{'combination':>34}{'Sharpe':>9}{'at -20%':>10}{'weights':>8}")
    rows = []
    for tag, w in (
        ("equal weight, all streams", np.ones(len(R.columns))),
        ("equal weight, positive streams only",
         np.array([1.0 if R[c].mean() > 0 else 0.0 for c in R.columns])),
        ("ORACLE max-Sharpe (hindsight)", maxsharpe(Z)),
        ("ORACLE max-Sharpe, long-only (hindsight)", maxsharpe(Z, True)),
    ):
        w = np.asarray(w, float)
        if np.abs(w).sum() == 0:
            continue
        w = w / np.abs(w).sum()
        comb = (Z * w).sum(axis=1)
        sh, g = gate_of(comb)
        rows.append((tag, sh, g, w))
        gs = "n/a" if not np.isfinite(g) else f"{g:.1f}%"
        print(f"{tag:>34}{sh:>9.2f}{gs:>10}")
        print(f"{'':>34}  " + "  ".join(
            f"{c}={x:+.2f}" for c, x in zip(R.columns, w)))

    rows = [r for r in rows if np.isfinite(r[2])]
    if not rows:
        print("\nno combination produced a valid gate. nothing to report.")
        sys.exit(0)
    best = max(rows, key=lambda r: r[2])
    print("\n" + "=" * 70)
    print(f"CEILING: the best of these, including the hindsight oracle, is "
          f"{best[2]:.1f}% at a 20% drawdown.")
    print(f"The brief asks for 300%. Shortfall: {300/best[2]:.1f}x.")
    need = np.sqrt(15.0 / 1.4)
    print(f"\nCalmar 15 needs Sharpe ~{need:.2f}. Best combination reaches "
          f"{best[1]:.2f}.")

    # ---- what would actually be required -------------------------------
    # For N streams of equal Sharpe s at average pairwise correlation rho,
    # the combination reaches s * sqrt(N / (1 + (N-1) rho)).
    base = max(gate_of(R[c])[0] for c in R.columns)
    print("\n" + "=" * 70)
    print("WHAT WOULD REACH IT. Combining N streams each of Sharpe "
          f"{base:.2f} (V7's), at\naverage pairwise correlation rho. Bold "
          f"cell = clears Sharpe {need:.2f}.\n")
    print(f"{'rho':>8}" + "".join(f"{'N=' + str(n):>10}" for n in (1, 2, 3, 4, 5)))
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4):
        cells = []
        for n in (1, 2, 3, 4, 5):
            s = base * np.sqrt(n / (1 + (n - 1) * rho))
            cells.append(f"{s:>9.2f}" + ("*" if s >= need else " "))
        print(f"{rho:>8.1f}" + "".join(cells))
    print("\n  * clears the brief. Read the row for the correlation you believe.")
    print(f"  measured correlation among the streams here: "
          f"{Cm.to_numpy()[np.triu_indices(len(Cm), 1)].mean():+.2f} average, "
          f"{Cm.to_numpy()[np.triu_indices(len(Cm), 1)].max():+.2f} worst.")
    print("\ndone: ceiling")
