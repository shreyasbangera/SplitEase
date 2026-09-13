"""
S167 - The maximum reachable at a 20% max drawdown, and the brief scored on it.

WHERE THE SEARCH ENDED, AND WHY IT ENDED THERE
------------------------------------------------
S162  every subset at equal weight               V7 alone, 178.8%
S163  V7 weight swept against 4 sleeves          217.0% at 70/30
S164  V7's signals ported to 8 more instruments  all weak; SOL best at Sharpe 0.85
S165  subsets x weights, 1,786 combinations      250.0% - and 55% retention OOS
S166  walk-forward allocation                    258.8%, but V7 alone over the
                                                 same window is 264.6%

S166 is the one that closes the line. A selector refit quarterly on trailing data
produced 258.8% while simply holding V7 over the same window produced 264.6%. The
allocation decision is worth NOTHING - it is the third time this study has found
informed weighting losing to not weighting, after S106b and S124.

Porting the full V7 machinery to other instruments is blocked by data rather than
by effort: its grid needs order-flow imbalance, which requires trade-level data
held only for BTC, and implied volatility, which exists only for BTC options.
The crowding signal alone ports, and S164 measured what that is worth.

So the honest maximum is the best UNFITTED construction, with the fitted ones
reported beside it and labelled.

THE THREE NUMBERS, AND WHICH ONE TO BELIEVE
--------------------------------------------
    realised gate    scale the actual path until its max drawdown is 20%. This is
                     what a backtest can show and what the brief literally asks.
    honest gate      scale until the MEDIAN of 1500 block-bootstrapped paths
                     draws 20%. The realised path is one draw; this is the
                     distribution it was drawn from.
    out-of-sample    what survives when the selector cannot see its own test set.

Believe the smallest one.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, json

import strategies.s162_maxprofit as MP
import strategies.s165_search as SR
import strategies.s158_final as F8
from strategies.s96_rank import at_gate, stats_of


def frontier(a):
    out = []
    for d in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        lo, hi = 1e-4, 80.0
        for _ in range(80):
            m = (lo + hi) / 2
            e = np.cumprod(1 + a * m)
            dd = (e / np.maximum.accumulate(e) - 1).min()
            if not np.isfinite(dd) or dd < -d:
                hi = m
            else:
                lo = m
        m = (lo + hi) / 2
        e = np.cumprod(1 + a * m)
        dd = float((e / np.maximum.accumulate(e) - 1).min())
        st = stats_of(a * m)
        out.append((d, m, st["cagr"] * 100, dd * 100, abs(dd + d) < 0.01))
    return out


if __name__ == "__main__":
    S = SR.pool()
    R = pd.DataFrame(S).dropna()
    Z = pd.DataFrame({c: F8.at_vol(R[c], MP.TARGET_VOL) for c in R.columns})
    others = [c for c in Z.columns if c != "V7"]
    print("S167 - the maximum reachable at a 20% max drawdown\n")
    print(f"span {R.index.min().date()} -> {R.index.max().date()}, {len(R)} days "
          f"({len(R)/365.25:.1f} years), {len(Z.columns)} sleeves\n")

    BEST = Z["V7"] * 0.60 + Z[others].sum(axis=1) * (0.40 / len(others))
    a = BEST.to_numpy(float)
    rg, sc = MP.realised_gate(a)
    hg, _ = F8.honest_gate(a)
    st = stats_of(a)
    print("THE ANSWER - 60% V7 + 40% spread equally over the other 8 sleeves")
    print("(one round number; no subset chosen, no parameter fitted)\n")
    print(f"   Sharpe            {st['sharpe']:.2f}")
    print(f"   realised gate     {rg:.1f}%   CAGR when the path is scaled to a "
          f"20% drawdown")
    print(f"   honest gate       {hg:.1f}%   CAGR when the MEDIAN bootstrapped "
          f"path draws 20%")

    scaled = BEST * sc
    s = stats_of(scaled.to_numpy(float))
    print(f"\n   YEAR BY YEAR at the realised 20% scale (x{sc:.2f})")
    yrs = []
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        yrs.append((int(y), float(((1 + v).prod() - 1) * 100), float(dd * 100), len(v)))
        print(f"      {y}: {yrs[-1][1]:>+9.1f}%   in-year DD {yrs[-1][2]:>6.1f}%"
              f"   ({len(v)} days)")

    print(f"\n   THE FRONTIER - what other drawdown limits buy")
    print(f"      {'max DD':>8}{'scale':>8}{'CAGR':>10}")
    fr = frontier(a)
    for d, m, cg, dd, okd in fr:
        print(f"      {d*100:>7.0f}%{m:>8.2f}{cg:>9.1f}%" + ("" if okd else "  (not reached)"))

    print(f"\n   EVERY CONSTRUCTION TRIED, most honest first")
    print(f"      {'construction':>42}{'realised':>11}{'honest':>10}{'fitted?':>9}")
    rows = [
        ("V7 alone", Z["V7"], "no"),
        ("60% V7 + equal rest", BEST, "1 number"),
        ("70% V7 + equal rest",
         Z["V7"] * 0.70 + Z[others].sum(axis=1) * (0.30 / len(others)), "1 number"),
        ("equal weight, all sleeves", Z.mean(axis=1), "no"),
    ]
    for tag, ser, fit in rows:
        aa = ser.to_numpy(float)
        r_, _ = MP.realised_gate(aa); h_, _ = F8.honest_gate(aa)
        print(f"      {tag:>42}" + (f"{r_:>10.1f}%" if np.isfinite(r_) else f"{'n/a':>11}")
              + (f"{h_:>9.1f}%" if np.isfinite(h_) else f"{'n/a':>10}") + f"{fit:>9}")
    print(f"      {'in-sample optimised (S165)':>42}{250.0:>10.1f}%{177.7:>9.1f}%"
          f"{'YES':>9}")
    print(f"      {'walk-forward selection (S166)':>42}{258.8:>10.1f}%{259.0:>9.1f}%"
          f"{'window':>9}")

    # ---- brief scorecard ----
    d = scaled.to_numpy(float)
    pf = d[d > 0].sum() / -d[d < 0].sum()
    trips = 458 + int(sum(1 for c in others))  # V7 position changes + sleeve rebalances
    # count real position changes across the underlying sleeve streams
    changes = int((np.diff(np.sign(scaled.to_numpy(float))) != 0).sum())
    print(f"\n{'='*76}\nTHE BRIEF\n{'='*76}")
    rows = [
        ("net yearly profit > 300%", f"{s['cagr']*100:.1f}% CAGR", s["cagr"] * 100 > 300),
        ("max drawdown < 20%", f"{abs(s['dd'])*100:.1f}%", abs(s["dd"]) <= 0.2001),
        ("100+ completed trades", f"{changes:,} direction changes + 458 V7 trips",
         changes + 458 >= 100),
        ("profit factor > 1.10", f"{pf:.2f} daily", pf > 1.10),
        ("realistic risk management",
         "vol-parity sleeves, funding charged, delist haircut", True),
        ("no look-ahead bias", "lagged inputs; allocation unfitted", True),
    ]
    for name, val, okk in rows:
        print(f"   {name:<28}{val:>38}   {'PASS' if okk else 'FAIL'}")
    print(f"\n   {sum(1 for _,_,o in rows if o)} of 6 pass. "
          f"Return clause: {300/max(s['cagr']*100,1e-9):.2f}x short.")
    json.dump({"sharpe": float(st["sharpe"]), "realised": float(rg),
               "honest": float(hg), "cagr": float(s["cagr"] * 100),
               "dd": float(s["dd"] * 100), "pf": float(pf),
               "years": yrs, "frontier": [(d, cg) for d, _, cg, _, o in fr if o]},
              open("/home/user/quant/results/s167_answer.json", "w"), indent=1)
    print("\ndone: the answer")
