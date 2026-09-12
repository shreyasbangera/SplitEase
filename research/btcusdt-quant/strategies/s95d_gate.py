"""
S95d - Is the VIX result information, or is it just dilution?

S95c produced the first Calmar improvement in a long while. Adding
s_vixchg = -z(10-day change in VIX) as a sixth signal at threshold 1.0:

    control  43.7%  -17.0%  PF 1.85  Sharpe 1.84  Calmar 2.57  P(DD>20%) 34%
    +vixchg  39.4%  -12.0%  PF 1.97  Sharpe 1.90  Calmar 3.27  P(DD>20%) 10%

with a correlation to the existing five-signal composite of +0.011, against
+0.39 to +0.68 for every signal already in the book. Out-of-sample profit
factor rose 1.88 -> 2.08 and OOS gave up only 1.5 points of CAGR against the
in-sample 6.0, which is the direction you want.

THE OBJECTION THAT HAS TO BE KILLED FIRST
-----------------------------------------
The composite is an EQUAL-WEIGHT AVERAGE. Adding a sixth name cuts every
existing signal's weight from 1/5 to 1/6, and s_vixchg is non-zero on only 26%
of bars. So on roughly three bars in four the new book is simply the OLD book
at 5/6 of the size. Smaller size means smaller drawdown and smaller CAGR - the
exact pattern observed - and it requires no information whatsoever.

Two controls settle it, and neither was run in S95c:

  1. DILUTION CONTROL. The five-signal composite multiplied by 5/6. Identical
     size reduction, zero new information. If this reproduces the improvement,
     s_vixchg is decoration.

  2. AT THE GATE. Calmar measured at different realised drawdowns flatters
     whichever book drew down less - S94 records this as the reason Calmar is
     never used to choose a risk level. So the risk is bisected for EVERY row
     until realised max drawdown lands on -20%, and CAGR is read there. That is
     the only comparison in which size is not doing the arguing.

  3. And the split, which is what killed the options chain at S50: a result
     that reverses between the halves of the sample is a property of the
     sample.

The result is judged against the dilution control, not against the raw control.
"""
import sys
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

import strategies.s45_single as S
import strategies.s46_net as S46
from research.harness import backtest, OOS_END, IS_END
from research.robust import bootstrap_dd
from strategies.s95c_book import add_vix

START = S.FULL_START
HOLD, STP, RR = 21, 3.0, 2.0


def run(g, net, risk, start=START, end=OOS_END):
    a = S.book(g, net, stp=STP, rr=RR)
    a["exit"] = (np.abs(np.nan_to_num(net)) <= 0.0).astype(float)
    return backtest(g, a, "12h", start=start, end=end, risk=risk,
                    max_lev=10.0, max_bars_h=HOLD * 24)


def at_gate(g, net, target=-0.20, lo=0.02, hi=0.40, iters=22):
    """Bisect the RISK until realised max drawdown sits on the gate."""
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = run(g, net, mid)
        m["risk"] = mid
        if best is None or abs(m["max_dd"] - target) < abs(best["max_dd"] - target):
            best = m
        if abs(m["max_dd"] - target) < 2e-4:
            break
        if m["max_dd"] < target:
            hi = mid
        else:
            lo = mid
    return best


def line(tag, m, base=None):
    d = "" if base is None else f"{(m['cagr']-base)*100:+7.1f}pt"
    print(f"{tag:>30} | risk {m['risk']*100:5.2f}%  CAGR {m['cagr']*100:7.1f}%"
          f"  DD {m['max_dd']*100:6.1f}%  PF {m['profit_factor']:5.2f}"
          f"  N {m['trades']:5d}  Shp {m['sharpe']:5.2f}"
          f"  Clm {m['cagr']/abs(m['max_dd']):5.2f}{d}", flush=True)
    return m


if __name__ == "__main__":
    g = add_vix(S.grid(START))
    base5 = S.composite(g, S46.LONG)
    S.THR["vixchg"] = 1.0
    net6 = S.composite(g, S46.LONG + ["vixchg"])
    dilute = base5 * (5.0 / 6.0)

    BOOKS = [("5 signals (control)", base5),
             ("5 signals x 5/6 (dilution)", dilute),
             ("6 signals with vixchg", net6)]

    print("1. AT THE -20% GATE - risk bisected so every row is at the same")
    print("   realised drawdown, which is the only way size stops arguing.\n")
    res = {}
    b0 = None
    for tag, net in BOOKS:
        m = at_gate(g, net)
        res[tag] = m
        b0 = m["cagr"] if b0 is None else b0
        line(tag, m, None if tag.startswith("5 signals (") else b0)

    print("\n   The dilution control is the comparison that matters: it has the")
    print("   same position size and none of the information.")

    # --- 2. the split ----------------------------------------------------
    print("\n2. DOES IT SURVIVE THE SPLIT? Same bisection, each half separately.\n")
    mid = "2023-12-01"
    for lab, s, e in (("first half", START, mid), ("second half", mid, OOS_END)):
        print(f"   {lab}: {s} -> {e}\n")
        bb = None
        for tag, net in BOOKS:
            def _gate(net=net, s=s, e=e):
                best = None
                lo, hi = 0.02, 0.40
                for _ in range(22):
                    r = (lo + hi) / 2
                    m = run(g, net, r, start=s, end=e); m["risk"] = r
                    if best is None or abs(m["max_dd"] + 0.20) < abs(best["max_dd"] + 0.20):
                        best = m
                    if abs(m["max_dd"] + 0.20) < 2e-4: break
                    if m["max_dd"] < -0.20: hi = r
                    else: lo = r
                return best
            m = _gate()
            bb = m["cagr"] if bb is None else bb
            line(tag, m, None if tag.startswith("5 signals (") else bb)
        print()

    # --- 3. year by year at a common risk --------------------------------
    print("3. YEAR BY YEAR at a common 8% risk - where does the difference live?\n")
    eq = {}
    for tag, net in BOOKS:
        m = run(g, net, 0.08)
        s = pd.Series(m["equity"], index=pd.to_datetime(m["dt"]))
        eq[tag] = s.resample("1D").last().dropna()
    yrs = sorted({d.year for d in eq[BOOKS[0][0]].index})
    print(f"{'year':>6}" + "".join(f"{t.split(' (')[0][:16]:>18}" for t, _ in BOOKS))
    for y in yrs:
        row = f"{y:>6}"
        for tag, _ in BOOKS:
            s = eq[tag][eq[tag].index.year == y]
            row += f"{(s.iloc[-1]/s.iloc[0]-1)*100:17.1f}%" if len(s) > 1 else f"{'-':>18}"
        print(row)

    print("\n4. BOOTSTRAP at the gate risk of each row\n")
    for tag, _ in BOOKS:
        m = res[tag]
        r = pd.Series(m["equity"], index=pd.to_datetime(m["dt"])).resample("1D").last(
            ).dropna().pct_change().fillna(0).to_numpy()
        b = bootstrap_dd(r, n=2000)
        print(f"{tag:>30} | median DD {b['dd_median']*100:6.1f}%"
              f"  5th pct {b['dd_p05']*100:6.1f}%"
              f"  P(DD>20%) {b['p_dd_worse_than_20']*100:3.0f}%")

    print("\ndone: dilution vs information")
