"""
S95c - The macro read put in the book, which is the only test that decides.

S95's IC screen did not kill the VIX candidate and did not validate it. S95b
explains why it could not do either: under an overlap-corrected t-statistic,
*three of the five signals already deployed* are themselves indistinguishable
from noise - fundz t = -0.33, basis t = -0.02, ofi96 t = +1.38, against VIX's
best of |t| = 0.97. This book has never worked by any single signal being
strong. It works by netting five weak, largely uncorrelated reads, and an IC
table cannot see that. So the screen is demoted to what it actually is, a
filter against garbage, and the decision is made where the study always makes
it: in the netted book, against the identical control, on the same window.

WHAT IS ADDED, FIXED IN ADVANCE
-------------------------------
Two candidate signals, built in the book's own idiom (z-score, dead band,
scaled by |z|/threshold, capped at 2), each with the sign the screen indicated
under the punitive `strict` lag:

    s_vixchg = -z(10-day change in VIX, 120d)   rising vol -> short
    s_vixlvl = +z(VIX level, 120d)              elevated vol -> long

They point in opposite directions on purpose and that is the economics, not a
hedge: an elevated level is fear already in the price, a rising level is fear
arriving. Both were stable in sign across both sample splits in S95; neither
was significant.

Thresholds are not searched. Each is tried at 1.0, the value five of the six
existing signals use, and at 0.7, the value the two weakest use. Four variants
per signal set, no grid, no tuning - because the whole risk here is finding a
sixth signal by searching until one appears.

WHAT WOULD COUNT
----------------
The control is the same five-signal book on the same window at the same risk.
A sixth signal counts if it raises Sharpe AND out-of-sample profit factor,
because the study's own precedent (the price-position candidate, 7th of 232 on
a +0.023 in-sample margin) is that anything less is inside the multiple-testing
noise and does not get adopted.

And the arithmetic is worth stating before the run rather than after. The book
is at 179.0% CAGR on the 20% drawdown gate and the brief asks 300%, so the
Sharpe has to go from 2.13 to roughly 3.3 - up 55%. Adding one uncorrelated
signal of equal quality to five moves Sharpe by sqrt(6/5) = 9.5%. Even a clean
win here is worth about 196%, not 300%. This is run to find out whether the
macro source class is empty, which is worth knowing on its own terms; it is not
run in the expectation that it closes the gap.
"""
import sys
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

import strategies.s45_single as S
import strategies.s46_net as S46
from research.harness import OOS_END, IS_END
from research import macro

RISK = 0.08
START = S.FULL_START


def zs(x, n):
    s = pd.Series(np.asarray(x, float))
    return ((s - s.rolling(n).mean()) / s.rolling(n).std()).to_numpy(float)


def add_vix(g):
    """Attach the two macro signals to the book's own 12h decision panel.

    The alignment is macro.onto_bars in `strict` mode: a VIX close dated D is
    unknowable until D+1 22:00 UTC, ~25 hours later, so nothing here can be
    reading a print before it exists.
    """
    vix = macro.load_csv("vix")
    v = macro.onto_bars(g.dt, vix, mode="strict")
    g = g.copy()
    g["s_vixchg"] = -zs(pd.Series(v).diff(20).to_numpy(float), 240)   # 10 days, 120d window
    g["s_vixlvl"] = zs(v, 240)
    return g


if __name__ == "__main__":
    g = add_vix(S.grid(START))
    n_ok = np.isfinite(g.s_vixchg.to_numpy(float)).sum()
    print(f"panel {len(g)} bars, {g.dt.iloc[0].date()} -> {g.dt.iloc[-1].date()}, "
          f"VIX signal finite on {n_ok} ({n_ok/len(g)*100:.0f}%)\n")

    print(f"risk {RISK*100:.0f}%, hold 21d, stop 3 ATR, target 2R - the S46 book\n")
    print(f"{'variant':>34} | {'CAGR':>7}{'MaxDD':>7}{'PF':>6}{'N':>6}"
          f"{'Shp':>6}{'Clm':>6} | {'IS':>6}{'OOS':>7}{'oPF':>6}")

    base = S.composite(g, S46.LONG)
    ctl = S.show("control - 5 signals", g, base, START, RISK, boot=True)

    for thr in (1.0, 0.7):
        for extra in (["vixchg"], ["vixlvl"], ["vixchg", "vixlvl"]):
            for e in extra:
                S.THR[e] = thr
            names = S46.LONG + extra
            net = S.composite(g, names)
            tag = "+" + "+".join(extra) + f" thr {thr}"
            S.show(tag, g, net, START, RISK, boot=True)

    # --- is it even uncorrelated? the entire case rests on this ------------
    print("\nCORRELATION to the existing composite - the premise of the whole test")
    for e in ("vixchg", "vixlvl"):
        S.THR[e] = 1.0
        u = S.unit(g, e)
        ok = np.isfinite(u) & np.isfinite(base)
        act = np.abs(u) > 0
        print(f"  s_{e:<8} corr to 5-signal net {np.corrcoef(u[ok], base[ok])[0,1]:+.3f}"
              f"   non-zero on {act.mean()*100:4.1f}% of bars")
    for n in S46.LONG:
        u = S.unit(g, n)
        ok = np.isfinite(u) & np.isfinite(base)
        print(f"  s_{n:<8} corr to 5-signal net {np.corrcoef(u[ok], base[ok])[0,1]:+.3f}"
              f"   non-zero on {(np.abs(u)>0).mean()*100:4.1f}% of bars")

    print("\ndone: macro in the book")
