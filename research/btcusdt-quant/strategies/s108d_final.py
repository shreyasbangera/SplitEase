"""
S108d - The candidate, judged on the book that is actually deployed.

S108c separated two numbers that S108b had run together.

    against ONE configuration      116.9% -> 332.4%    +215.5
    against V7's TOP-3 BLEND       168.8% -> 197.5%     +28.7

The second is the real one. V7 already holds three configurations at risk/3, and
a blend that is already smoothed has much less room for a fourth stream than a
single sleeve does. **+28.7 points, not +215.5** - and S96c's selection-noise
band has sd 24 points, so that is 1.2 sd. Marginal, not decisive.

What survived the controls and what did not:

    PASSED  split half, on the single-config basis, by +186.9 in the WORSE half
            - the control that killed S105, and the first candidate to pass it
    PASSED  the admission bar: Sharpe 2.21 at rho 0.499 needs 1.58 (S102c), and
            the blend's Sharpe 2.51 matches the decorrelation formula's 2.52
    FAILED  the realised -7.5% drawdown. Bootstrap median is -12.5%, so roughly
            half the drawdown improvement is this path rather than the blend.
    MIXED   delay. One bar late costs the event book 65% against the clock
            book's 43% - from a SHORTER delay (8h against 12h). Construction is
            causal by inspection, and the clock book loses heavily too, so this
            is edge concentrated at the boundary rather than leakage - but it is
            an operational warning, not a clean pass.

Three things remain to settle before this can be called a result.

    1  the split half on the FULL STACK, which is the candidate, rather than on
       the single configuration the earlier test used
    2  whether 3 bars a day is a spike or a plateau - S99, S105 and S106 all
       produced interior spikes that were noise
    3  what it costs if the drawdown is the bootstrap's -12.5% and not -7.5%
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
import strategies.s45_single as S45
import strategies.s46_net as S46
import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s108_event as EV
from research.harness import OOS_END
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

REF = (2.0, 3.0, 2.0, 21)
RATES = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
RISK = 0.08


def use_clock():
    g = S45.grid(S45.FULL_START)
    v = S45.composite(g, S46.LONG); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    H._ctxc.clear()


def event_daily(per_day, risk):
    EV.install_bars(per_day=per_day)
    EV.install(EV.grid_ev())
    return S69.daily(S69.sim(REF, EV.FULL_START, OOS_END, risk))


def comb(a, b):
    return pd.DataFrame({"a": a, "b": b}).fillna(0.0).sum(axis=1)


def halves(a):
    a = np.asarray(a, float); h = len(a) // 2
    return (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
            at_gate(a[h:])["cagr"] * 100)


if __name__ == "__main__":
    use_clock()
    R = S87.rankings()
    v7, _ = S87.blend(R, 3, RISK)
    use_clock()
    v7h, _ = S87.blend(R, 3, RISK / 2)
    ev = {p: event_daily(p, RISK / 2) for p in RATES}

    bf, b1, b2 = halves(v7)
    print("1. the sweep and the split half, both on V7's ACTUAL top-3 blend\n")
    print(f"{'book':>28}{'Shp':>7}{'Clm':>7}{'realDD':>9}{'full':>10}"
          f"{'1st half':>11}{'2nd half':>11}{'worse half':>12}")
    st = stats_of(np.asarray(v7, float))
    print(f"{'V7 as deployed':>28}{st['sharpe']:7.2f}{st['calmar']:7.2f}"
          f"{st['dd']*100:8.1f}%{bf:9.1f}%{b1:10.1f}%{b2:10.1f}%{'':>12}")
    for p in RATES:
        c = comb(v7h, ev[p])
        f_, h1, h2 = halves(c)
        s = stats_of(np.asarray(c, float))
        print(f"{f'V7 + event {p:g}/day':>28}{s['sharpe']:7.2f}{s['calmar']:7.2f}"
              f"{s['dd']*100:8.1f}%{f_:9.1f}%{h1:10.1f}%{h2:10.1f}%"
              f"{min(h1-b1, h2-b2):+11.1f}", flush=True)

    print("\n2. drawdown: realised against bootstrap, and the gate re-read on the\n"
          "   bootstrap median instead of the single realised path\n")
    print(f"{'book':>28}{'realDD':>9}{'bootDD':>9}{'P>20%':>8}"
          f"{'CAGR':>9}{'gate on real':>14}{'gate on boot':>14}")
    for tag, r in [("V7 as deployed", v7)] + [
            (f"V7 + event {p:g}/day", comb(v7h, ev[p])) for p in RATES]:
        a = np.asarray(r, float)
        s = stats_of(a); b = bootstrap_dd(a, n=4000)
        # what the gate would read if the drawdown were the bootstrap median:
        # CAGR scales with the bisection factor, which is linear to first order
        g_real = at_gate(a)["cagr"] * 100
        g_boot = g_real * abs(s["dd"]) / abs(b["dd_median"])
        print(f"{tag:>28}{s['dd']*100:8.1f}%{b['dd_median']*100:8.1f}%"
              f"{b['p_dd_worse_than_20']*100:7.0f}%{s['cagr']*100:8.1f}%"
              f"{g_real:13.1f}%{g_boot:13.1f}%", flush=True)

    print("\n3. the correlation that the whole thing rests on\n")
    print(f"{'bars/day':>10}{'rho vs V7':>12}{'ev Sharpe':>12}{'needs':>8}{'clears':>8}")
    s7 = stats_of(np.asarray(v7, float))["sharpe"]
    for p in RATES:
        rho = float(pd.DataFrame({"a": v7, "b": ev[p]}).fillna(0.0).corr().iloc[0, 1])
        se = stats_of(np.asarray(ev[p], float))["sharpe"]
        nd = s7 * (np.sqrt(2 + 2 * rho) - 1)
        print(f"{p:10g}{rho:12.3f}{se:12.2f}{nd:8.2f}"
              f"{('YES' if se > nd else 'no'):>8}", flush=True)
    print("\ndone: event-bar candidate")
