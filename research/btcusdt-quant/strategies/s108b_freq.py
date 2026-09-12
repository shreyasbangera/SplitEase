"""
S108b - How many event bars, and are they different enough to hold beside the clock book?

S108's screen says two things that do not fit together on first reading.

    at the -20% gate, three fixed configs      mean
    12h clock (deployed)      98.8 / 116.9 / 113.0     109.6
    event, 2 bars/day          63.8 /  41.9 /  47.4      51.0
    event, 4 bars/day          97.1 /  79.3 /  94.6      90.3

Two bars a day is the MATCHED comparison - 4,979 bars against the clock's 4,870,
within 10% in every year - and there the clock wins clearly. Yet doubling the
event-bar rate recovers most of the gap, while doubling the CLOCK rate destroys
the book (6h reads 43.8% on the full stack, 8h 29.5%). The two families respond
to frequency in opposite directions, which no result in this log predicts.

So the frequency is swept properly rather than assumed. Either it peaks below
the clock book - in which case the clock is doing real work and S101's fragility
is a property of the edge rather than of the sampling - or it peaks above, and
the deployed grid was never the right one.

The second question is the one that could actually matter for the brief. The
event book reaches Sharpe 1.8-2.1, and S102c set the bar a second sleeve must
clear beside the deployed book:

        S2  >  2.19 * (sqrt(2 + 2*rho) - 1)

which is 1.61 at rho = 0.5 and 1.79 at rho = 0.65. **Every horizon sleeve failed
this on Sharpe.** An event-bar sleeve is the first candidate in this study that
might clear it on Sharpe - so whether it clears depends entirely on a
correlation nobody has measured, and that is measured here.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s69_calsel as S69
import strategies.s108_event as EV
import strategies.s45_single as S45
import strategies.s46_net as S46
import research.harness as H
from research.harness import OOS_END
from strategies.s96_rank import at_gate, stats_of

CFGS = ((1.0, 3.0, 2.0, 21), (2.0, 3.0, 2.0, 21), (2.0, 2.5, 3.0, 14))
REF = (2.0, 3.0, 2.0, 21)          # the sleeve carried into the blend test
RATES = (2.0, 3.0, 4.0, 6.0, 8.0)
RISK = 0.08


def clock_book(risk):
    """The deployed 12h book, rebuilt in this process for a like-for-like blend."""
    g = S45.grid(S45.FULL_START)
    v = S45.composite(g, S46.LONG); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    H._ctxc.clear()
    return S69.daily(S69.sim(REF, S45.FULL_START, OOS_END, risk))


def event_book(per_day, risk, cfg=REF):
    EV.install_bars(per_day=per_day)
    EV.install(EV.grid_ev())
    return S69.daily(S69.sim(cfg, EV.FULL_START, OOS_END, risk))


def need(s1, rho):
    return s1 * (np.sqrt(2 + 2 * rho) - 1)


if __name__ == "__main__":
    print("1. the event-bar frequency sweep, three fixed configs, risk 8%\n")
    print(f"{'bars/day':>10}{'n':>7}{'exp':>5}{'CAGR':>9}{'MaxDD':>8}{'N':>7}"
          f"{'Shp':>7}{'Clm':>7}{'at -20%':>10}")
    best = {}
    for pd_ in RATES:
        EV.install_bars(per_day=pd_)
        g = EV.grid_ev()
        EV.install(g)
        gates = []
        for cfg in CFGS:
            m = S69.sim(cfg, EV.FULL_START, OOS_END, RISK)
            r = np.asarray(S69.daily(m), float)
            gt = at_gate(r)
            gates.append(gt["cagr"] * 100)
            print(f"{pd_:10.0f}{len(g):7d}{cfg[0]:5.1f}{m['cagr']*100:8.1f}%"
                  f"{m['max_dd']*100:7.1f}%{m['trades']:7d}{m['sharpe']:7.2f}"
                  f"{m['calmar']:7.2f}{gt['cagr']*100:9.1f}%", flush=True)
        best[pd_] = float(np.mean(gates))
        print(f"{'':10}mean at the gate {best[pd_]:.1f}%   "
              f"(12h clock reference 109.6%)\n", flush=True)

    top = max(best, key=best.get)
    print(f"best event-bar rate: {top:.0f} bars/day at {best[top]:.1f}% mean, "
          f"against the clock's 109.6%\n")

    print("2. is the event book a DIFFERENT stream, or the same one resampled?\n")
    ck = clock_book(RISK)
    s_clock = stats_of(np.asarray(ck, float))["sharpe"]
    print(f"{'bars/day':>10}{'Sharpe':>9}{'rho vs clock':>14}{'needs':>9}"
          f"{'clears?':>9}")
    ev = {}
    for pd_ in RATES:
        e = event_book(pd_, RISK)
        ev[pd_] = e
        both = pd.DataFrame({"c": ck, "e": e}).fillna(0.0)
        rho = float(both.corr().iloc[0, 1])
        se = stats_of(np.asarray(e, float))["sharpe"]
        nd = need(s_clock, rho)
        print(f"{pd_:10.0f}{se:9.2f}{rho:14.3f}{nd:9.2f}"
              f"{('YES' if se > nd else 'no'):>9}", flush=True)

    print(f"\n(clock sleeve Sharpe {s_clock:.2f}; the bar is "
          f"S2 > {s_clock:.2f} x (sqrt(2+2rho) - 1))\n")

    print("3. the blend, measured the way S87 blends configurations\n")
    print(f"{'book':>26}{'CAGR':>9}{'MaxDD':>8}{'Shp':>7}{'Clm':>7}"
          f"{'at -20%':>10}{'vs clock':>10}")
    ck2 = clock_book(RISK / 2)
    base = None
    a = np.asarray(ck, float)
    st, gt = stats_of(a), at_gate(a)
    base = gt["cagr"] * 100
    print(f"{'12h clock alone':>26}{st['cagr']*100:8.1f}%{st['dd']*100:7.1f}%"
          f"{st['sharpe']:7.2f}{st['calmar']:7.2f}{base:9.1f}%")
    for pd_ in RATES:
        e2 = event_book(pd_, RISK / 2)
        comb = pd.DataFrame({"c": ck2, "e": e2}).fillna(0.0).sum(axis=1)
        a = np.asarray(comb, float)
        st, gt = stats_of(a), at_gate(a)
        print(f"{f'clock + event {pd_:.0f}/day':>26}{st['cagr']*100:8.1f}%"
              f"{st['dd']*100:7.1f}%{st['sharpe']:7.2f}{st['calmar']:7.2f}"
              f"{gt['cagr']*100:9.1f}%{gt['cagr']*100-base:+10.1f}", flush=True)
    print("\ndone: event frequency and blend")
