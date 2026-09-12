"""
S108c - The controls on the first candidate that has ever cleared the bar.

S108b produced this, and it is the first result in 108 experiments that beats
the deployed book by more than noise:

    12h clock alone                 63.5%  -12.5%  Shp 2.16  Clm  5.10   116.9%
    clock + event 3/day             76.2%   -7.5%  Shp 2.51  Clm 10.17   332.4%

One half of it is already explained and is not in doubt. Two sleeves at Sharpe
2.16 and 2.21, correlated 0.499, combine to a predicted Sharpe of
(2.16+2.21)/sqrt(2+2*0.499) = **2.52**, against **2.51** observed. The Sharpe
gain is textbook decorrelation, exactly the size the arithmetic says, and it is
the first time any sleeve in this log has cleared S102c's admission bar.

The other half is the problem. Sharpe rose 16%; **Calmar doubled**, because the
blend's worst drawdown is -7.5% against -12.5% and -21.5% for the two sleeves
alone. A 40% cut in the denominator is more than decorrelation buys, and at the
gate the denominator is the whole number - which is precisely the trap S105b
caught, where a cap that clipped one 2024 episode read +8.7 on the full sample
and was NEGATIVE in both halves of it.

And the sweep shape is the other thing this log has learned to distrust:
+18.5, **+215.5**, +60.1, -24.7, +52.4 across 2/3/4/6/8 bars a day. The winning
cell is 3.5x its best neighbour. Every interior spike in this study - S99's
thresholds, S105's caps, S106's weights - has been noise.

So: four controls, and the honest prediction stated first so it cannot be
rewritten afterwards. **If only the Sharpe gain is real, the sustainable figure
is about 135-160%, not 332%.**

    1  SPLIT HALF      the control that killed S105. A gain that lives in one
                       half is one lucky path, whatever it reads on the full
                       sample.
    2  FULL STACK      S108b compared against ONE configuration at 116.9%. V7
                       deploys the quarterly top-3 blend at 168.8%, and that is
                       the number a candidate has to beat.
    3  EXTRA LAG       act one whole decision bar later. Event-bar boundaries
                       depend on traded volume, so if anything about a bar's
                       close leaked into its own signal, a delay will not
                       degrade the book gracefully - it will collapse it.
    4  BOOTSTRAP DD    is -7.5% a property of the blend or of this path?
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import engine.data as ED
import research.harness as H
import strategies.s45_single as S45
import strategies.s46_net as S46
import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s108_event as EV
from research.harness import backtest, OOS_END
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

REF = (2.0, 3.0, 2.0, 21)
RATES = (2.0, 3.0, 4.0, 8.0)
RISK = 0.08


def use_clock():
    g = S45.grid(S45.FULL_START)
    v = S45.composite(g, S46.LONG); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    H._ctxc.clear()


def use_event(per_day):
    EV.install_bars(per_day=per_day)
    EV.install(EV.grid_ev())


def sim_lag(cfg, start, end, risk, lag=1):
    """S69.sim with the alignment lag exposed, to act a whole bar later."""
    c = S69.ctx(); u = np.nan_to_num(S69.shape(cfg[0])); a = c["a"]
    arr = dict(entry=u, stop=cfg[1] * a, tp=cfg[1] * cfg[2] * a,
               exit=(np.abs(u) <= 0.0).astype(float))
    orig = ED.align_to_exec
    if lag != 1:
        H.align_to_exec = lambda s, e, ar, lag=1, _o=orig, _l=lag: _o(s, e, ar, lag=_l)
    try:
        return backtest(c["g"], arr, "x", start=start, end=end, risk=risk,
                        max_lev=10.0, max_bars_h=cfg[3] * 24)
    finally:
        H.align_to_exec = orig


def daily_of(kind, risk, per_day=None, lag=1):
    if kind == "clock":
        use_clock(); s = S45.FULL_START
    else:
        use_event(per_day); s = EV.FULL_START
    H._ctxc.clear()
    return S69.daily(sim_lag(REF, s, OOS_END, risk, lag=lag))


def comb(a, b):
    return pd.DataFrame({"a": a, "b": b}).fillna(0.0).sum(axis=1)


def row(tag, r, base=None, extra=""):
    a = np.asarray(r, float)
    st, gt = stats_of(a), at_gate(a)
    d = "" if base is None else f"{gt['cagr']*100-base:+10.1f}"
    print(f"{tag:>30}{st['cagr']*100:9.1f}%{st['dd']*100:8.1f}%{st['sharpe']:7.2f}"
          f"{st['calmar']:8.2f}{gt['cagr']*100:10.1f}%{d}{extra}", flush=True)
    return gt["cagr"] * 100


if __name__ == "__main__":
    print("PREDICTION, stated before the controls run: if only the Sharpe gain is")
    print("real, the sustainable figure is about 135-160%, not 332%.\n")

    ck = daily_of("clock", RISK)
    ck2 = daily_of("clock", RISK / 2)
    ev2 = {p: daily_of("event", RISK / 2, p) for p in RATES}

    print("1. SPLIT HALF - each half scaled to the -20% gate separately\n")
    print(f"{'book':>30}{'full':>10}{'1st half':>11}{'2nd half':>11}"
          f"{'worse half':>12}")

    def halves(a):
        a = np.asarray(a, float); h = len(a) // 2
        return (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
                at_gate(a[h:])["cagr"] * 100)

    bf, b1, b2 = halves(ck)
    print(f"{'12h clock alone':>30}{bf:9.1f}%{b1:10.1f}%{b2:10.1f}%{'':>12}")
    for p in RATES:
        f_, h1, h2 = halves(comb(ck2, ev2[p]))
        print(f"{f'clock + event {p:.0f}/day':>30}{f_:9.1f}%{h1:10.1f}%{h2:10.1f}%"
              f"{min(h1-b1, h2-b2):+11.1f}", flush=True)

    print("\n2. FULL STACK - against what V7 actually deploys, not one config\n")
    print(f"{'book':>30}{'CAGR':>10}{'MaxDD':>9}{'Shp':>7}{'Clm':>8}"
          f"{'at -20%':>10}{'vs V7':>11}")
    use_clock()
    R = S87.rankings()
    v7, _ = S87.blend(R, 3, RISK)
    base = row("V7 (clock, top-3 blend)", v7)
    use_clock()
    v7h, _ = S87.blend(R, 3, RISK / 2)
    for p in RATES:
        row(f"V7 + event {p:.0f}/day", comb(v7h, ev2[p]), base)

    print("\n3. EXTRA LAG - act one whole decision bar later than the book does\n")
    print(f"{'book':>30}{'CAGR':>10}{'MaxDD':>9}{'Shp':>7}{'Clm':>8}"
          f"{'at -20%':>10}{'vs lag 1':>11}")
    for p in (3.0, 4.0):
        b1_ = row(f"event {p:.0f}/day, lag 1", daily_of("event", RISK, p, lag=1))
        row(f"event {p:.0f}/day, lag 2", daily_of("event", RISK, p, lag=2), b1_)
    ckb = row("12h clock, lag 1", ck)
    row("12h clock, lag 2", daily_of("clock", RISK, lag=2), ckb)

    print("\n4. BOOTSTRAP DRAWDOWN - is -7.5% the blend, or this path?\n")
    for tag, r in (("12h clock alone", ck),
                   *[(f"clock + event {p:.0f}/day", comb(ck2, ev2[p])) for p in RATES]):
        a = np.asarray(r, float)
        b = bootstrap_dd(a, n=4000)
        print(f"{tag:>30}  realised {stats_of(a)['dd']*100:6.1f}%   "
              f"bootstrap median {b['dd_median']*100:6.1f}%   "
              f"P(worse than 20%) {b['p_dd_worse_than_20']*100:3.0f}%", flush=True)
    print("\ndone: event-bar controls")
