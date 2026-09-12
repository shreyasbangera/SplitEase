"""
S108e - The fair comparison: give the event sleeve the same machinery the clock book has.

Every number so far has handicapped the event sleeve. V7's clock side is the
full apparatus - 200 configurations, quarterly Calmar selection, top-3 held at
risk/3, trend gate - and the event side has been ONE hand-picked configuration
with no selection and no gate. That the blend still read +28.7 on the gate and
took P(drawdown > 20%) from 33% to 7% was achieved under that handicap.

It also explains the one control the candidate failed. On the full stack the
blend is worse than V7 in the first half by 5.6 points, and the frequency
profile is unstable - 128.4% at 3.5 bars a day against 197.5% at 3 and 182.3%
at 4. A single fixed configuration has no way to adapt to a changing regime,
which is exactly what the quarterly selection exists to do and exactly what the
first half of the sample demands.

So the event panel gets its own rankings, recomputed from scratch on its own
bars - never reused from the clock panel, which would be a selection fitted on
one grid and applied to another, measuring neither.

The bar to clear is unchanged and is stated before the run:

    1  higher CAGR at the -20% gate than V7's 168.8%
    2  survives the dilution control - the blend already halves each sleeve
    3  holds in BOTH halves of the sample, which is what the fixed-config
       version failed
    4  clears S96c's 24-point noise band
"""
import sys, os, json; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
import strategies.s45_single as S45
import strategies.s46_net as S46
import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s108_event as EV
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of, quarters

RISK, K = 0.08, 3
RATES = (3.0, 4.0)


def use_clock():
    g = S45.grid(S45.FULL_START)
    v = S45.composite(g, S46.LONG); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    H._ctxc.clear()


def rankings_resumable(tag):
    """Checkpointed per quarter; a reclaimed container costs one quarter, not all."""
    path = f"/home/user/quant/results/ranks_ev{tag}.json"
    done = []
    if os.path.exists(path):
        done = [(s, e, [tuple(c) for c in cs]) for s, e, cs in json.load(open(path))]
    Q = quarters()
    if len(done) >= len(Q):
        return done
    for s, e in Q[len(done):]:
        tr0 = str((pd.Timestamp(s, tz="UTC") - pd.DateOffset(months=12)).date())
        sc = sorted(((S69.calmar_of(S87.sim(cfg, tr0, s, 0.10)), cfg)
                     for cfg in S87.GRID), key=lambda x: -x[0])
        done.append((s, e, [tuple(c) for c in (c for _, c in sc)]))
        json.dump([[a, b, [list(c) for c in cs]] for a, b, cs in done], open(path, "w"))
        print(f"    {s[:7]}  best Calmar {sc[0][0]:6.2f}  [{len(done)}/{len(Q)}]",
              flush=True)
    return done


def event_stack(per_day, risk):
    EV.install_bars(per_day=per_day)
    EV.install(EV.grid_ev())
    R = rankings_resumable(f"{per_day:g}")
    r, pnl = S87.blend(R, K, risk)
    return r, pnl


def comb(a, b):
    return pd.DataFrame({"a": a, "b": b}).fillna(0.0).sum(axis=1)


def halves(a):
    a = np.asarray(a, float); h = len(a) // 2
    return (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
            at_gate(a[h:])["cagr"] * 100)


def report(tag, r, base=None):
    a = np.asarray(r, float)
    s = stats_of(a); b = bootstrap_dd(a, n=4000)
    f_, h1, h2 = halves(a)
    d = "" if base is None else f"{f_-base[0]:+9.1f}{min(h1-base[1], h2-base[2]):+12.1f}"
    print(f"{tag:>30}{s['sharpe']:7.2f}{s['calmar']:7.2f}{s['dd']*100:8.1f}%"
          f"{b['dd_median']*100:8.1f}%{b['p_dd_worse_than_20']*100:6.0f}%"
          f"{f_:9.1f}%{h1:9.1f}%{h2:9.1f}%{d}", flush=True)
    return (f_, h1, h2)


if __name__ == "__main__":
    use_clock()
    R = S87.rankings()
    v7, _ = S87.blend(R, K, RISK)
    use_clock()
    v7h, _ = S87.blend(R, K, RISK / 2)

    ev_full, ev_half = {}, {}
    for p in RATES:
        print(f"\nranking the event panel at {p:g} bars/day")
        ev_full[p], _ = event_stack(p, RISK)
        ev_half[p], _ = event_stack(p, RISK / 2)

    print(f"\n{'book':>30}{'Shp':>7}{'Clm':>7}{'realDD':>9}{'bootDD':>9}"
          f"{'P>20%':>7}{'full':>9}{'1st h':>9}{'2nd h':>9}{'vs V7':>9}{'worse half':>12}")
    base = report("V7 as deployed (clock)", v7)
    for p in RATES:
        report(f"event {p:g}/day stack alone", ev_full[p], base)
    for p in RATES:
        report(f"V7 + event {p:g}/day stack", comb(v7h, ev_half[p]), base)

    print("\ncorrelation and the S102c admission bar, full stack against full stack")
    s7 = stats_of(np.asarray(v7, float))["sharpe"]
    print(f"{'bars/day':>10}{'rho vs V7':>12}{'ev Sharpe':>12}{'needs':>8}{'clears':>8}")
    for p in RATES:
        rho = float(pd.DataFrame({"a": v7, "b": ev_full[p]}).fillna(0.0).corr().iloc[0, 1])
        se = stats_of(np.asarray(ev_full[p], float))["sharpe"]
        nd = s7 * (np.sqrt(2 + 2 * rho) - 1)
        print(f"{p:10g}{rho:12.3f}{se:12.2f}{nd:8.2f}"
              f"{('YES' if se > nd else 'no'):>8}")
    print("\ndone: event stack")
