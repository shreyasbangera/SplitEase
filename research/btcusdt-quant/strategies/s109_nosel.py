"""
S109 - How many configurations should be held, when the selection is the problem?

S108e killed the event-bar candidate in a specific way. Giving the event sleeve
V7's quarterly machinery took it from Sharpe 2.21 to 1.49 and from 86.4% to
32.9% at the gate. The diagnosis was that trailing Calmar is a MAXIMUM over 200
cells, the event panel's training Calmars run 42-69 against the clock panel's
3-40, and the noisier the cells the more of that maximum is noise.

That diagnosis makes a prediction nobody has tested, and the prediction is not
about event bars at all. **If the selection is mostly noise, the right number of
configurations to hold is large, not three.** S86 swept k over {1, 3, 5, 8, 12}
on the clock panel and stopped at 12 because 3 won. It never asked what happens
at 50, or at 200 - which is the same as not selecting at all.

The detail that makes this worth running: on event bars a SINGLE hand-picked
configuration reached Sharpe 2.21, against the deployed clock book's 2.15. Not
selecting already beat selecting there. So the question is whether the quarterly
apparatus - the thing responsible for S96c's 24-point noise band, S96b's oracle
ceiling and S106's weight instability - is load-bearing or is a tax.

    top-k       hold the k best by trailing 12m Calmar, at risk/k   (V7 uses 3)
    all         hold all 200 equally: no selection, nothing refit
    random-k    hold k at random each quarter, resampled

The random control is the point. S86b established that V7's blend gain is the
RANKING rather than diversification, by showing random-3 far below top-3. If
top-k and random-k converge as k grows, the ranking is contributing nothing that
holding more cells does not, and a book with no quarterly selection is a
strictly better object: nothing is refit, there is no selection-noise band, and
neither oracle bound applies to it.

Every (quarter, configuration) return series is computed once and cached, so
every k, and every rule anyone asks about later, is free after the first run.
"""
import sys, os; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
import strategies.s45_single as S45
import strategies.s46_net as S46
import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s108_event as EV
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

RISK = 0.08
# Every series is simulated once at a small reference risk and scaled. Under
# "k sleeves at risk/k, summed", linearity makes the blend (RISK/REF) times the
# MEAN of the k series, with the scale independent of k - so one cache serves
# every k. S94 validated the linear rescaling directly (daily correlation
# 0.9962, ~2% optimistic in level), and at_gate rescales again anyway, so only
# the shape is being compared. REF is kept small to stay in the linear regime.
REF = 0.01
KS = (1, 3, 5, 8, 12, 25, 50, 100, 200)
NDRAW = 60


def use_clock():
    g = S45.grid(S45.FULL_START)
    v = S45.composite(g, S46.LONG); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    H._ctxc.clear()


def panel(tag):
    """Install a panel and return its cached quarterly rankings."""
    if tag == "clock":
        use_clock()
        S87.RANKS = "/home/user/quant/results/ranks_gate200.json"
        return S87.rankings()
    p = float(tag)
    EV.install_bars(per_day=p)
    EV.install(EV.grid_ev())
    import json
    return [(s, e, [tuple(c) for c in cs]) for s, e, cs in
            json.load(open(f"/home/user/quant/results/ranks_ev{p:g}.json"))]


def returns_cache(tag, R):
    """Daily returns for every (quarter, configuration). The whole cost."""
    npz = f"/home/user/quant/results/s109_{tag}.npz"
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        return z["R"].item()
    out = {}
    for qi, (s, e, cfgs) in enumerate(R):
        for cfg in cfgs:                       # all 200, in ranked order
            out[(s, cfg)] = np.asarray(S69.daily(S87.sim(cfg, s, e, REF)), float)
        print(f"    {s[:7]} ({qi+1}/{len(R)})", flush=True)
        np.savez(npz, R=np.array(out, dtype=object))
    return out


def blend(R, C, pick):
    """k sleeves at risk/k, summed - which under linear rescaling is (RISK/REF)
    times the MEAN of the k cached series, the scale falling out independent of
    k. The engine's own nonlinearity in risk is therefore held out of the
    comparison, so that only the COUNT differs between rows."""
    segs = []
    for s, e, cfgs in R:
        sel = pick(s, cfgs)
        segs.append(np.mean([C[(s, c)] for c in sel], axis=0) * (RISK / REF))
    return np.concatenate(segs)


def row(tag, a, base=None):
    st, gt = stats_of(a), at_gate(a)
    h = len(a) // 2
    g1, g2 = at_gate(a[:h])["cagr"] * 100, at_gate(a[h:])["cagr"] * 100
    d = "" if base is None else f"{gt['cagr']*100-base[0]:+9.1f}"
    print(f"{tag:>22}{st['sharpe']:7.2f}{st['calmar']:7.2f}{st['dd']*100:8.1f}%"
          f"{gt['cagr']*100:10.1f}%{g1:9.1f}%{g2:9.1f}%{d}", flush=True)
    return (gt["cagr"] * 100, g1, g2)


if __name__ == "__main__":
    for tag in ("clock", "3", "4"):
        R = panel(tag)
        print(f"\n=== panel: {tag} " + "=" * 50)
        C = returns_cache(tag, R)
        print(f"\n{'book':>22}{'Shp':>7}{'Clm':>7}{'realDD':>9}{'at -20%':>10}"
              f"{'1st h':>9}{'2nd h':>9}{'vs top-3':>9}")
        base = None
        res = {}
        for k in KS:
            a = blend(R, C, lambda s, cfgs, k=k: cfgs[:k])
            r = row(f"top-{k}" + (" = all, no sel" if k == 200 else ""), a,
                    base if k != 3 else None)
            res[k] = r
            if k == 3:
                base = r
        print()
        rng = np.random.default_rng(0)
        for k in (3, 12, 50):
            v = []
            for _ in range(NDRAW):
                a = blend(R, C, lambda s, cfgs, k=k: [cfgs[i] for i in
                                                      rng.choice(len(cfgs), k, replace=False)])
                v.append(at_gate(a)["cagr"] * 100)
            v = np.array(v)
            top = res[k][0]
            print(f"{f'random-{k}':>22}{'':14}{'':9}{np.median(v):10.1f}%"
                  f"{'':18}   top-{k} is {(top-v.mean())/max(v.std(ddof=1),1e-9):+.1f} sd "
                  f"above random (sd {v.std(ddof=1):.1f})", flush=True)
    print("\ndone: how many configurations")
