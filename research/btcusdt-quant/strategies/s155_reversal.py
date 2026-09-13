"""
S155 - The full screen on the honest universe, including short-term reversal.

WHAT S147 MISSED BY SETTING ITS BAR AT THE WRONG HORIZON
---------------------------------------------------------
S147 screened 17 features and required |t| > 2 at a FIVE-day horizon. Two
features cleared it. But look at what it printed at ONE day and discarded:

    rev3    IC +0.0216   t +2.97
    rev5    IC +0.0211   t +2.91
    vol90   IC +0.0748   t +8.98

Short-term reversal was significant at one day and was thrown away because it
decays by five - which is exactly what a reversal effect is supposed to do. The
bar was set at the wrong horizon for that family and the family was dismissed on
it. Reversal is the single best-documented cross-sectional effect in crypto, it
is a liquidity-provision premium rather than a forecast, and it lives at one to
three days.

That is a search error, not a data limitation, so this file re-screens the whole
feature set on the point-in-time universe at the horizon each family actually
lives at, with the surrogate control applied to every survivor rather than to a
chosen few.

THE COST PROBLEM, STATED UP FRONT
----------------------------------
A one-day reversal book rebalances daily and its turnover is near 2.0 gross per
day, against 0.07 for low-vol. At 10bps a side that is roughly 73% a year in
costs. Reversal has to be very strong to survive that, and this file charges it
in full and sweeps the cost rather than assuming a friendly number.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s151_pit as PIT
import strategies.s148_lowvol as L
import strategies.s147_panel as P
import strategies.s153_honest as H
import strategies.s154_stack as ST
from strategies.s96_rank import at_gate, stats_of


def features(px, dv, ok, fund):
    lr = np.log(px); r1 = lr.diff(1)
    F = {}
    for k in (1, 2, 3, 5, 10):
        F[f"rev{k}"] = -lr.diff(k)
    # reversal standardised by the asset's own volatility: a 10% move in a quiet
    # coin is a bigger surprise than in a wild one
    sd = r1.rolling(30, min_periods=20).std()
    for k in (1, 3, 5):
        F[f"srev{k}"] = -(lr.diff(k) / (sd * np.sqrt(k) + 1e-12))
    # reversal against the market rather than against zero
    mkt = r1.mean(axis=1)
    for k in (1, 3, 5):
        F[f"irev{k}"] = -(lr.diff(k).sub(mkt.rolling(k).sum(), axis=0))
    for k in (7, 30, 90, 180):
        F[f"mom{k}"] = lr.diff(k)
    F["vol30"] = -r1.rolling(30, min_periods=20).std()
    F["vol90"] = -r1.rolling(90, min_periods=60).std()
    F["carry1"] = -fund
    F["carry3"] = -fund.rolling(3, min_periods=1).mean()
    F["carry7"] = -fund.rolling(7, min_periods=3).mean()
    F["dvol"] = -np.log(dv.clip(lower=1.0)).diff(1)     # volume spike fade
    F["amihud"] = (r1.abs() / dv.replace(0, np.nan)).rolling(30, min_periods=20).mean()
    return {k: v.where(ok) for k, v in F.items()}


if __name__ == "__main__":
    pan = PIT.build()
    px = pan["close"]; dv = pan["qv"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, dv, ok = px.loc[first:], dv.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = H.pit_funding(list(px.columns), px.index)
    dead = ST.death_window(px)
    lr = np.log(px)

    print("S155 - full screen on the honest universe, reversal included\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days, "
          f"median {int(ok.sum(axis=1).median())} tradeable/day\n")

    F = features(px, dv, ok, fund)
    print("CROSS-SECTIONAL IC. t counts days and is deflated for overlap.")
    print(f"   {'feature':>9}" + "".join(f"{'IC'+str(k)+'d':>9}{'t':>7}"
                                         for k in (1, 3, 5, 20)) +
          f"{'1st h':>8}{'2nd h':>8}{'best':>6}")
    surv = []
    for name, f in F.items():
        row, ics = "", {}
        for k in (1, 3, 5, 20):
            ic = P.xs_ic(f.shift(1), lr.diff(k).shift(-k), ok)
            m, t, _ = P.ic_stat(ic, k)
            ics[k] = (m, t, ic); row += f"{m:>9.4f}{t:>7.2f}"
        bk = max(ics, key=lambda k: abs(ics[k][1]))
        v = ics[bk][2].dropna(); h = len(v) // 2
        i1, i2 = v.iloc[:h].mean(), v.iloc[h:].mean()
        ok_sign = np.sign(i1) == np.sign(i2)
        if abs(ics[bk][1]) > 2.5 and ok_sign:
            surv.append((name, bk, ics[bk][0], ics[bk][1]))
        print(f"   {name:>9}{row}{i1:>8.4f}{i2:>8.4f}{str(bk)+'d':>6}")

    print(f"\nsurvivors (|t| > 2.5 at their best horizon, sign stable across "
          f"halves): {len(surv)}")
    print("\nSURROGATE CONTROL on every survivor - 100 phase-randomised panels")
    real = []
    for name, k, m, t in sorted(surv, key=lambda x: -abs(x[3])):
        ts = []
        for _ in range(100):
            s = P.phase_randomise(F[name])
            ic = P.xs_ic(s.shift(1), lr.diff(k).shift(-k), ok)
            _, tt, _ = P.ic_stat(ic, k)
            if np.isfinite(tt):
                ts.append(abs(tt))
        ts = np.array(ts); pv = float((ts >= abs(t)).mean())
        tag = "REAL" if pv < 0.05 else "not distinguishable"
        if pv < 0.05:
            real.append((name, k, m, t))
        print(f"   {name:>9} @{k:>2}d  real |t| {abs(t):>6.2f}  surrogate median "
              f"{np.median(ts):>5.2f}, 95th {np.percentile(ts,95):>5.2f}  "
              f"p = {pv:.3f}  {tag}")

    print(f"\nBOOKS for the {len(real)} features that survive BOTH tests.")
    print("position limit 5%, full haircut on dying names, 5bps fee + slip swept")
    for name, k, m, t in real:
        print(f"\n   --- {name} (best at {k}d, IC {m:+.4f}, t {t:+.2f})")
        print(f"   {'construction':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
              f"{'turn':>8}{'at -20%':>9}")
        for mode in ("zscore", "dollarneutral"):
            for sl in (5.0, 10.0):
                w = L.weights(F[name], ok, mode)
                net, turn = ST.book(px, w, fund if "carry" in name else None,
                                    dead=dead, haircut=1.0, max_w=0.05,
                                    slip_bps=sl)
                H.show(net, turn, f"{mode}, {sl:.0f}bps slip", w=26)
    print("\ndone: reversal screen")
