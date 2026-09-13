"""
S157 - Where the money actually is: the right tail, not the rank.

THE CORRECTION THIS FILE MAKES TO S147 AND S155
------------------------------------------------
Both screened with a cross-sectional RANK IC and both concluded that
cross-sectional momentum is absent or negative in this universe. On a rank
statistic that is exactly what the data says: mom90 came in at t = -4.80, and
short-term reversal was positive and survived a surrogate control at p = 0.01.

Sorting the same universe into deciles by its 5-day return and measuring what
each decile actually PAYS says something different and incompatible:

    decile 1  (biggest losers)    mean  -6.0 bp/day   median 0.0 bp   skew +0.31
    decile 10 (biggest winners)   mean +21.8 bp/day   median 0.0 bp   skew +1.51

The medians are both zero. The rank IC is measuring the middle of the
distribution, where losers do modestly outperform. The money is entirely in the
right tail, where recent winners keep winning and occasionally by enormous
amounts. Rank and dollars point in opposite directions, and a long-D1/short-D10
book built on the rank signal loses 27.8 bp a day.

**A rank IC is not a tradeable edge when the payoff is this skewed.** That is the
single most useful thing this portfolio line has produced, and it invalidates the
screening method used in S147, S149 and S155 - not their arithmetic, their
choice of statistic. Everything below is measured in dollars.

THE OTHER THING THE DECILES SHOW
---------------------------------
Every decile has 86-112% annualised volatility, and eight of the ten have a
NEGATIVE CAGR despite a positive mean daily return. That is volatility drag:
a portfolio averaging +5bp a day at 90% vol compounds at roughly -20% a year.
An equal-weighted basket of altcoins is a machine for converting positive
average returns into losses, which is why every construction in S148-S156 that
equal-weighted anything struggled. Sizing by risk is not a refinement here, it
is the difference between a positive and a negative number.

WHAT IS BUILT
-------------
Long the top slice by trailing return, sized by inverse volatility so no single
name supplies the risk, hedged three ways (unhedged, against the universe mean,
against the bottom slice), with lookback and slice width swept rather than
chosen. Costs, funding, a 5% name limit and a full haircut on dying contracts
throughout.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s151_pit as PIT
import strategies.s153_honest as H
import strategies.s154_stack as ST
from strategies.s96_rank import at_gate, stats_of


def slice_weights(score, px, ok, frac=0.10, side="top", risk_parity=True,
                  vol_win=30):
    """Equal-RISK weights over the top (or bottom) `frac` of the cross-section."""
    r = score.where(ok).rank(axis=1, pct=True)
    m = (r > 1 - frac) if side == "top" else (r <= frac)
    m = m & ok
    w = m.astype(float)
    if risk_parity:
        rv = px.pct_change().rolling(vol_win, min_periods=20).std()
        floor = float(np.nanquantile(rv.to_numpy(), 0.05))
        w = w / rv.clip(lower=floor)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


if __name__ == "__main__":
    pan = PIT.build()
    px = pan["close"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = H.pit_funding(list(px.columns), px.index)
    dead = ST.death_window(px)
    lr = np.log(px)

    print("S157 - the right tail, measured in dollars\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days, "
          f"median {int(ok.sum(axis=1).median())} tradeable/day\n")

    print("0. VOLATILITY DRAG, since it governs every construction below")
    ew = ok.astype(float).div(ok.sum(axis=1).replace(0, np.nan), axis=0)
    pr = (ew.shift(1).fillna(0.0) * px.pct_change()).sum(axis=1)
    rp = slice_weights(pd.DataFrame(1.0, index=px.index, columns=px.columns),
                       px, ok, frac=1.0, side="top")
    pr2 = (rp.shift(1).fillna(0.0) * px.pct_change()).sum(axis=1)
    for tag, s in (("equal DOLLAR weight, all names", pr),
                   ("equal RISK weight, all names", pr2)):
        st = stats_of(s.to_numpy(float))
        print(f"   {tag:>32}: mean {np.mean(s)*1e4:>+6.1f}bp/day -> CAGR "
              f"{st['cagr']*100:>+7.1f}%  vol {np.std(s)*np.sqrt(365.25)*100:>5.1f}%"
              f"  Sharpe {st['sharpe']:>+5.2f}")

    print("\n1. LONG THE TOP SLICE, sized by risk. Lookback and width swept.")
    print(f"   {'lookback':>9}{'width':>7}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>7}{'at -20%':>9}")
    best = None
    for lb in (5, 10, 20, 60, 120):
        for fr in (0.05, 0.10, 0.20):
            w = slice_weights(lr.diff(lb), px, ok, frac=fr)
            net, turn = ST.book(px, w, fund, dead=dead, haircut=1.0, max_w=0.05)
            st, g = H.show(net, turn, f"{lb}d{fr*100:>6.0f}%", w=16)
            if np.isfinite(g) and (best is None or g > best[0]):
                best = (g, lb, fr, net)

    print("\n2. HEDGED - the same long slice against three shorts")
    lb, fr = best[1], best[2]
    print(f"   using the best long slice from above: {lb}d lookback, "
          f"{fr*100:.0f}% width")
    print(f"   {'hedge':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>7}"
          f"{'at -20%':>9}")
    wl = slice_weights(lr.diff(lb), px, ok, frac=fr)
    hedges = {
        "none (long only)": None,
        "universe mean (risk-weighted)": slice_weights(
            pd.DataFrame(1.0, index=px.index, columns=px.columns), px, ok,
            frac=1.0, side="top"),
        "bottom slice": slice_weights(lr.diff(lb), px, ok, frac=fr, side="bottom"),
    }
    sleeves = {}
    for tag, ws in hedges.items():
        w = wl if ws is None else (wl - ws)
        net, turn = ST.book(px, w, fund, dead=dead, haircut=1.0, max_w=0.05)
        H.show(net, turn, tag, w=28)
        sleeves[tag] = net

    print("\n3. AVERAGED ACROSS LOOKBACKS - parameter noise removed, which is the "
          "one\n   form of combination this study has found not to dilute")
    print(f"   {'book':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>7}{'at -20%':>9}")
    for hedge in ("none", "mean"):
        acc = []
        for lb2 in (5, 10, 20, 60, 120):
            w = slice_weights(lr.diff(lb2), px, ok, frac=fr)
            if hedge == "mean":
                w = w - hedges["universe mean (risk-weighted)"]
            nb, _ = ST.book(px, w, fund, dead=dead, haircut=1.0, max_w=0.05)
            acc.append(nb)
        a = sum(acc) / len(acc)
        H.show(a, pd.Series(0.0, index=a.index), f"avg over 5 lookbacks, {hedge}", w=28)
        sleeves[f"avg_{hedge}"] = a

    print("\n4. VOL-TARGETED, which is how the gate is actually reached")
    print(f"   {'book @ target':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>7}"
          f"{'at -20%':>9}")
    for tv in (0.20, 0.30, 0.50):
        acc = []
        for lb2 in (5, 10, 20, 60, 120):
            w = slice_weights(lr.diff(lb2), px, ok, frac=fr) \
                - hedges["universe mean (risk-weighted)"]
            nb, _ = ST.book(px, w, fund, dead=dead, haircut=1.0, max_w=0.05,
                            target_vol=tv)
            acc.append(nb)
        a = sum(acc) / len(acc)
        H.show(a, pd.Series(0.0, index=a.index), f"hedged avg @ {tv*100:.0f}%", w=28)

    print("\n5. COST SENSITIVITY on the hedged average")
    print(f"   {'slippage/side':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>7}"
          f"{'at -20%':>9}")
    for sl in (2.0, 5.0, 10.0, 20.0):
        acc, tt = [], []
        for lb2 in (5, 10, 20, 60, 120):
            w = slice_weights(lr.diff(lb2), px, ok, frac=fr) \
                - hedges["universe mean (risk-weighted)"]
            nb, tb = ST.book(px, w, fund, dead=dead, haircut=1.0, max_w=0.05,
                             slip_bps=sl)
            acc.append(nb); tt.append(tb)
        a = sum(acc) / len(acc)
        H.show(a, sum(tt) / len(tt), f"{sl:.0f}bps", w=28)
    print("\ndone: tail momentum")
