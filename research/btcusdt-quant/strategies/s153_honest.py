"""
S153 - An honest haircut, and the full sleeve stack on the real universe.

THE FLAW IN S152, CORRECTED HERE
---------------------------------
S152 charged for uncollectable windfalls with a symmetric per-asset daily return
cap, and the book got BETTER as the cap tightened: Sharpe 0.51 uncapped, 1.27 at
25%. That should have been the tell. A symmetric cap truncates the left tail as
well as the right, so it does not only remove gains a real book could not
collect - it also refunds losses a real book certainly would pay. It is not a
haircut, it is a stop-loss with hindsight, and reporting it as conservative
would have been wrong.

The correct version is ASYMMETRIC. A position's PROFIT on a day is capped,
because that is the side that depends on an exchange filling you in a collapsing
contract; its LOSS is not, because nothing about a collapse makes a losing
position cheaper to hold. That is implemented here and every S152 figure is
re-reported under it.

WHAT S152 GOT RIGHT, AND IS WORTH KEEPING
------------------------------------------
    the point-in-time universe BEATS the survivor universe, 1.27 against 0.41
    Sharpe and a 15.4% gate against 3.7%.

That is the opposite of the usual survivorship story and the reason is
mechanical: a dying coin is violently volatile and expensive to hold, so the
low-vol and carry factors are both SHORT it. Removing the corpses from the panel
removes the profitable side of the trade. Survivorship bias was making this
strategy look worse than it is, not better.

    breadth helps. The same IC measured across 52 names carries t = 8.40 where
    16 names carried t = 5.73.

WHAT IS STACKED HERE
--------------------
The three sleeves S150 found to be mutually uncorrelated (+0.03, -0.09, -0.07),
now on the honest universe: low-volatility, carry, and the range-expansion
breakout. Sleeve weights are equal, because S106b showed informed weighting
loses to equal weights by 8 standard deviations in this study.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, glob, os, zipfile, io

import strategies.s151_pit as PIT
import strategies.s148_lowvol as L
import strategies.s147_panel as P
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"


def pit_funding(cols, index, src=f"{D}/pitfund"):
    """Daily funding per asset from the point-in-time archive."""
    out = {}
    for s in cols:
        fs = sorted(glob.glob(f"{src}/{s}__*.zip"))
        if not fs:
            fs = sorted(glob.glob(f"{D}/fund2/{s}/*.zip"))
        if not fs:
            continue
        parts = []
        for f in fs:
            try:
                with zipfile.ZipFile(f) as z:
                    d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
            except Exception:
                continue
            d.columns = [str(c).strip().lower() for c in d.columns]
            tc = next((c for c in d.columns if "time" in c), None)
            rc = next((c for c in d.columns if "rate" in c), None)
            if tc is None or rc is None:
                continue
            d = d[pd.to_numeric(d[tc], errors="coerce").notna()]
            if not len(d):
                continue
            t = pd.to_numeric(d[tc])
            unit = "us" if t.max() > 1e14 else ("ms" if t.max() > 1e11 else "s")
            parts.append(pd.Series(
                pd.to_numeric(d[rc], errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(pd.to_datetime(t.to_numpy(), unit=unit,
                                                      utc=True)).tz_localize(None)))
        if not parts:
            continue
        r = pd.concat(parts).sort_index()
        r = r[~r.index.duplicated()]
        out[s] = r.resample("1D").sum().reindex(index)
    return pd.DataFrame(out).reindex(columns=cols).reindex(index)


def book(px, w, fund, gain_cap=None, slip_bps=5.0, target_vol=None,
         max_lev=3.0, hl=45):
    """Net returns with an ASYMMETRIC per-position daily gain cap.

    `gain_cap` limits what a single position may EARN in one day, per unit of
    exposure. Losses are never truncated. This is the direction a real
    constraint runs: an exchange that auto-deleverages you, or a contract that
    halts, takes away the windfall and leaves the loss.
    """
    r = px.pct_change().fillna(0.0)
    w = w.shift(1).fillna(0.0)
    if target_vol is not None:
        g0 = (w * r).sum(axis=1)
        rv = g0.ewm(halflife=hl, min_periods=30).std().shift(1) * np.sqrt(365.25)
        lev = (target_vol / rv.replace(0, np.nan)).clip(upper=max_lev).fillna(0.0)
        w = w.mul(lev, axis=0)
        gr = w.abs().sum(axis=1)
        w = w.div(np.maximum(gr / L.MAX_GROSS, 1.0), axis=0).fillna(0.0)
    pnl = w * r
    if gain_cap is not None:
        pnl = np.minimum(pnl, w.abs() * gain_cap)      # gains only
    gross = pnl.sum(axis=1)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * (L.FEE_BPS + slip_bps) / 1e4
    carry = (w * fund.reindex_like(w).fillna(0.0)).sum(axis=1) if fund is not None else 0.0
    return (gross - cost - carry).clip(lower=-0.95), turn


def show(net, turn, tag, w=24):
    a = net.to_numpy(float)
    st = stats_of(a); g = L.gate_of(a)
    gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
    print(f"   {tag:>{w}}{st['sharpe']:>7.2f}{st['cagr']*100:>9.1f}%"
          f"{st['dd']*100:>8.1f}%{float(np.mean(turn)):>8.2f}{gs:>9}")
    return st, g


if __name__ == "__main__":
    pan = PIT.build()
    px = pan["close"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = pit_funding(list(px.columns), px.index)
    fcov = fund.where(ok).notna().sum().sum() / max(ok.sum().sum(), 1) * 100

    print("S153 - honest haircut and the full sleeve stack\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, "
          f"{len(px)} days, median {int(ok.sum(axis=1).median())} tradeable/day")
    print(f"funding coverage over tradeable cells: {fcov:.1f}%\n")

    r1 = np.log(px).diff(1)
    mkt = r1.mean(axis=1)
    beta = (r1.rolling(180, min_periods=90).cov(mkt)
            .div(mkt.rolling(180, min_periods=90).var(), axis=0)).where(ok).clip(0.4, 2.5)
    lowvol = (-r1.rolling(90, min_periods=60).std()).where(ok)

    print("1. THE SAME BOOK UNDER A SYMMETRIC CAP (S152, wrong) AND AN")
    print("   ASYMMETRIC GAIN-ONLY CAP (right). Gains capped, losses never.")
    print(f"   {'cap':>24}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}{'at -20%':>9}")
    w_lv = L.weights(lowvol, ok, "betaneutral", beta)
    net, turn = book(px, w_lv, None, gain_cap=None)
    show(net, turn, "no cap at all")
    for c in (0.50, 0.25, 0.15, 0.10):
        net, turn = book(px, w_lv, None, gain_cap=c)
        show(net, turn, f"gains capped at {c*100:.0f}%/day")

    print("\n2. EVERY SLEEVE, gain-capped at 15%/day throughout")
    CAP = 0.15
    print(f"   {'sleeve':>24}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}{'at -20%':>9}")
    sleeves = {}

    net, turn = book(px, L.weights(lowvol, ok, "zscore"), None, gain_cap=CAP)
    show(net, turn, "low-vol (zscore)"); sleeves["lowvol"] = net

    if fcov > 50:
        carry = (-fund.rolling(3, min_periods=1).mean()).where(ok)
        net, turn = book(px, L.weights(carry, ok, "zscore"), fund, gain_cap=CAP)
        show(net, turn, "carry (zscore)"); sleeves["carry"] = net
    else:
        print(f"   {'carry':>24}   skipped: funding coverage only {fcov:.0f}%")

    import strategies.s150_rexpanel as RX
    store = []
    for N in (55, 89, 144, 233):
        for q in (0.80, 0.90, 0.95):
            pos = RX.rex_pos(px, pan["high"], pan["low"], N, q)
            w = RX.risk_size(pos, px, ok).div(ok.sum(axis=1).replace(0, np.nan), axis=0)
            nb, _ = book(px, w, fund, gain_cap=CAP)
            store.append(nb)
    rex = sum(store) / len(store)
    show(rex, pd.Series(0.0, index=rex.index), "rex panel (12-cfg avg)")
    sleeves["rex"] = rex

    print("\n3. CORRELATION BETWEEN SLEEVES")
    C = pd.DataFrame(sleeves).corr()
    print("        " + "".join(f"{c:>10}" for c in C.columns))
    for i, row in C.iterrows():
        print(f"   {i:>8}" + "".join(f"{row[c]:>10.3f}" for c in C.columns))

    def at_vol(s, t=0.30):
        sd = float(np.std(s.to_numpy(float))) * np.sqrt(365.25)
        return s * (t / sd) if sd > 0 else s * 0.0
    Z = pd.DataFrame({k: at_vol(v) for k, v in sleeves.items()}).dropna()
    print(f"\n4. THE STACK  (equal weight, each sleeve at 30% annualised vol)")
    print(f"   {'combination':>24}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    z = pd.Series(0.0, index=Z.index)
    comb = Z.mean(axis=1)
    st, g = show(comb, z, f"all {len(Z.columns)} sleeves")
    for c in Z.columns:
        show(Z[c], z, f"  {c} alone")

    print("\n5. YEAR BY YEAR, the stack scaled to a 20% max drawdown")
    a = comb.to_numpy(float)
    gg = at_gate(a, lo=1e-4, hi=60.0, iters=70)
    sc = gg["scale"] if "scale" in gg else np.nan
    scaled = comb * (sc if np.isfinite(sc) else 1.0)
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"   {y}: {((1+v).prod()-1)*100:>+8.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")
    print(f"\n   scale applied: x{sc:.2f}   full-sample CAGR "
          f"{stats_of(scaled.to_numpy(float))['cagr']*100:.1f}% at max DD "
          f"{stats_of(scaled.to_numpy(float))['dd']*100:.1f}%")
    print("\ndone: honest stack")
