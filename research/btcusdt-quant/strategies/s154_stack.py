"""
S154 - Targeted delisting haircut, causal risk limits, and the sleeve stack.

TWO WRONG HAIRCUTS, AND WHY
----------------------------
S152 charged for uncollectable gains with a SYMMETRIC daily return cap. The book
improved as the cap tightened, which was the tell: a symmetric cap refunds losses
as well as confiscating gains, so it is a hindsight stop-loss, not a haircut.

S153 corrected it to an ASYMMETRIC gain-only cap and the book collapsed to Sharpe
-1.20. That is also wrong, in the opposite direction. A blanket gain cap
confiscates every large favourable move in every name on every day, while leaving
every adverse move intact. Applied to a roughly symmetric return distribution it
manufactures a large negative drift by construction. It does not measure a real
constraint; it measures the asymmetry of the test.

THE CONSTRAINT THAT IS ACTUALLY REAL
-------------------------------------
You cannot reliably collect a move in a contract that is being HALTED, SETTLED or
AUTO-DELEVERAGED. That is narrow and datable: it happens in the final window of a
contract's life, and this universe knows exactly which contracts died and when.
So the haircut is applied there and nowhere else:

    in the last `DEAD_WIN` days of any contract that stops trading, a position's
    GAIN is scaled by (1 - haircut). Losses are untouched. Swept from 0 to 1,
    where 1 means every gain in every dying contract's final window is assumed
    uncollectable - which is more punitive than reality, since Binance settles
    delisted perps at a published mark and most positions do get out.

FAT TAILS ARE A RISK PROBLEM, NOT A HAIRCUT PROBLEM
-----------------------------------------------------
The reason the symmetric cap flattered the book so much is that the uncapped book
has genuinely dangerous tails: one name moving 80% against a concentrated leg.
The legitimate answer to that is a POSITION LIMIT, which is causal, is what any
real book runs, and costs nothing in hindsight:

    no single name may carry more than MAX_W of gross exposure.

That is applied throughout, and the limit is swept rather than chosen.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s151_pit as PIT
import strategies.s148_lowvol as L
import strategies.s153_honest as H
import strategies.s150_rexpanel as RX
from strategies.s96_rank import at_gate, stats_of

DEAD_WIN = 10          # days before a contract's last bar that count as its death


def death_window(px, win=DEAD_WIN):
    """Boolean panel: True in the final `win` days of a contract that dies."""
    end = px.index.max()
    last = px.apply(lambda s: s.last_valid_index())
    M = pd.DataFrame(False, index=px.index, columns=px.columns)
    for s, t in last.items():
        if pd.isna(t) or t >= end - pd.Timedelta(days=30):
            continue
        M.loc[(M.index > t - pd.Timedelta(days=win)) & (M.index <= t), s] = True
    return M


def cap_weights(w, max_w):
    """Per-name gross limit, renormalised so total exposure is preserved."""
    if max_w is None:
        return w
    g0 = w.abs().sum(axis=1)
    c = w.clip(-max_w, max_w)
    g1 = c.abs().sum(axis=1).replace(0, np.nan)
    return c.mul((g0 / g1).clip(upper=3.0), axis=0).fillna(0.0)


def book(px, w, fund, dead=None, haircut=0.0, max_w=None, slip_bps=5.0,
         target_vol=None, hl=45):
    r = px.pct_change().fillna(0.0)
    w = cap_weights(w, max_w).shift(1).fillna(0.0)
    if target_vol is not None:
        g0 = (w * r).sum(axis=1)
        rv = g0.ewm(halflife=hl, min_periods=30).std().shift(1) * np.sqrt(365.25)
        lev = (target_vol / rv.replace(0, np.nan)).clip(upper=3.0).fillna(0.0)
        w = w.mul(lev, axis=0)
        gr = w.abs().sum(axis=1)
        w = w.div(np.maximum(gr / L.MAX_GROSS, 1.0), axis=0).fillna(0.0)
    pnl = w * r
    if dead is not None and haircut > 0:
        d = dead.reindex_like(pnl).fillna(False).to_numpy()
        v = pnl.to_numpy().copy()
        hit = d & (v > 0)
        v[hit] *= (1.0 - haircut)
        pnl = pd.DataFrame(v, index=pnl.index, columns=pnl.columns)
    gross = pnl.sum(axis=1)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * (L.FEE_BPS + slip_bps) / 1e4
    carry = (w * fund.reindex_like(w).fillna(0.0)).sum(axis=1) if fund is not None else 0.0
    return (gross - cost - carry).clip(lower=-0.95), turn


if __name__ == "__main__":
    pan = PIT.build()
    px = pan["close"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = H.pit_funding(list(px.columns), px.index)
    dead = death_window(px)

    print("S154 - targeted haircut, causal position limits, sleeve stack\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, "
          f"{len(px)} days, median {int(ok.sum(axis=1).median())} tradeable/day")
    print(f"contracts that die: {int((dead.any()).sum())}, "
          f"{int(dead.to_numpy().sum())} asset-days inside a death window")
    print(f"funding coverage: "
          f"{fund.where(ok).notna().sum().sum()/max(ok.sum().sum(),1)*100:.1f}%\n")

    r1 = np.log(px).diff(1)
    lowvol = (-r1.rolling(90, min_periods=60).std()).where(ok)
    carry = (-fund.rolling(3, min_periods=1).mean()).where(ok)

    print("1. POSITION LIMIT SWEEP - the causal answer to fat tails")
    print(f"   {'max gross per name':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>8}{'at -20%':>9}")
    w0 = L.weights(lowvol, ok, "zscore")
    for mw in (None, 0.10, 0.05, 0.03, 0.02):
        net, turn = book(px, w0, None, max_w=mw)
        H.show(net, turn, "none" if mw is None else f"{mw*100:.0f}%", w=26)

    MAXW = 0.05
    print(f"\n2. DELISTING HAIRCUT SWEEP, position limit fixed at {MAXW*100:.0f}%")
    print("   haircut 1.00 = every gain in every dying contract's last 10 days "
          "is assumed\n   uncollectable. That is harsher than reality.")
    print(f"   {'haircut on dying names':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>8}{'at -20%':>9}")
    for hc in (0.0, 0.5, 1.0):
        net, turn = book(px, w0, None, dead=dead, haircut=hc, max_w=MAXW)
        H.show(net, turn, f"{hc:.2f}", w=26)

    print(f"\n3. EVERY SLEEVE, position limit {MAXW*100:.0f}%, full 1.00 haircut "
          f"on dying names")
    print(f"   {'sleeve':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    sleeves = {}
    for tag, sig, f_ in (("low-vol", lowvol, None), ("carry", carry, fund)):
        for mode in ("zscore", "betaneutral"):
            mkt = r1.mean(axis=1)
            beta = (r1.rolling(180, min_periods=90).cov(mkt)
                    .div(mkt.rolling(180, min_periods=90).var(), axis=0)
                    ).where(ok).clip(0.4, 2.5)
            w = L.weights(sig, ok, mode, beta)
            net, turn = book(px, w, f_, dead=dead, haircut=1.0, max_w=MAXW)
            H.show(net, turn, f"{tag} ({mode})", w=26)
            if mode == "zscore":
                sleeves[tag] = net

    store = []
    for N in (55, 89, 144, 233):
        for q in (0.80, 0.90, 0.95):
            pos = RX.rex_pos(px, pan["high"], pan["low"], N, q)
            w = RX.risk_size(pos, px, ok).div(ok.sum(axis=1).replace(0, np.nan), axis=0)
            nb, _ = book(px, w, fund, dead=dead, haircut=1.0, max_w=MAXW)
            store.append(nb)
    rex = sum(store) / len(store)
    H.show(rex, pd.Series(0.0, index=rex.index), "rex panel (12-cfg avg)", w=26)
    sleeves["rex"] = rex

    print("\n4. CORRELATION")
    C = pd.DataFrame(sleeves).corr()
    print("          " + "".join(f"{c:>10}" for c in C.columns))
    for i, row in C.iterrows():
        print(f"   {i:>8}" + "".join(f"{row[c]:>10.3f}" for c in C.columns))

    def at_vol(s, t=0.30):
        sd = float(np.std(s.to_numpy(float))) * np.sqrt(365.25)
        return s * (t / sd) if sd > 0 else s * 0.0
    Z = pd.DataFrame({k: at_vol(v) for k, v in sleeves.items()}).dropna()
    z = pd.Series(0.0, index=Z.index)
    print(f"\n5. THE STACK, equal weight")
    print(f"   {'combination':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    comb = Z.mean(axis=1)
    H.show(comb, z, f"all {len(Z.columns)} sleeves", w=26)
    for c in Z.columns:
        H.show(Z[c], z, f"  {c} alone", w=26)

    a = comb.to_numpy(float)
    g = at_gate(a, lo=1e-4, hi=60.0, iters=70)
    if np.isfinite(g["dd"]) and abs(g["dd"] + 0.20) < 0.01:
        sc = g["scale"]
        scaled = comb * sc
        print(f"\n6. YEAR BY YEAR at the 20%-drawdown scale (x{sc:.2f})")
        for y, v in scaled.groupby(scaled.index.year):
            eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
            print(f"   {y}: {((1+v).prod()-1)*100:>+8.1f}%   in-year DD "
                  f"{dd*100:>6.1f}%   ({len(v)} days)")
        s = stats_of(scaled.to_numpy(float))
        print(f"   full sample: CAGR {s['cagr']*100:.1f}%, max DD "
              f"{s['dd']*100:.1f}%, Sharpe {s['sharpe']:.2f}")
    print("\ndone: stack")
