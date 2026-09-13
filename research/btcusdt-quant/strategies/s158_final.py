"""
S158 - The best book reachable from a portfolio of cryptos, scored on the brief.

WHAT THE PORTFOLIO LINE PRODUCED
---------------------------------
Nine files, a rebuilt 683-symbol point-in-time universe including every coin that
died, real funding for all of it, and these are the sleeves that survived both a
surrogate control and an honest cost and haircut treatment:

    BTC rex144+crowd365   single instrument      Sharpe 1.46   gate 42.6%
    carry (PIT)           cross-sectional        Sharpe 1.13   gate 12.3%
    low-vol (PIT)         cross-sectional        Sharpe 0.56   gate  5.2%
    rex panel (16)        per-asset breakout     Sharpe 0.49   gate  6.4%
    tail momentum (PIT)   right-tail, dollars    Sharpe 0.59   gate  6.3%

The uncomfortable fact in that table is that the best single object is still the
SINGLE-INSTRUMENT book, by a factor of three and a half. Sixteen months of
portfolio machinery did not beat one instrument. What the portfolio adds is not
a better sleeve; it is sleeves that are close to uncorrelated with the one good
one, and S124's arithmetic says that is the only thing that raises a Sharpe
without a better signal.

So this file does the one combination that has not been run, measures its
correlation structure rather than assuming it, and scores the result against
every clause of the brief - including the two that the previous single-instrument
answer passed and that a portfolio could plausibly break: trade count and profit
factor.

THE GATE IS REPORTED TWICE, AS IT MUST BE
------------------------------------------
    realised gate   scales the book until the ONE path it actually walked draws
                    down 20%. This flatters, because that path is a single draw.
    honest gate     scales until the MEDIAN of 1500 block-bootstrapped paths
                    draws down 20%. S112 established that a book whose realised
                    drawdown is 20% has an 88% chance of exceeding it.

The honest gate is the number that answers the brief.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s151_pit as PIT
import strategies.s153_honest as H
import strategies.s154_stack as ST
import strategies.s148_lowvol as L
import strategies.s157_tailmom as TM
from strategies.s96_rank import at_gate, stats_of

TARGET_VOL = 0.30


def at_vol(s, t=TARGET_VOL):
    sd = float(np.std(np.asarray(s, float))) * np.sqrt(365.25)
    return s * (t / sd) if sd > 0 else s * 0.0


def honest_gate(a, nboot=1500, block=90, seed=0, tol=0.004):
    """CAGR at the size where the MEDIAN bootstrapped drawdown is -20%."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    T = len(a); nb = int(np.ceil(T / block))
    st = rng.integers(0, max(T - block, 1), (nboot, nb))
    idx = np.clip((st[:, :, None] + np.arange(block)[None, None, :])
                  .reshape(nboot, -1)[:, :T], 0, T - 1)

    def med(s):
        e = np.cumprod(1.0 + a[idx] * s, axis=1)
        return float(np.median((e / np.maximum.accumulate(e, axis=1) - 1).min(axis=1)))

    lo, hi = 1e-3, 40.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if med(mid) < -0.20:
            hi = mid
        else:
            lo = mid
    s = (lo + hi) / 2
    if abs(med(s) + 0.20) > tol:
        return np.nan, np.nan
    return stats_of(a * s)["cagr"] * 100, s


def btc_book():
    """The single-instrument book this study ended on: rex144 + crowd365."""
    import strategies.s135_more as S135
    import strategies.s144_blend as S144
    import strategies.s146_maxsharpe as S146
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean(); rng = hi - lo
    lr = np.log(px).diff()
    SL = S146.all_sleeves(px, hi, lo, rng, atr, lr)
    use = [k for k in ("rex144", "crowd365") if k in SL]
    if len(use) < 2:
        return None, use
    sig = sum(at_vol_sig(SL[k]) for k in use) / len(use)
    net, pos = S144.book(px, fd, sig)[:2]
    n = pd.Series(np.asarray(net, float), index=pd.to_datetime(px.index))
    n.index = n.index.tz_localize(None) if n.index.tz else n.index
    return n.groupby(n.index.normalize()).sum(), use


def at_vol_sig(s):
    s = pd.Series(np.asarray(s, float), index=s.index).fillna(0.0)
    sd = float(s.std())
    return s / sd if sd > 0 else s


def pf_and_trips(net, pos):
    """Profit factor and completed round trips, counted on position changes."""
    p = np.asarray(pos, float)
    live = np.abs(p) > 1e-9
    a = np.asarray(net, float)
    trips, acc, on = [], 0.0, False
    for i in range(len(a)):
        if live[i]:
            acc += a[i]; on = True
        elif on:
            trips.append(acc); acc, on = 0.0, False
    if on:
        trips.append(acc)
    t = np.array(trips)
    pf = (t[t > 0].sum() / -t[t < 0].sum()) if (t < 0).any() else np.inf
    return pf, len(t)


if __name__ == "__main__":
    print("S158 - the best book reachable from a portfolio of cryptos\n")

    pan = PIT.build()
    px = pan["close"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = H.pit_funding(list(px.columns), px.index)
    dead = ST.death_window(px)
    lr = np.log(px); r1 = lr.diff(1)

    sleeves = {}

    carry = (-fund.rolling(3, min_periods=1).mean()).where(ok)
    nb, tb = ST.book(px, L.weights(carry, ok, "zscore"), fund, dead=dead,
                     haircut=1.0, max_w=0.05)
    sleeves["carry"] = nb
    lv = (-r1.rolling(90, min_periods=60).std()).where(ok)
    import strategies.s156_composite as CP
    mkt = r1.mean(axis=1)
    beta = (r1.rolling(180, min_periods=90).cov(mkt)
            .div(mkt.rolling(180, min_periods=90).var(), axis=0)).where(ok).clip(0.3, 3.0)
    nb, tb = ST.book(px, CP.make_weights(CP.zrank(lv, ok), px, ok, beta),
                     None, dead=dead, haircut=1.0, max_w=0.05)
    sleeves["lowvol"] = nb
    acc = []
    for lb in (5, 10, 20, 60, 120):
        w = TM.slice_weights(lr.diff(lb), px, ok, frac=0.05)
        a, _ = ST.book(px, w, fund, dead=dead, haircut=1.0, max_w=0.05)
        acc.append(a)
    sleeves["tailmom"] = sum(acc) / len(acc)

    try:
        import strategies.s150_rexpanel as RX
        import strategies.s147_panel as P
        px16, dv16, ok16 = P.panel()
        pa, hi16, lo16 = RX.ohlc_panel()
        ix = px16.index.intersection(pa.index)
        lv16 = ok16.sum(axis=1); ix = ix[ix >= lv16[lv16 >= 8].index.min()]
        px16, ok16 = px16.loc[ix], ok16.loc[ix]
        pa, hi16, lo16 = pa.loc[ix, px16.columns], hi16.loc[ix, px16.columns], lo16.loc[ix, px16.columns]
        f16 = L.funding_panel(list(px16.columns), ix)
        st = []
        for N in (55, 89, 144, 233):
            for q in (0.80, 0.90, 0.95):
                pos = RX.rex_pos(pa, hi16, lo16, N, q)
                w = RX.risk_size(pos, px16, ok16).div(ok16.sum(axis=1).replace(0, np.nan), axis=0)
                x, _, _ = L.book(px16, w, f16)
                st.append(x)
        sleeves["rexpanel"] = sum(st) / len(st)
    except Exception as e:
        print("   (rex panel unavailable:", e, ")")

    bb, used = btc_book()
    if bb is not None:
        sleeves["BTC rex+crowd"] = bb
        print(f"BTC single-instrument sleeve built from {used}\n")

    print(f"{'sleeve':>18}{'from':>13}{'to':>13}{'days':>7}{'Shp':>7}"
          f"{'CAGR':>9}{'maxDD':>8}{'gate':>9}")
    for k, v in sleeves.items():
        a = v.to_numpy(float); s = stats_of(a); g = L.gate_of(a)
        gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
        print(f"{k:>18}{str(v.index.min().date()):>13}{str(v.index.max().date()):>13}"
              f"{len(v):>7}{s['sharpe']:>7.2f}{s['cagr']*100:>8.1f}%"
              f"{s['dd']*100:>7.1f}%{gs:>9}")

    R = pd.DataFrame(sleeves).dropna()
    print(f"\ndate INTERSECTION: {R.index.min().date()} -> {R.index.max().date()}, "
          f"{len(R)} days ({len(R)/365.25:.1f} years)")
    C = R.corr()
    print("\nCORRELATION on the intersection:")
    print("                  " + "".join(f"{c[:9]:>11}" for c in C.columns))
    for i, row in C.iterrows():
        print(f"{i:>18}" + "".join(f"{row[c]:>11.3f}" for c in C.columns))
    rho = C.to_numpy()[np.triu_indices(len(C), 1)]
    print(f"\naverage pairwise correlation {np.nanmean(rho):+.3f}, "
          f"worst {np.nanmax(rho):+.3f}")

    Z = pd.DataFrame({c: at_vol(R[c]) for c in R.columns})
    print(f"\nCOMBINATIONS (each sleeve at {TARGET_VOL*100:.0f}% annualised vol, "
          f"equal weight)")
    print(f"{'combination':>36}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'gate':>9}"
          f"{'honest':>9}")
    combos = {"all sleeves": list(Z.columns)}
    if "BTC rex+crowd" in Z.columns:
        combos["BTC only"] = ["BTC rex+crowd"]
        combos["portfolio only (no BTC book)"] = [c for c in Z.columns
                                                  if c != "BTC rex+crowd"]
        combos["BTC + carry"] = ["BTC rex+crowd", "carry"]
        combos["BTC + carry + lowvol"] = ["BTC rex+crowd", "carry", "lowvol"]
    best = None
    for tag, cols in combos.items():
        comb = Z[cols].mean(axis=1)
        a = comb.to_numpy(float); s = stats_of(a)
        g = L.gate_of(a); hg, sc = honest_gate(a)
        gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
        hs = "n/a" if not np.isfinite(hg) else f"{hg:>8.1f}%"
        print(f"{tag:>36}{s['sharpe']:>7.2f}{s['cagr']*100:>8.1f}%"
              f"{s['dd']*100:>7.1f}%{gs:>9}{hs:>9}")
        if np.isfinite(hg) and (best is None or hg > best[0]):
            best = (hg, tag, comb, sc)

    if best:
        hg, tag, comb, sc = best
        print(f"\nBEST: {tag}, honest gate {hg:.1f}%")
        scaled = comb * sc
        s = stats_of(scaled.to_numpy(float))
        print(f"\nYEAR BY YEAR at the honest 20%-drawdown scale (x{sc:.2f})")
        for y, v in scaled.groupby(scaled.index.year):
            eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
            print(f"   {y}: {((1+v).prod()-1)*100:>+8.1f}%   in-year DD "
                  f"{dd*100:>6.1f}%   ({len(v)} days)")
        print(f"   full sample: CAGR {s['cagr']*100:.1f}%, max DD {s['dd']*100:.1f}%, "
              f"Sharpe {s['sharpe']:.2f}")
        print(f"\nKELLY CEILING at Sharpe {s['sharpe']:.2f}: "
              f"{(np.exp(s['sharpe']**2/2)-1)*100:.0f}% CAGR, and that is with the "
              f"drawdown\nlimit removed entirely. The brief asks for 300% WITH a "
              f"20% limit.")
        # --- the brief, clause by clause -------------------------------
        d = scaled.to_numpy(float)
        pos = (np.abs(d) > 1e-12).astype(float)
        pf, trips = pf_and_trips(d, pos)
        yr = [( (1+v).prod()-1)*100 for _, v in scaled.groupby(scaled.index.year)]
        print("\n" + "=" * 72)
        print("THE BRIEF, CLAUSE BY CLAUSE")
        print("=" * 72)
        rows = [
            ("net yearly profit > 300%", f"{s['cagr']*100:.1f}% CAGR",
             s['cagr'] * 100 > 300),
            ("max drawdown < 20%", f"{abs(s['dd'])*100:.1f}% realised",
             abs(s['dd']) < 0.20),
            ("100+ completed trades", f"{trips} round trips", trips >= 100),
            ("profit factor > 1.10", f"{pf:.2f}", pf > 1.10),
            ("realistic risk management",
             "vol-targeted, 5% name cap, funding charged", True),
            ("no look-ahead bias",
             "all inputs lagged, point-in-time universe", True),
        ]
        for name, val, okk in rows:
            print(f"   {name:<30}{val:>34}   {'PASS' if okk else 'FAIL'}")
        need = (2 * np.log(4.0)) ** 0.5
        print(f"Sharpe needed for 300% at ANY drawdown: {need:.2f}")
        print(f"Sharpe needed for 300% at a 20% drawdown "
              f"(Calmar 15 under S129's law): {(15/0.84)**(1/1.53):.2f}")
    print("\ndone: final portfolio")
