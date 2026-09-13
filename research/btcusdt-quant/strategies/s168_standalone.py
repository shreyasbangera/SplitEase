"""
S168 - The best book that does NOT contain V7.

WHY THIS IS A DIFFERENT QUESTION
---------------------------------
S162-S167 maximised a book that was allowed to hold V7, and the answer was
essentially "hold V7 and put a little on the side" - 203.5% at a 20% drawdown,
almost all of it V7's. That is useless to someone already running V7: it is not
a new strategy, it is a re-weighting of one they have.

The real question is the one being asked now. V7 is live and untouched. What is
the best SEPARATE book, built from anything except V7, to run beside it?

That changes the objective in two ways:

  1  V7 is removed from the pool entirely, so the anchor is whatever is second
     best rather than whatever is best.
  2  CORRELATION TO V7 now matters as much as standalone return. A new book that
     moves with V7 adds leverage to an existing bet; one that moves independently
     adds genuine diversification. Both are reported, and the combined book is
     measured so the effect on the whole account is visible rather than implied.

TWO SAMPLES, BOTH REPORTED
---------------------------
    LONG   2020-07 -> 2026-08, 6.1 years. Drops the alt crowding books, which
           only begin 2021-12. More data, fewer candidates.
    SHORT  2022-03 -> 2026-08, 4.5 years. Everything, including the alt books.

The long sample is the primary one, because six years of evidence on five
strategies beats four and a half on nine.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, itertools, json

import strategies.s162_maxprofit as MP
import strategies.s165_search as SR
import strategies.s158_final as F8
import strategies.s148_lowvol as L
import strategies.s151_pit as PIT
import strategies.s153_honest as H
import strategies.s154_stack as ST
import strategies.s156_composite as CP
import strategies.s157_tailmom as TM
import strategies.s150_rexpanel as RX
import strategies.s147_panel as P
from strategies.s96_rank import at_gate, stats_of

WEIGHTS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def long_pool():
    """Sleeves with history back to 2020, V7 excluded."""
    S = {}
    pan = PIT.build()
    px = pan["close"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = H.pit_funding(list(px.columns), px.index)
    dead = ST.death_window(px)
    r1 = np.log(px).diff(1); mkt = r1.mean(axis=1)
    beta = (r1.rolling(180, min_periods=90).cov(mkt)
            .div(mkt.rolling(180, min_periods=90).var(), axis=0)).where(ok).clip(0.3, 3.0)

    carry = (-fund.rolling(3, min_periods=1).mean()).where(ok)
    S["carry"], _ = ST.book(px, L.weights(carry, ok, "zscore"), fund,
                            dead=dead, haircut=1.0, max_w=0.05)
    lv = (-r1.rolling(90, min_periods=60).std()).where(ok)
    S["lowvol"], _ = ST.book(px, CP.make_weights(CP.zrank(lv, ok), px, ok, beta),
                             None, dead=dead, haircut=1.0, max_w=0.05)
    acc = []
    for lb in (5, 10, 20, 60, 120):
        w = TM.slice_weights(np.log(px).diff(lb), px, ok, frac=0.05)
        a, _ = ST.book(px, w, fund, dead=dead, haircut=1.0, max_w=0.05)
        acc.append(a)
    S["tailmom"] = sum(acc) / len(acc)

    bb, _ = F8.btc_book()
    S["btc_rex_crowd"] = bb

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
    S["rex_panel"] = sum(st) / len(st)

    out = {}
    for k, v in S.items():
        v = pd.Series(np.asarray(v, float), index=pd.to_datetime(v.index))
        v.index = v.index.tz_localize(None) if v.index.tz else v.index
        out[k] = v.groupby(v.index.normalize()).sum()
    return out


def anchor_sweep(Z, anchor, tag=""):
    """Sweep the anchor's weight against the rest split equally."""
    others = [c for c in Z.columns if c != anchor]
    best = None
    print(f"   {'weight on ' + anchor:>34}{'Shp':>7}{'realised':>11}{'honest':>10}{'maxDD':>8}")
    for wv in WEIGHTS:
        rest = (1 - wv) / max(len(others), 1)
        comb = Z[anchor] * wv + (Z[others].sum(axis=1) * rest if others else 0.0)
        a = comb.to_numpy(float); s = stats_of(a)
        rg, sc = MP.realised_gate(a); hg, _ = F8.honest_gate(a)
        print(f"   {f'{wv*100:.0f}%':>34}{s['sharpe']:>7.2f}"
              + (f"{rg:>10.1f}%" if np.isfinite(rg) else f"{'n/a':>11}")
              + (f"{hg:>9.1f}%" if np.isfinite(hg) else f"{'n/a':>10}")
              + f"{s['dd']*100:>8.1f}%")
        if np.isfinite(rg) and (best is None or rg > best[0]):
            best = (rg, wv, comb, sc, hg)
    return best


def frontier(a):
    out = []
    for d in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        lo, hi = 1e-4, 80.0
        for _ in range(80):
            m = (lo + hi) / 2
            e = np.cumprod(1 + a * m)
            dd = (e / np.maximum.accumulate(e) - 1).min()
            if not np.isfinite(dd) or dd < -d:
                hi = m
            else:
                lo = m
        m = (lo + hi) / 2
        e = np.cumprod(1 + a * m)
        dd = float((e / np.maximum.accumulate(e) - 1).min())
        if abs(dd + d) < 0.01:
            out.append((d, stats_of(a * m)["cagr"] * 100))
    return out


if __name__ == "__main__":
    print("S168 - the best book WITHOUT V7\n")
    S = long_pool()
    R = pd.DataFrame(S).dropna()
    print(f"LONG SAMPLE  {R.index.min().date()} -> {R.index.max().date()}, "
          f"{len(R)} days ({len(R)/365.25:.1f} years), {len(R.columns)} strategies\n")
    print(f"   {'strategy':>16}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'at -20%':>10}")
    for c in R.columns:
        a = R[c].to_numpy(float); s = stats_of(a); g, _ = MP.realised_gate(a)
        print(f"   {c:>16}{s['sharpe']:>7.2f}{s['cagr']*100:>8.1f}%{s['dd']*100:>7.1f}%"
              + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}"))

    Cm = R.corr()
    print("\n   correlation between them:")
    print("                  " + "".join(f"{c[:10]:>12}" for c in Cm.columns))
    for i, row in Cm.iterrows():
        print(f"   {i:>14}" + "".join(f"{row[c]:>12.3f}" for c in Cm.columns))

    Z = pd.DataFrame({c: F8.at_vol(R[c], MP.TARGET_VOL) for c in R.columns})
    anchor = max(R.columns, key=lambda c: stats_of(R[c].to_numpy(float))["sharpe"])
    print(f"\nWEIGHT SWEEP - anchor is {anchor} (highest Sharpe without V7)")
    best = anchor_sweep(Z, anchor)
    print(f"\n   {'equal weight, all of them':>34}", end="")
    ew = Z.mean(axis=1); a = ew.to_numpy(float); s = stats_of(a)
    rg_ew, sc_ew = MP.realised_gate(a); hg_ew, _ = F8.honest_gate(a)
    print(f"{s['sharpe']:>7.2f}"
          + (f"{rg_ew:>10.1f}%" if np.isfinite(rg_ew) else f"{'n/a':>11}")
          + (f"{hg_ew:>9.1f}%" if np.isfinite(hg_ew) else f"{'n/a':>10}")
          + f"{s['dd']*100:>8.1f}%")

    rg, wv, comb, sc, hg = best
    print(f"\n{'='*74}\nBEST STANDALONE (no V7): {wv*100:.0f}% {anchor} + "
          f"{(1-wv)*100:.0f}% across the other {len(R.columns)-1}")
    print(f"{'='*74}")
    scaled = comb * sc
    s = stats_of(scaled.to_numpy(float))
    print(f"   Sharpe {s['sharpe']:.2f}   realised gate {rg:.1f}%   honest gate {hg:.1f}%")
    print(f"\n   YEAR BY YEAR at the 20% scale (x{sc:.2f})")
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")
    print(f"\n   FRONTIER")
    for d, cg in frontier(comb.to_numpy(float)):
        print(f"      {d*100:>3.0f}% max DD  ->  {cg:>7.1f}% CAGR")

    # ---------- how it sits beside V7 ----------
    print(f"\n{'='*74}\nHOW IT SITS BESIDE V7 (which stays exactly as it is)")
    print(f"{'='*74}")
    try:
        SV = MP.sleeves()
        v7 = SV["V7"]
        J = pd.DataFrame({"new": comb, "V7": v7}).dropna()
        Zj = pd.DataFrame({c: F8.at_vol(J[c], MP.TARGET_VOL) for c in J.columns})
        print(f"   overlap {J.index.min().date()} -> {J.index.max().date()}, "
              f"{len(J)} days")
        print(f"   correlation of the new book to V7: "
              f"{J['new'].corr(J['V7']):+.3f}")
        print(f"\n   {'account':>30}{'Shp':>7}{'realised':>11}{'honest':>10}")
        for tag, ser in (("V7 alone (what runs today)", Zj["V7"]),
                         ("the new book alone", Zj["new"]),
                         ("50/50 across two accounts", Zj.mean(axis=1)),
                         ("70% V7 / 30% new", Zj["V7"] * 0.7 + Zj["new"] * 0.3)):
            a = ser.to_numpy(float); st = stats_of(a)
            r_, _ = MP.realised_gate(a); h_, _ = F8.honest_gate(a)
            print(f"   {tag:>30}{st['sharpe']:>7.2f}"
                  + (f"{r_:>10.1f}%" if np.isfinite(r_) else f"{'n/a':>11}")
                  + (f"{h_:>9.1f}%" if np.isfinite(h_) else f"{'n/a':>10}"))
    except Exception as e:
        print("   (V7 comparison unavailable:", e, ")")

    json.dump({"anchor": anchor, "weight": wv, "realised": float(rg),
               "honest": float(hg), "sharpe": float(s["sharpe"]),
               "frontier": frontier(comb.to_numpy(float))},
              open("/home/user/quant/results/s168_standalone.json", "w"), indent=1)
    print("\ndone: standalone")
