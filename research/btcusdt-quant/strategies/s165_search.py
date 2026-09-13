"""
S165 - Subsets x weights, and what the search itself costs.

THE GAP
-------
S162 searched every SUBSET at equal weight and found V7 alone best at 178.8%.
S163 swept the V7 WEIGHT against all four other sleeves and reached 217.0% at
70/30. Neither searched both, and they disagree about diversification precisely
because of that: equal weight makes extra sleeves harmful, the right weight makes
them helpful. So this searches subsets and weights together.

S164 added eight alt crowding books and they came in at Sharpe 0.41, 0.52, 0.85,
0.54, 0.27, 0.03, 0.12 and -0.31. Only SOL is worth anything. Including all of
them dropped the book to 194.1%, so the pool here keeps the four that clear
Sharpe 0.5 and drops the rest rather than carrying dead weight into the search.

THE PART THAT MATTERS MORE THAN THE NUMBER
-------------------------------------------
Searching 500+ combinations over 4.5 years and reporting the best one is
optimisation, and the winner is a fitted number. Saying so is not enough; it has
to be measured. So the same search is run under a split:

    SELECT on 2022-03 -> 2024-05  (first half)
    MEASURE on 2024-06 -> 2026-08 (second half, never seen by the selector)

The gap between the in-sample winner and what that winner then does out of
sample is the cost of the search, and it is the honest discount on every figure
in S162, S163 and this file. If the out-of-sample number collapses, the 217%
was a story about 4.5 years of one asset, not a strategy.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, itertools, json

import strategies.s162_maxprofit as MP
import strategies.s164_multicrowd as MC
import strategies.s158_final as F8
import strategies.s118_crowd as C
from strategies.s96_rank import at_gate, stats_of

WEIGHTS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def pool():
    S = MP.sleeves()
    for sym in ("SOLUSDT", "BNBUSDT", "XRPUSDT", "ETHUSDT"):
        try:
            px, fd, F = MC.alt_panel(sym)
            if px is None or len(px) < 400:
                continue
            cols = [c for c in C.SIGNS if c in F.columns and F[c].notna().sum() > 300]
            st = F[cols].dropna(thresh=max(len(cols) - 1, 1)).index.min()
            px, fd, F = px[px.index >= st], fd[fd.index >= st], F[F.index >= st]
            res = C.run(px, fd, C.zsig(F, cols, 365), 0.30)
            n = res[0] if isinstance(res, tuple) else res
            n = pd.Series(np.asarray(n, float), index=pd.to_datetime(px.index))
            n.index = n.index.tz_localize(None) if n.index.tz else n.index
            S[sym.replace("USDT", "") + "_crowd"] = n.groupby(n.index.normalize()).sum()
        except Exception as e:
            print(f"   ({sym} unavailable: {e})")
    return S


def search(Z, cols, v7="V7"):
    """Every subset containing V7, at every V7 weight. Returns ranked results."""
    others = [c for c in cols if c != v7]
    out = []
    for r in range(0, len(others) + 1):
        for combo in itertools.combinations(others, r):
            for wv in (WEIGHTS if combo else (1.0,)):
                rest = (1 - wv) / len(combo) if combo else 0.0
                comb = Z[v7] * wv + (Z[list(combo)].sum(axis=1) * rest
                                     if combo else 0.0)
                a = comb.to_numpy(float)
                rg, sc = MP.realised_gate(a)
                if not np.isfinite(rg):
                    continue
                out.append((rg, stats_of(a)["sharpe"], wv, combo, comb, sc))
    out.sort(key=lambda x: -x[0])
    return out


if __name__ == "__main__":
    S = pool()
    R = pd.DataFrame(S).dropna()
    Z = pd.DataFrame({c: F8.at_vol(R[c], MP.TARGET_VOL) for c in R.columns})
    cols = list(Z.columns)
    print("S165 - subsets x weights, and the cost of the search\n")
    print(f"pool: {len(cols)} sleeves, {R.index.min().date()} -> "
          f"{R.index.max().date()}, {len(R)} days ({len(R)/365.25:.1f} years)")
    print(f"   {', '.join(cols)}\n")

    res = search(Z, cols)
    print(f"searched {len(res)} valid subset-weight combinations\n")
    print(f"   {'#':>3}{'V7 wt':>7}{'Shp':>7}{'realised':>11}{'honest':>10}"
          f"   sleeves alongside V7")
    for i, (rg, sh, wv, combo, comb, sc) in enumerate(res[:12], 1):
        hg, _ = F8.honest_gate(comb.to_numpy(float))
        print(f"   {i:>3}{wv*100:>6.0f}%{sh:>7.2f}{rg:>10.1f}%"
              + (f"{hg:>9.1f}%" if np.isfinite(hg) else f"{'n/a':>10}")
              + f"   {'+'.join(combo) if combo else '(none)'}")

    best = res[0]
    rg, sh, wv, combo, comb, sc = best
    scaled = comb * sc
    s = stats_of(scaled.to_numpy(float))
    print(f"\n{'='*76}\nIN-SAMPLE BEST: {wv*100:.0f}% V7 + "
          f"{'+'.join(combo) if combo else 'nothing'}")
    print(f"{'='*76}")
    print(f"   Sharpe {sh:.2f}   CAGR {s['cagr']*100:.1f}%   maxDD {s['dd']*100:.1f}%")
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")

    # ---------------- what the search costs ----------------------------
    h = len(Z) // 2
    Z1, Z2 = Z.iloc[:h], Z.iloc[h:]
    print(f"\n{'='*76}\nWHAT THE SEARCH COSTS")
    print(f"{'='*76}")
    print(f"   select on {Z1.index.min().date()} -> {Z1.index.max().date()} "
          f"({len(Z1)} days)")
    print(f"   measure on {Z2.index.min().date()} -> {Z2.index.max().date()} "
          f"({len(Z2)} days), never seen by the selector\n")
    r1 = search(Z1, cols)
    if not r1:
        print("   no valid combination in the first half."); sys.exit(0)
    _, _, wv1, combo1, _, _ = r1[0]
    rest = (1 - wv1) / len(combo1) if combo1 else 0.0
    oos = Z2["V7"] * wv1 + (Z2[list(combo1)].sum(axis=1) * rest if combo1 else 0.0)
    rg_oos, _ = MP.realised_gate(oos.to_numpy(float))
    rg_is, _ = MP.realised_gate(
        (Z1["V7"] * wv1 + (Z1[list(combo1)].sum(axis=1) * rest if combo1 else 0.0)
         ).to_numpy(float))
    # what the FULL-SAMPLE winner does out of sample, for comparison
    rest_f = (1 - wv) / len(combo) if combo else 0.0
    oos_f = Z2["V7"] * wv + (Z2[list(combo)].sum(axis=1) * rest_f if combo else 0.0)
    rg_oosf, _ = MP.realised_gate(oos_f.to_numpy(float))
    print(f"   first-half winner: {wv1*100:.0f}% V7 + "
          f"{'+'.join(combo1) if combo1 else 'nothing'}")
    print(f"      in sample  (first half)  {rg_is:>8.1f}%")
    print(f"      OUT OF SAMPLE (2nd half) "
          + (f"{rg_oos:>8.1f}%" if np.isfinite(rg_oos) else f"{'n/a':>9}"))
    if np.isfinite(rg_oos) and rg_is > 0:
        print(f"      retention {rg_oos/rg_is*100:.0f}% of the in-sample figure")
    print(f"\n   full-sample winner measured on the 2nd half alone: "
          + (f"{rg_oosf:.1f}%" if np.isfinite(rg_oosf) else "n/a"))
    print(f"   full-sample winner in-sample headline: {rg:.1f}%")
    if np.isfinite(rg_oosf):
        print(f"   -> the honest discount on the headline is "
              f"{rg/max(rg_oosf,1e-9):.2f}x")
    print(f"\n   target 300%: "
          + ("REACHED in sample" if s["cagr"] * 100 >= 300
             else f"{300/max(s['cagr']*100,1e-9):.2f}x short in sample"))
    json.dump({"wv": wv, "combo": list(combo), "gate": float(rg),
               "sharpe": float(sh), "oos": float(rg_oosf) if np.isfinite(rg_oosf) else None},
              open("/home/user/quant/results/s165_best.json", "w"), indent=1)
    print("\ndone: subset x weight search")
