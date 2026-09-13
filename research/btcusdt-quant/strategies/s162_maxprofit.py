"""
S162 - What is the maximum net profit reachable at a 20% max drawdown?

THE QUESTION, POSED PROPERLY
-----------------------------
Not "can it reach 300%" - that has been asked and the arithmetic answered it
three times. This asks the well-posed version: scale every book and every
combination of books until the realised max drawdown is exactly 20%, and report
the largest CAGR that survives. Whatever that number is, it is the answer.

WHAT GOES IN THIS TIME THAT DID NOT BEFORE
-------------------------------------------
V7. It was excluded from S147-S161 because the brief for those files was new
strategies, and it kept being reached for by reflex. But the question now is
the maximum reachable, V7 is the highest-Sharpe object this study has produced
(2.13 against rex144+crowd365's 1.51), and leaving the best component out of a
maximisation is not discipline, it is just a worse answer.

Its caveats travel with it and are applied, not mentioned:

    IN-SAMPLE SELECTION   V7 picks the top 3 of a 200-config grid on trailing
                          12-month Calmar, refit quarterly. The selection is
                          causal but the GRID was built with the full sample
                          in view.
    DRAWDOWN OPTIMISM     its realised -19.99% is one path. The stationary block
                          bootstrap puts P(worse than 20%) at 98% at the 14.4%
                          risk that produced the 179% headline, 34% at 8%.
    SHORT SPAN            2022-03 onward, 4.5 years, against 6.1 for the rest.

So every figure is reported twice - at the realised gate, which is what a
backtest can show, and at the honest gate, which scales to the MEDIAN of 1500
block-bootstrapped paths and is the number that survives contact with a second
sample.

THE LEVER THAT MATTERS MOST
----------------------------
S161 measured the gap between the idealised frontier and reality: a Sharpe 1.51
book should reach 122% at a 20% drawdown under GBM and reaches 42.6%. That 2.9x
penalty is fat tails and volatility clustering, and it is worth more than any
plausible Sharpe improvement. Halving it is worth more than adding a whole
uncorrelated sleeve, so the combinations here are searched on the gate itself
rather than on Sharpe.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, itertools, json

import strategies.s151_pit as PIT
import strategies.s153_honest as H
import strategies.s154_stack as ST
import strategies.s148_lowvol as L
import strategies.s156_composite as CP
import strategies.s157_tailmom as TM
import strategies.s158_final as F8
from strategies.s96_rank import at_gate, stats_of

TARGET_VOL = 0.30


def realised_gate(a, tol=0.01):
    """CAGR when the ACTUAL path is scaled to a 20% max drawdown."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return np.nan, np.nan
    g = at_gate(a, lo=1e-4, hi=80.0, iters=80)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan, np.nan
    return g["cagr"] * 100, g["scale"]


def sleeves():
    """Every return stream this study has produced that is worth combining."""
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

    bb, used = F8.btc_book()
    if bb is not None:
        S["btc_rex_crowd"] = bb

    try:
        import strategies.s110_meta as M
        import strategies.s87_combined as S87
        import strategies.s119_dev as S119
        M.use_clock()
        R = S87.rankings()
        vr, _ = S87.blend(R, 3, 0.144)
        vr = pd.Series(np.asarray(vr, float))
        cpx, cfd, F = S119.build("1D")
        end = cpx.index.max()
        vr.index = pd.date_range(end=end, periods=len(vr), freq="1D", tz=cpx.index.tz)
        vr.index = vr.index.tz_localize(None) if vr.index.tz else vr.index
        S["V7"] = vr.groupby(vr.index.normalize()).sum()
    except Exception as e:
        print("   (V7 unavailable:", e, ")")
    out = {}
    for k, v in S.items():
        v = pd.Series(np.asarray(v, float), index=pd.to_datetime(v.index))
        v.index = v.index.tz_localize(None) if v.index.tz else v.index
        out[k] = v.groupby(v.index.normalize()).sum()
    return out


if __name__ == "__main__":
    S = sleeves()
    print("S162 - maximum net profit at a 20% max drawdown\n")
    print(f"{'sleeve':>16}{'from':>12}{'to':>12}{'days':>7}{'Shp':>7}"
          f"{'realised':>11}{'honest':>10}")
    for k, v in S.items():
        a = v.to_numpy(float)
        st = stats_of(a); rg, _ = realised_gate(a); hg, _ = F8.honest_gate(a)
        print(f"{k:>16}{str(v.index.min().date()):>12}{str(v.index.max().date()):>12}"
              f"{len(v):>7}{st['sharpe']:>7.2f}"
              + (f"{rg:>10.1f}%" if np.isfinite(rg) else f"{'n/a':>11}")
              + (f"{hg:>9.1f}%" if np.isfinite(hg) else f"{'n/a':>10}"))

    R = pd.DataFrame(S).dropna()
    print(f"\nintersection: {R.index.min().date()} -> {R.index.max().date()}, "
          f"{len(R)} days ({len(R)/365.25:.1f} years)")
    C = R.corr()
    print("\ncorrelation:")
    print("                " + "".join(f"{c[:11]:>13}" for c in C.columns))
    for i, row in C.iterrows():
        print(f"{i:>16}" + "".join(f"{row[c]:>13.3f}" for c in C.columns))

    Z = pd.DataFrame({c: F8.at_vol(R[c], TARGET_VOL) for c in R.columns})
    cols = list(Z.columns)
    print(f"\nEVERY SUBSET, equal weight, each sleeve at {TARGET_VOL*100:.0f}% vol")
    print(f"   {'combination':>46}{'Shp':>7}{'realised':>11}{'honest':>10}")
    rows = []
    for r in range(1, len(cols) + 1):
        for combo in itertools.combinations(cols, r):
            comb = Z[list(combo)].mean(axis=1)
            a = comb.to_numpy(float)
            st = stats_of(a); rg, sc = realised_gate(a); hg, _ = F8.honest_gate(a)
            rows.append((rg if np.isfinite(rg) else -1, hg if np.isfinite(hg) else -1,
                         st["sharpe"], combo, comb, sc))
    rows.sort(key=lambda x: -x[0])
    for rg, hg, sh, combo, _, _ in rows[:14]:
        tag = "+".join(c[:9] for c in combo)
        print(f"   {tag:>46}{sh:>7.2f}"
              + (f"{rg:>10.1f}%" if rg > 0 else f"{'n/a':>11}")
              + (f"{hg:>9.1f}%" if hg > 0 else f"{'n/a':>10}"))

    best = rows[0]
    rg, hg, sh, combo, comb, sc = best
    print(f"\n{'='*74}\nBEST AT A REALISED 20% DRAWDOWN: {'+'.join(combo)}")
    print(f"{'='*74}")
    scaled = comb * sc
    s = stats_of(scaled.to_numpy(float))
    print(f"   Sharpe {sh:.2f}   CAGR {s['cagr']*100:.1f}%   max DD {s['dd']*100:.1f}%"
          f"   scale x{sc:.2f}")
    print(f"\n   YEAR BY YEAR")
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")
    print(f"\n   target 300%: {'REACHED' if s['cagr']*100 >= 300 else 'not reached'}"
          f"   ({300/max(s['cagr']*100,1e-9):.1f}x short)")
    print(f"   Kelly ceiling at Sharpe {sh:.2f}: {(np.exp(sh**2/2)-1)*100:.0f}%")
    idl = 2 * 0.20 * sh / (np.log(2) + 0.20)
    g = sh * idl - idl * idl / 2
    print(f"   idealised frontier at this Sharpe: {(np.exp(g)-1)*100:.0f}%  "
          f"-> tail penalty {(np.exp(g)-1)/(s['cagr']):.1f}x")
    json.dump({"best": list(combo), "sharpe": float(sh), "cagr": float(s["cagr"]*100),
               "dd": float(s["dd"]*100), "realised_gate": float(rg),
               "honest_gate": float(hg)},
              open("/home/user/quant/results/s162_best.json", "w"), indent=1)
    print("\ndone: maximum at 20% drawdown")
