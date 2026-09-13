"""
S169 - The best book that is genuinely DIFFERENT from V7, not just without it.

THE PROBLEM WITH S168's ANSWER
-------------------------------
S168 found the best V7-free book: 70% btc_rex_crowd plus a spread of four others,
54.4% at a 20% drawdown over 6.1 years. But its correlation to V7 is +0.377, the
highest pair in the whole study, and the reason is structural rather than
accidental: btc_rex_crowd is half `crowd365`, which reads the SAME crowd
positioning data V7 reads. Deploying it beside a live V7 buys more of a bet that
is already on, not a second bet.

So this file asks the question that actually matters when one strategy is already
running: what is the best book whose returns barely move with V7's?

Every candidate is measured on two axes at once - what it earns standalone, and
how much it moves with V7 - and the crowding component is split out so the
breakout can be judged without the part that duplicates V7.

WHY BOTH AXES, RATHER THAN JUST RETURN
---------------------------------------
Two strategies at correlation 0 combine into something with materially smaller
drawdowns than either. Two at correlation 0.4 barely do. Since the gate converts
drawdown into permitted size linearly, a weaker but independent book can be worth
more to the whole account than a stronger correlated one - and that is measured
here rather than argued.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, itertools, json

import strategies.s168_standalone as SA
import strategies.s162_maxprofit as MP
import strategies.s164_multicrowd as MC
import strategies.s158_final as F8
import strategies.s118_crowd as C
import strategies.s135_more as S135
import strategies.s144_blend as S144
import strategies.s146_maxsharpe as S146
from strategies.s96_rank import at_gate, stats_of

MAXRHO = 0.20


def split_btc_book():
    """rex144 and crowd365 as SEPARATE streams, so the breakout can be used
    without the crowding half that duplicates V7."""
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean(); rng = hi - lo
    lr = np.log(px).diff()
    SL = S146.all_sleeves(px, hi, lo, rng, atr, lr)
    out = {}
    for name in ("rex55", "rex89", "rex144", "crowd365", "trend", "expos"):
        if name not in SL:
            continue
        s = pd.Series(np.asarray(SL[name], float), index=px.index).fillna(0.0)
        sd = float(s.std())
        if sd <= 0:
            continue
        net, pos = S144.book(px, fd, s / sd)[:2]
        n = pd.Series(np.asarray(net, float), index=pd.to_datetime(px.index))
        n.index = n.index.tz_localize(None) if n.index.tz else n.index
        out[name] = n.groupby(n.index.normalize()).sum()
    # the rex family averaged across lookbacks - S146's one non-diluting blend
    rexes = [v for k, v in out.items() if k.startswith("rex")]
    if rexes:
        out["rex_avg"] = sum(rexes) / len(rexes)
    return out


if __name__ == "__main__":
    print("S169 - the best book genuinely different from V7\n")
    S = SA.long_pool()
    S.pop("btc_rex_crowd", None)            # rebuilt split, below
    try:
        S.update(split_btc_book())
    except Exception as e:
        print("   (split unavailable:", e, ")")
    SV = MP.sleeves()
    v7 = SV["V7"]

    rows = []
    print(f"{'strategy':>14}{'from':>12}{'days':>7}{'Shp':>7}{'at -20%':>10}"
          f"{'rho to V7':>11}")
    for k, v in sorted(S.items()):
        a = v.to_numpy(float)
        if len(a) < 400 or np.allclose(a, 0):
            continue
        st = stats_of(a); g, _ = MP.realised_gate(a)
        J = pd.DataFrame({"x": v, "v": v7}).dropna()
        rho = float(J["x"].corr(J["v"])) if len(J) > 200 else np.nan
        rows.append((k, st["sharpe"], g, rho, v))
        print(f"{k:>14}{str(v.index.min().date()):>12}{len(v):>7}{st['sharpe']:>7.2f}"
              + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}")
              + (f"{rho:>+11.3f}" if np.isfinite(rho) else f"{'n/a':>11}"))

    keep = [r for r in rows if np.isfinite(r[3]) and abs(r[3]) <= MAXRHO
            and r[1] > 0.2]
    print(f"\ncandidates with |correlation to V7| <= {MAXRHO:.2f} and a positive "
          f"Sharpe: {len(keep)}")
    for k, sh, g, rho, _ in sorted(keep, key=lambda r: -r[1]):
        print(f"   {k:>14}  Sharpe {sh:>5.2f}   gate "
              + (f"{g:>6.1f}%" if np.isfinite(g) else "  n/a ")
              + f"   rho {rho:+.3f}")
    if not keep:
        print("   none."); sys.exit(0)

    R = pd.DataFrame({k: v for k, _, _, _, v in keep}).dropna()
    Z = pd.DataFrame({c: F8.at_vol(R[c], MP.TARGET_VOL) for c in R.columns})
    print(f"\nintersection {R.index.min().date()} -> {R.index.max().date()}, "
          f"{len(R)} days ({len(R)/365.25:.1f} years)")
    Cm = R.corr()
    print("\ncorrelation among them:")
    print("              " + "".join(f"{c[:10]:>12}" for c in Cm.columns))
    for i, row in Cm.iterrows():
        print(f"   {i:>10}" + "".join(f"{row[c]:>12.3f}" for c in Cm.columns))

    print(f"\nCOMBINATIONS (equal weight within each subset)")
    print(f"   {'subset':>44}{'Shp':>7}{'realised':>11}{'honest':>10}{'rho V7':>9}")
    best = None
    cols = list(Z.columns)
    for r in range(1, len(cols) + 1):
        for cb in itertools.combinations(cols, r):
            comb = Z[list(cb)].mean(axis=1)
            a = comb.to_numpy(float)
            rg, sc = MP.realised_gate(a)
            if not np.isfinite(rg):
                continue
            J = pd.DataFrame({"x": comb, "v": v7}).dropna()
            rho = float(J["x"].corr(J["v"]))
            best = best or (rg, cb, comb, sc, rho)
            if rg > best[0]:
                best = (rg, cb, comb, sc, rho)
    # rank on the cheap metric first, then pay for the bootstrap on the top 12
    # only - 1,023 subsets x 1,500 resampled paths each is the bottleneck and
    # the ranking does not need it.
    rough = []
    for r in range(1, len(cols) + 1):
        for cb in itertools.combinations(cols, r):
            comb = Z[list(cb)].mean(axis=1)
            a = comb.to_numpy(float)
            rg, sc = MP.realised_gate(a)
            if not np.isfinite(rg):
                continue
            rough.append((rg, cb, comb, sc))
    rough.sort(key=lambda x: -x[0])
    scored = []
    for rg, cb, comb, sc in rough[:12]:
        a = comb.to_numpy(float)
        hg, _ = F8.honest_gate(a)
        J = pd.DataFrame({"x": comb, "v": v7}).dropna()
        scored.append((rg, hg, stats_of(a)["sharpe"], cb,
                       float(J["x"].corr(J["v"])), comb, sc))
    scored.sort(key=lambda x: -x[0])
    for rg, hg, sh, cb, rho, _, _ in scored[:10]:
        print(f"   {'+'.join(cb):>44}{sh:>7.2f}{rg:>10.1f}%"
              + (f"{hg:>9.1f}%" if np.isfinite(hg) else f"{'n/a':>10}")
              + f"{rho:>+9.3f}")

    rg, hg, sh, cb, rho, comb, sc = scored[0]
    scaled = comb * sc
    s = stats_of(scaled.to_numpy(float))
    print(f"\n{'='*76}\nBEST LOW-CORRELATION BOOK: {' + '.join(cb)}")
    print(f"{'='*76}")
    print(f"   Sharpe {sh:.2f}   realised gate {rg:.1f}%   honest gate {hg:.1f}%"
          f"   correlation to V7 {rho:+.3f}")
    print(f"\n   YEAR BY YEAR at the 20% scale (x{sc:.2f})")
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")
    print(f"\n   FRONTIER")
    for d, cg in SA.frontier(comb.to_numpy(float)):
        print(f"      {d*100:>3.0f}% max DD  ->  {cg:>7.1f}% CAGR")

    print(f"\n{'='*76}\nWHAT IT DOES FOR THE WHOLE ACCOUNT (V7 untouched)")
    print(f"{'='*76}")
    J = pd.DataFrame({"new": comb, "V7": v7}).dropna()
    Zj = pd.DataFrame({c: F8.at_vol(J[c], MP.TARGET_VOL) for c in J.columns})
    print(f"   overlap {J.index.min().date()} -> {J.index.max().date()}, {len(J)} days")
    print(f"   {'split':>30}{'Shp':>7}{'realised':>11}{'honest':>10}{'maxDD':>8}")
    for tag, w in (("100% V7 (today)", 1.0), ("90% V7 / 10% new", 0.9),
                   ("80% V7 / 20% new", 0.8), ("70% V7 / 30% new", 0.7),
                   ("50/50", 0.5), ("100% new", 0.0)):
        ser = Zj["V7"] * w + Zj["new"] * (1 - w)
        a = ser.to_numpy(float); st = stats_of(a)
        r_, _ = MP.realised_gate(a); h_, _ = F8.honest_gate(a)
        print(f"   {tag:>30}{st['sharpe']:>7.2f}"
              + (f"{r_:>10.1f}%" if np.isfinite(r_) else f"{'n/a':>11}")
              + (f"{h_:>9.1f}%" if np.isfinite(h_) else f"{'n/a':>10}")
              + f"{st['dd']*100:>8.1f}%")
    json.dump({"subset": list(cb), "realised": float(rg), "honest": float(hg),
               "sharpe": float(sh), "rho_v7": float(rho),
               "frontier": SA.frontier(comb.to_numpy(float))},
              open("/home/user/quant/results/s169_uncorr.json", "w"), indent=1)
    print("\ndone: uncorrelated book")
