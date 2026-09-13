"""
S163 - Three levers on the 178.8%, and which of them is real.

WHERE S162 LEFT IT
-------------------
V7 alone reaches 178.8% at a realised 20% drawdown, Sharpe 2.13. Every equal-
weighted combination with the portfolio sleeves made it WORSE, because equal-
weighting a Sharpe 2.13 book against Sharpe 0.5 books is a way of throwing away
the good one. And the idealised frontier at Sharpe 2.13 is 386%, so the book is
paying a 2.2x tail penalty. That penalty is worth more than any plausible Sharpe
improvement: halving it reaches 290% on its own.

Three levers, each tested as a full sweep with the whole curve printed, because
the peak of a swept parameter is a fitted number and saying so is the only thing
that keeps it honest.

  WEIGHTS       S162 searched subsets at equal weight only. Here the V7 weight
                is swept from 50% to 100% with the remainder split equally
                among the other sleeves. One parameter, and the shape of the
                curve says whether the diversification is worth anything at all.

  VOLATILITY    scale the combined stream by target/realised vol, estimated
                causally on trailing data. This attacks volatility CLUSTERING,
                which is half of what the tail penalty is made of. It cannot
                add return; it can only move return from the drawdown into the
                gate, which is exactly what is wanted.

  DRAWDOWN RAIL cut size as the running drawdown deepens, using the drawdown as
                it stood at the START of the day. S126 found this helps only
                low-Sharpe books, but S126 measured at the honest gate and this
                is the realised gate, so it is re-tested rather than assumed.

Every gate figure refuses to report unless the bisection actually reached -20%.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s162_maxprofit as MP
import strategies.s158_final as F8
from strategies.s96_rank import at_gate, stats_of


def voltarget(r, target=0.30, hl=45, max_lev=3.0, min_periods=60):
    """Scale a return stream to constant volatility, causally."""
    r = pd.Series(np.asarray(r, float), index=r.index)
    rv = r.ewm(halflife=hl, min_periods=min_periods).std().shift(1) * np.sqrt(365.25)
    lev = (target / rv.replace(0, np.nan)).clip(upper=max_lev).fillna(0.0)
    return r * lev


def rail(r, limit, floor=0.0, shape=1.0):
    """Causal drawdown rail: the weight on day t uses the drawdown at the START
    of day t, never including that day's own return."""
    a = np.asarray(r, float)
    out = np.empty(len(a)); eq, peak = 1.0, 1.0
    for i in range(len(a)):
        dd = eq / peak - 1.0
        w = np.clip(1.0 + dd / limit, floor, 1.0) ** shape
        out[i] = w * a[i]
        eq *= (1.0 + out[i]); peak = max(peak, eq)
    return pd.Series(out, index=r.index)


def rep(a, tag, w=34):
    a_ = np.asarray(a, float)
    st = stats_of(a_); rg, sc = MP.realised_gate(a_); hg, _ = F8.honest_gate(a_)
    print(f"   {tag:>{w}}{st['sharpe']:>7.2f}"
          + (f"{rg:>10.1f}%" if np.isfinite(rg) else f"{'n/a':>11}")
          + (f"{hg:>9.1f}%" if np.isfinite(hg) else f"{'n/a':>10}")
          + f"{st['dd']*100:>8.1f}%")
    return rg if np.isfinite(rg) else -1, hg


if __name__ == "__main__":
    S = MP.sleeves()
    R = pd.DataFrame(S).dropna()
    Z = pd.DataFrame({c: F8.at_vol(R[c], MP.TARGET_VOL) for c in R.columns})
    others = [c for c in Z.columns if c != "V7"]
    v7 = Z["V7"]
    print("S163 - three levers on the 178.8%\n")
    print(f"intersection {R.index.min().date()} -> {R.index.max().date()}, "
          f"{len(R)} days ({len(R)/365.25:.1f} years)\n")

    print("1. WEIGHT SWEEP - how much of the book should NOT be V7?")
    print(f"   {'V7 weight':>34}{'Shp':>7}{'realised':>11}{'honest':>10}{'maxDD':>8}")
    best_w = None
    for wv in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00):
        rest = (1 - wv) / max(len(others), 1)
        comb = v7 * wv + Z[others].sum(axis=1) * rest
        rg, hg = rep(comb, f"{wv*100:.0f}% V7 / {(1-wv)*100:.0f}% rest")
        if best_w is None or rg > best_w[0]:
            best_w = (rg, wv, comb)

    base = best_w[2]
    print(f"\n2. VOLATILITY TARGETING on the {best_w[1]*100:.0f}% V7 blend")
    print(f"   {'target / halflife':>34}{'Shp':>7}{'realised':>11}{'honest':>10}{'maxDD':>8}")
    rep(base, "none (baseline)")
    best_v = (best_w[0], None, base)
    for tv in (0.20, 0.30, 0.40, 0.60):
        for hl in (20, 45, 90):
            vt = voltarget(base, tv, hl)
            rg, hg = rep(vt, f"{tv*100:.0f}% vol, halflife {hl}d")
            if rg > best_v[0]:
                best_v = (rg, (tv, hl), vt)

    print(f"\n3. DRAWDOWN RAIL on the best stream so far "
          f"({'vol-targeted' if best_v[1] else 'unscaled'})")
    print(f"   {'limit / floor / shape':>34}{'Shp':>7}{'realised':>11}{'honest':>10}{'maxDD':>8}")
    cur = best_v[2]
    rep(cur, "no rail (baseline)")
    best_r = (best_v[0], None, cur)
    for lim in (0.10, 0.15, 0.20, 0.30):
        for fl in (0.0, 0.25, 0.50):
            for sh in (1.0, 2.0):
                rr = rail(cur, lim, fl, sh)
                rg, hg = rep(rr, f"{lim*100:.0f}% / {fl:.2f} / {sh:.0f}")
                if rg > best_r[0]:
                    best_r = (rg, (lim, fl, sh), rr)

    rg, params, final = best_r
    a = final.to_numpy(float)
    st = stats_of(a); _, sc = MP.realised_gate(a); hg, _ = F8.honest_gate(a)
    scaled = final * sc
    s = stats_of(scaled.to_numpy(float))
    print(f"\n{'='*76}\nBEST STREAM")
    print(f"{'='*76}")
    print(f"   V7 weight {best_w[1]*100:.0f}%, "
          f"vol target {best_v[1] if best_v[1] else 'none'}, "
          f"rail {params if params else 'none'}")
    print(f"   Sharpe {st['sharpe']:.2f}   realised gate {rg:.1f}%   "
          f"honest gate {hg:.1f}%")
    print(f"\n   YEAR BY YEAR at the realised 20% scale (x{sc:.2f})")
    for y, v in scaled.groupby(scaled.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
              f"{dd*100:>6.1f}%   ({len(v)} days)")
    print(f"   full sample: CAGR {s['cagr']*100:.1f}%, max DD {s['dd']*100:.1f}%")
    idl = 2 * 0.20 * st["sharpe"] / (np.log(2) + 0.20)
    g = st["sharpe"] * idl - idl * idl / 2
    print(f"\n   idealised frontier at Sharpe {st['sharpe']:.2f}: {(np.exp(g)-1)*100:.0f}%")
    print(f"   tail penalty now {(np.exp(g)-1)/max(s['cagr'],1e-9):.1f}x "
          f"(was 2.2x)")
    hit = s["cagr"] * 100 >= 300
    print(f"   target 300%: "
          + ("REACHED" if hit else f"{300/max(s['cagr']*100, 1e-9):.2f}x short"))
    print("\ndone: levers")
