"""
S131 - Does quarterly Calmar-ranked selection transfer off V7?

THE CHAIN THAT LEADS HERE
-------------------------
S129 fitted Calmar = 0.84 x Sharpe^1.53 across 332 books, R^2 0.969. Every book
in this study sits 0.70-1.55x of that line. **V7 sits at 3.33x** (2.40x on the
stricter bootstrap gate). That residual is the largest unexplained quantity in
the study and it is worth more than any signal in it - 2.4x on Calmar is 2.4x on
the gate figure.

Three explanations tested and killed:
    shape       skew correlates with the residual at r = 0.005; no shape
                variable explains it (S129)
    stops       per-trade loss truncation lifts the Calmar ratio 1.03x on real
                signals and 1.06x on random ones - nothing, and identical (S130)
    sample      the same books measured on V7's 2022-03+ window instead of
                2021+ have median ratio 0.97x either way

What remains is the machinery S95 already measured and this log never connected
to the Calmar law: V1, five signals on one fixed configuration, is Sharpe 2.15
at 71.2%. V7, the same signals under **quarterly Calmar-ranked top-3 selection**,
is Sharpe 2.13 at 179.0%. Identical Sharpe, 2.5x the gate. Everything that moved,
moved Calmar.

WHY THIS IS A GENERAL MECHANISM AND NOT A V7 INTERNAL
------------------------------------------------------
"Rank your configurations quarterly on trailing Calmar and run the best three"
is an allocation rule. It says nothing about crowding data, 12-hour bars, ATR
stops or trend gates. It has never been applied to any other signal family in
this study - every non-V7 book here is a single fixed configuration, which S95
says is the version that gives up the 2.5x.

So: build a configuration grid for the **crowding family** - not V7, no ATR
stops, no trend gate, no 12h clock, daily and vol-targeted - and run the same
allocation rule over it.

THE CONTROLS, WHICH ARE THE ACTUAL EXPERIMENT
---------------------------------------------
This log has caught selection being worthless three times (S106b weights, S109
configs, S118b signs), so the controls decide it, not the headline:

    RANDOM 3     three configurations drawn at random each quarter. S86b's
                 control: if this matches the ranked version, the gain is
                 diversification and the ranking is decoration.
    EQUAL ALL    every configuration, equally weighted, always. No selection.
    BEST HINDSIGHT  the single configuration with the best full-sample result.
                 Unattainable; an upper bound on what picking one can do.

Ranking uses a trailing window that ends BEFORE the quarter it allocates for.
Nothing else is refitted.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s119_dev as S119
import strategies.s118_crowd as C
import strategies.s130_stops as ST
from strategies.s96_rank import at_gate, stats_of

A, B = 0.84, 1.53
LOOKBACK = 365          # days of trailing history the ranking sees
TOPK = 3


def gate(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


def grid(px, fund, F, cols):
    """A configuration grid for the CROWDING family. Not V7's grid."""
    out = {}
    for zwin in (90, 180, 365):
        sig = S119.signal(F, cols, zwin)
        for thr in (0.0, 0.20, 0.40, 0.60):
            for tv in (0.20, 0.40):
                for hl in (16, 32, 64):
                    nb = ST.book(px, fund, sig, tv, thr, hl, stop=None)
                    a = nb.to_numpy()
                    if np.allclose(a, 0):
                        continue
                    out[f"z{zwin}_t{thr:.1f}_v{tv:.1f}_h{hl}"] = nb
    return pd.DataFrame(out)


def trailing_calmar(a):
    a = np.asarray(a, float)
    if not len(a) or np.allclose(a, 0):
        return -9e9
    e = np.cumprod(1.0 + a)
    if e[-1] <= 0:
        return -9e9
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    yrs = len(a) / 365.25
    cagr = e[-1] ** (1 / yrs) - 1
    return cagr / abs(dd) if dd < -1e-9 else -9e9


def allocate(G, mode, seed=0):
    """Walk quarters forward. The ranking for a quarter uses only days before it."""
    rng = np.random.default_rng(seed)
    idx = G.index
    qs = pd.Series(idx, index=idx).groupby([idx.year, idx.quarter]).min().to_numpy()
    out = pd.Series(0.0, index=idx)
    chosen = []
    for i, qstart in enumerate(qs):
        qend = qs[i + 1] if i + 1 < len(qs) else idx.max() + pd.Timedelta(days=1)
        hist = G.loc[(idx < qstart) & (idx >= qstart - pd.Timedelta(days=LOOKBACK))]
        live = (idx >= qstart) & (idx < qend)
        if len(hist) < 120:
            continue
        if mode == "equal":
            pick = list(G.columns)
        elif mode == "random":
            pick = list(rng.choice(G.columns, size=min(TOPK, len(G.columns)),
                                   replace=False))
        else:                                      # ranked on trailing Calmar
            sc = {c: trailing_calmar(hist[c].to_numpy()) for c in G.columns}
            pick = [c for c, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:TOPK]]
        chosen.append(len(pick))
        out.loc[live] = G.loc[live, pick].mean(axis=1)
    return out


def report(tag, a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    st = stats_of(a); g = gate(a)
    if not np.isfinite(g):
        print(f"   {tag:>28}{st['sharpe']:>8.2f}{'n/a':>10}{'n/a':>10}{'n/a':>9}")
        return np.nan
    cal = g / 20.0
    ratio = cal / (A * max(st["sharpe"], 1e-9) ** B)
    h = len(a) // 2
    g1, g2 = gate(a[:h]), gate(a[h:])
    s1 = "n/a" if not np.isfinite(g1) else f"{g1:.0f}%"
    s2 = "n/a" if not np.isfinite(g2) else f"{g2:.0f}%"
    print(f"   {tag:>28}{st['sharpe']:>8.2f}{cal:>10.2f}{g:>9.1f}%{ratio:>9.2f}x"
          f"{s1:>9}{s2:>9}")
    return g


if __name__ == "__main__":
    px, fund, F = S119.build("1D")
    cols = [c for c in C.SIGNS if c in F.columns]
    G = grid(px, fund, F, cols)
    print("S131 - does quarterly Calmar-ranked selection transfer off V7?\n")
    print(f"crowding family, {G.shape[1]} configurations, "
          f"{G.index.min().date()} -> {G.index.max().date()}")
    print(f"ranked on trailing {LOOKBACK}d Calmar, top {TOPK}, "
          f"reselected each quarter, nothing refitted\n")

    print(f"   {'allocation':>28}{'Sharpe':>8}{'Calmar':>10}{'gate':>9}"
          f"{'vs law':>10}{'1st h':>9}{'2nd h':>9}")

    best_single = max(G.columns, key=lambda c: gate(G[c].to_numpy())
                      if np.isfinite(gate(G[c].to_numpy())) else -9e9)
    report("BEST single (hindsight)", G[best_single].to_numpy())
    report("EQUAL all, no selection", allocate(G, "equal").to_numpy())
    rs = [allocate(G, "random", seed=s) for s in range(8)]
    for i, r in enumerate(rs[:3]):
        report(f"RANDOM {TOPK}, draw {i}", r.to_numpy())
    med = np.median([gate(r.to_numpy()) for r in rs
                     if np.isfinite(gate(r.to_numpy()))])
    print(f"   {'RANDOM ' + str(TOPK) + ', median of 8':>28}{'':>8}{'':>10}"
          f"{med:>9.1f}%")
    print()
    report("RANKED top-3 (causal)", allocate(G, "ranked").to_numpy())

    print(f"\n   reference: V7 sits at 3.33x the law; every other book in this")
    print(f"   study sits 0.70-1.55x. crowding as one fixed config is ~1.0x.")
    print("\ndone: selection transfer")
