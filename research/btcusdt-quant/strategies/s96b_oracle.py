"""
S96b - How much is the ranking channel worth AT MOST? A deliberate oracle.

THE POINT OF USING LOOK-AHEAD ON PURPOSE
----------------------------------------
S96 tries to denoise the ranking estimator. Before spending more of the loop on
that direction it is worth knowing what the direction can possibly be worth,
and there is an exact answer: rank each quarter's 200 configurations by what
they ACTUALLY GO ON TO DELIVER, and blend the top k.

That is flagrant look-ahead and it is not a strategy. It is an upper bound. No
estimator built from past data can beat a ranking built from the answer, so
whatever the oracle returns at the -20% gate is the ceiling on everything S96
and any successor could achieve by improving the ranking. Reported with that
label attached everywhere it appears, because a look-ahead number loose in a
research log is exactly how a study poisons itself.

WHAT THE ORACLE FORESEES TURNS OUT TO DECIDE EVERYTHING
-------------------------------------------------------
The obvious oracle - foresight of next-quarter Calmar, the quantity the
deployed rule is estimating - builds a book WORSE than the causal control, and
so does foresight of next-quarter CAGR. That is not a bug, it is what those
objectives are over a three-month window:

  * realised quarterly Calmar rewards a lucky denominator. A quarter with a
    freak -0.5% drawdown outranks one with +40% and -8%.
  * realised quarterly CAGR picks the most aggressive configurations, whose
    drawdowns then chain ACROSS quarter boundaries, so the blended path has to
    be scaled down to reach -20% and ends up lower.

Sharpe is the only one of the three that aggregates correctly across
concatenated quarters, and its oracle is the real ceiling. Section 3 prints
what the losing oracles actually selected, so the mechanism is measured rather
than asserted.

The consequence for the causal rule is the interesting one: trailing Calmar
cannot be working by PREDICTING next-quarter Calmar, because perfect prediction
of it loses. It works by SMOOTHING - twelve months of it selects configurations
that are structurally sound rather than momentarily lucky.

THE DIAGNOSTIC THAT GOES WITH IT
--------------------------------
Section 1 measures the cross-sectional rank correlation between a
configuration's trailing Calmar at the selection date and its realised Calmar
over the following quarter - per quarter and pooled - which says how much
signal the estimator has to work with at all.
"""
import sys, os, time
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd
from scipy import stats

from strategies.s87_combined import GRID, sim, blend
from strategies.s96_rank import quarters, at_gate, train_metrics, MIN_TRADES, BAD

CACHE = "/home/user/quant/results/s96b_test.npz"


def test_metrics():
    """Every configuration's REALISED metrics over each quarter it would be held.

    This is the look-ahead the oracle is built from.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    Q = quarters()
    shape = (len(Q), len(GRID))
    out = {k: np.full(shape, np.nan) for k in ("cagr", "dd", "sharpe", "trades")}
    t0 = time.time()
    for qi, (s, e) in enumerate(Q):
        for ci, cfg in enumerate(GRID):
            m = sim(cfg, s, e, 0.10)
            out["cagr"][qi, ci] = m["cagr"]
            out["dd"][qi, ci] = m["max_dd"]
            out["sharpe"][qi, ci] = m["sharpe"]
            out["trades"][qi, ci] = m["trades"]
        print(f"  quarter {qi+1:2d}/{len(Q)}  {time.time()-t0:6.0f}s", flush=True)
    np.savez_compressed(CACHE, **out)
    return out


def calmar_from(cagr, dd, trades, min_trades=MIN_TRADES):
    ok = (trades >= min_trades) & (dd < 0)
    return np.where(ok, cagr / np.maximum(np.abs(dd), 1e-9), BAD)


def ranked_plan(score_fn):
    Q = quarters()
    out = []
    for qi, (s, e) in enumerate(Q):
        sc = score_fn(qi)
        order = np.argsort(-sc, kind="stable")
        out.append((s, e, [tuple(GRID[i]) for i in order]))
    return out


if __name__ == "__main__":
    print(f"{len(GRID)} configurations x {len(quarters())} quarters, "
          f"realised metrics (LOOK-AHEAD, used only as a bound)\n", flush=True)
    T = test_metrics()
    M = train_metrics()
    Q = quarters()

    tr_cal = np.array([calmar_from(M["cagr"][2, qi], M["dd"][2, qi],
                                   M["trades"][2, qi]) for qi in range(len(Q))])
    te_cal = np.array([calmar_from(T["cagr"][qi], T["dd"][qi],
                                   T["trades"][qi]) for qi in range(len(Q))])
    ok = (T["trades"] >= MIN_TRADES) & (T["dd"] < 0)

    # ---------------- 1. how much signal does the estimator actually have?
    print("1. DOES TRAILING CALMAR PREDICT NEXT-QUARTER CALMAR?")
    print("   Cross-sectional rank correlation across 200 configurations.\n")
    rhos = []
    for qi, (s, e) in enumerate(Q):
        a, b = tr_cal[qi], te_cal[qi]
        m = (a > BAD) & (b > BAD)
        if m.sum() < 20:
            continue
        r = stats.spearmanr(a[m], b[m])[0]
        rhos.append(r)
        print(f"   {s} -> {e}   rho {r:+.3f}   on {m.sum():3d} configs")
    rhos = np.array(rhos)
    t = rhos.mean() / (rhos.std(ddof=1) / np.sqrt(len(rhos)))
    print(f"\n   mean rho {rhos.mean():+.3f}   sd {rhos.std(ddof=1):.3f}   "
          f"t {t:+.2f}   positive in {int((rhos>0).sum())}/{len(rhos)} quarters")

    # ---------------- 2. WHAT the oracle foresees decides everything
    print("\n2. THE BOUND. Same look-ahead, different quantity foreseen.")
    print("   Every row blends top-k at risk/k and is put on the -20% gate by")
    print("   scaling daily returns, control measured the same way.\n")
    ctl = ranked_plan(lambda qi: tr_cal[qi])
    oracles = {
        "realised CAGR":   np.where(ok, T["cagr"], BAD),
        "realised Calmar": te_cal,
        "realised Sharpe": np.where(ok, T["sharpe"], BAD),
    }
    print(f"{'ranked by':>28}{'k':>4}{'CAGR':>9}{'DD':>8}{'Shp':>7}{'Clm':>7}{'vs V7':>10}")
    r, _ = blend(ctl, 3, 0.08)
    g = at_gate(r.to_numpy(float)); b0 = g["cagr"]
    print(f"{'V7 trailing Calmar (causal)':>28}{3:>4}{g['cagr']*100:8.1f}%"
          f"{g['dd']*100:7.1f}%{g['sharpe']:7.2f}{g['calmar']:7.2f}{0.0:+9.1f}pt",
          flush=True)
    for tag, sc in oracles.items():
        for k in (3, 10):
            r, _ = blend(ranked_plan(lambda qi, sc=sc: sc[qi]), k, 0.08)
            g = at_gate(r.to_numpy(float))
            print(f"{'ORACLE ' + tag:>28}{k:>4}{g['cagr']*100:8.1f}%{g['dd']*100:7.1f}%"
                  f"{g['sharpe']:7.2f}{g['calmar']:7.2f}"
                  f"{(g['cagr']-b0)*100:+9.1f}pt", flush=True)

    # ---------------- 3. what the losing oracles are actually picking
    print("\n3. WHY THE CAGR AND CALMAR ORACLES LOSE - median over quarters\n")
    for tag, sc in (("Calmar oracle", te_cal), ("CAGR oracle", oracles["realised CAGR"])):
        top = [np.argsort(-sc[i], kind="stable")[:3] for i in range(len(Q))]
        f = lambda k: np.median([T[k][i, j] for i, t in enumerate(top) for j in t])
        print(f"   {tag:>16}: trades {f('trades'):5.0f}   quarterly DD "
              f"{f('dd')*100:6.1f}%   quarterly CAGR {f('cagr')*100:7.1f}%")
    print(f"   {'all 200 configs':>16}: trades {np.median(T['trades']):5.0f}"
          f"   quarterly DD {np.median(T['dd'])*100:6.1f}%"
          f"   quarterly CAGR {np.median(T['cagr'])*100:7.1f}%")

    # ---------------- 4. the Sharpe oracle swept over k
    print("\n4. THE CEILING: perfect foresight of next-quarter SHARPE, over k\n")
    shp = np.where(ok, T["sharpe"], BAD)
    print(f"{'k':>4}{'CAGR':>9}{'DD':>8}{'Shp':>7}{'Clm':>7}")
    best = 0.0
    for k in (1, 2, 3, 5, 8, 20):
        r, _ = blend(ranked_plan(lambda qi: shp[qi]), k, 0.08)
        g = at_gate(r.to_numpy(float))
        best = max(best, g["cagr"])
        print(f"{k:>4}{g['cagr']*100:8.1f}%{g['dd']*100:7.1f}%"
              f"{g['sharpe']:7.2f}{g['calmar']:7.2f}", flush=True)
    print(f"\n   ceiling over k: {best*100:.1f}% against the brief's 300%.")
    print("   The ORACLE rows are NOT strategies - they read the answer. They")
    print("   bound what any causal ranking rule can ever be worth.")
    print("\ndone: oracle bound")
