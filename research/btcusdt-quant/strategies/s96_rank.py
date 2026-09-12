"""
S96 - The ranking ESTIMATOR, which is the one mechanism with demonstrated
Calmar leverage that has never been varied.

WHY THE RANKING AND NOT SOMETHING ELSE
--------------------------------------
S95 recorded the arithmetic the loop had been avoiding. Two versions of this
book at the -20% gate:

    V1  5 signals, one fixed configuration    71.2% CAGR   Sharpe 2.15
    V7  quarterly selection, top-3 blend     179.0% CAGR   Sharpe 2.13

The study's entire 2.5x improvement came with NO Sharpe improvement. Everything
that moved, moved Calmar, and S86b located where: blending five configurations
drawn at RANDOM is worth nothing (71.7-75.8% against 73.2% for the single best)
while blending the top five by trailing Calmar is worth 18%. **The gain is the
ranking, not the diversification.**

So the ranking is the load-bearing mechanism, and it is the one thing in the
pipeline nobody has varied. S68-S69 varied the OBJECTIVE - Sharpe against
Calmar against their product - and Calmar won by a distance, which is settled.
This is a different question: given that we rank on Calmar, is trailing Calmar
a good ESTIMATOR of it?

THE SPECIFIC COMPLAINT
----------------------
Calmar = CAGR / |max drawdown|, and the denominator is measured over a 12-month
training window carrying roughly 40 trades. Max drawdown is the single worst
point of one realised path - the noisiest statistic in the whole set, with no
averaging in it at all. Every other number the selection could use (Sharpe,
profit factor, downside deviation, the ulcer index) pools information across
the entire path.

Ranking 200 configurations on a statistic that noisy means the winner is partly
just the luckiest, which is the textbook winner's curse, and the textbook fix is
to denoise the estimator rather than to change what is being estimated.

WHAT IS TESTED, AND HOW IT AVOIDS BEING A LUCKY CELL
----------------------------------------------------
Eight ranking rules, in three families with a stated hypothesis each. The test
is whether a FAMILY beats the control together - the defence S68-S69 used when
"the whole 24-month drawdown-aware family wins together, rather than one lucky
cell" - not whether some cell wins.

  control       calmar        CAGR / |max DD|, exactly what V7 deploys

  stable denominator - same numerator, a denominator that uses the whole path
                ulcer         CAGR / sqrt(mean(drawdown_t^2))
                sortino       CAGR / downside deviation
                bootdd        CAGR / |bootstrap median drawdown|

  rank averaging - combine noisy estimators instead of trusting one
                rank_cs       mean rank of (calmar, sharpe)
                rank_cup      mean rank of (calmar, ulcer, profit factor)
                multilb       mean rank of calmar over 6, 9 and 12-month windows

  shrinkage     shrunk        calmar pulled halfway to the grid mean

Nothing is tuned. The shrinkage weight is fixed at 0.5 rather than searched, the
multi-lookback windows are the three that fit before the panel starts, and every
rule uses the same 200-configuration grid, the same quarterly dates, the same
top-3 blend at risk/3 and the same training sims. Only the sort order differs.

EVALUATION
----------
Every rule is compared AT THE -20% GATE, because a rule that merely sizes
differently is not a better rule - the lesson S95 paid for. Following S94, the
gate is found by scaling the blend's daily returns and the CONTROL IS MEASURED
THE SAME WAY, so the comparison is like-for-like even though the method runs
about 2% optimistic in level against a true risk bisection. Any rule that wins
is then re-run with a real risk bisection before it is believed.
"""
import sys, os, json, time
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

import strategies.s45_single as S
from research.harness import OOS_END
from research.robust import bootstrap_dd
from strategies.s87_combined import GRID, sim, blend
from strategies.s69_calsel import daily

LOOKBACK, RESELECT = 12, 3
MIN_TRADES = 15
CACHE = "/home/user/quant/results/s96_train.npz"
BAD = -9e9


# ------------------------------------------------------- the expensive part

def quarters():
    t0 = pd.Timestamp(S.FULL_START, tz="UTC")
    t = t0 + pd.DateOffset(months=LOOKBACK)
    end = pd.Timestamp(OOS_END, tz="UTC")
    out = []
    while t < end:
        te = min(t + pd.DateOffset(months=RESELECT), end)
        out.append((str(t.date()), str(te.date())))
        t = te
    return out


def train_metrics(lookbacks=(6, 9, 12)):
    """Every configuration's trailing metrics at every selection date.

    Run once and cached. A ranking rule is then a different sort of the same
    numbers, which is what makes eight rules affordable and, more importantly,
    what makes them a fair comparison - no rule gets a different backtest.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    Q = quarters()
    shape = (len(lookbacks), len(Q), len(GRID))
    out = {k: np.full(shape, np.nan) for k in
           ("cagr", "dd", "sharpe", "pf", "trades", "ulcer", "dsd", "bootdd")}
    t0 = time.time()
    for li, lb in enumerate(lookbacks):
        for qi, (s, e) in enumerate(Q):
            tr0 = str((pd.Timestamp(s, tz="UTC") - pd.DateOffset(months=lb)).date())
            for ci, cfg in enumerate(GRID):
                m = sim(cfg, tr0, s, 0.10)
                out["cagr"][li, qi, ci] = m["cagr"]
                out["dd"][li, qi, ci] = m["max_dd"]
                out["sharpe"][li, qi, ci] = m["sharpe"]
                out["pf"][li, qi, ci] = m["profit_factor"]
                out["trades"][li, qi, ci] = m["trades"]
                r = daily(m).to_numpy(float)
                if len(r) > 5:
                    eq = np.cumprod(1.0 + r)
                    ddp = eq / np.maximum.accumulate(eq) - 1.0
                    out["ulcer"][li, qi, ci] = np.sqrt(np.mean(ddp ** 2))
                    neg = r[r < 0]
                    out["dsd"][li, qi, ci] = (neg.std() * np.sqrt(365.25)
                                              if len(neg) > 2 else np.nan)
                    out["bootdd"][li, qi, ci] = bootstrap_dd(r, n=200)["dd_median"]
            print(f"  lookback {lb:2d}m  quarter {qi+1:2d}/{len(Q)}  "
                  f"{time.time()-t0:6.0f}s", flush=True)
    np.savez_compressed(CACHE, **out)
    return out


# ------------------------------------------------------------ ranking rules

def _ranks(x, higher_is_better=True):
    """Average ranks, NaNs last. 0 = best."""
    v = np.where(np.isfinite(x), x, -np.inf if higher_is_better else np.inf)
    if not higher_is_better:
        v = -v
    order = np.argsort(-v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    return r


def scores(M, qi, rule):
    """One rule's score for every configuration at one selection date.

    Index 2 of every cached array is the 12-month lookback, which is what the
    deployed book uses; 0 and 1 are 6 and 9 months and only `multilb` reads them.
    """
    L12 = 2
    cagr = M["cagr"][L12, qi]
    dd = np.abs(M["dd"][L12, qi])
    ok = (M["trades"][L12, qi] >= MIN_TRADES) & (M["dd"][L12, qi] < 0)
    calmar = np.where(ok & (dd > 0), cagr / np.maximum(dd, 1e-9), BAD)

    if rule == "calmar":
        s = calmar
    elif rule == "ulcer":
        u = M["ulcer"][L12, qi]
        s = np.where(ok & (u > 0), cagr / np.maximum(u, 1e-9), BAD)
    elif rule == "sortino":
        d = M["dsd"][L12, qi]
        s = np.where(ok & (d > 0), cagr / np.maximum(d, 1e-9), BAD)
    elif rule == "bootdd":
        b = np.abs(M["bootdd"][L12, qi])
        s = np.where(ok & (b > 0), cagr / np.maximum(b, 1e-9), BAD)
    elif rule == "shrunk":
        # Shrinking every estimate toward the grid mean by the SAME factor is
        # an affine transform and leaves the ranking identical - the shrinkage
        # has to be per-configuration to reorder anything. Weight by how much
        # evidence each configuration actually has: a config that traded 80
        # times keeps most of its estimate, one that traded 16 is pulled most
        # of the way to the mean. k = 40 is the grid's rough median trade
        # count and is fixed, not searched.
        n = M["trades"][L12, qi]
        good = calmar[calmar > BAD]
        mu = good.mean() if len(good) else 0.0
        lam = n / (n + 40.0)
        s = np.where(ok, lam * calmar + (1.0 - lam) * mu, BAD)
    elif rule == "rank_cs":
        r = _ranks(np.where(ok, calmar, np.nan)) + _ranks(np.where(ok, M["sharpe"][L12, qi], np.nan))
        s = np.where(ok, -r, BAD)
    elif rule == "rank_cup":
        u = M["ulcer"][L12, qi]
        cu = np.where(ok & (u > 0), cagr / np.maximum(u, 1e-9), np.nan)
        r = (_ranks(np.where(ok, calmar, np.nan)) + _ranks(cu)
             + _ranks(np.where(ok, M["pf"][L12, qi], np.nan)))
        s = np.where(ok, -r, BAD)
    elif rule == "multilb":
        r = np.zeros(len(calmar))
        for li in (0, 1, 2):
            c = M["cagr"][li, qi]
            d = np.abs(M["dd"][li, qi])
            o = (M["trades"][li, qi] >= MIN_TRADES) & (M["dd"][li, qi] < 0)
            r = r + _ranks(np.where(o & (d > 0), c / np.maximum(d, 1e-9), np.nan))
        s = np.where(ok, -r, BAD)
    else:
        raise ValueError(rule)
    return s


def plan_for(M, rule):
    Q = quarters()
    out = []
    for qi, (s, e) in enumerate(Q):
        sc = scores(M, qi, rule)
        order = np.argsort(-sc, kind="stable")
        out.append((s, e, [tuple(GRID[i]) for i in order]))
    return out


# -------------------------------------------------------------- evaluation

def stats_of(r):
    r = np.asarray(r, float)
    eq = np.cumprod(1.0 + r)
    yrs = len(r) / 365.25
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    cagr = float(eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0
    sd = r.std()
    return dict(cagr=cagr, dd=dd,
                sharpe=float(r.mean() / sd * np.sqrt(365.25)) if sd else np.nan,
                calmar=cagr / abs(dd) if dd else np.nan)


def at_gate(r, target=-0.20, lo=0.2, hi=6.0, iters=40):
    """Scale the daily returns until realised max drawdown sits on the gate.

    S94's method, and the control is measured the same way, so the comparison
    is like-for-like even though the level runs ~2% optimistic.
    """
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = stats_of(np.asarray(r, float) * mid)
        m["scale"] = mid
        if best is None or abs(m["dd"] - target) < abs(best["dd"] - target):
            best = m
        if abs(m["dd"] - target) < 1e-4:
            break
        if m["dd"] < target:
            hi = mid
        else:
            lo = mid
    return best


RULES = [("calmar", "control - what V7 deploys"),
         ("ulcer", "stable denominator"),
         ("sortino", "stable denominator"),
         ("bootdd", "stable denominator"),
         ("rank_cs", "rank averaging"),
         ("rank_cup", "rank averaging"),
         ("multilb", "rank averaging"),
         ("shrunk", "shrinkage")]


if __name__ == "__main__":
    print(f"{len(GRID)} configurations x {len(quarters())} quarters x 3 lookbacks")
    print("caching every training sim once (~40 min cold, instant after)\n", flush=True)
    M = train_metrics()
    print("\ncached.\n")

    # how different are these rankings from each other at all?
    print("AGREEMENT with the control's top 3, averaged over quarters:\n")
    base = plan_for(M, "calmar")
    plans = {}
    for rule, fam in RULES:
        p = plan_for(M, rule)
        plans[rule] = p
        share = np.mean([len(set(a[2][:3]) & set(b[2][:3])) / 3.0
                         for a, b in zip(base, p)])
        print(f"  {rule:>9} ({fam:<24}) shares {share*100:5.1f}% of the top 3")

    print("\nAT THE -20% GATE - daily returns scaled, control measured the same way\n")
    print(f"{'rule':>9}  {'family':<24} {'CAGR':>8}{'DD':>8}{'Shp':>7}{'Clm':>7}{'vs ctl':>9}")
    b0 = None
    res = {}
    for rule, fam in RULES:
        r, _ = blend(plans[rule], 3, 0.08)
        res[rule] = r
        g = at_gate(r.to_numpy(float))
        if b0 is None:
            b0 = g["cagr"]
        print(f"{rule:>9}  {fam:<24} {g['cagr']*100:7.1f}%{g['dd']*100:7.1f}%"
              f"{g['sharpe']:7.2f}{g['calmar']:7.2f}"
              f"{(g['cagr']-b0)*100:+8.1f}pt", flush=True)

    print("\nA rule counts only if its FAMILY beats the control together.")
    print("done: ranking estimator")
