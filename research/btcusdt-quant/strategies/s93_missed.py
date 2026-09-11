"""
S93 - What do MISSED bars cost, and what do they cost you to not know about?

S92 asked what acting LATE costs. This asks what not acting at all costs, which
is the failure mode of the only machine now available: a laptop. It sleeps. Some
12h bars get no decision.

WHAT A MISS ACTUALLY IS
-----------------------
Not a deleted trade. A SUBSTITUTED one.

On a missed bar the position is carried forward - you keep whatever you were
already holding - and only the signal-driven actions are lost: the reversal, the
flat-signal exit, the rebalance. The stop and the take-profit are reduce-only
orders already resting on the exchange and fire whether or not anything of yours
is running, so the loss tail stays capped.

So a missed bar quietly converts the book from "exit when the edge decays" into
"exit only on a stop, a target, or the holding cap". That is a different system.
It is modelled here exactly that way: `entry` is zeroed at the missed decision
bar and `exit` is masked across it, while stop and tp are left alone.

WHY BLOCKS
----------
A laptop is not off for independent bars. It is off for a night, a weekend, a
trip. Six consecutive missed bars are three days of ONE market regime, not six
independent draws, so the same miss RATE does different damage depending on how
it clusters. Both are measured: block=1 (scattered) and block=6 (three-day
outages).

THE SECOND QUESTION IS THE IMPORTANT ONE
----------------------------------------
The average cost of missing bars is worth knowing but is not the real problem.
The real problem is DISPERSION: at a fixed miss rate, how much does the answer
depend on WHICH bars you happened to miss? If that spread is wide, then a single
realised result - your live P&L after three months - tells you very little about
the strategy, because most of what you are looking at is which days your laptop
was open. Part B measures that spread directly.

Base: the 12-month-lookback quarterly plan at 8% risk, which is the deployed
setting. Control must reproduce 73.2% CAGR, -14.8% DD, PF 2.64, N 771.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from research.harness import backtest
from research.planscache import cached_plan
from strategies.s69_calsel import ctx, shape, daily
from research.robust import bootstrap_dd


def sim(cfg, start, end, risk, miss=0.0, seed=0, block=1):
    p, stp, rr, hold = cfg
    c = ctx(); u = np.nan_to_num(shape(p)); a = c["a"]
    arr = dict(entry=u, stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u) <= 0.0).astype(float))
    return backtest(c["g"], arr, "12h", start=start, end=end, risk=risk,
                    max_lev=10.0, max_bars_h=hold * 24,
                    miss_prob=miss, miss_seed=seed, miss_block=block)


def replay(plan, risk, miss=0.0, seed=0, block=1):
    segs, pnl, rates = [], [], []
    for i, (s, e, cfg) in enumerate(plan):
        # A different seed per quarter, derived from the run's seed, so the
        # outages are not the same calendar pattern in every segment.
        m = sim(cfg, s, e, risk, miss, seed * 1000 + i, block)
        segs.append(daily(m))
        if "miss_rate" in m: rates.append(m["miss_rate"])
        td = m["trades_df"]
        if td is not None and len(td): pnl.append(td["pnl"].to_numpy(float))
    return (pd.concat(segs), np.concatenate(pnl) if pnl else np.array([]),
            float(np.mean(rates)) if rates else 0.0)


def metrics(r, P):
    e = np.cumprod(1 + r.to_numpy()); yrs = (r.index[-1] - r.index[0]).days / 365.25
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    cagr = e[-1] ** (1 / yrs) - 1
    pf = P[P > 0].sum() / max(-P[P < 0].sum(), 1e-9) if len(P) else float("nan")
    return dict(cagr=cagr, dd=dd, pf=pf, n=len(P),
                sharpe=float(r.mean() / r.std() * np.sqrt(365.25)),
                calmar=cagr / abs(dd) if dd else float("nan"))


def line(tag, m, rate=None):
    r = f"  miss {rate*100:4.1f}%" if rate is not None else " " * 12
    print(f"{tag:>26}{r} | CAGR {m['cagr']*100:7.1f}%  DD {m['dd']*100:6.1f}%"
          f"  PF {m['pf']:5.2f}  N {m['n']:4d}  Shp {m['sharpe']:5.2f}"
          f"  Clm {m['calmar']:5.2f}", flush=True)


if __name__ == "__main__":
    P = cached_plan(12)
    RISK = 0.08

    print("CONTROL - must reproduce 73.2% / -14.8% / PF 2.64 / N 771\n")
    r, pl, _ = replay(P, RISK)
    base = metrics(r, pl)
    line("control, nothing missed", base)

    print("\n\nPART A - the average cost of a miss rate")
    print("8 seeds per cell; the figure is the MEAN, the bracket the range.\n")
    SEEDS = range(1, 9)
    for block, blabel in ((1, "scattered single bars"), (6, "three-day outages")):
        print(f"  --- outages as {blabel} (block={block}) ---")
        for miss in (0.05, 0.10, 0.20, 0.33, 0.50):
            ms, rates = [], []
            for sd in SEEDS:
                r, pl, rate = replay(P, RISK, miss, sd, block)
                ms.append(metrics(r, pl)); rates.append(rate)
            cg = np.array([m["cagr"] for m in ms]) * 100
            pf = np.array([m["pf"] for m in ms])
            dd = np.array([m["dd"] for m in ms]) * 100
            nn = np.array([m["n"] for m in ms])
            print(f"    miss {np.mean(rates)*100:4.1f}%  "
                  f"CAGR {cg.mean():6.1f}% [{cg.min():5.1f} .. {cg.max():5.1f}]  "
                  f"PF {pf.mean():4.2f} [{pf.min():.2f} .. {pf.max():.2f}]  "
                  f"DD {dd.mean():6.1f}%  N {nn.mean():5.0f}", flush=True)

    print("\n\nPART B - the dispersion at one realistic setting")
    print("20% of bars missed in three-day blocks - a laptop shut most weekends")
    print("and away one week in five. 40 seeds. SAME strategy, SAME period, SAME")
    print("data: the only thing that differs is WHICH bars were missed.\n")
    out = []
    for sd in range(101, 141):
        r, pl, rate = replay(P, RISK, 0.20, sd, 6)
        out.append(metrics(r, pl))
        print(f"    seed {sd}  CAGR {out[-1]['cagr']*100:7.1f}%  "
              f"PF {out[-1]['pf']:5.2f}  DD {out[-1]['dd']*100:6.1f}%  "
              f"N {out[-1]['n']:4d}", flush=True)

    cg = np.sort(np.array([m["cagr"] for m in out]) * 100)
    pf = np.sort(np.array([m["pf"] for m in out]))
    print(f"\n    control CAGR {base['cagr']*100:.1f}%   PF {base['pf']:.2f}")
    print(f"    missed  CAGR  min {cg.min():6.1f}%  p10 {np.percentile(cg,10):6.1f}%"
          f"  median {np.median(cg):6.1f}%  p90 {np.percentile(cg,90):6.1f}%"
          f"  max {cg.max():6.1f}%")
    print(f"            PF   min {pf.min():5.2f}   p10 {np.percentile(pf,10):5.2f}"
          f"   median {np.median(pf):5.2f}   p90 {np.percentile(pf,90):5.2f}"
          f"   max {pf.max():5.2f}")
    print(f"\n    spread in CAGR: {cg.max()-cg.min():.0f} points, on one strategy over "
          f"one period.\n    Anything you conclude from a single run has to clear that.")
    print("\ndone: missed bars")
