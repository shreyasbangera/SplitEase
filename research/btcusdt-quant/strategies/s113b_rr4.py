"""
S113b - Widening the one fence that might bind, and choosing it causally.

S113 swept the three grid edges the quarterly selection presses against and only
one of them is a fence rather than a preference:

    stop    optimum INSIDE {2.5, 3.0} - 3.0 best, and tightening collapses the
            book (1.25 ATR: -65 points, 17.9% of trades stopped out). The
            selection preferring 2.5 to 3.0 is a within-grid preference.
    hold    14 days is a sharp interior peak - 10 days -21.5, 21 days -20.0,
            35 days -21.5. Not a fence.
    R:R     3.0 is the grid MAXIMUM, and **4.0 scores +9.9 points** outside it,
            with 5.0 and 6.0 falling away again.

So the reward:risk axis gets widened from {2.0, 3.0} to {2.0, 3.0, 4.0} - 200
configurations become 300 - and the quarterly selection is re-run from scratch
so the new value has to be EARNED on trailing Calmar, quarter by quarter, with
no knowledge of the full sample. That is S85's discipline, and it is the only
reason the trend gate counts as a result.

THE PRIOR, STATED BEFORE THE RUN
--------------------------------
I expect this to fail, for three reasons that should be on the record first.

    1  +9.9 is inside S96c's 24-point selection-noise band.
    2  The screen was IN-SAMPLE and single-configuration.
    3  R:R is nearly inert up here. Take-profits fire on 3.3% of trades, and at
       R:R 3, 4, 5 and 6 the maximum drawdown is IDENTICAL (-22.6%) with trade
       counts of 654, 645, 641, 636. The whole +9.9 rests on a handful of trades
       that reached target at 4R and not at 3R. A sharp peak resting on a few
       events is what S99, S105 and S106 all produced, and all three were noise.

Against that: it is the only fence in the grid that is actually binding, it is
cheap, and "the menu was too narrow" is the one objection S96b's oracle bound
cannot answer. Worth settling rather than assuming.

The bar is unchanged: beat 168.8% by more than the noise band, hold in BOTH
halves, and survive the dilution the wider menu itself causes - S109 showed that
enlarging the pool a selection draws from makes more of its maximum noise, so a
wider grid can lose even when it contains something better.
"""
import sys, os, json; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s110_meta as M
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of, quarters

RISK, K = 0.08, 3
GATES = S87.GATES
WIDE = [(p, stp, rr, hold, sp, md)
        for p in (1.0, 1.5, 2.0, 2.5, 3.0)
        for stp in (2.5, 3.0) for rr in (2.0, 3.0, 4.0) for hold in (14, 21)
        for (sp, md) in GATES]
RANKS = "/home/user/quant/results/ranks_rr4.json"


def rankings():
    """Checkpointed per quarter; a reclaimed container costs one quarter."""
    done = []
    if os.path.exists(RANKS):
        done = [(s, e, [tuple(c) for c in cs]) for s, e, cs in json.load(open(RANKS))]
    Q = quarters()
    if len(done) >= len(Q):
        return done
    for s, e in Q[len(done):]:
        tr0 = str((pd.Timestamp(s, tz="UTC") - pd.DateOffset(months=12)).date())
        sc = sorted(((S69.calmar_of(S87.sim(cfg, tr0, s, 0.10)), cfg) for cfg in WIDE),
                    key=lambda x: -x[0])
        done.append((s, e, [tuple(c) for c in (c for _, c in sc)]))
        json.dump([[a, b, [list(c) for c in cs]] for a, b, cs in done], open(RANKS, "w"))
        print(f"    {s[:7]}  best {sc[0][1]}  Calmar {sc[0][0]:6.2f}  "
              f"[{len(done)}/{len(Q)}]", flush=True)
    return done


def row(tag, r, pl, base=None):
    a = np.asarray(r, float)
    st = stats_of(a); b = bootstrap_dd(a, n=4000, block=90)
    h = len(a) // 2
    f_, h1, h2 = (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
                  at_gate(a[h:])["cagr"] * 100)
    pf = pl[pl > 0].sum() / max(-pl[pl < 0].sum(), 1e-9) if len(pl) else np.nan
    d = "" if base is None else f"{f_-base[0]:+9.1f}{min(h1-base[1], h2-base[2]):+12.1f}"
    print(f"{tag:>26}{len(pl):7d}{pf:6.2f}{st['sharpe']:7.2f}{st['calmar']:7.2f}"
          f"{st['dd']*100:8.1f}%{b['dd_median']*100:9.1f}%{f_:9.1f}%{h1:9.1f}%"
          f"{h2:9.1f}%{d}", flush=True)
    return (f_, h1, h2)


if __name__ == "__main__":
    M.use_clock()
    print(f"{len(WIDE)} configurations (was {len(S87.GRID)}); R:R menu now "
          f"{{2.0, 3.0, 4.0}}\n")
    R4 = rankings()

    from collections import Counter
    c = Counter(cfg[2] for _, _, cs in R4 for cfg in cs[:K])
    print(f"\nwhat the causal selection picks for R:R among its top {K}: "
          f"{dict(sorted(c.items()))}")
    c1 = Counter(cs[0][2] for _, _, cs in R4)
    print(f"and for the single best each quarter: {dict(sorted(c1.items()))}\n")

    print(f"{'book':>26}{'N':>7}{'PF':>6}{'Shp':>7}{'Clm':>7}{'realDD':>8}"
          f"{'medDD':>9}{'full':>9}{'1st h':>9}{'2nd h':>9}{'vs V7':>9}{'worse half':>12}")
    S87.RANKS = "/home/user/quant/results/ranks_gate200.json"
    base = row("V7 (R:R 2 or 3)", *S87.blend(S87.rankings(), K, RISK))
    r, pl = S87.blend(R4, K, RISK)
    row("wide (R:R 2, 3 or 4)", r, pl, base)
    print("\ndone: widened grid")
