"""
S96c - How much of V7's 168.8% is the ranking, and how much is luck?

S96 tested eight defensible ranking rules and found neither family beat the
control together - so the estimator direction is a negative. But it produced a
more uncomfortable number on the way: those eight rules span **73.1% to 181.2%**
at the -20% gate. A 108-point range from changing nothing but the sort order.

V7's own rule sits at 168.8%, near the top of that range. Two readings, and
they have opposite consequences:

  1. Trailing Calmar is genuinely a good estimator and deserves its position.
  2. The quarterly selection is noise-dominated, the spread is what noise looks
     like, and V7's placement near the top is partly the luck of one draw.

If (2), then 168.8% is an optimistic reading of the deployed book and the
honest number is lower - which matters for the live account, not just for the
log.

METHOD
------
Perturb the trailing-Calmar scores with Gaussian noise scaled to their own
cross-sectional spread, re-rank, re-blend top-3, and measure at the gate. At
zero noise every draw collapses onto the control. As noise grows the spread of
outcomes opens up, and its width says how much of the result the ranking is
actually holding.

This is a sensitivity measurement, not a strategy. The noise is not something a
trader suffers; it stands in for how much the ranking would move under an
equally defensible estimate of the same quantity.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np
from strategies.s87_combined import GRID, blend
from strategies.s96_rank import quarters, at_gate, train_metrics, scores, BAD

if __name__ == "__main__":
    M = train_metrics(); Q = quarters()
    base = np.array([scores(M, qi, "calmar") for qi in range(len(Q))])
    valid = base > BAD
    sd = np.array([base[qi][valid[qi]].std() if valid[qi].sum() > 2 else 0.0
                   for qi in range(len(Q))])
    print(f"{len(Q)} quarters, cross-sectional sd of trailing Calmar "
          f"{sd.mean():.2f} on average\n")
    print(f"{'noise':>7}{'seeds':>7}{'median':>9}{'min':>9}{'max':>9}{'sd':>8}"
          f"{'>= ctl':>8}")
    ctl = None
    for lvl in (0.0, 0.10, 0.25, 0.50, 1.00):
        outs = []
        for seed in range(12 if lvl > 0 else 1):
            rng = np.random.default_rng(seed)
            plan = []
            for qi, (s, e) in enumerate(Q):
                sc = base[qi].copy()
                sc = np.where(valid[qi], sc + rng.normal(0, lvl * sd[qi], len(sc)), BAD)
                order = np.argsort(-sc, kind="stable")
                plan.append((s, e, [tuple(GRID[i]) for i in order]))
            r, _ = blend(plan, 3, 0.08)
            outs.append(at_gate(r.to_numpy(float))["cagr"])
        o = np.array(outs) * 100
        if ctl is None:
            ctl = o[0]
        print(f"{lvl:7.2f}{len(o):7d}{np.median(o):8.1f}%{o.min():8.1f}%"
              f"{o.max():8.1f}%{o.std(ddof=1) if len(o)>1 else 0:7.1f}"
              f"{int((o >= ctl).sum()):5d}/{len(o)}", flush=True)
    print(f"\ncontrol (zero noise) = {ctl:.1f}%")
    print("done: how much of it is the ranking")
