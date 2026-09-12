"""
S98 - Splitting the composite into disjoint books, which is a different
decorrelation from the two this study has already closed.

WHAT IS AND IS NOT ALREADY KNOWN
--------------------------------
Two decorrelation ideas are closed. S86b blended CONFIGURATIONS of the same
five-signal composite and found the gain is the ranking, not the diversity -
random blending is worth nothing, because two configurations of one composite
take nearly the same positions. S42 blended PHASES of the same book and found
cross-phase correlation 0.70-0.82, so there was almost nothing to diversify.

Neither is this. Here the five signals are PARTITIONED into disjoint groups,
each group is netted into its own composite, and each is traded as its own book
with its own entry, stop and target at risk/m. The books then hold genuinely
different positions - frequently opposite ones - because they are reading
different markets.

The premise is measurable rather than assumed, and it is measured below: each
signal's unit series correlates +0.39 to +0.68 with the five-signal composite
(S95c), which is far from the ~0.9 that two configurations of one composite
share. Two books on disjoint halves should therefore decorrelate substantially,
and decorrelated equity paths at the same Sharpe is exactly the Calmar lever
the brief needs - the one thing S94 could not get from portfolio overlays and
S96 could not get from the ranking.

THE COSTS, STATED FIRST
-----------------------
Splitting is not free and the test has to pay for all of it:

  * m books pay m sets of fees where the netted composite paid one. At 16 bps a
    round turn and ~180 trades a year per book this is a real charge, and it is
    charged in full - the backtest fees each book separately.
  * each book trades only when ITS signals fire, so a two-signal book is in the
    market far less often than the five-signal composite.
  * S46's leave-one-out showed every signal is load-bearing (Sharpe falls to
    1.50 without the stablecoin basis, 1.51 without positioning), so the groups
    are all weaker than the whole.

WHAT WOULD COUNT
----------------
Comparison at the -20% gate against the SAME fixed configuration running the
netted five-signal composite, so the only difference is the partition. The
partitions are the natural groupings written down before running, not searched.
A win has to clear the noise band before it means anything.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

import strategies.s45_single as S
import strategies.s46_net as S46
from research.harness import backtest, OOS_END
from strategies.s69_calsel import daily
from strategies.s96_rank import at_gate

START = S.FULL_START
HOLD, STP, RR = 21, 3.0, 2.0

PARTITIONS = {
    "netted 5 (control)":      [["flow", "cmpx", "btcdom", "fundz", "posn"]],
    "flow+cmpx | rest":        [["flow", "cmpx"], ["btcdom", "fundz", "posn"]],
    "flow+btcdom | rest":      [["flow", "btcdom"], ["cmpx", "fundz", "posn"]],
    "flow | cmpx+dom | fnd+pos": [["flow"], ["cmpx", "btcdom"], ["fundz", "posn"]],
    "all five separate":       [["flow"], ["cmpx"], ["btcdom"], ["fundz"], ["posn"]],
}


def book_returns(g, names, risk):
    net = S.composite(g, names)
    a = S.book(g, net, stp=STP, rr=RR)
    a["exit"] = (np.abs(np.nan_to_num(net)) <= 0.0).astype(float)
    m = backtest(g, a, "12h", start=START, end=OOS_END, risk=risk,
                 max_lev=10.0, max_bars_h=HOLD * 24)
    return daily(m), m


if __name__ == "__main__":
    g = S.grid(START)
    print(f"panel {len(g)} bars, {g.dt.iloc[0].date()} -> {g.dt.iloc[-1].date()}")
    print("every book at risk/m so total risk is held constant across rows\n")
    print(f"{'partition':>28}{'m':>3}{'CAGR':>9}{'DD':>8}{'Shp':>7}{'Clm':>7}"
          f"{'N':>7}{'vs ctl':>9}")

    b0 = None
    curves = {}
    for tag, groups in PARTITIONS.items():
        m = len(groups)
        rs, n = [], 0
        for grp in groups:
            r, mm = book_returns(g, grp, 0.08 / m)
            rs.append(r); n += mm["trades"]
        tot = pd.concat(rs, axis=1).fillna(0.0).sum(axis=1)
        curves[tag] = [r for r in rs]
        gt = at_gate(tot.to_numpy(float))
        if b0 is None:
            b0 = gt["cagr"]
        print(f"{tag:>28}{m:>3}{gt['cagr']*100:8.1f}%{gt['dd']*100:7.1f}%"
              f"{gt['sharpe']:7.2f}{gt['calmar']:7.2f}{n:7d}"
              f"{(gt['cagr']-b0)*100:+8.1f}pt", flush=True)

    print("\nTHE PREMISE: how decorrelated are the split books actually?\n")
    for tag, rs in curves.items():
        if len(rs) < 2:
            continue
        D = pd.concat(rs, axis=1).fillna(0.0)
        C = D.corr().to_numpy()
        off = C[np.triu_indices_from(C, 1)]
        print(f"   {tag:>28}: pairwise daily corr mean {off.mean():+.3f}  "
              f"range {off.min():+.3f} to {off.max():+.3f}")
    print("\ndone: disjoint books")
