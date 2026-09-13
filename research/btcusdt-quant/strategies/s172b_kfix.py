"""
S172b - The k-sweep, redone at constant total risk.

THE ERROR IN S172
------------------
S172 built each configuration's return stream at a fixed 4.8% risk (= 14.4%/3)
and then SUMMED the top k. At k=3 that reproduces V7's 14.4% total exactly, which
is why every other test in that file is sound - they all ran at k=3. But the
k-sweep itself varied k while holding PER-CONFIG risk fixed, so total risk went

    k=1  ->  4.8%      k=3  ->  14.4%      k=5  ->  24%
    k=8  -> 38.4%      k=12 -> 57.6%

V7's actual rule is top-k at risk/k: the total stays 14.4% whatever k is. The gate
rescales to a 20% drawdown so leverage is largely normalised away, but not
cleanly - the engine's max_lev=10 cap binds harder at higher risk, penalising the
high-k rows for a reason that has nothing to do with k.

So each k is rebuilt here with per-config risk set to 0.144/k, which is what V7
does, and the comparison becomes like-for-like.
"""
import sys, os; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s172_v7stress as V
import strategies.s46_net as S46
from strategies.s96_rank import stats_of

if __name__ == "__main__":
    print("S172b - k-sweep at constant 14.4% total risk\n")
    print(f"{'top k':>7}{'per-config risk':>18}{'total risk':>13}{'Shp':>7}"
          f"{'CAGR':>9}{'maxDD':>8}{'at -20%':>10}")
    rows = []
    for k in (1, 2, 3, 5, 8, 12):
        per = 0.144 / k
        B = V.streams(S46.LONG, f"k{k}", risk=per)
        r = V.wf(B, k=k)
        if not len(r):
            print(f"   k={k}: no windows"); continue
        a = r.to_numpy(float)
        s = stats_of(a); g = V.gate_of(a)
        rows.append((k, g, s["sharpe"]))
        print(f"{k:>7}{per*100:>17.2f}%{0.144*100:>12.1f}%{s['sharpe']:>7.2f}"
              f"{s['cagr']*100:>8.1f}%{s['dd']*100:>7.1f}%"
              + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}"))

    ok = [r for r in rows if np.isfinite(r[1])]
    if ok:
        best = max(ok, key=lambda r: r[1])
        v7 = next((r for r in ok if r[0] == 3), None)
        print(f"\n   best k = {best[0]} at {best[1]:.1f}%")
        if v7:
            print(f"   V7's k = 3 at {v7[1]:.1f}%"
                  + ("  <- V7 is the peak" if best[0] == 3
                     else f"  <- {best[1]-v7[1]:+.1f}pp above V7's choice"))
        print("\n   S172's confounded version reported: k=1 123.1%, k=3 117.4%,")
        print("   k=5 98.5%, k=8 92.5%, k=12 100.0% - at total risks of 4.8% to")
        print("   57.6% rather than a constant 14.4%.")
    print("\ndone: k-sweep corrected")
