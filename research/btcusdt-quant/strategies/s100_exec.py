"""
S100 - V7 on a 1-minute execution grid. The deployed book's own conservatism, priced.

The engine resolves stops and targets on a 15-MINUTE grid and, whenever one bar
straddles both levels, assumes the STOP filled first. That is a deliberate
worst case, and it has never been priced for V7. The execution-granularity
validation in this log was run on S7, S15 and S4 - earlier books with different
stops, targets and composites - and found the conclusions held. V7 is a
different book: quarterly-selected, conviction-sized, a three-config blend.

Re-running on the 1-MINUTE grid (3.5M bars, no gaps) resolves the actual path
instead of assuming the bad half of it. The bias does NOT obviously run one
way, which is why it is worth measuring rather than asserting:

  * fewer bars straddle both levels, so the worst-case assumption fires far
    less often  -> favours the finer grid
  * but stops also trigger on wicks a 15-minute bar smoothed over
    -> favours the coarser grid

This is not a new strategy and cannot "beat" V7; it is the same book measured
more accurately. Whichever way it moves, the number it produces is the more
honest one, and it applies to the live account as it stands.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

import research.harness as H
from strategies.s87_combined import rankings, blend
from strategies.s96_rank import at_gate, stats_of

RISK = 0.08


def v7_on(grid):
    H.EXEC_TF[0] = grid
    H._ctxc.clear()
    r, pnl = blend(rankings(), 3, RISK)
    return r, pnl


if __name__ == "__main__":
    print("V7 = quarterly Calmar selection, top-3 blend at risk/3, 8% total risk")
    print("same plan, same signals, same configurations - only the grid on which")
    print("stops and targets are resolved changes.\n")
    print(f"{'exec grid':>12}{'CAGR':>9}{'MaxDD':>8}{'Sharpe':>8}{'Calmar':>8}"
          f"{'trades':>8}{'win%':>7}{'at -20% gate':>15}")

    out = {}
    for grid in ("fut_15m", "fut_1m"):
        r, pnl = v7_on(grid)
        a = np.asarray(r, float)
        s = stats_of(a)
        g = at_gate(a)
        wins = float((pnl > 0).mean()) * 100 if len(pnl) else float("nan")
        out[grid] = (s, g, pnl)
        print(f"{grid.replace('fut_',''):>12}{s['cagr']*100:8.1f}%{s['dd']*100:7.1f}%"
              f"{s['sharpe']:8.2f}{s['calmar']:8.2f}{len(pnl):8d}{wins:6.1f}%"
              f"{g['cagr']*100:14.1f}%", flush=True)

    a, b = out["fut_15m"], out["fut_1m"]
    print(f"\n{'delta (1m - 15m)':>12} {(b[0]['cagr']-a[0]['cagr'])*100:+7.1f}pp"
          f"{(b[0]['dd']-a[0]['dd'])*100:+7.1f}pp{b[0]['sharpe']-a[0]['sharpe']:+8.2f}"
          f"{b[0]['calmar']-a[0]['calmar']:+8.2f}{len(b[2])-len(a[2]):+8d}"
          f"{'':7}{(b[1]['cagr']-a[1]['cagr'])*100:+14.1f}pp")

    print("\nper-trade P&L distribution, which is where a fill assumption shows up\n")
    for grid, (s, g, pnl) in out.items():
        if not len(pnl):
            continue
        print(f"   {grid.replace('fut_',''):>5}: mean {pnl.mean():+8.2f}  median {np.median(pnl):+8.2f}"
              f"  5th {np.percentile(pnl,5):+9.2f}  95th {np.percentile(pnl,95):+9.2f}"
              f"  worst {pnl.min():+10.2f}")
    H.EXEC_TF[0] = "fut_15m"
    print("\ndone: execution granularity on V7")
