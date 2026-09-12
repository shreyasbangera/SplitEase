"""
S99 - The signal thresholds, fixed since S45 and never once varied.

THR = {flow 1.0, cmpx 1.0, btcdom 1.0, fundz 1.0, posn 0.7} was set when the
five-signal book was first assembled and has been carried unchanged through
every experiment since - through the conviction exponent, the quarterly
selection, the trend gate, the blend, and every negative in S94-S98. The grid
V7 selects from varies the exponent, stop, reward:risk, hold and gate. It does
not vary this.

The mechanism is real and points both ways. Each signal contributes 0 inside
its band and +/-1 outside, scaled by |z|/threshold and capped at 2, so the
threshold sets two things at once: how OFTEN a signal speaks, and how its
conviction is scaled once it does.

  * lower  -> signals fire more often, the composite is rarely flat, more
              trades, but more of them on marginal readings.
  * higher -> fewer and more confident trades. The drawdown anatomy says the
              top 10% of trades carry 130% of net profit and the bottom 90%
              lose money together, which is a direct argument for concentrating
              into the strong readings.

That asymmetry is why this is worth running despite S94-S98 all being negative:
the profit concentration is a measured property of this book, and raising the
bar is the one change that acts on it directly.

Also measured here, because it has never been checked: how often the 10x
leverage cap actually binds. If it binds frequently the conviction sizing is
being truncated and every exponent result in the study is partly a statement
about the cap rather than about conviction.

Compared at the -20% gate against the unchanged thresholds. A scale multiplier
is applied to all five together rather than tuning them individually - five
free parameters searched separately on one sample is how this study would
manufacture a result rather than find one.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

import strategies.s45_single as S
import strategies.s46_net as S46
from research.harness import backtest, OOS_END, IS_END
from strategies.s69_calsel import daily
from strategies.s96_rank import at_gate

START = S.FULL_START
HOLD, STP, RR = 21, 3.0, 2.0
BASE = dict(S.THR)


def run(g, scale, risk=0.08, start=START, end=OOS_END):
    for k, v in BASE.items():
        S.THR[k] = v * scale
    net = S.composite(g, S46.LONG)
    a = S.book(g, net, stp=STP, rr=RR)
    a["exit"] = (np.abs(np.nan_to_num(net)) <= 0.0).astype(float)
    m = backtest(g, a, "12h", start=start, end=end, risk=risk,
                 max_lev=10.0, max_bars_h=HOLD * 24)
    return daily(m), m, net


if __name__ == "__main__":
    g = S.grid(START)
    print(f"panel {len(g)} bars, {g.dt.iloc[0].date()} -> {g.dt.iloc[-1].date()}")
    print(f"base thresholds {BASE}\n")
    print(f"{'THR scale':>11}{'CAGR':>9}{'DD':>8}{'PF':>6}{'N':>7}{'Shp':>7}"
          f"{'Clm':>7}{'flat%':>8}{'mean|net|':>10}{'vs base':>9}")

    # the baseline first, so every row below can be quoted against it
    b0 = at_gate(run(g, 1.0)[0].to_numpy(float))["cagr"]

    for scale in (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5):
        r, m, net = run(g, scale)
        gt = at_gate(r.to_numpy(float))
        nz = np.abs(np.nan_to_num(net)) > 0
        tag = f"{scale:.2f}" + (" *" if scale == 1.0 else "")
        print(f"{tag:>11}{gt['cagr']*100:8.1f}%{gt['dd']*100:7.1f}%"
              f"{m['profit_factor']:6.2f}{m['trades']:7d}{gt['sharpe']:7.2f}"
              f"{gt['calmar']:7.2f}{(1-nz.mean())*100:7.1f}%"
              f"{np.abs(net[nz]).mean():10.3f}"
              f"{(gt['cagr']-b0)*100:+8.1f}pt", flush=True)

    print("\n* = the deployed thresholds")
    print("\nDOES THE 10x LEVERAGE CAP BIND? (never checked before)\n")
    for scale in (0.5, 1.0, 2.0):
        r, m, net = run(g, scale)
        td = m["trades_df"]
        if td is not None and len(td) and "lev" in td.columns:
            lv = td["lev"].to_numpy(float)
            print(f"   THR x{scale:.2f}: median leverage {np.median(lv):5.2f}x  "
                  f"at cap on {np.mean(lv >= 9.99)*100:5.1f}% of trades")
        else:
            cols = list(td.columns) if td is not None else []
            print(f"   THR x{scale:.2f}: trades_df columns = {cols}")
            break
    for k, v in BASE.items():
        S.THR[k] = v
    print("\ndone: thresholds")
