"""
S112 - What size actually respects a 20% drawdown limit? Not the one in the headline.

Every "at the gate" figure in this log, the deployed 179.0% at -19.99% included,
is a **single realised path**. The book was run at 14.4% risk, the worst drawdown
that particular sequence of trades happened to produce was -19.99%, and the
number was recorded as satisfying the brief.

S103 is the reason to distrust that. V7's maximum drawdown is a three-day
mark-to-market excursion on a position that went on to make money: BTC fell 4.1%
while the netted account held 3.05x. Nothing about that episode is a property of
the strategy. Reorder the same trades and it lands somewhere else.

The bootstrap has been printing the warning all along and it was never followed
up. At 8% risk V7's realised drawdown is -12.3% but the bootstrap **median** is
-18.1%, and **P(drawdown worse than 20%) is 33%**. If the median path at 8% risk
is already close to the limit, the deployed 14.4% is not respecting a 20% limit
in any distributional sense - it cleared it once.

So this asks the question the brief actually implies, rather than the one the
single path answers:

    at what risk is the MEDIAN max drawdown -20%, and what does the book earn there?

and the stricter version a person sizing real money would want:

    at what risk is P(drawdown worse than 20%) only 1 in 10?

Nothing here is a new strategy. It is the same book, measured against a
distribution instead of against one lucky sequence, and it revises the headline
of this entire study downward.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s87_combined as S87
import strategies.s110_meta as M
from research.robust import bootstrap_dd
from strategies.s96_rank import stats_of

RISKS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.144, 0.16, 0.20)
K = 3


def book(risk):
    R = S87.rankings()
    r, pnl = S87.blend(R, K, risk)
    return np.asarray(r, float), pnl


if __name__ == "__main__":
    M.use_clock()
    print("V7, the deployed book, measured against the DISTRIBUTION of drawdowns\n")
    print(f"{'risk':>7}{'CAGR':>9}{'realised DD':>13}{'boot median':>13}"
          f"{'worst 5%':>10}{'P(DD>20%)':>11}{'Sharpe':>8}{'trades':>8}")
    rows = []
    for rk in RISKS:
        a, pnl = book(rk)
        st = stats_of(a)
        b = bootstrap_dd(a, n=6000)
        p05 = b["dd_p05"]          # the pessimistic tail: drawdowns are negative
        rows.append((rk, st["cagr"], st["dd"], b["dd_median"],
                     b["p_dd_worse_than_20"], st["sharpe"], len(pnl)))
        print(f"{rk*100:6.1f}%{st['cagr']*100:8.1f}%{st['dd']*100:12.1f}%"
              f"{b['dd_median']*100:12.1f}%{p05*100:9.1f}%"
              f"{b['p_dd_worse_than_20']*100:10.0f}%{st['sharpe']:8.2f}{len(pnl):8d}",
              flush=True)

    D = pd.DataFrame(rows, columns=["risk", "cagr", "dd", "bmed", "p20", "shp", "n"])

    def solve(col, target):
        """Risk at which `col` crosses `target`, linearly between the two
        bracketing rows. Reported as a range when it falls outside the ladder."""
        x, y = D.risk.to_numpy(), D[col].to_numpy()
        for i in range(len(x) - 1):
            if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
                w = (target - y[i]) / (y[i + 1] - y[i])
                rk = x[i] + w * (x[i + 1] - x[i])
                cg = D.cagr.to_numpy()[i] + w * (D.cagr.to_numpy()[i + 1]
                                                 - D.cagr.to_numpy()[i])
                return rk, cg
        return None, None

    print("\nthree readings of the same 20% limit\n")
    r0, c0 = solve("dd", -0.20)
    print(f"  realised path      max DD hits -20% at risk {r0*100:5.1f}%  "
          f"-> CAGR {c0*100:6.1f}%   <- what this log has been reporting")
    r1, c1 = solve("bmed", -0.20)
    if r1:
        print(f"  median path        max DD hits -20% at risk {r1*100:5.1f}%  "
              f"-> CAGR {c1*100:6.1f}%")
    else:
        print(f"  median path        already worse than -20% at the lowest risk "
              f"tested ({D.risk.iloc[0]*100:.0f}%, median {D.bmed.iloc[0]*100:.1f}%)")
    r2, c2 = solve("p20", 0.10)
    if r2:
        print(f"  1-in-10 confidence P(DD>20%) = 10% at risk {r2*100:5.1f}%  "
              f"-> CAGR {c2*100:6.1f}%")
    else:
        print(f"  1-in-10 confidence never reached on this ladder "
              f"(lowest P(DD>20%) is {D.p20.min()*100:.0f}% at {D.risk.iloc[0]*100:.0f}% risk)")

    print(f"\n  at the DEPLOYED 14.4% risk: P(drawdown worse than 20%) = "
          f"{D.p20[np.isclose(D.risk, 0.144)].iloc[0]*100:.0f}%")
    print("\ndone: the honest gate")
