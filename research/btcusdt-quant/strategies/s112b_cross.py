"""
S112b - Cross-checking S112 without the bootstrap's assumptions.

S112 says the deployed book's realised drawdown is a favourable path: at 8% risk
the realised max is -12.3% while the block bootstrap's median is -18.1% and
P(worse than 20%) is 34%.

That rests on a model. `bootstrap_dd` resamples 5-day blocks, which preserves
dependence up to a week and destroys everything longer. If V7's equity curve is
mean-reverting over months - drawdowns that systematically recover, which S103
says is exactly what its deepest episodes did - then the bootstrap will
manufacture drawdowns the real process would have healed, and overstate the risk.
The objection is real and the number should not be reported without testing it.

So the same question is asked with no resampling at all: **the worst drawdown
inside each rolling window of the ACTUAL series, in its actual order.** If the
realised full-sample maximum was lucky, then many individual windows will contain
drawdowns close to or worse than it, since a longer run is a maximum over more
windows. If the bootstrap is inventing risk, the real windows will be tame.

Reported per risk level, so it can be read against S112's table row by row.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s87_combined as S87
import strategies.s110_meta as M
from research.robust import bootstrap_dd
from strategies.s96_rank import stats_of

RISKS = (0.06, 0.08, 0.10, 0.144)
WINDOWS = (180, 365, 540)


def maxdd(r):
    e = np.cumprod(1 + np.asarray(r, float))
    return float((e / np.maximum.accumulate(e) - 1).min())


def rolling_dd(a, w):
    return np.array([maxdd(a[i:i + w]) for i in range(0, len(a) - w + 1, 5)])


if __name__ == "__main__":
    M.use_clock()
    R = S87.rankings()
    print("worst drawdown inside each rolling window of the REAL series, no resampling\n")
    print(f"{'risk':>7}{'window':>9}{'n win':>7}{'best':>9}{'median':>9}{'worst':>9}"
          f"{'% of windows':>14}{'  full-sample realised':>23}")
    for rk in RISKS:
        r, _ = S87.blend(R, 3, rk)
        a = np.asarray(r, float)
        full = maxdd(a)
        for w in WINDOWS:
            d = rolling_dd(a, w)
            if not len(d):
                continue
            print(f"{rk*100:6.1f}%{w:9d}{len(d):7d}{d.max()*100:8.1f}%"
                  f"{np.median(d)*100:8.1f}%{d.min()*100:8.1f}%"
                  f"{(d < -0.20).mean()*100:12.0f}%  "
                  f"{full*100:20.1f}%", flush=True)
        b = bootstrap_dd(a, n=4000)
        print(f"{'':7}{'bootstrap':>9}{'':7}{'':9}{b['dd_median']*100:8.1f}%"
              f"{b['dd_p05']*100:8.1f}%{b['p_dd_worse_than_20']*100:12.0f}%\n")

    print("reading: the '% of windows' column is the fraction of real, in-order")
    print("windows whose own worst drawdown breached 20%. If it tracks the")
    print("bootstrap's P(DD>20%), the bootstrap is not inventing risk.")
    print("\ndone: cross-check")
