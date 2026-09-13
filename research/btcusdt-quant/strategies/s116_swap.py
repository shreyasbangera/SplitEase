"""
S116 - Replace the defensive gate with an offensive complement.

S115 built V7's opposite and failed, but HOW it failed is a lead. Six textbook
trend rules all correlated POSITIVELY with V7 when a fade-the-crowd book should
have correlated negatively, and the reason offered there was **V7's trend
gate**: S84 bolted it on after finding the worst drawdown was 29 shorts into the
strongest trend of the sample, and it works by suppressing shorts above a long
moving average - which makes V7 partly trend-aligned by construction.

If that diagnosis is right, the gate is standing in the way of its own
replacement. And the gate is a **veto**: in a melt-up it steps aside and earns
nothing, where a trend sleeve beside an ungated book would take the other side
of that regime and be paid for it.

    incumbent   V7 = top-3 of 200 configurations, gate menu {none,100,150,200,300}
    candidate   ungated V7 (top-3 of the 40 no-gate cells) + a trend sleeve

The first question is the only one that matters and it is cheap: with the gate
gone, does the crowding book correlate negatively with trend? If not, the
mechanism S115 blamed was not the mechanism.

A NOTE ON ALIGNMENT, BECAUSE THE FIRST VERSION OF THIS FILE GOT IT WRONG
------------------------------------------------------------------------
V7's returns begin 2022-03, after the selection's 12-month lookback. The trend
sleeves begin 2021-03. The first draft compared them POSITIONALLY - `v7[:n]`
against `trend[:n]` - which lines V7's 2022 up against trend's 2021 and drove
every correlation to about zero. It looked like a clean result and it was a
year of misalignment.

S115's own method was not clean either: it built a DataFrame from two series of
different span and filled the gap with zeros, which adds a year of (0, r) pairs
and biases toward zero. Measured on the date INTERSECTION, the real figures are
+0.197 to +0.413 against S115's reported +0.184 to +0.382. S115's conclusion
survives; its numbers were mildly understated.

Everything here is joined on dates with an inner join, and the overlap is
printed so the alignment can be seen rather than trusted.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s110_meta as M
import strategies.s115_trend as T
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

RISK, K = 0.08, 3


def ser(x):
    return pd.Series(np.asarray(x, float), index=pd.to_datetime(x.index))


def ungated(R):
    """The same selection rule on a menu with the gate removed.

    Not a differently fitted book: the quarterly ranking is untouched and simply
    restricted to the 40 no-gate configurations before the top three are taken.
    """
    return [(s, e, [c for c in cs if c[4] == 0]) for s, e, cs in R]


def rho(a, b):
    J = pd.concat([a, b], axis=1, join="inner").dropna()
    return float(J.corr().iloc[0, 1]), len(J)


def halves(a):
    h = len(a) // 2
    return (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
            at_gate(a[h:])["cagr"] * 100)


def row(tag, s, base=None):
    a = np.asarray(s, float)
    st = stats_of(a); b = bootstrap_dd(a, n=3000, block=90)
    f_, h1, h2 = halves(a)
    d = "" if base is None else f"{f_-base[0]:+9.1f}{min(h1-base[1], h2-base[2]):+12.1f}"
    print(f"{tag:>28}{st['cagr']*100:8.1f}%{st['dd']*100:8.1f}%{b['dd_median']*100:9.1f}%"
          f"{st['sharpe']:7.2f}{st['calmar']:7.2f}{f_:9.1f}%{h1:9.1f}%{h2:9.1f}%{d}",
          flush=True)
    return (f_, h1, h2)


if __name__ == "__main__":
    M.use_clock()
    g = S69.ctx()["g"]
    R = S87.rankings()

    v7 = ser(S87.blend(R, K, RISK)[0])
    ung = ser(S87.blend(ungated(R), K, RISK)[0])

    from collections import Counter
    cg = Counter(c[4] for _, _, cs in R for c in cs[:K])
    print(f"gate choices in the incumbent's top-{K}: {dict(sorted(cg.items()))}")
    print(f"  (0 = no gate)  -> the gate is actually used in "
          f"{sum(v for k, v in cg.items() if k)}/{sum(cg.values())} picks\n")

    print("1. with the gate gone, does the correlation turn negative?\n")
    print(f"{'trend sleeve':>14}{'Sharpe':>9}{'overlap':>9}{'rho GATED':>12}"
          f"{'rho UNGATED':>14}{'change':>9}")
    sl = {}
    for tag, u in T.signals(g).items():
        d = ser(S69.daily(T.run(g, u, RISK)))
        sl[tag] = d
        rg, n = rho(v7, d)
        ru, _ = rho(ung, d)
        print(f"{tag:>14}{stats_of(np.asarray(d, float))['sharpe']:9.2f}{n:9d}"
              f"{rg:12.3f}{ru:14.3f}{ru-rg:+9.3f}")

    print(f"\n2. the two books on their own\n")
    print(f"{'book':>28}{'CAGR':>9}{'realDD':>8}{'medDD':>9}{'Shp':>7}{'Clm':>7}"
          f"{'full':>9}{'1st h':>9}{'2nd h':>9}{'vs V7':>9}{'worse half':>12}")
    base = row("V7 (gated) - incumbent", v7)
    row("V7 ungated, top-3 of 40", ung, base)

    print(f"\n3. the swap: ungated book + a trend sleeve, 50/50\n")
    ungh = ser(S87.blend(ungated(R), K, RISK / 2)[0])
    for tag, u in T.signals(g).items():
        half = ser(S69.daily(T.run(g, u, RISK / 2)))
        J = pd.concat([ungh, half], axis=1, join="inner").dropna()
        row(f"ungated + {tag}", J.iloc[:, 0] + J.iloc[:, 1], base)
    print("\ndone: gate swap")
