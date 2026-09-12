"""
S109b - Closing the event-bar line: does the repaired sleeve blend?

S109 confirmed S108e's diagnosis and then overshot it. On the 3-bars-a-day event
panel the trailing-Calmar ranking is not merely uninformative, it is
ANTI-informative: top-3 sits 1.9 sd BELOW random-3, top-12 4.6 sd below
random-12, top-50 4.7 sd below random-50, and the gate reading rises monotonically
with k from 27.6% at three configurations to 57.8% at all two hundred. Selecting
the highest trailing Calmar on a panel whose training Calmars run 42-69 is
selecting for overfitting, and doing less of it helps every time.

The same rule on the clock panel is strongly POSITIVE - top-3 at 167.4%, 3.6 sd
above random-3, and falling monotonically to 110.0% at k=200. One selection rule,
two samplings of the same five signals, opposite signs.

So the event sleeve can be repaired by removing the selection: 27.6% becomes
57.8%, and Sharpe 1.22 becomes 1.80. This file asks the only question left on
this line. **A repaired sleeve is still less than half of V7. Does it blend?**

S102c's bar is unchanged and it is the thing to watch:

        S2  >  S1 * (sqrt(2 + 2*rho) - 1)

Everything here reuses S109's cached per-configuration returns, so nothing new is
simulated and the two files cannot disagree.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

RISK, REF = 0.08, 0.01
Q = None


def load(tag):
    z = np.load(f"/home/user/quant/results/s109_{tag}.npz", allow_pickle=True)
    return z["R"].item()


def book(C, k):
    """top-k at risk/k per quarter, from the cached ranked order. Kept as a dict
    of quarter -> series rather than one array, because the clock and event
    panels do not produce identical day counts inside a quarter and concatenating
    first would silently misalign the two books by a day here and there."""
    qs = {}
    for (s, cfg) in C:
        qs.setdefault(s, []).append(cfg)
    return {s: np.mean([C[(s, c)] for c in qs[s][:k]], axis=0) * (RISK / REF)
            for s in sorted(qs)}


def flat(B):
    return np.concatenate([B[s] for s in sorted(B)])


def mix(A, B, w=0.5):
    """Blend two books quarter by quarter, padding to the longer of the two."""
    out = []
    for s in sorted(A):
        a, b = A[s], B[s]
        n = max(len(a), len(b))
        pa = np.r_[a, np.zeros(n - len(a))]
        pb = np.r_[b, np.zeros(n - len(b))]
        out.append(w * pa + (1 - w) * pb)
    return np.concatenate(out)


def halves(a):
    h = len(a) // 2
    return (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
            at_gate(a[h:])["cagr"] * 100)


def row(tag, a, base=None):
    st = stats_of(a); b = bootstrap_dd(a, n=3000)
    f_, h1, h2 = halves(a)
    d = "" if base is None else f"{f_-base[0]:+9.1f}{min(h1-base[1], h2-base[2]):+12.1f}"
    print(f"{tag:>30}{st['sharpe']:7.2f}{st['calmar']:7.2f}{st['dd']*100:8.1f}%"
          f"{b['dd_median']*100:8.1f}%{b['p_dd_worse_than_20']*100:6.0f}%"
          f"{f_:9.1f}%{h1:9.1f}%{h2:9.1f}%{d}", flush=True)
    return (f_, h1, h2)


if __name__ == "__main__":
    CK, E3, E4 = load("clock"), load("3"), load("4")
    v7 = book(CK, 3)                                   # the deployed book
    ev = {"3/day no-sel": book(E3, 200), "3/day top-50": book(E3, 50),
          "4/day no-sel": book(E4, 200), "4/day top-50": book(E4, 50)}

    print(f"{'book':>30}{'Shp':>7}{'Clm':>7}{'realDD':>9}{'bootDD':>9}{'P>20%':>7}"
          f"{'full':>9}{'1st h':>9}{'2nd h':>9}{'vs V7':>9}{'worse half':>12}")
    base = row("V7 (clock top-3)", flat(v7))
    for t, a in ev.items():
        row(f"event {t} alone", flat(a), base)
    print()
    for t, a in ev.items():
        row(f"V7 + event {t}", mix(v7, a), base)

    print("\nthe admission bar, against the repaired sleeves")
    f7 = flat(v7)
    s7 = stats_of(f7)["sharpe"]
    print(f"{'sleeve':>16}{'Sharpe':>9}{'rho vs V7':>12}{'needs':>8}{'clears':>8}")
    for t, a in ev.items():
        pair = np.column_stack([mix(v7, a, 1.0), mix(v7, a, 0.0)])
        rho = float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1])
        se = stats_of(flat(a))["sharpe"]
        nd = s7 * (np.sqrt(2 + 2 * rho) - 1)
        print(f"{t:>16}{se:9.2f}{rho:12.3f}{nd:8.2f}"
              f"{('YES' if se > nd else 'no'):>8}")
    print("\ndone: event line closed")
