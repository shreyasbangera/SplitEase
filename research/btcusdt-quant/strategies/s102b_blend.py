"""
S102b - The only constructive reading of S102: can two bar lengths be held at once?

Stage A closed the structural question and opened a smaller one. 12h is a spike:
6h reaches 37% at the gate, 8h 17%, 24h 7%, against 12h's 110% on the same fixed
configurations. None of them is a book worth trading on its own.

But the target does not need another book that is as good as V7. It needs a
higher CAGR at a FIXED -20% drawdown, and that is a statement about Calmar, not
about CAGR. A weak sleeve with a genuinely different return stream raises the
blend's Calmar even while lowering its mean - which is the mechanism that made
S86's top-3 blend worth holding in the first place.

So the question is not "is 6h good" - it is not - but "is 6h DIFFERENT". Two
prior results bracket the answer and disagree:

    S42   phase-shifted 12h books correlate 0.70-0.82: nothing to diversify
    S98   disjoint-signal books correlate 0.068, and degrade monotonically

This is neither. Same five signals, same thresholds, same shape - a different
clock. If the correlation lands near S42's the idea is dead on arrival; if it
lands nearer S98's there is something to size.

Measured exactly the way S87 blends configurations: each sleeve is re-simulated
at risk/n and the daily returns summed, rather than averaging returns computed
at full risk, so the position sizing is the sizing that would actually be used.
The comparison is at the -20% gate throughout, because that is the only place a
weaker-but-different sleeve can win.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import itertools
import numpy as np, pandas as pd

import strategies.s69_calsel as S69
import strategies.s102_horizon as S102
from research.harness import OOS_END
from strategies.s96_rank import at_gate, stats_of

HORIZONS = ("6h", "8h", "12h", "24h")
CFGS = [(2.0, 3.0, 2.0, 21), (1.0, 3.0, 2.0, 21)]
RISK = 0.08


_G = {}


def use(tf):
    """Install horizon `tf`, building its panel at most once per process.

    grid_h() deliberately drops the harness cache so the panel is rebuilt under
    the right z-window scaling; that is right once and wasteful seven times, and
    the panel build is the whole cost of this file.
    """
    if tf not in _G:
        _G[tf] = S102.grid_h(tf)
    S102._K["k"] = 12.0 / S102.hours(tf)
    S102.install(_G[tf])


def sleeves(cfg, risk):
    """One daily-return Series per horizon, each simulated at `risk`."""
    out = {}
    for tf in HORIZONS:
        use(tf)
        out[tf] = S69.daily(S69.sim(cfg, S102.FULL_START, OOS_END, risk))
    return out


def combine(parts):
    """Sum daily returns across sleeves, exactly as S87.blend does."""
    return pd.DataFrame(parts).fillna(0.0).sum(axis=1)


def line(tag, r, base=None):
    a = np.asarray(r, float)
    s, gt = stats_of(a), at_gate(a)
    d = "" if base is None else f"{gt['cagr']*100 - base:+8.1f}"
    print(f"{tag:>26}{s['cagr']*100:8.1f}%{s['dd']*100:7.1f}%{s['sharpe']:7.2f}"
          f"{s['calmar']:7.2f}{gt['cagr']*100:10.1f}%{d}", flush=True)
    return gt["cagr"] * 100


if __name__ == "__main__":
    for cfg in CFGS:
        print(f"\n===== config exp {cfg[0]}  {cfg[1]}ATR x{cfg[2]}R  {cfg[3]}d, "
              f"total risk {RISK*100:.0f}% split evenly across the sleeves held")
        full = sleeves(cfg, RISK)

        print("\ndaily-return correlation between horizons")
        C = pd.DataFrame(full).fillna(0.0).corr()
        print(C.round(3).to_string())

        print(f"\n{'book':>26}{'CAGR':>9}{'MaxDD':>8}{'Shp':>7}{'Clm':>7}"
              f"{'at -20%':>10}{'vs 12h':>9}")
        base = line("12h alone", full["12h"])
        for n in (2, 3, 4):
            parts = sleeves(cfg, RISK / n)          # depends on n, not on which
            for combo in itertools.combinations(HORIZONS, n):
                if "12h" not in combo:
                    continue
                line(" + ".join(combo), combine({t: parts[t] for t in combo}), base)
    print("\ndone: horizon blend")
