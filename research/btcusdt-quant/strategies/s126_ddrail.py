"""
S126 - A drawdown rail. The one risk technique this study kept naming and never
built.

THE ARGUMENT FOR IT, WHICH IS DIFFERENT FROM S123's
----------------------------------------------------
S123 sized off a better VOLATILITY forecast and it failed, for a reason worth
keeping: the gate rewards shrinking before LOSSES, and volatility and losses
coincide less than the sizing literature assumes. A drawdown rail does not
forecast anything at all. It responds to losses that have already happened,
which removes the forecasting problem entirely.

It should matter specifically at the HONEST gate. That gate scales a book until
the *median bootstrapped* drawdown is 20%, so it is set by the middle of the
drawdown distribution and dragged by its tail. A rail truncates the left tail
without touching paths that never draw down, so it should cut the median
drawdown by more than it cuts return - and every unit of drawdown saved is a
unit of size bought back, linearly.

The standard objection is real and is why this is a test rather than a
deployment: a rail de-risks at the bottom and is small through the recovery.
Whether truncating the tail beats missing the rebound is exactly the empirical
question, and it is answered here rather than assumed in either direction.

WHAT IT IS
----------
    w_t = clip(1 - |D_t| / LIMIT, FLOOR, 1)  ^ SHAPE

where D_t is the running drawdown computed from returns strictly BEFORE t. At
zero drawdown the book is full size; at LIMIT it is at FLOOR. SHAPE 1 is linear,
2 de-risks later and harder. **Nothing here is fitted to a result** - the grid is
round numbers, every cell is printed, and the honest gate with S124's guard is
the only figure quoted.

Applied as an OVERLAY to every stream this study has produced, because a rail is
a property of a return series and not of any one strategy's internals. If it
works it works on all of them, and if it only works on one it is a fluke.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s124_ceiling as C
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of


def rail(r, limit, floor=0.0, shape=1.0):
    """Apply a causal drawdown rail to a daily return series.

    The weight in force on day t uses only returns through t-1: the equity path
    is stepped forward one day at a time and the weight is computed BEFORE that
    day's return is applied. Computing the drawdown including today's return and
    then using it to size today is the obvious way to get a beautiful and
    entirely fake result.
    """
    r = np.asarray(r, float)
    n = len(r)
    out = np.empty(n)
    eq, peak = 1.0, 1.0
    for i in range(n):
        dd = eq / peak - 1.0                       # known at the START of day i
        w = np.clip(1.0 + dd / limit, floor, 1.0) ** shape
        out[i] = w * r[i]
        eq *= (1.0 + w * r[i])
        peak = max(peak, eq)
    return out


def honest_gate(a, nboot=1200, block=90, seed=0, tol=0.004):
    """CAGR at the size where the MEDIAN bootstrapped drawdown is -20%.

    Paths are resampled once and re-used across the bisection, and the result is
    refused unless the achieved median drawdown really is -20% (S124's guard).
    """
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan
    rng = np.random.default_rng(seed)
    T = len(a)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, max(T - block, 1), (nboot, nb))
    idx = np.clip((starts[:, :, None] + np.arange(block)[None, None, :])
                  .reshape(nboot, -1)[:, :T], 0, T - 1)

    def med(s):
        e = np.cumprod(1.0 + a[idx] * s, axis=1)
        return float(np.median((e / np.maximum.accumulate(e, axis=1) - 1).min(axis=1)))

    lo, hi = 1e-3, 40.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if med(mid) < -0.20:
            hi = mid
        else:
            lo = mid
    s = (lo + hi) / 2
    if abs(med(s) + 0.20) > tol:
        return np.nan
    return stats_of(a * s)["cagr"] * 100


if __name__ == "__main__":
    S = C.streams()
    keep = {k: v for k, v in S.items() if k in ("V7", "crowding", "trend", "hold")}
    print("S126 - does a drawdown rail buy size at the honest gate?\n")
    print("the rail is causal: the weight on day t uses the drawdown as it stood "
          "at the\nSTART of day t. gate figures refused unless the median "
          "bootstrapped drawdown\nreally reaches -20%.\n")

    for name, r in keep.items():
        a = np.asarray(r, float); a = a[np.isfinite(a)]
        base = honest_gate(a)
        st = stats_of(a)
        print(f"=== {name}  (Sharpe {st['sharpe']:.2f}, realised DD "
              f"{st['dd']*100:.1f}%)   no rail: "
              + ("n/a" if not np.isfinite(base) else f"{base:.1f}%"))
        print(f"   {'limit':>7}{'floor':>7}{'shape':>7}{'Shp':>7}{'realDD':>8}"
              f"{'medDD':>8}{'honest gate':>13}{'vs no rail':>12}")
        rows = []
        for limit in (0.10, 0.15, 0.20, 0.30):
            for floor in (0.0, 0.25, 0.50):
                for shape in (1.0, 2.0):
                    b = rail(a, limit, floor, shape)
                    g = honest_gate(b)
                    sb = stats_of(b)
                    bd = bootstrap_dd(b, n=600, block=90)
                    d = ("" if not (np.isfinite(g) and np.isfinite(base))
                         else f"{(g/base - 1)*100:>+11.0f}%")
                    gs = "n/a" if not np.isfinite(g) else f"{g:>12.1f}%"
                    rows.append((g, limit, floor, shape))
                    print(f"   {limit*100:>6.0f}%{floor:>7.2f}{shape:>7.0f}"
                          f"{sb['sharpe']:>7.2f}{sb['dd']*100:>7.1f}%"
                          f"{bd['dd_median']*100:>7.1f}%{gs}{d}")
        ok = [x for x in rows if np.isfinite(x[0])]
        if ok and np.isfinite(base):
            b = max(ok)
            verdict = "HELPS" if b[0] > base * 1.02 else "does not help"
            print(f"   -> best rail {b[0]:.1f}% against {base:.1f}% unrailed: "
                  f"{verdict}\n")
        else:
            print()
    print("done: drawdown rail")
