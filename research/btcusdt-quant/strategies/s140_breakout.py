"""
S140 - Developing the breakout family, because it is the only thing still paying.

THE REASON THIS AND NOT SOMETHING ELSE
---------------------------------------
Net return by year, every mechanism in this study at the same risk:

                        2022    2023    2024    2025    2026   last 2y
    crowding           12.6%   17.4%    9.0%    1.1%    7.4%      4.7%
    exposure timing    -6.8%   34.5%   39.0%  -10.2%   -1.6%      4.4%
    BREAKOUT            3.7%   12.6%   15.9%    6.5%    5.2%     16.9%
    buy & hold        -38.9%   79.4%   61.8%  -12.6%  -10.6%      1.7%

**Breakout is the only book that has not decayed.** Positive every year since
2021, and it made money through 2025 and 2026 while the asset itself fell 12.6%
and 10.6%. Everything else in this study is living off 2020-2024.

That matters more than the headline figure. A 30% book whose last two years are
flat is a description of the past; a 13% book still paying is a strategy. So the
development effort goes here.

ITS TWO WEAKNESSES, AND WHAT IS TRIED
--------------------------------------
    LOW RETURN     13.1% at the gate. It is flat most of the time by design, so
                   the capital is idle. Tried: volatility-expansion filters to
                   take only the breakouts worth taking, pyramiding into
                   continuation, and a wider set of channel definitions.
    FEW TRADES     32 completed round trips, which fails the brief outright.
                   S135 already showed naive multi-horizon ensembling makes this
                   WORSE, not better (13.1% -> 10.0%), so that is not retried.

FOUR CHANNEL FAMILIES, because "breakout" is not one thing:
    donchian   N-day high/low - pure price extremes
    keltner    EMA +/- k x ATR - volatility-scaled bands around a trend
    bollinger  SMA +/- k x standard deviation - statistical extremes
    range      breakout of the recent RANGE relative to its own history, which
               fires on expansion rather than on level

Each with an ATR trailing exit. Every cell prints its year-by-year durability,
because on this evidence durability is the property worth selecting for and the
full-sample number is the one that misleads.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s135_more as S135
import strategies.s136_ensemble as S136
from strategies.s96_rank import stats_of

gate, run, score, HDR = S135.gate, S135.run, S136.score, S136.HDR


def trail(px, entry_long, entry_short, atr, k):
    """Shared engine: enter on a trigger, exit on an ATR trail from the extreme."""
    c, a = px.to_numpy(), atr.to_numpy()
    el, es = np.asarray(entry_long, bool), np.asarray(entry_short, bool)
    s = np.zeros(len(px)); cur = 0.0; peak = 0.0
    for i in range(len(px)):
        if not np.isfinite(a[i]):
            continue
        if cur == 0.0:
            if el[i]: cur, peak = 1.0, c[i]
            elif es[i]: cur, peak = -1.0, c[i]
        elif cur > 0:
            peak = max(peak, c[i])
            if c[i] < peak - k * a[i]: cur = 0.0
        else:
            peak = min(peak, c[i])
            if c[i] > peak + k * a[i]: cur = 0.0
        s[i] = cur
    return pd.Series(s, index=px.index).shift(1).fillna(0.0)


def yearly(net):
    out = []
    for y, g in net.groupby(net.index.year):
        if len(g) < 60:
            continue
        out.append((y, (np.prod(1 + g.to_numpy()) - 1) * 100))
    return out


def last2(net):
    r = net[net.index >= net.index.max() - pd.Timedelta(days=730)]
    if len(r) < 200:
        return np.nan
    return (np.prod(1 + r.to_numpy()) ** (365.25 / len(r)) - 1) * 100


if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    lp = np.log(px)
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    rng = (hi - lo)

    print("S140 - developing the breakout family\n")
    print("1. FOUR CHANNEL DEFINITIONS — 'breakout' is not one thing")
    print(HDR + f"{'last2y':>9}")
    cands = {}
    for N in (34, 55, 89):
        up = hi.rolling(N).max().shift(1); dn = lo.rolling(N).min().shift(1)
        cands[f"donchian {N}"] = (px > up, px < dn)
        ma = px.rolling(N).mean().shift(1); sd = px.rolling(N).std().shift(1)
        cands[f"bollinger {N}"] = (px > ma + 2 * sd, px < ma - 2 * sd)
        e = px.ewm(span=N, adjust=False).mean().shift(1)
        cands[f"keltner {N}"] = (px > e + 2 * atr.shift(1), px < e - 2 * atr.shift(1))
        rq = rng.rolling(N).quantile(0.9).shift(1)
        cands[f"range-exp {N}"] = ((rng > rq) & (px > px.shift(1)),
                                   (rng > rq) & (px < px.shift(1)))
    rows = []
    for name, (el, es) in cands.items():
        sig = trail(px, el.fillna(False), es.fillna(False), atr, 3.0)
        net, pos = run(px, fd, sig, 0.30)
        g = score(name, net, pos)
        l2 = last2(net)
        print(f"{'':>96}{l2:>8.1f}%" if np.isfinite(l2) else "")
        rows.append((name, g, l2, sig))

    print("\n2. VOLATILITY-EXPANSION FILTER on the best channel")
    print(HDR + f"{'last2y':>9}")
    ok = [r for r in rows if np.isfinite(r[1])]
    bestname, _, _, bestsig = max(ok, key=lambda r: (r[2] if np.isfinite(r[2]) else -9e9))
    print(f"   (selecting on LAST-2-YEAR durability, not full sample: {bestname})")
    volr = (tr.rolling(10).mean() / tr.rolling(60).mean()).shift(1)
    for thr in (0.0, 1.0, 1.2):
        s = bestsig * (volr > thr).astype(float) if thr > 0 else bestsig
        net, pos = run(px, fd, s, 0.30)
        score(f"{bestname}, vol-expansion > {thr:.1f}", net, pos)
        l2 = last2(net)
        print(f"{'':>96}{l2:>8.1f}%" if np.isfinite(l2) else "")

    print("\n3. THE BEST BOOK, YEAR BY YEAR")
    net, pos = run(px, fd, bestsig, 0.30)
    print(f"   {bestname}: " + "  ".join(f"{y} {v:+.1f}%" for y, v in yearly(net)))
    print(f"   gate {gate(net.to_numpy()):.1f}%, "
          f"last 2y annualised {last2(net):.1f}%, "
          f"Sharpe {stats_of(net.to_numpy())['sharpe']:.2f}")

    print("\n4. RISK DIAL on it")
    print(HDR + f"{'last2y':>9}")
    for tv in (0.30, 0.50, 0.80):
        net, pos = run(px, fd, bestsig, tv, max_lev=5.0)
        score(f"target vol {tv*100:.0f}%", net, pos)
        l2 = last2(net)
        print(f"{'':>96}{l2:>8.1f}%" if np.isfinite(l2) else "")
    print("\ndone: S140")
