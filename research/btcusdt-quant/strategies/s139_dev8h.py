"""
S139 - Developing the best non-V7 book: crowding on an 8-hour clock + exposure.

WHERE IT STANDS
---------------
S138 landed the best standalone result in this study:

    crowding 8h + exposure timing    Sharpe 1.46   30.0% at the gate   DD -15.1%
    at double the cost assumption    Sharpe 1.28   23.5%

and it survived both traps set for it. The cost sweep says the frequency edge is
real but not free (8-hourly beats daily at 8, 16 and 30 bps and LOSES at 50), and
the clock-phase sweep says it is not S101's artefact - all eight offsets of the
8-hour grid land between 13.5% and 21.6%, median 16.6%, sd 2.5. Four-hourly
collapses at every phase, so the sweet spot is specifically eight hours, which is
exactly the funding settlement interval: the clock matches when information
actually arrives.

WHAT IS DEVELOPED, AND THE ONE THING THAT WORRIES ME
-----------------------------------------------------
The halves read 84% and 21%. That is a large decay and S119 watched the daily
crowding book fade monotonically to nothing over the same period, so the first
thing this file does is print the year-by-year - before any development, because
if the recent years are empty then tuning the thing is polishing a corpse.

Then three levers, each tested on its own:

    SELECTIVITY   S119 found a dead band was the single biggest lever on the
                  daily book's drawdown. It has never been tried on the 8h one.
    RISK          the target volatility, which the gate normalises away but
                  which interacts with the leverage cap.
    A THIRD LEG   S136 found adding sleeves to a good pair DILUTES it - the
                  three-way ensemble fell to 19.1% from 29.1%. Retested here on
                  the 8h clock rather than assumed to carry over.

Everything charged at 16bps round trip and re-checked at 30.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s138_fast as F8
import strategies.s135_more as S135
import strategies.s136_ensemble as S136
from strategies.s96_rank import stats_of

gate, score, HDR = S135.gate, S136.score, S136.HDR
RULE, BYR = "8h", 1095.75


def build():
    px, fd, F = F8.panel(RULE, BYR)
    sg = F8.csig(F, BYR)
    lp = np.log(px)
    span = lambda d: max(2, int(d * BYR / 365.25))
    agree3 = (sum(np.sign(lp.ewm(span=span(a), adjust=False).mean()
                          - lp.ewm(span=span(b), adjust=False).mean())
                  for a, b in ((8, 32), (16, 64), (32, 128))) / 3)
    ex = agree3.clip(0, 1).shift(1).fillna(0.0)
    return px, fd, F, sg, ex, lp, span


def run(px, fd, sig, tv=0.30, rt=16.0, thr=0.0, max_lev=3.0, band=0.10):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = np.maximum(r.ewm(halflife=max(4, int(32 * BYR / 365.25)), adjust=False)
                    .std().shift(1).bfill().to_numpy() * np.sqrt(BYR), 0.05)
    s = np.asarray(sig, float).copy()
    s[np.abs(s) < thr] = 0.0
    want = np.clip(s * tv / rv, -max_lev, max_lev)
    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * px.pct_change().fillna(0.0).to_numpy() \
        - pos * fd.to_numpy() - turn * (rt / 2.0) / 1e4
    n = pd.Series(net, index=px.index)
    nd = n.groupby(n.index.normalize()).sum()
    p = pd.Series(pos, index=px.index)
    return nd, p.groupby(p.index.normalize()).last().to_numpy()


if __name__ == "__main__":
    px, fd, F, sg, ex, lp, span = build()
    combo = (sg + ex) / 2.0
    print("S139 - developing crowding-8h + exposure\n")

    print("0. IS IT DECAYING? net return by calendar year, before any development")
    nd, _ = run(px, fd, combo)
    print(f"   {'year':>6}{'8h combo':>12}{'exposure of days':>19}")
    for y, g in nd.groupby(nd.index.year):
        if len(g) < 60:
            continue
        print(f"   {y:>6}{(np.prod(1 + g.to_numpy()) - 1) * 100:>11.1f}%"
              f"{(g != 0).mean() * 100:>17.0f}%")

    print("\n1. SELECTIVITY — a dead band, the biggest lever on the daily book")
    print(HDR)
    for thr in (0.0, 0.10, 0.20, 0.30, 0.45):
        score(f"threshold {thr:.2f}", *run(px, fd, combo, thr=thr))

    print("\n2. RISK DIAL, at the best threshold so far")
    print(HDR)
    for tv in (0.20, 0.30, 0.50, 0.80):
        score(f"target vol {tv*100:.0f}%", *run(px, fd, combo, tv=tv, max_lev=5.0))

    print("\n3. A THIRD LEG — S136 found extra sleeves dilute. retested on 8h")
    print(HDR)
    d = pd.read_parquet("/home/user/quant/data/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True); d = d.set_index("dt")
    hi = d["high"].resample(RULE).max().reindex(px.index)
    lo = d["low"].resample(RULE).min().reindex(px.index)
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=span(20), adjust=False).mean()
    bo = S135.donchian(px, hi, lo, span(55), 3.0, atr)
    score("crowd + expos (reference)", *run(px, fd, combo))
    score("crowd + expos + breakout", *run(px, fd, (sg + ex + bo) / 3.0))
    score("crowd + breakout", *run(px, fd, (sg + bo) / 2.0))

    print("\n4. THE BOOK AT BOTH COST ASSUMPTIONS, with the best threshold")
    print(HDR)
    for thr in (0.0, 0.20):
        for rt in (16.0, 30.0, 50.0):
            score(f"thr {thr:.2f}, rt {rt:.0f}bps",
                  *run(px, fd, combo, thr=thr, rt=rt))
    print("\ndone: S139")
