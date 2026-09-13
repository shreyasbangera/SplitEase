"""
S141 - Developing the range-expansion book, the only one still paying.

S140 established durability as the property worth selecting for: every mechanism
in this study except breakout is living off 2020-2024, while `range-exp 89` is
positive in all seven years and annualises 19.9% over the trailing two while the
asset itself fell 12.6% and 10.6%.

Its weakness is the level: 12.5% at the gate, Sharpe 0.80, and a -31.2% drawdown
at 30% target vol, which is deep for that Sharpe. The gate scales it down to -20%
and that is where the 12.5% comes from. So the levers worth trying are the ones
that change the SHAPE of its losses, not the ones that change its signal.

FIVE LEVERS
    trigger      the range percentile that counts as expansion. 0.90 was picked
                 for being a round number, never tested.
    lookback     55 / 89 / 144 days of range history.
    exit         the ATR trail multiple. This is the drawdown lever - a tighter
                 trail cuts losses and cuts winners, and which dominates is an
                 empirical question.
    pyramiding   adding to a position that keeps going. The classic turtle
                 construction, and the one development idea that raises return
                 without touching the entry rule.
    asymmetry    BTC drifts up over this sample, so equal-sized longs and shorts
                 are not obviously right. Tested in both directions rather than
                 assumed.

A WARNING ABOUT WHAT I AM SELECTING ON
---------------------------------------
S140 chose this book on trailing-two-year performance. That is a 730-day sample,
and selecting on it is the same winner's-curse this log has caught four times
(S106b, S109, S118b, S131). So every cell below prints BOTH the full-sample gate
and the trailing-two-year figure, and the full sweep is shown rather than a
winner. If a lever only helps the recent window it is noise, and the way to see
that is to have both columns side by side.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s135_more as S135
import strategies.s136_ensemble as S136
import strategies.s140_breakout as B
from strategies.s96_rank import stats_of

gate, run, score = S135.gate, S135.run, S136.score


def rex(px, hi, lo, rng, N, q):
    r = rng.rolling(N).quantile(q).shift(1)
    return (rng > r) & (px > px.shift(1)), (rng > r) & (px < px.shift(1))


def trail_pyr(px, el, es, atr, k, pyr=0, step=1.0, cap=3.0):
    """ATR trailing exit, optionally adding to a position that keeps running.

    `pyr` is how many times the book may add; each add happens once price has
    moved another `step` x ATR in favour of the position, and total size is
    capped at `cap` units. Adds use the SAME trail from the running extreme, so
    a pyramided position exits all at once rather than piecewise.
    """
    c, a = px.to_numpy(), atr.to_numpy()
    el, es = np.asarray(el, bool), np.asarray(es, bool)
    s = np.zeros(len(px))
    cur, peak, base, adds = 0.0, 0.0, 0.0, 0
    for i in range(len(px)):
        if not np.isfinite(a[i]):
            continue
        if cur == 0.0:
            if el[i]: cur, peak, base, adds = 1.0, c[i], c[i], 0
            elif es[i]: cur, peak, base, adds = -1.0, c[i], c[i], 0
        elif cur > 0:
            peak = max(peak, c[i])
            if adds < pyr and c[i] >= base + (adds + 1) * step * a[i]:
                adds += 1
                cur = min(cur + 1.0, cap)
            if c[i] < peak - k * a[i]:
                cur, adds = 0.0, 0
        else:
            peak = min(peak, c[i])
            if adds < pyr and c[i] <= base - (adds + 1) * step * a[i]:
                adds += 1
                cur = max(cur - 1.0, -cap)
            if c[i] > peak + k * a[i]:
                cur, adds = 0.0, 0
        s[i] = cur
    return pd.Series(s, index=px.index).shift(1).fillna(0.0)


def row(tag, sig, px, fd, tv=0.30, lev=3.0):
    net, pos = run(px, fd, sig, tv, max_lev=lev)
    a = net.to_numpy()
    g, l2 = gate(a), B.last2(net)
    st = stats_of(a)
    live = np.asarray(pos, float) != 0
    flips = int((np.abs(np.diff(np.r_[0.0, np.asarray(pos, float)])) > 1e-9).sum())
    yrs = B.yearly(net)
    neg = sum(1 for _, v in yrs if v < 0)
    gs = "n/a" if not np.isfinite(g) else f"{g:>7.1f}%"
    ls = "n/a" if not np.isfinite(l2) else f"{l2:>8.1f}%"
    print(f"   {tag:>30}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%{flips:>7}"
          f"{gs:>8}{ls:>9}{neg:>8}")
    return g, l2


HDR = (f"   {'variant':>30}{'Shp':>7}{'realDD':>8}{'flips':>7}{'gate':>8}"
       f"{'last2y':>9}{'neg yrs':>8}")

if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    rng = hi - lo

    print("S141 - developing range-expansion. both columns shown throughout,\n"
          "because selecting on the trailing two years is itself a selection.\n")

    print("1. TRIGGER AND LOOKBACK")
    print(HDR)
    for N in (55, 89, 144):
        for q in (0.80, 0.90, 0.95):
            el, es = rex(px, hi, lo, rng, N, q)
            sig = trail_pyr(px, el.fillna(False), es.fillna(False), atr, 3.0)
            row(f"N={N}, q={q:.2f}", sig, px, fd)

    print("\n2. THE EXIT — the drawdown lever")
    print(HDR)
    el, es = rex(px, hi, lo, rng, 89, 0.90)
    el, es = el.fillna(False), es.fillna(False)
    for k in (1.5, 2.0, 3.0, 4.0, 6.0):
        row(f"ATR trail {k:.1f}x", trail_pyr(px, el, es, atr, k), px, fd)

    print("\n3. PYRAMIDING — adding to a position that keeps running")
    print(HDR)
    for pyr, step in ((0, 1.0), (1, 1.0), (2, 1.0), (2, 2.0), (3, 1.0)):
        row(f"{pyr} adds, every {step:.0f}xATR",
            trail_pyr(px, el, es, atr, 3.0, pyr=pyr, step=step), px, fd, lev=5.0)

    print("\n4. ASYMMETRY — BTC drifts up, so equal long/short is not obvious")
    print(HDR)
    base = trail_pyr(px, el, es, atr, 3.0)
    for lw, sw, tag in ((1.0, 1.0, "symmetric"), (1.0, 0.5, "shorts at half"),
                        (1.0, 0.0, "long only"), (0.5, 1.0, "longs at half")):
        s = base.copy()
        s[s > 0] *= lw
        s[s < 0] *= sw
        row(tag, s, px, fd)

    print("\n5. BEST-BY-DURABILITY, YEAR BY YEAR")
    cands = {"N89 q0.90 trail3 (S140)": base,
             "N89 q0.90 trail2": trail_pyr(px, el, es, atr, 2.0),
             "N89 q0.90 trail3 +2 adds": trail_pyr(px, el, es, atr, 3.0, pyr=2)}
    for tag, s in cands.items():
        net, _ = run(px, fd, s, 0.30, max_lev=5.0)
        print(f"   {tag}")
        print(f"      " + "  ".join(f"{y} {v:+.1f}%" for y, v in B.yearly(net)))
        print(f"      gate {gate(net.to_numpy()):.1f}%, last2y {B.last2(net):.1f}%, "
              f"Sharpe {stats_of(net.to_numpy())['sharpe']:.2f}")
    print("\ndone: S141")
