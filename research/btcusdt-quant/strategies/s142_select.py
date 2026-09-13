"""
S142 - Is the selectivity result real? And can it clear 100 trades?

S141 found that raising the range-expansion trigger from the 90th to the 95th
percentile roughly doubles the book, and it did so at EVERY lookback - 8.2/9.2/
15.3 at N=55, 7.3/12.5/23.1 at N=89, 21.9/13.8/26.0 at N=144. A family effect
rather than a lucky cell, which is the defence this log has used since S68.

Two things now have to be established before that number is worth anything.

  IS IT MONOTONIC?
      A real selectivity effect should improve smoothly as the bar rises and
      then fall away when the sample gets too thin. A spiky profile - good at
      0.95, bad at 0.93 and 0.97 - is noise wearing a family's clothes. The
      sweep here is fine-grained (0.85 to 0.98) across five lookbacks, and the
      shape of the curve is the evidence, not the best cell in it.

  DOES IT STILL TRADE ENOUGH?
      This is the catch. N=144 at q=0.95 makes 80 position changes and N=89 at
      q=0.95 makes 89 - both short of the brief's 100. Selectivity bought return
      by trading less, which is exactly the trade-off the brief forbids. Three
      fixes are tried: a shorter lookback at the same selectivity, combining the
      q=0.95 family across lookbacks (S140 says combination dilutes here, so
      this is a test of whether that holds when the components are equally
      good), and a faster bar.

Durability columns throughout, since S140 established that as the property worth
having and the full-sample figure as the one that misleads.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s135_more as S135
import strategies.s140_breakout as B
import strategies.s141_rexdev as R
from strategies.s96_rank import stats_of

gate, run = S135.gate, S135.run

if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    rng = hi - lo

    QS = (0.85, 0.90, 0.93, 0.95, 0.97, 0.98)
    NS = (34, 55, 89, 144, 233)
    print("S142 - is selectivity monotonic, and does it still trade?\n")
    print("1. THE SHAPE OF THE CURVE — gate %, and (flips) beneath")
    print(f"   {'N':>6}" + "".join(f"{'q=' + format(q, '.2f'):>11}" for q in QS))
    grid = {}
    for N in NS:
        gr, fr = [], []
        for q in QS:
            el, es = R.rex(px, hi, lo, rng, N, q)
            sig = R.trail_pyr(px, el.fillna(False), es.fillna(False), atr, 3.0)
            net, pos = run(px, fd, sig, 0.30)
            g = gate(net.to_numpy())
            f = int((np.abs(np.diff(np.r_[0.0, np.asarray(pos, float)])) > 1e-9).sum())
            grid[(N, q)] = (g, f, net, sig)
            gr.append(g); fr.append(f)
        print(f"   {N:>6}" + "".join((f"{v:>10.1f}%" if np.isfinite(v) else f"{'n/a':>11}")
                                     for v in gr))
        print(f"   {'':>6}" + "".join(f"{'(' + str(v) + ')':>11}" for v in fr))

    print("\n   monotone in q? (fraction of adjacent steps that increase)")
    for N in NS:
        v = [grid[(N, q)][0] for q in QS]
        v = [x if np.isfinite(x) else -9e9 for x in v]
        inc = sum(1 for i in range(len(v) - 1) if v[i + 1] >= v[i])
        print(f"      N={N:>4}: {inc}/{len(v)-1} steps rising   "
              f"peak at q={QS[int(np.argmax(v))]:.2f}")

    print("\n2. THE 100-TRADE PROBLEM")
    print(f"   {'variant':>34}{'Shp':>7}{'realDD':>8}{'flips':>7}{'gate':>8}"
          f"{'last2y':>9}{'negyr':>7}{'100+':>6}")

    def show(tag, sig):
        net, pos = run(px, fd, sig, 0.30)
        a = net.to_numpy(); g, l2 = gate(a), B.last2(net)
        f = int((np.abs(np.diff(np.r_[0.0, np.asarray(pos, float)])) > 1e-9).sum())
        neg = sum(1 for _, v in B.yearly(net) if v < 0)
        st = stats_of(a)
        gs = "n/a" if not np.isfinite(g) else f"{g:>7.1f}%"
        ls = "n/a" if not np.isfinite(l2) else f"{l2:>8.1f}%"
        print(f"   {tag:>34}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%{f:>7}{gs:>8}"
              f"{ls:>9}{neg:>7}{'YES' if f >= 100 else 'no':>6}")
        return net

    for N in NS:
        if np.isfinite(grid[(N, 0.95)][0]):
            show(f"N={N}, q=0.95 alone", grid[(N, 0.95)][3])

    print("\n   combining the q=0.95 family across lookbacks:")
    for use in ((55, 89), (55, 89, 144), (34, 55, 89, 144), (34, 55, 89, 144, 233)):
        s = sum(grid[(N, 0.95)][3] for N in use) / len(use)
        show(f"q=0.95, N in {use}", s)

    print("\n3. A FASTER BAR — same rule on 12h and 8h decisions")
    print(f"   {'variant':>34}{'Shp':>7}{'realDD':>8}{'flips':>7}{'gate':>8}"
          f"{'last2y':>9}{'negyr':>7}{'100+':>6}")
    d = pd.read_parquet("/home/user/quant/data/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True); d = d.set_index("dt")
    f_ = pd.read_parquet("/home/user/quant/data/funding.parquet")
    f_["dt"] = pd.to_datetime(f_.dt, utc=True); fr_ = f_.set_index("dt")["rate"]
    for rule, byr, lab in (("12h", 730.5, "12h"), ("8h", 1095.75, "8h")):
        p2 = d["close"].resample(rule).last().dropna()
        h2 = d["high"].resample(rule).max().reindex(p2.index)
        l2_ = d["low"].resample(rule).min().reindex(p2.index)
        fd2 = fr_.resample(rule).sum().reindex(p2.index).fillna(0.0)
        tr2 = pd.concat([h2 - l2_, (h2 - p2.shift(1)).abs(),
                         (l2_ - p2.shift(1)).abs()], axis=1).max(axis=1)
        atr2 = tr2.ewm(span=max(2, int(20 * byr / 365.25)), adjust=False).mean()
        rng2 = h2 - l2_
        for N in (89, 144):
            NN = max(10, int(N * byr / 365.25))
            el, es = R.rex(p2, h2, l2_, rng2, NN, 0.95)
            sig = R.trail_pyr(p2, el.fillna(False), es.fillna(False), atr2, 3.0)
            r2 = np.log(p2 / p2.shift(1)).fillna(0.0)
            rv = np.maximum(r2.ewm(halflife=max(4, int(32 * byr / 365.25)),
                                   adjust=False).std().shift(1).bfill().to_numpy()
                            * np.sqrt(byr), 0.05)
            want = np.clip(np.asarray(sig, float) * 0.30 / rv, -3.0, 3.0)
            pos, cur = np.zeros(len(p2)), 0.0
            for i in range(len(p2)):
                if abs(want[i] - cur) > 0.10 or (want[i] == 0.0 and cur != 0.0):
                    cur = want[i]
                pos[i] = cur
            turn = np.abs(np.diff(np.r_[0.0, pos]))
            net = (pos * p2.pct_change().fillna(0.0).to_numpy() - pos * fd2.to_numpy()
                   - turn * 8.0 / 1e4)
            nn = pd.Series(net, index=p2.index)
            nd = nn.groupby(nn.index.normalize()).sum()
            g, l2v = gate(nd.to_numpy()), B.last2(nd)
            fl = int((turn > 1e-9).sum())
            neg = sum(1 for _, v in B.yearly(nd) if v < 0)
            st = stats_of(nd.to_numpy())
            gs = "n/a" if not np.isfinite(g) else f"{g:>7.1f}%"
            ls = "n/a" if not np.isfinite(l2v) else f"{l2v:>8.1f}%"
            print(f"   {f'{lab} bars, N~{N}d, q=0.95':>34}{st['sharpe']:>7.2f}"
                  f"{st['dd']*100:>8.1f}%{fl:>7}{gs:>8}{ls:>9}{neg:>7}"
                  f"{'YES' if fl >= 100 else 'no':>6}")
    print("\ndone: S142")
