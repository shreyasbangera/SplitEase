"""
S143 - Four more: dollar-bar breakout, vol-of-vol, dip buying, alt confirmation.

  1  DOLLAR-BAR BREAKOUT
     `research/eventbars.py` was built earlier in this study to answer a
     fragility question and was never used to build a strategy. A dollar bar
     closes when a fixed amount of value has changed hands, so it has no phase
     and no length - the "which twelve hours?" question cannot be asked of it.

     That matters for a BREAKOUT specifically, more than for anything else this
     study has run on it. A range expansion is a statement about how far price
     travelled; in clock time a big move on dead volume and a big move on heavy
     volume look identical, and in volume time they cannot. Range expansion
     measured per unit of value traded is a different and better-posed quantity
     than range expansion measured per unit of wall time, and S142 already found
     the book is sensitive to the bar it runs on (12h 24.0%, 8h 4.2%).

  2  VOLATILITY OF VOLATILITY
     Never tested here. Vol-of-vol says whether the volatility regime itself is
     stable. High vol-of-vol means the market cannot decide what regime it is
     in, which is when trend books historically suffer.

  3  SYSTEMATIC DIP BUYING
     The most common retail strategy in this asset and never tested in this log.
     Buy after a decline of X% from the running peak, exit on recovery or after
     a fixed hold. It has a real mechanism - forced deleveraging overshoots -
     and a real failure mode, which is that it is short gamma and 2022 exists.

  4  ALT BREADTH CONFIRMING BTC BREAKOUTS
     S125 tested alt POSITIONING as a signal and found noise. This is different:
     alt PRICE breadth as a confirmation filter. A BTC breakout with the whole
     complex moving is a different event from BTC moving alone, and breadth
     cannot be computed from BTC's own series at any window length.

Judged against the brief alone, with durability columns since S140 established
that as the property worth having.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s135_more as S135
import strategies.s140_breakout as B
import strategies.s141_rexdev as R
from research import eventbars as EB
from strategies.s96_rank import stats_of

gate, run = S135.gate, S135.run
D = "/home/user/quant/data"


def show(tag, net, pos, n_flips=None):
    a = np.asarray(net, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        print(f"   {tag:>34}   no position"); return np.nan
    g, l2 = gate(a), B.last2(pd.Series(a, index=net.index))
    st = stats_of(a)
    f = n_flips if n_flips is not None else int(
        (np.abs(np.diff(np.r_[0.0, np.asarray(pos, float)])) > 1e-9).sum())
    neg = sum(1 for _, v in B.yearly(net) if v < 0)
    gs = "n/a" if not np.isfinite(g) else f"{g:>7.1f}%"
    ls = "n/a" if not np.isfinite(l2) else f"{l2:>8.1f}%"
    print(f"   {tag:>34}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%{f:>7}{gs:>8}"
          f"{ls:>9}{neg:>7}{'YES' if f >= 100 else 'no':>6}")
    return g


HDR = (f"   {'variant':>34}{'Shp':>7}{'realDD':>8}{'flips':>7}{'gate':>8}"
       f"{'last2y':>9}{'negyr':>7}{'100+':>6}")

if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    rng = hi - lo
    print("S143 - four more strategies\n")

    # ------------------------------------------------ 1. dollar-bar breakout
    print("1. DOLLAR-BAR BREAKOUT — range expansion in volume time, no clock")
    print(HDR)
    d5 = pd.read_parquet(f"{D}/fut_5m.parquet")
    d5["dt"] = pd.to_datetime(d5.dt, utc=True)
    f_ = pd.read_parquet(f"{D}/funding.parquet")
    f_["dt"] = pd.to_datetime(f_.dt, utc=True)
    frate = f_.set_index("dt")["rate"]
    for per_day in (1.0, 2.0, 4.0):
        try:
            bars = EB.build_bars(d5, mode="adaptive", per_day=per_day)
        except Exception as e:
            print(f"   (per_day {per_day}: {e})"); continue
        bars = bars.dropna(subset=["close"]).copy()
        bi = pd.to_datetime(bars["dt"] if "dt" in bars.columns else bars.index, utc=True)
        bars.index = bi
        bp, bh, bl = bars["close"], bars["high"], bars["low"]
        bfd = frate.reindex(bi, method="ffill").fillna(0.0) * 0.0
        # funding accrued between consecutive bar closes
        acc = frate.cumsum().reindex(bi, method="ffill").fillna(0.0)
        bfd = acc.diff().fillna(0.0)
        btr = pd.concat([bh - bl, (bh - bp.shift(1)).abs(),
                         (bl - bp.shift(1)).abs()], axis=1).max(axis=1)
        batr = btr.ewm(span=20, adjust=False).mean()
        brng = bh - bl
        byr = 365.25 * per_day
        for N in (89, 144):
            el, es = R.rex(bp, bh, bl, brng, N, 0.95)
            sig = R.trail_pyr(bp, el.fillna(False), es.fillna(False), batr, 3.0)
            r2 = np.log(bp / bp.shift(1)).fillna(0.0)
            rv = np.maximum(r2.ewm(halflife=max(4, int(32 * per_day)), adjust=False)
                            .std().shift(1).bfill().to_numpy() * np.sqrt(byr), 0.05)
            want = np.clip(np.asarray(sig, float) * 0.30 / rv, -3.0, 3.0)
            pos, cur = np.zeros(len(bp)), 0.0
            for i in range(len(bp)):
                if abs(want[i] - cur) > 0.10 or (want[i] == 0.0 and cur != 0.0):
                    cur = want[i]
                pos[i] = cur
            turn = np.abs(np.diff(np.r_[0.0, pos]))
            net = (pos * bp.pct_change().fillna(0.0).to_numpy()
                   - pos * bfd.to_numpy() - turn * 8.0 / 1e4)
            nn = pd.Series(net, index=bi)
            nd = nn.groupby(nn.index.normalize()).sum()
            show(f"{per_day:.0f} bars/day, N={N}, q=0.95", nd, None,
                 n_flips=int((turn > 1e-9).sum()))

    # ------------------------------------------------------- 2. vol of vol
    print("\n2. VOLATILITY OF VOLATILITY")
    print(HDR)
    lr = np.log(px).diff()
    rv20 = lr.rolling(20).std()
    vov = rv20.rolling(60).std() / rv20.rolling(60).mean()
    zv = ((vov - vov.rolling(365, min_periods=120).mean())
          / (vov.rolling(365, min_periods=120).std() + 1e-12)).clip(-2, 2).shift(1).fillna(0.0)
    base = R.trail_pyr(px, *[x.fillna(False) for x in R.rex(px, hi, lo, rng, 89, 0.95)],
                       atr, 3.0)
    for tag, s in (("vov as signal (follow)", zv),
                   ("vov as signal (fade)", -zv),
                   ("breakout, calm vov only", base * (zv < 0).astype(float)),
                   ("breakout, unstable vov only", base * (zv > 0).astype(float))):
        show(tag, *run(px, fd, s, 0.30))

    # ----------------------------------------------------- 3. dip buying
    print("\n3. SYSTEMATIC DIP BUYING — the most common retail strategy here")
    print(HDR)
    dd = (px / px.cummax() - 1.0)
    for thr in (-0.10, -0.20, -0.30):
        for hold in (10, 30):
            s = np.zeros(len(px)); left = 0
            ddv = dd.to_numpy()
            for i in range(len(px)):
                if left > 0:
                    s[i] = 1.0; left -= 1
                elif ddv[i] <= thr:
                    s[i] = 1.0; left = hold - 1
            sig = pd.Series(s, index=px.index).shift(1).fillna(0.0)
            show(f"buy {thr*100:.0f}% dip, hold {hold}d", *run(px, fd, sig, 0.30))

    # ------------------------------------------- 4. alt breadth confirmation
    print("\n4. ALT BREADTH CONFIRMING BTC BREAKOUTS")
    print(HDR)
    try:
        ac = pd.read_parquet(f"{D}/alts_close.parquet")
        if not isinstance(ac.index, pd.DatetimeIndex):
            ac.index = pd.to_datetime(ac.index, utc=True)
        ac.index = ac.index.tz_convert("UTC") if ac.index.tz else ac.index.tz_localize("UTC")
        ad = ac.resample("1D").last()
        up = (np.log(ad).diff(5) > 0).sum(axis=1) / ad.notna().sum(axis=1).replace(0, np.nan)
        breadth = up.reindex(px.index).ffill().shift(1)
        print(f"   (alt breadth from {ac.shape[1]} symbols, "
              f"coverage {breadth.notna().mean()*100:.0f}%)")
        show("breakout alone (reference)", *run(px, fd, base, 0.30))
        for lo_, hi_ in ((0.5, 1.01), (0.6, 1.01)):
            m = ((breadth >= lo_) & (base > 0)) | ((breadth <= 1 - lo_) & (base < 0))
            show(f"breakout, breadth confirms (>{lo_:.0%})",
                 *run(px, fd, base * m.astype(float).fillna(0.0), 0.30))
        show("breakout, breadth CONTRADICTS",
             *run(px, fd, base * (~(((breadth >= 0.5) & (base > 0))
                                    | ((breadth <= 0.5) & (base < 0)))).astype(float).fillna(0.0),
                  0.30))
    except Exception as e:
        print("   (alt data unavailable:", e, ")")
    print("\ndone: S143")
