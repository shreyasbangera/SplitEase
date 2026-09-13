"""
S152 - The validated factors on the point-in-time universe.

Three questions, each of which could overturn everything before it:

  1  DOES LOW-VOL SURVIVE A UNIVERSE WHERE COINS DIE? S148 measured it on 16
     survivors. Here it runs on ~48 tradeable names a day including LUNA, DODO,
     AKRO, BZRX and the rest of the graveyard.

  2  DOES BREADTH HELP? The same factor on 16 names and on 48. A cross-sectional
     rank is a better estimate of relative position the more names it ranks, so
     this should improve - and if it does not, the cross-sectional story is
     weaker than it looks.

  3  HOW MUCH OF IT IS UNCOLLECTABLE? A short in a coin that gaps to zero over a
     weekend is not a trade anyone fills. Binance auto-deleverages positions
     when the insurance fund cannot absorb a liquidation cascade, and the
     profitable side is what gets deleveraged. This charges for that explicitly
     rather than booking a 99% gain on a corpse:

         a per-asset daily return cap, swept, which truncates exactly the
         windfalls that a real book would not have collected.

Every figure is reported with and without that cap, so the reader can see how
much of the result depends on the least believable trades in it.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s151_pit as PIT
import strategies.s148_lowvol as L
import strategies.s147_panel as P
from strategies.s96_rank import at_gate, stats_of

SURVIVORS = ["ADAUSDT", "ATOMUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT",
             "DOGEUSDT", "DOTUSDT", "ETHUSDT", "FILUSDT", "LINKUSDT", "LTCUSDT",
             "NEARUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT"]


def book_capped(px, w, fund, cap=None, slip_bps=5.0, target_vol=None, band=0.10):
    """L.book, but with an optional per-asset daily return cap.

    The cap stands in for everything that stops a book collecting the full move
    in a collapsing contract: auto-deleveraging, position limits, a market that
    gaps through the stop, and an exchange that halts the symbol.
    """
    r = px.pct_change()
    if cap is not None:
        r = r.clip(-cap, cap)
    r = r.fillna(0.0)
    w = w.shift(1).fillna(0.0)
    if target_vol is not None:
        g0 = (w * r).sum(axis=1)
        rv = g0.ewm(halflife=45, min_periods=30).std().shift(1) * np.sqrt(365.25)
        lev = (target_vol / rv.replace(0, np.nan)).clip(upper=3.0).fillna(0.0)
        w = w.mul(lev, axis=0)
        gr = w.abs().sum(axis=1)
        w = w.div(np.maximum(gr / L.MAX_GROSS, 1.0), axis=0).fillna(0.0)
    gross = (w * r).sum(axis=1)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * (L.FEE_BPS + slip_bps) / 1e4
    carry = (w * fund.reindex_like(w).fillna(0.0)).sum(axis=1) if fund is not None else 0.0
    return (gross - cost - carry).clip(lower=-0.95), turn


def run(px, ok, sig, mode, fund, cap=None, tv=None, beta=None):
    w = L.weights(sig, ok, mode, beta)
    return book_capped(px, w, fund, cap=cap, target_vol=tv)


if __name__ == "__main__":
    pan = PIT.build()
    px = pan["close"]
    ok = PIT.mask(pan)
    n = ok.sum(axis=1)
    first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    ok = ok.loc[px.index]

    print("S152 - validated factors on the point-in-time universe\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, "
          f"{len(px)} days, {px.shape[1]} symbols ever, "
          f"median {int(ok.sum(axis=1).median())} tradeable/day\n")

    r1 = np.log(px).diff(1)
    lowvol = (-r1.rolling(90, min_periods=60).std()).where(ok)
    mkt = r1.mean(axis=1)
    beta = (r1.rolling(180, min_periods=90).cov(mkt)
            .div(mkt.rolling(180, min_periods=90).var(), axis=0)).where(ok).clip(0.4, 2.5)

    print("1. IS THE LOW-VOL IC STILL THERE WITH ~48 NAMES INSTEAD OF 16?")
    print(f"   {'universe':>28}{'names':>7}" +
          "".join(f"{'IC'+str(k)+'d':>9}{'t':>7}" for k in (1, 5, 20)))
    surv = pd.DataFrame(np.tile(ok.columns.isin(SURVIVORS), (len(ok), 1)),
                        index=ok.index, columns=ok.columns)
    for tag, m in (("point-in-time (all)", ok),
                   ("survivors only (S148's 16)", ok & surv)):
        s = (-r1.rolling(90, min_periods=60).std()).where(m)
        row = ""
        for k in (1, 5, 20):
            ic = P.xs_ic(s.shift(1), np.log(px).diff(k).shift(-k), m)
            mm, t, _ = P.ic_stat(ic, k)
            row += f"{mm:>9.4f}{t:>7.2f}"
        print(f"   {tag:>28}{int(m.sum(axis=1).median()):>7}{row}")

    print("\n2. THE BOOK, AND WHAT THE DEAD NAMES ARE WORTH")
    print(f"   {'variant':>28}{'cap':>7}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>8}{'at -20%':>9}")
    masks = {"point-in-time (all)": ok, "survivors only": ok & surv}
    res = {}
    for tag, m in masks.items():
        s = (-r1.rolling(90, min_periods=60).std()).where(m)
        for cap in (None, 0.50, 0.25):
            net, turn = run(px, m, s, "betaneutral", None, cap=cap, beta=beta)
            st = stats_of(net.to_numpy(float)); g = L.gate_of(net.to_numpy(float))
            cs = "none" if cap is None else f"{cap*100:.0f}%"
            gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
            print(f"   {tag:>28}{cs:>7}{st['sharpe']:>7.2f}{st['cagr']*100:>8.1f}%"
                  f"{st['dd']*100:>7.1f}%{turn.mean():>8.2f}{gs:>9}")
            if cap == 0.25:
                res[tag] = net

    print("\n3. CONSTRUCTIONS ON THE FULL UNIVERSE, 25% daily cap throughout")
    print(f"   {'construction':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    sleeves = {}
    for mode in ("longonly", "dollarneutral", "betaneutral", "zscore"):
        for tv in (None, 0.30):
            net, turn = run(px, ok, lowvol, mode, None, cap=0.25, tv=tv, beta=beta)
            st = stats_of(net.to_numpy(float)); g = L.gate_of(net.to_numpy(float))
            gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
            lbl = f"{mode}" + ("" if tv is None else f" @{tv*100:.0f}%")
            print(f"   {lbl:>28}{st['sharpe']:>7.2f}{st['cagr']*100:>8.1f}%"
                  f"{st['dd']*100:>7.1f}%{turn.mean():>8.2f}{gs:>9}")
            if tv is None:
                sleeves[mode] = net

    print("\n4. DEEPER CROSS-SECTION: does ranking finer help? tercile vs decile")
    print(f"   {'split':>28}{'names/leg':>11}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'at -20%':>9}")
    for frac, name in ((2, "halves"), (3, "terciles"), (5, "quintiles"),
                       (10, "deciles")):
        w = L.weights(lowvol, ok, "betaneutral", beta, frac=frac)
        net, turn = book_capped(px, w, None, cap=0.25)
        st = stats_of(net.to_numpy(float)); g = L.gate_of(net.to_numpy(float))
        gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
        k = int((ok.sum(axis=1) // frac).median())
        print(f"   {name:>28}{k:>11}{st['sharpe']:>7.2f}{st['cagr']*100:>8.1f}%"
              f"{st['dd']*100:>7.1f}%{gs:>9}")
    print("\ndone: factors on the point-in-time universe")
