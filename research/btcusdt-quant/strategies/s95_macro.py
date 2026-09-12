"""
S95 - The macro read. The one source class the study has never touched.

WHY THIS, AND WHY NOW
---------------------
S50 ended with "the source list is closed for real: OHLCV, taker volume, trade
count, funding, open interest, trader positioning, quarterly futures, the
coin-margined contract, tick prints, the order book, the volatility index, the
options chain." Every entry on that list is a Binance endpoint. The study has
read crypto's own plumbing twelve ways and never once looked outside it.

Two facts make that gap worth closing rather than merely noting:

  * The book earns +6% in 2022 and +1% to +3% at every risk setting on the
    2022 slice. It is inert through the one year in the sample when BTC was
    trading as a long-duration risk asset against rates and the dollar rather
    than on its own microstructure.
  * The ceiling is a CORRELATION ceiling. S56 put the reachable Sharpe near 3
    given the existing pool, and the five deployed signals are all crypto
    microstructure or positioning - flow, stablecoin basis, turnover rotation,
    funding, positioning. A macro read is the only candidate left whose
    correlation to that pool should be near zero, which is the only thing that
    moves a correlation ceiling.

WHAT IS BEING TESTED, FIXED BEFORE LOOKING
------------------------------------------
VIX, as the screen. It is the single most-used macro factor, it is daily back
to 1990, and it is the only macro series reachable from this environment at no
cost. If the macro hypothesis has life, the equity-vol regime read is where it
should show first. If VIX shows nothing stable, that is evidence about the
direction, not just about VIX.

Sign prior, stated in advance: **VIX up -> BTC down.** BTC has traded as a
risk asset for the whole sample; a rising equity-vol regime should precede
weakness. The contrarian reading (extreme VIX = capitulation = buy) exists too,
so a positive IC is not by itself disqualifying - but a feature whose sign
FLIPS between sample halves is dead either way, which is exactly the test that
killed the options chain at S50.

THE TEST
--------
Spearman IC against forward 12h-bar returns at 1, 2, 4, 8 and 24 bars (12h to
12 days), computed three ways:

  full sample | first half vs second half | in-sample vs out-of-sample

A feature has to hold its sign in BOTH splits to be worth a backtest. Nothing
here is traded yet; this is the diagnostic that decides whether the macro
direction earns a backtest at all.

The alignment carries a deliberately punitive lag - a VIX close dated D is
treated as unknowable until D+1 22:00 UTC, about 25 hours later - so no result
below can be an artefact of reading a close before it printed. The `prompt`
alignment (~45 minutes after the close, which is the truth) is reported beside
it as a contrast.
"""
import sys
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd
from scipy import stats

from research.harness import panel, IS_END, OOS_END
from research import macro

HORIZONS = (1, 2, 4, 8, 24)          # bars of 12h -> 12h .. 12 days
START = "2021-03-01"                 # the book's own long window


def fwd(close, h):
    c = pd.Series(np.asarray(close, float))
    return (np.log(c.shift(-h)) - np.log(c)).to_numpy()


def ic(x, y, m):
    ok = np.isfinite(x) & np.isfinite(y) & m
    if ok.sum() < 300:
        return np.nan, 0
    return stats.spearmanr(x[ok], y[ok])[0], int(ok.sum())


def screen(feat, close, dt, title):
    dt = pd.Series(pd.to_datetime(dt))
    n = len(dt)
    mid = dt.iloc[n // 2]
    halves = dict(
        full=np.ones(n, bool),
        h1=(dt < mid).to_numpy(),
        h2=(dt >= mid).to_numpy(),
        IS=(dt < IS_END).to_numpy(),
        OOS=(dt >= IS_END).to_numpy(),
    )
    print(f"\n{title}")
    print(f"  split at {mid.date()} (halves), {IS_END} (IS/OOS)")
    print(f"{'feature':>10}{'h':>4}{'full':>9}{'1st':>9}{'2nd':>9}"
          f"{'IS':>9}{'OOS':>9}   stable")
    for c in feat.columns:
        x = feat[c].to_numpy(float)
        for h in HORIZONS:
            y = fwd(close, h)
            vals = {k: ic(x, y, m)[0] for k, m in halves.items()}
            same_half = np.sign(vals["h1"]) == np.sign(vals["h2"])
            same_oos = np.sign(vals["IS"]) == np.sign(vals["OOS"])
            flag = "BOTH" if (same_half and same_oos) else (
                   "half" if same_half else ("oos" if same_oos else ""))
            print(f"{c:>10}{h:>4}" + "".join(
                f"{vals[k]:+9.4f}" for k in ("full", "h1", "h2", "IS", "OOS"))
                + f"   {flag}")
    return


if __name__ == "__main__":
    fut, f12 = panel("12h")
    f = f12[f12.dt >= START].reset_index(drop=True)
    print(f"12h panel: {len(f)} bars, {f.dt.iloc[0]} -> {f.dt.iloc[-1]}")

    vix = macro.load_csv("vix")
    vix = vix[(vix.date >= "2020-06-01")].reset_index(drop=True)
    print(f"VIX daily: {len(vix)} rows, {vix.date.iloc[0].date()} -> "
          f"{vix.date.iloc[-1].date()}, mean {vix.val.mean():.1f}")

    for mode in ("strict", "prompt"):
        v = macro.onto_bars(f.dt, vix, mode=mode)
        cov = np.isfinite(v).mean()
        feat = macro.features(v)
        screen(feat, f.close.to_numpy(float), f.dt,
               f"VIX on the 12h grid - {mode} alignment "
               f"(coverage {cov*100:.1f}%)")

    # How stale is the strict alignment actually allowed to get?
    v_s = macro.onto_bars(f.dt, vix, "strict")
    v_p = macro.onto_bars(f.dt, vix, "prompt")
    diff = (v_s != v_p) & np.isfinite(v_s) & np.isfinite(v_p)
    print(f"\nstrict and prompt differ on {diff.mean()*100:.0f}% of bars "
          f"(the rest are weekends/holidays where both are the same stale value)")
    print("\ndone: macro screen")
