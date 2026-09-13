"""
S117 - A standalone managed-futures book on BTCUSDT. Judged on its own.

Not a V7 sleeve, not a V7 overlay, not a V7 complement. A different strategy,
with a different edge, a different architecture, and its own verdict against the
brief: 300% net a year, max drawdown under 20%, 100+ trades, profit factor
above 1.10.

WHAT IT IS, AND HOW IT DIFFERS
------------------------------
V7 forecasts direction from crowding state and bets a fixed fraction of equity
to an ATR stop. This does neither. It is the classic managed-futures
construction:

    position = trend_sign x (target_vol / realised_vol)

**No stops. No take-profits. No holding cap. No crowding signals.** Size is
continuous and inversely proportional to recent volatility, so the book takes
the same RISK in every regime rather than the same notional - it shrinks into
turbulence automatically instead of being stopped out of it. Direction comes
from price alone, blended across three speeds the way a trend follower actually
trades rather than one moving average.

It also runs on a **longer sample**: 2020-01 rather than 2021-03, because it
needs no positioning or basis data. 6.7 years, including the 2021 top and the
2022 bear market in full.

WHAT IS CHARGED
---------------
    funding     real settled BTCUSDT rates on the perp position, every 8h. A
                long-biased perp book pays about 11.8% a year for the privilege
                and any backtest omitting it is fiction.
    costs       fee + slippage on every change in position size, not just on
                direction flips - a vol-targeted book trades constantly.
    no stops    so no stop-fill optimism to argue about.

THE PRIOR, WRITTEN DOWN FIRST
-----------------------------
BTC buy-and-hold over this window is CAGR 43.1% at 61.7% volatility, so the raw
asset has a Sharpe around 0.7, and a long-biased book inherits that. Trend and
vol targeting should roughly double it - the literature says 1.0 to 1.4 on a
single asset - which at the -20% gate is worth perhaps 30-60% a year. **That is
far short of 300% and I expect it to stay short.** It is built because the
question asked was "find a different strategy", and a different strategy
measured honestly is an answer even when the answer is no.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from engine.data import load
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

FEE_BPS, SLIP_BPS = 5.0, 3.0
SPEEDS = ((8, 32), (16, 64), (32, 128))     # fast/slow EMA pairs, days
VOL_WIN = 32                                # days, EWMA halflife for realised vol
MAX_LEV = 3.0
BAND = 0.10                                 # no-trade band, fraction of equity


def data():
    d = load("fut_1h").copy()
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    px = d.set_index("dt")["close"].resample("1D").last().dropna()
    f = load("funding").copy()
    f["dt"] = pd.to_datetime(f.dt, utc=True)
    # total funding paid per calendar day, aligned to the daily grid
    fd = f.set_index("dt")["rate"].resample("1D").sum().reindex(px.index).fillna(0.0)
    return px, fd


def signal(px, mode):
    """Trend direction in [-1, +1], blended over three speeds, strictly past."""
    lp = np.log(px)
    sig = np.zeros(len(px))
    for fast, slow in SPEEDS:
        f = lp.ewm(span=fast, adjust=False).mean()
        s = lp.ewm(span=slow, adjust=False).mean()
        sig += np.sign((f - s).to_numpy())
    sig /= len(SPEEDS)
    sig = pd.Series(sig, index=px.index).shift(1).fillna(0.0).to_numpy()
    if mode == "long":
        return np.ones(len(px))                          # always long
    if mode == "longflat":
        return np.clip(sig, 0.0, None)
    return sig                                                 # long/short


def run(px, fd, mode, target_vol, band=BAND, max_lev=MAX_LEV):
    r = np.log(px / px.shift(1)).fillna(0.0)
    # Realised vol from strictly past returns, annualised.
    rv = (r.ewm(halflife=VOL_WIN, adjust=False).std()
            .shift(1).bfill().to_numpy() * np.sqrt(365.25))
    rv = np.maximum(rv, 0.05)
    s = signal(px, mode)
    want = np.clip(s * target_vol / rv, -max_lev, max_lev)

    # No-trade band: only move when the target is more than `band` of equity
    # away. A vol-targeted book otherwise trades every single day for nothing.
    pos = np.zeros(len(px))
    cur = 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band:
            cur = want[i]
        pos[i] = cur

    cost = (FEE_BPS + SLIP_BPS) / 1e4
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    simple = px.pct_change().fillna(0.0).to_numpy()
    # long pays funding when the rate is positive; short receives it
    net = pos * simple - pos * fd.to_numpy() - turn * cost
    turns = int((turn > 1e-9).sum())
    return pd.Series(net, index=px.index), turns, pos


def report(tag, r, turns, base=None):
    a = np.asarray(r, float)
    st = stats_of(a)
    b = bootstrap_dd(a, n=3000, block=90)
    g = at_gate(a)
    h = len(a) // 2
    g1, g2 = at_gate(a[:h])["cagr"] * 100, at_gate(a[h:])["cagr"] * 100
    print(f"{tag:>34}{st['cagr']*100:8.1f}%{st['dd']*100:8.1f}%{b['dd_median']*100:9.1f}%"
          f"{st['sharpe']:7.2f}{st['calmar']:7.2f}{turns:7d}{g['cagr']*100:9.1f}%"
          f"{g1:9.1f}%{g2:9.1f}%", flush=True)
    return g["cagr"] * 100


if __name__ == "__main__":
    px, fd = data()
    print(f"BTCUSDT perp, daily, {px.index.min().date()} -> {px.index.max().date()} "
          f"({len(px)} days, {len(px)/365.25:.1f} years)")
    print(f"trend: mean of sign(EMA{SPEEDS[0][0]}-EMA{SPEEDS[0][1]}), "
          f"{SPEEDS[1]}, {SPEEDS[2]}   vol: EWMA halflife {VOL_WIN}d   "
          f"cap {MAX_LEV}x   band {BAND}\n")

    print(f"{'book':>34}{'CAGR':>9}{'realDD':>8}{'medDD':>9}{'Shp':>7}{'Clm':>7}"
          f"{'turns':>7}{'at -20%':>9}{'1st h':>9}{'2nd h':>9}")
    # the asset itself, for scale
    bh = pd.Series(px.pct_change().fillna(0.0).to_numpy() - fd.to_numpy(),
                   index=px.index)
    report("buy & hold (perp, funding paid)", bh, 0)
    print()
    for mode, lab in (("long", "vol-target only, always long"),
                      ("longflat", "trend long/flat"),
                      ("longshort", "trend long/SHORT")):
        for tv in (0.20, 0.40, 0.60):
            report(f"{lab}, tgt vol {tv*100:.0f}%", *run(px, fd, mode, tv)[:2])
        print()
    print("done: managed futures")
