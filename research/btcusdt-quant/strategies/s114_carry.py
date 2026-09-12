"""
S114 - A different strategy, not a different way of holding V7.

Everything since S45 has been one edge: five crowding/flow signals netted into a
directional BTCUSDT position. The last several experiments all claimed to be new
directions and were not - a filter on V7's trades, a sizing overlay on V7, a
measurement of V7, a sweep of V7's grid, and V7's own signals on differently
shaped bars. This is a different EDGE.

    long the front quarterly future, short the perp, in equal notional.

Delta-neutral in BTC. The position earns the perp's funding (paid by the longs
who are crowding the perp) and pays the quarterly's basis as it converges to spot
at expiry. Net carry is therefore **funding minus basis**, and nothing in it
depends on predicting the direction of BTC at all.

Both legs are BTCUSDT futures on Binance, so this stays inside the single-
instrument scope. S5 built the SPOT version (long spot, short perp) and rejected
it on trade count and CAGR - 29 trades and 15-29% - while recording Sharpe 5-6.7
and Calmar ~4 as "by far the best risk-adjusted numbers in this study". The
futures-only version has never been built, and it avoids S5's largest cost: no
USDT borrow on a levered spot leg, which alone cut S5's 3x variant from 41.5% to
29.0%.

WHAT THE ARITHMETIC SAYS BEFORE THE RUN
---------------------------------------
Stated first so the result cannot be rationalised afterwards. The quarterly basis
runs a median of 6.3% annualised over this sample. Typical perp funding is around
0.01% per 8 hours, which is roughly 11% annualised. Net carry is therefore of
order **4-5% a year gross**, against roughly 1.3% a year in costs from four legs
rolled quarterly - so perhaps 3% unlevered.

That is nowhere near the brief, and leverage cannot rescue it: the spread itself
moves, reaching -46% annualised at its worst in this sample, which on a 90-day
contract is about -11% in price terms and would wipe out anything levered past
about 8x.

So the expected verdict is that this is a real edge that is far too small to be a
book. **It is built anyway, and built honestly, because the arithmetic above is an
estimate and the question was asked properly: measure it rather than talk yourself
out of it.** And there is a second reason that does not depend on its size - a
sleeve uncorrelated with V7 needs only Sharpe 0.91 to pass S102c's admission bar,
where every directional sleeve tested has needed 1.4 or more.

WHAT IS MODELLED
----------------
    funding     real settled BTCUSDT rates, received on the short perp leg
    basis       marked hourly from the actual front-quarterly print
    roll        into the next contract when the front has under `MIN_DTE` days
    costs       fee + slippage on all four legs, at entry, exit and every roll
    no borrow   both legs are futures margin, so S5's USDT borrow does not apply
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from engine.data import load

FEE_BPS, SLIP_BPS = 5.0, 3.0       # per leg, per side - the same as every other book here
MIN_DTE = 3.0                      # roll out of a contract this close to expiry
D = "/home/user/quant/data"


def panel():
    """Hourly spread panel: front quarterly, perp, basis, and the funding due."""
    q = pd.read_parquet(f"{D}/quarterly_front.parquet")
    q["dt"] = pd.to_datetime(q.dt, utc=True)
    q = q[(q.dte > MIN_DTE) & (q.basis.abs() < 0.25)].sort_values("dt")
    q = q.drop_duplicates("dt", keep="first").reset_index(drop=True)

    f = load("funding").copy()
    f["dt"] = pd.to_datetime(f.dt, utc=True)
    # A rate settled at time T is paid by longs to shorts at T. Short perp
    # RECEIVES it. Attach it to the hour it settles in, zero elsewhere.
    q["fund"] = 0.0
    idx = np.searchsorted(q.dt.to_numpy(), f.dt.to_numpy())
    ok = idx < len(q)
    q.loc[q.index[idx[ok]], "fund"] = f.rate.to_numpy()[ok]

    q["ann_fund"] = q.fund.replace(0.0, np.nan).ffill() * 3 * 365      # 3 settlements a day
    q["carry"] = q.ann_fund - q.ann          # what the trade earns, annualised
    return q


def run(q, thr=0.0, lev=1.0, side=+1):
    """side=+1 is long quarterly / short perp (earn funding, pay basis).

    P&L is computed on the SPREAD directly rather than through the OHLCV engine,
    because the engine trades one instrument and this trades the difference of
    two. Both legs are equal notional at entry, so the BTC delta is zero and the
    only exposure is the basis itself.
    """
    dtn = q.dt.to_numpy()
    perp = q.perp.to_numpy(float)
    fut = q.close.to_numpy(float)
    fund = q.fund.to_numpy(float)
    sym = q.sym.to_numpy()
    carry = q.carry.to_numpy(float)

    cost = (FEE_BPS + SLIP_BPS) / 1e4 * 2      # two legs, one direction
    eq = 1.0
    on = False
    cur = None
    p0 = f0 = 0.0
    out = np.empty(len(q))
    trades = 0
    for i in range(len(q)):
        want = np.isfinite(carry[i]) and (side * carry[i] > thr)
        rolled = on and sym[i] != cur
        if on and (not want or rolled):
            eq *= (1.0 - cost * lev)             # close both legs
            on = False
        if not on and want:
            eq *= (1.0 - cost * lev)             # open both legs
            on, cur, p0, f0 = True, sym[i], perp[i], fut[i]
            trades += 1
        if on:
            # Spread return over this hour plus the funding received on the short
            # perp leg, applied MULTIPLICATIVELY to equity. The first version
            # added them, which is an account that can go through zero and keep
            # trading - it printed -100% CAGR with a -3483% drawdown, which is
            # not a result, it is an account that went bankrupt and carried on.
            r = 0.0
            if i > 0 and sym[i] == sym[i - 1]:
                r += ((fut[i] - fut[i - 1]) / f0) - ((perp[i] - perp[i - 1]) / p0)
            r += fund[i]                         # short perp receives positive funding
            eq *= (1.0 + side * lev * r)
            if eq <= 1e-6:                       # liquidated; stop rather than invert
                out[i:] = 0.0
                return pd.Series(out, index=pd.to_datetime(q.dt)), trades
        out[i] = eq
    return pd.Series(out, index=pd.to_datetime(q.dt)), trades


def stats(e, trades, tag):
    d = e.resample("1D").last().dropna()
    r = d.pct_change().fillna(0.0).to_numpy()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    cagr = d.iloc[-1] ** (1 / yrs) - 1 if d.iloc[-1] > 0 else -1.0
    dd = float((d / d.cummax() - 1).min())
    sh = r.mean() / r.std() * np.sqrt(365.25) if r.std() > 0 else 0.0
    print(f"{tag:>34}{cagr*100:9.1f}%{dd*100:8.1f}%{sh:8.2f}"
          f"{(cagr/abs(dd) if dd < 0 else np.nan):8.2f}{trades:8d}", flush=True)
    return dict(cagr=cagr, dd=dd, sharpe=sh, daily=d.pct_change().fillna(0.0))


if __name__ == "__main__":
    q = panel()
    print(f"{len(q)} hourly rows, {q.sym.nunique()} contracts, "
          f"{q.dt.min().date()} -> {q.dt.max().date()}\n")
    print("the carry itself, annualised (funding received minus basis paid):")
    c = q.carry.dropna()
    print(f"   median {c.median()*100:5.1f}%   mean {c.mean()*100:5.1f}%   "
          f"positive {(c > 0).mean()*100:4.0f}% of hours   "
          f"p5 {c.quantile(.05)*100:6.1f}%   p95 {c.quantile(.95)*100:5.1f}%\n")

    print(f"{'book':>34}{'CAGR':>10}{'MaxDD':>8}{'Shp':>8}{'Clm':>8}{'N':>8}")
    # Always on, rolling only at expiry: the cleanest read on the carry itself,
    # with the minimum number of round turns the trade can possibly pay.
    for lev in (1, 3, 5):
        e, n = run(q, thr=-9e9, lev=lev, side=+1)
        stats(e, n, f"ALWAYS ON, roll only, {lev}x")
    print()
    res = {}
    for lev in (1, 2, 3, 5, 8):
        e, n = run(q, thr=0.0, lev=lev, side=+1)
        res[lev] = stats(e, n, f"long qtr / short perp, {lev}x")
    print()
    for thr in (0.02, 0.05, 0.08):
        e, n = run(q, thr=thr, lev=3, side=+1)
        stats(e, n, f"same, 3x, only when carry > {thr*100:.0f}%")
    print()
    e, n = run(q, thr=0.0, lev=3, side=-1)
    stats(e, n, "REVERSE: short qtr / long perp, 3x")
    print("\ndone: futures-only carry")
