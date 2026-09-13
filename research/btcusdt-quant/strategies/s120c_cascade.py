"""
S120c - The cascade book, built causally, and priced at honest slippage.

WHAT S120b FOUND, AND WHAT IS SUSPECT ABOUT IT
----------------------------------------------
Fading ordinary intraday moves loses the round trip in every one of the 96 cells
covering the top 10% to the top 1% of moves: -13 to -19 net basis points, no
exceptions. But the far tail flips, and it does not flip in one cell - at hourly
bars the whole region beyond the top 1% is positive across every hold length, 20
contiguous cells. That is the shape a real effect makes, and it has a mechanism:
a 3.7-sigma hourly candle on a perpetual is forced deleveraging, and standing on
the other side of forced flow is the oldest paid risk in markets.

Three things are wrong with taking that at face value.

    SAMPLE      the positive region is 59 to 585 events over 6.7 years.
    DECAY       the best cells halve or worse between halves: 172.8 -> 27.0 at
                1h, 76.1 -> 11.1 at 30m, 59.2 -> 14.0 at 15m.
    SELECTION   those were the best cells of a 144-cell surface, and this log
                has already watched a +9.9 in-sample edge become a -48.8 causal
                one (S113) for exactly that reason.

**And a fourth, which is specific to this book and worse than the other three.**
This strategy deliberately trades at the single worst moment of liquidity in the
year. Every other book in this study trades a calm market at a scheduled time;
this one puts on risk in the middle of a cascade, when the book is thin, the
spread is wide, and every other participant is trying to do the same thing. The
3 basis points of slippage carried through the rest of this log is a number for
normal conditions and it is fiction here.

So slippage is not a footnote in this file, it is the experiment. The book is
priced at 3, 10, 25, 50 and 100 basis points of one-way slippage, and the
question is not whether it makes money at 3 - it is where it stops.

HOW IT IS BUILT, CAUSALLY
-------------------------
    trigger     |bar return| >= K x trailing vol, where the vol is an EWMA over
                the previous 500 bars and EXCLUDES the trigger bar. K is 3.0,
                3.5 and 4.0 - round numbers picked for being round.
    entry       the OPEN of the NEXT bar, never the close just observed.
    exit        the open of the bar H later, no stop, no target.
    one at a time   a trigger firing while a position is open is ignored, so
                every trade is a clean countable round trip.
    sizing      risk-based: notional scaled so the position's exposure to a
                one-sigma bar move is a fixed fraction of equity, capped at 3x.
                This is the "realistic risk management" the brief asks for -
                the position shrinks when the market is violent, which is
                exactly when this book trades.
    charged     taker fee both ways, slippage both ways at the swept rate, and
                funding accrued over the hold.

Every bar size and every K is printed. Nothing is selected on its own result.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

FEE_BPS = 5.0
VOL_HL = 500          # bars, trailing volatility for the trigger and for sizing
RISK = 0.02           # fraction of equity exposed to a one-sigma bar move
MAX_LEV = 3.0
SLIPS = (3.0, 10.0, 25.0, 50.0, 100.0)


def bars(minutes):
    """OHLC on an N-minute grid, plus funding accrued in each bar."""
    d = pd.read_parquet("/home/user/quant/data/fut_1m.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    d = d.set_index("dt")
    rule = f"{minutes}min"
    o = pd.DataFrame({
        "open": d["open"].resample(rule).first(),
        "close": d["close"].resample(rule).last(),
        "qv": d["quote_volume"].resample(rule).sum(),
    }).dropna(subset=["open", "close"])

    f = pd.read_parquet("/home/user/quant/data/funding.parquet")
    f["dt"] = pd.to_datetime(f.dt, utc=True)
    fr = f.set_index("dt")["rate"].resample(rule).sum().reindex(o.index).fillna(0.0)
    o["fund"] = fr
    return o


def events(o, k):
    """Trigger indices and side. Strictly past: the vol estimate excludes the bar."""
    c = np.log(o.close.to_numpy())
    r = np.diff(c, prepend=c[0])
    sd = pd.Series(r).ewm(halflife=VOL_HL, adjust=False).std().shift(1).bfill().to_numpy()
    sd = np.maximum(sd, 1e-9)
    z = r / sd
    fire = np.abs(z) >= k
    return z, sd, fire


def book(o, k, hold, slip_bps):
    """One pass. Returns per-bar net returns on the bar index, and the trades."""
    op = o.open.to_numpy()
    fund = o.fund.to_numpy()
    z, sd, fire = events(o, k)
    n = len(o)
    cost = (FEE_BPS + slip_bps) / 1e4

    net = np.zeros(n)
    trades = []
    i = 0
    while i < n - hold - 2:
        if not fire[i]:
            i += 1
            continue
        # observed at the close of bar i -> act at the open of bar i+1
        e, x = i + 1, i + 1 + hold
        if x >= n:
            break
        side = -np.sign(z[i])
        # risk-based size: a one-sigma bar move moves the book by RISK of equity
        lev = min(RISK / max(sd[i], 1e-9), MAX_LEV)
        gross = side * lev * (op[x] / op[e] - 1.0)
        fees = 2.0 * lev * cost
        carry = side * lev * float(fund[e:x].sum())
        pnl = gross - fees - carry
        net[x - 1] += pnl
        trades.append(dict(i=i, dt=o.index[e], side=side, lev=lev,
                           gross=gross, fees=fees, carry=carry, pnl=pnl))
        i = x                                   # one position at a time
    return pd.Series(net, index=o.index), pd.DataFrame(trades)


def to_daily(net):
    """Bar returns summed onto a calendar-day grid before anything scores them.

    stats_of and at_gate both compute years as len(r)/365.25 and annualise by
    sqrt(365.25). Handed a series on a 15-minute index they read 6.7 years as
    0.6 and inflate CAGR beyond recognition - the S119 units error. Summing to
    calendar days puts every bar size on one clock.
    """
    s = pd.Series(np.asarray(net, float), index=pd.to_datetime(net.index))
    d = s.groupby(s.index.normalize()).sum()
    idx = pd.date_range(d.index.min(), d.index.max(), freq="1D", tz=d.index.tz)
    return d.reindex(idx).fillna(0.0)


def score(tag, net, T):
    a = to_daily(net).to_numpy()
    a = a[np.isfinite(a)]
    if not len(T) or not np.any(a):
        print(f"{tag:>26}{len(T):>8}{'-':>9}{'-':>8}{'-':>8}{'-':>7}{'-':>9}"
              f"{'-':>9}{'-':>9}")
        return
    st, g = stats_of(a), at_gate(a)
    b = bootstrap_dd(a, n=2000, block=90)
    p = T.pnl.to_numpy()
    win, loss = p[p > 0].sum(), -p[p < 0].sum()
    pf = win / loss if loss > 0 else np.inf
    h = len(a) // 2
    g1, g2 = at_gate(a[:h])["cagr"] * 100, at_gate(a[h:])["cagr"] * 100
    print(f"{tag:>26}{len(T):>8}{st['cagr']*100:>8.1f}%{st['dd']*100:>7.1f}%"
          f"{b['dd_median']*100:>7.1f}%{pf:>7.2f}{g['cagr']*100:>8.1f}%"
          f"{g1:>8.1f}%{g2:>8.1f}%", flush=True)


HDR = (f"{'book':>26}{'trades':>8}{'CAGR':>9}{'realDD':>8}{'medDD':>8}"
       f"{'PF':>7}{'at -20%':>9}{'1st h':>9}{'2nd h':>9}")

if __name__ == "__main__":
    print("S120c - fading liquidation cascades, priced at honest slippage\n")
    print(f"risk {RISK*100:.0f}% of equity per sigma of bar move, cap "
          f"{MAX_LEV:.0f}x, taker fee {FEE_BPS:.0f}bps each way, funding "
          f"charged over the hold.")
    print("entry at the open AFTER the trigger bar. one position at a time.\n")

    for mins, holds in ((15, (8, 16)), (30, (4, 8)), (60, (2, 4))):
        o = bars(mins)
        lab = f"{mins}m" if mins < 60 else "1h"
        print(f"=== {lab} bars, {len(o):,} of them, "
              f"{o.index.min().date()} -> {o.index.max().date()}")
        for k in (3.0, 3.5, 4.0):
            for hold in holds:
                print(f"\n  trigger {k:.1f} sigma, hold {hold} bars "
                      f"({hold*mins} minutes)")
                print(HDR)
                for slip in SLIPS:
                    net, T = book(o, k, hold, slip)
                    score(f"slip {slip:.0f}bps", net, T)
        print()
    print("done: cascade book")
