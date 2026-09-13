"""
S120 - The frequency axis. Does BTCUSDT have intraday structure worth trading?

WHY THIS, AND WHY IT IS NOT ANOTHER VERSION OF ANYTHING HERE
-----------------------------------------------------------
Every book in this log decides at 12 hours or one day. The brief asks for 300%
a year at a drawdown under 20%, which is **Calmar 15**. Nothing daily in this
study has cleared Calmar 1.3, and S103 explained why: at a daily decision
frequency the position is held through overnight gaps and multi-day slides, so
the worst single excursion is large relative to the year's return no matter how
good the direction call is. That is a property of the HOLDING PERIOD, not of the
signal, and no amount of better forecasting moves it.

Calmar is frequency-dependent. A book making 2,000 short, largely independent
bets a year accumulates the same expected return over a much smaller worst-path
excursion than one making 200 long ones. It is the only structural axis in this
study that has never been touched, and there are 3.5 million one-minute bars
from 2020-01 on disk that no strategy here has opened.

This file does not build a strategy. It asks the prior question, because
building an intraday book before knowing whether intraday structure exists is
how the last four files ended.

THE QUESTION
------------
At each horizon from 1 minute to 1 day: does the past return predict the next
one, in which direction, and is the effect bigger than the cost of acting on it?

    reversal     past up -> next down. Liquidity provision: someone needed to
                 trade urgently, paid up to do it, and the price comes back.
    momentum     past up -> next up. Continuation.

The cost hurdle is drawn on the same axis. A round trip pays fee + slippage
twice: at 5bps fee and 3bps slippage that is **16bps per completed trade**,
taker both ways. An effect of 4bps is not an edge; it is a way to pay Binance.
No maker rebate is assumed anywhere - a resting order is not a filled order,
and assuming otherwise is the single easiest way to invent an intraday edge
that does not exist.

WHAT IS MEASURED
----------------
    IC           Spearman rank correlation of past return against next return,
                 at the same horizon. Sign says reversal or momentum.
    edge_bps     the honest one: mean next-horizon return, signed by the
                 strategy's direction, conditional on the past return being in
                 its most extreme decile. This is what one trade actually earns
                 GROSS, in basis points, and it is what 16bps has to clear.
    by half      first and second half separately. An effect that exists only
                 in 2020-2022 is a fact about 2020-2022.

Split by regime too, because the reversal literature is mostly written on
markets that had a 2020 and a 2021 in them, and this study has already watched
one crowding effect decay monotonically to zero (S119).
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

FEE_BPS, SLIP_BPS = 5.0, 3.0
ROUND_TRIP_BPS = 2 * (FEE_BPS + SLIP_BPS)


def bars(minutes):
    """OHLCV + aggressive-flow imbalance resampled to an N-minute grid."""
    d = pd.read_parquet("/home/user/quant/data/fut_1m.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    d = d.set_index("dt")
    rule = f"{minutes}min"
    o = pd.DataFrame({
        "close": d["close"].resample(rule).last(),
        "qv": d["quote_volume"].resample(rule).sum(),
        "tbq": d["taker_buy_quote"].resample(rule).sum(),
        "n": d["count"].resample(rule).sum(),
    }).dropna(subset=["close"])
    o = o[o.qv > 0]
    # aggressive buy share, centred: +0.5 means every trade lifted the offer
    o["imb"] = o.tbq / o.qv - 0.5
    return o


def ic_table(o, label):
    """Past-return vs next-return structure at this bar size."""
    r = np.log(o.close).diff()
    fwd = r.shift(-1)
    m = np.isfinite(r) & np.isfinite(fwd)
    x, y = r[m], fwd[m]
    ic = pd.Series(x.to_numpy()).corr(pd.Series(y.to_numpy()), method="spearman")

    # direction the effect implies, then the gross edge of acting on it
    sgn = -1.0 if ic < 0 else 1.0
    q = x.quantile([0.10, 0.90])
    ext = (x <= q.iloc[0]) | (x >= q.iloc[1])
    # position = sgn * sign(past move), taken on the extreme decile only
    edge = (sgn * np.sign(x[ext]) * y[ext]).mean() * 1e4

    h = len(x) // 2
    ic1 = pd.Series(x.iloc[:h].to_numpy()).corr(
        pd.Series(y.iloc[:h].to_numpy()), method="spearman")
    ic2 = pd.Series(x.iloc[h:].to_numpy()).corr(
        pd.Series(y.iloc[h:].to_numpy()), method="spearman")
    e1 = (sgn * np.sign(x.iloc[:h][ext.iloc[:h]]) * y.iloc[:h][ext.iloc[:h]]).mean() * 1e4
    e2 = (sgn * np.sign(x.iloc[h:][ext.iloc[h:]]) * y.iloc[h:][ext.iloc[h:]]).mean() * 1e4

    kind = "reversal" if ic < 0 else "momentum"
    clears = "YES" if edge > ROUND_TRIP_BPS else "no"
    print(f"{label:>10}{len(x):>10,}{ic:>9.4f}{kind:>11}{edge:>10.1f}"
          f"{ic1:>9.4f}{ic2:>9.4f}{e1:>9.1f}{e2:>9.1f}{clears:>7}", flush=True)
    return dict(label=label, ic=ic, edge=edge, e1=e1, e2=e2)


def flow_table(o, label):
    """Does aggressive-flow imbalance add anything the past return does not?"""
    r = np.log(o.close).diff()
    fwd = r.shift(-1)
    imb = o.imb
    m = np.isfinite(imb) & np.isfinite(fwd)
    x, y = imb[m], fwd[m]
    ic = pd.Series(x.to_numpy()).corr(pd.Series(y.to_numpy()), method="spearman")
    sgn = -1.0 if ic < 0 else 1.0
    q = x.quantile([0.10, 0.90])
    ext = (x <= q.iloc[0]) | (x >= q.iloc[1])
    edge = (sgn * np.sign(x[ext]) * y[ext]).mean() * 1e4
    h = len(x) // 2
    ic1 = pd.Series(x.iloc[:h].to_numpy()).corr(
        pd.Series(y.iloc[:h].to_numpy()), method="spearman")
    ic2 = pd.Series(x.iloc[h:].to_numpy()).corr(
        pd.Series(y.iloc[h:].to_numpy()), method="spearman")
    kind = "fade" if ic < 0 else "follow"
    print(f"{label:>10}{len(x):>10,}{ic:>9.4f}{kind:>11}{edge:>10.1f}"
          f"{ic1:>9.4f}{ic2:>9.4f}{'':>18}{'YES' if edge > ROUND_TRIP_BPS else 'no':>7}",
          flush=True)


GRID = (1, 5, 15, 30, 60, 240, 1440)

if __name__ == "__main__":
    print("S120 - is there intraday structure on BTCUSDT perp, and does it "
          "clear costs?\n")
    print(f"round trip = 2 x ({FEE_BPS:.0f}bps fee + {SLIP_BPS:.0f}bps slip) = "
          f"{ROUND_TRIP_BPS:.0f}bps, taker both ways, no maker rebate assumed\n")

    print("PAST RETURN -> NEXT RETURN, same horizon")
    print(f"{'bar':>10}{'obs':>10}{'IC':>9}{'kind':>11}{'edge bps':>10}"
          f"{'IC 1st':>9}{'IC 2nd':>9}{'e 1st':>9}{'e 2nd':>9}{'>cost':>7}")
    cache = {}
    for mins in GRID:
        o = bars(mins)
        cache[mins] = o
        lab = f"{mins}m" if mins < 60 else (f"{mins//60}h" if mins < 1440 else "1d")
        ic_table(o, lab)

    print("\nAGGRESSIVE FLOW IMBALANCE -> NEXT RETURN")
    print(f"{'bar':>10}{'obs':>10}{'IC':>9}{'kind':>11}{'edge bps':>10}"
          f"{'IC 1st':>9}{'IC 2nd':>9}{'':>18}{'>cost':>7}")
    for mins in GRID:
        lab = f"{mins}m" if mins < 60 else (f"{mins//60}h" if mins < 1440 else "1d")
        flow_table(cache[mins], lab)

    print("\ndone: intraday structure diagnostic")
