"""
S125 - Cross-asset positioning as a signal for a BTCUSDT-only book.

A SCOPE DISTINCTION THIS STUDY HAS BEEN GETTING WRONG
-----------------------------------------------------
"Single instrument" has been read throughout this log as a constraint on the
data. It is not. It is a constraint on what is **traded**. A book that holds
nothing but BTCUSDT perpetual, and decides what to hold by looking at what is
happening in eight other markets, is a single-instrument book - one position,
one funding rate, one liquidation price, one thing to execute.

On that reading `altmetrics_1h.parquet` has been sitting unopened for the whole
study: 333,000 rows of positioning and open interest across ADA, AVAX, BNB,
DOGE, ETH, LINK, SOL and XRP, from 2021-12 to 2026-08, full coverage.

WHY THIS IS A DIFFERENT SIGNAL AND NOT MORE OF THE SAME
-------------------------------------------------------
S119 established that BTC's own crowding data is decaying monotonically - the
edge ran +21.9% in 2021 and +0.8% by 2025 while exposure stayed flat. Adding
more BTC positioning series to a BTC positioning book is what that result rules
out.

This is not that. **Cross-sectional structure is a different mathematical object
from the level of any one series.** Breadth (how many markets are doing the same
thing) and dispersion (how much they disagree) cannot be computed from BTC alone
at any window length, and they carry information that a single series does not:
whether a move is the whole complex or just this market.

The mechanism, stated before measuring it: alts are the leveraged end of this
market. They are smaller, thinner, and held with more leverage by less patient
money, so speculative froth shows up there first and liquidates there first.
When open interest is building across every alt at once, that is late-cycle
risk appetite and BTC is fragile. When it collapses across every alt at once,
that is forced selling working through the complex, which historically marks
capitulation rather than continuation.

EIGHT FEATURES, EACH WITH A REASON
----------------------------------
    oi_breadth      fraction of alts whose USD open interest is rising
    oi_agg          aggregate alt open interest, 7-day change
    pos_mean        cross-sectional mean top-trader position ratio - the complex
                    is one-sided
    pos_disp        cross-sectional dispersion of it - markets disagree
    retail_mean     cross-sectional mean all-account ratio - where the crowd is
    taker_mean      cross-sectional mean aggressive buy share
    taker_disp      dispersion of it
    rotation        alt positioning minus BTC's own - money moving down the
                    risk curve, which is the classic late-cycle tell

NOTHING IS BUILT HERE. This measures information content first, the way S120
and S122 did, because four files in this log were built before that question was
asked and all four should not have been. Every feature is lagged a full day and
reported on both halves; a feature that changes sign between halves is noise
this study has already been fooled by twice.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

D = "/home/user/quant/data"


def alt_panel():
    """Daily cross-sectional features from the eight alt markets."""
    a = pd.read_parquet(f"{D}/altmetrics_1h.parquet")
    a["dt"] = pd.to_datetime(a.dt, utc=True)
    a = a.rename(columns={
        "sum_open_interest_value": "oi",
        "sum_toptrader_long_short_ratio": "tt_pos",
        "count_long_short_ratio": "retail",
        "sum_taker_long_short_vol_ratio": "taker"})

    def wide(col):
        w = a.pivot_table(index="dt", columns="sym", values=col, aggfunc="last")
        return w.resample("1D").last()

    oi, tt, rt, tk = wide("oi"), wide("tt_pos"), wide("retail"), wide("taker")

    def zx(w, win=90):
        """Standardise each market against its OWN history, then compare across."""
        return (w - w.rolling(win, min_periods=30).mean()) / \
               (w.rolling(win, min_periods=30).std() + 1e-12)

    F = pd.DataFrame(index=oi.index)
    doi = np.log(oi.clip(lower=1e-9)).diff(7)
    F["oi_breadth"] = (doi > 0).sum(axis=1) / doi.notna().sum(axis=1).replace(0, np.nan)
    F["oi_agg"] = np.log(oi.sum(axis=1).clip(lower=1e-9)).diff(7)
    ztt, zrt, ztk = zx(tt), zx(rt), zx(tk)
    F["pos_mean"] = ztt.mean(axis=1)
    F["pos_disp"] = ztt.std(axis=1)
    F["retail_mean"] = zrt.mean(axis=1)
    F["taker_mean"] = ztk.mean(axis=1)
    F["taker_disp"] = ztk.std(axis=1)
    return F


def btc_panel():
    d = pd.read_parquet(f"{D}/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    px = d.set_index("dt")["close"].resample("1D").last().dropna()
    m = pd.read_parquet(f"{D}/metrics_1h.parquet")
    m["dt"] = pd.to_datetime(m.dt, utc=True)
    m = m.set_index("dt")
    btt = m["tt_pos"].resample("1D").last()
    return px, btt


def build():
    F = alt_panel()
    px, btt = btc_panel()
    ix = F.dropna(how="all").index.intersection(px.index)
    F = F.reindex(ix)
    px = px.reindex(ix)
    # rotation: how one-sided the alt complex is RELATIVE to BTC's own crowd
    bz = ((btt - btt.rolling(90, min_periods=30).mean())
          / (btt.rolling(90, min_periods=30).std() + 1e-12)).reindex(ix)
    F["rotation"] = F.pos_mean - bz
    return px, F


# Sign convention is NOT asserted here - the IC is reported raw and signed by
# what it actually is, because guessing signs is what S118b had to correct.
COLS = ["oi_breadth", "oi_agg", "pos_mean", "pos_disp", "retail_mean",
        "taker_mean", "taker_disp", "rotation"]

if __name__ == "__main__":
    px, F = build()
    print("S125 - cross-asset positioning as a BTCUSDT-only signal\n")
    print(f"daily, {px.index.min().date()} -> {px.index.max().date()}, "
          f"{len(px)} days ({len(px)/365.25:.1f} years)")
    print("eight alt markets, traded instrument is BTCUSDT perp and nothing else\n")

    lr = np.log(px)
    h = len(px) // 2
    print(f"{'feature':>14}{'cover':>7}" +
          "".join(f"{'IC ' + str(k) + 'd':>10}" for k in (1, 3, 5, 10)) +
          f"{'IC5 1st':>10}{'IC5 2nd':>10}{'stable':>8}")
    keep = []
    for c in COLS:
        x = F[c].shift(1)                      # strictly past
        cov = x.notna().mean() * 100
        ics = []
        for k in (1, 3, 5, 10):
            y = lr.diff(k).shift(-k)
            m = np.isfinite(x) & np.isfinite(y)
            ics.append(pd.Series(x[m].to_numpy()).corr(
                pd.Series(y[m].to_numpy()), method="spearman"))
        y5 = lr.diff(5).shift(-5)
        m = np.isfinite(x) & np.isfinite(y5)
        xa, ya = x[m], y5[m]
        hh = len(xa) // 2
        i1 = pd.Series(xa.iloc[:hh].to_numpy()).corr(
            pd.Series(ya.iloc[:hh].to_numpy()), method="spearman")
        i2 = pd.Series(xa.iloc[hh:].to_numpy()).corr(
            pd.Series(ya.iloc[hh:].to_numpy()), method="spearman")
        stable = np.sign(i1) == np.sign(i2) and abs(i1) > 0.02 and abs(i2) > 0.02
        if stable:
            keep.append((c, np.sign(i1)))
        print(f"{c:>14}{cov:>6.0f}%" + "".join(f"{v:>10.4f}" for v in ics) +
              f"{i1:>10.4f}{i2:>10.4f}{'YES' if stable else '':>8}")

    print(f"\nfeatures whose 5-day IC keeps its sign across both halves and "
          f"exceeds 0.02 in each: {len(keep)} of {len(COLS)}")
    for c, s in keep:
        print(f"   {c:>14}  sign {int(s):+d}")
    if not keep:
        print("   none. no book is built on this; there is nothing to build on.")
    print("\ndone: cross-asset diagnostic")
