"""Event-based bars: one bar per N dollars traded, instead of one per N hours.

S101 and S102 found the most serious fragility in this study. V7's result lives
in a single cell of a two-dimensional CLOCK grid - twelve hours, anchored at
00:00 UTC - and every neighbouring cell is 2.4x to 10x worse: 16.4% and 46.3%
at the adjacent phases, 43.8% and 29.5% at the adjacent lengths. The book's
headline depends on a sampling choice nobody deliberately made.

A dollar bar has no phase and no length. It closes when a fixed amount of value
has changed hands, so the grid is set by the market's own activity rather than
by a wall clock, and the question "which twelve hours?" cannot be asked of it.
If the edge survives here, that fragility is answered rather than argued about.

There is a second, older reason to prefer them. Returns sampled in volume time
are closer to independent and identically distributed than returns sampled in
clock time - clock bars oversample dead hours and undersample the minutes that
carry the information - so every z-score in the book is being computed over a
window whose statistical content varies with the time of day. Volume time
removes that too.

THE THRESHOLD, AND WHY IT HAS TO ADAPT
--------------------------------------
BTC dollar volume grew by more than an order of magnitude across this sample. A
FIXED threshold therefore produces bars of wildly different duration at the two
ends of the record - multi-day bars in 2021, hourly bars in 2026 - which would
confound "event sampling" with "sampling frequency rising through the sample".

So the threshold tracks a trailing 30-day mean of daily dollar volume, divided
to target a chosen number of bars per day. The trailing window is shifted a full
day into the past before it is used, so the threshold applied to a bar is fixed
before that bar opens. Both variants are built here and both are reported: the
adaptive one is the like-for-like comparison against a 12h book, the fixed one
is the check that the adaptation is not doing the work.
"""
import numpy as np, pandas as pd

_NB = ("open", "high", "low", "close", "volume", "quote_volume", "count",
       "taker_buy_base")


def _edges_fixed(qv, thr):
    """Bar index for each base bar under a constant dollar threshold."""
    c = np.cumsum(qv)
    return np.floor_divide(c, thr).astype(np.int64)


def _edges_adaptive(qv, thr):
    """Bar index under a per-base-bar threshold. Closes as soon as the value
    accumulated since the last close reaches the threshold in force AT THE OPEN
    of the current bar, so nothing about the future enters the boundary."""
    n = len(qv)
    out = np.empty(n, np.int64)
    k, acc, lim = 0, 0.0, thr[0]
    for i in range(n):
        out[i] = k
        acc += qv[i]
        if acc >= lim:
            k += 1
            acc = 0.0
            lim = thr[i]            # known at i, applies to the bar opening at i+1
    return out


def thresholds(df5, per_day=2.0, win_days=30, min_bars=6):
    """Trailing-mean dollar volume per bar, lagged a full day."""
    d = df5.set_index("dt")["quote_volume"]
    daily = d.resample("1D").sum()
    trail = daily.rolling(win_days, min_periods=min_bars).mean().shift(1)
    trail = trail.reindex(d.index, method="ffill")
    v = np.array((trail / per_day).to_numpy(float), copy=True)
    ok = np.isfinite(v) & (v > 0)
    if ok.any():                                  # warm-up before any history
        v[~ok] = v[ok][0]
    return v


def build_bars(df5, mode="adaptive", per_day=2.0, thr=None):
    """Aggregate 5-minute bars into event bars. Label = bar OPEN time."""
    d = df5.sort_values("dt").reset_index(drop=True)
    qv = d.quote_volume.to_numpy(float)
    if mode == "adaptive":
        idx = _edges_adaptive(qv, thresholds(d, per_day=per_day))
    else:
        idx = _edges_fixed(qv, thr if thr is not None else np.nanmean(qv) * 144.0)
    g = d.groupby(idx)
    out = pd.DataFrame({
        "dt": g.dt.first(),
        "open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
        "close": g.close.last(), "volume": g.volume.sum(),
        "quote_volume": g.quote_volume.sum(), "count": g["count"].sum(),
        "taker_buy_base": g.taker_buy_base.sum()}).reset_index(drop=True)
    # The final group is whatever had accumulated when the data ran out; it is
    # not a closed bar and must not be traded on.
    return out.iloc[:-1].reset_index(drop=True)


def to_edges(df, edges_dt):
    """Aggregate any OHLCV frame onto bar boundaries defined by `edges_dt`.

    Used for the spot leg, which has to be cut on exactly the same boundaries as
    the perp or the basis is a comparison between two different bars.
    """
    d = df.sort_values("dt").reset_index(drop=True)
    k = np.searchsorted(edges_dt, d.dt.to_numpy(), side="right") - 1
    m = k >= 0
    d, k = d[m], k[m]
    g = d.groupby(k)
    out = pd.DataFrame({
        "open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
        "close": g.close.last(), "volume": g.volume.sum()})
    out["dt"] = edges_dt[out.index.to_numpy()]
    return out.reset_index(drop=True)[["dt", "open", "high", "low", "close", "volume"]]


def describe(bars, label=""):
    dur = pd.Series(bars.dt).diff().dt.total_seconds() / 3600.0
    y = pd.Series(bars.dt).dt.year
    per_year = y.value_counts().sort_index()
    return (f"{label:>22}  {len(bars):6d} bars   duration h: median {dur.median():5.2f} "
            f"p10 {dur.quantile(0.1):5.2f} p90 {dur.quantile(0.9):6.2f} "
            f"max {dur.max():7.1f}   bars/yr " +
            " ".join(f"{int(k)}:{v}" for k, v in per_year.items()))
