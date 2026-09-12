"""Macro series aligned causally onto the 12h BTC decision grid.

Every one of the twelve sources the study calls closed - OHLCV, taker volume,
trade count, funding, open interest, trader positioning, quarterly futures, the
coin-margined contract, tick prints, the order book, the volatility index, the
options chain - is a Binance endpoint. Nothing outside crypto has ever been
tested. That matters here specifically because the book earns +6% in 2022, the
one year in the sample when macro, not crypto microstructure, set BTC's
direction.

THE ALIGNMENT IS THE WHOLE PROBLEM
----------------------------------
A macro series is daily and stamped with a trading DATE; the book decides on
12h bars stamped by OPEN time and acting at CLOSE. Getting that wrong is how a
macro signal manufactures an edge out of nothing, so the lag here is
deliberately more conservative than reality requires:

  * `strict`   - a value dated D is treated as knowable only from D+1 22:00 UTC.
                 That is ~25 hours after the US cash close on D. No publication,
                 revision or timezone subtlety can reach back through it.
  * `prompt`   - dated D knowable from D 22:00 UTC, ~45 minutes after the close.
                 Closer to the truth and used only as a contrast: if a feature
                 works under `prompt` and dies under `strict`, the edge lives
                 entirely in the 24 hours the strict lag removes, which for a
                 slow regime read is a warning, not a finding.

Weekends and holidays carry the last value forward, which is what a trader
would actually see.
"""
import numpy as np
import pandas as pd

DIR = "/home/user/quant/data/macro"

# A 12h bar is labelled by OPEN time and closes 12 hours later; the decision is
# taken at close. Everything below merges on that close.
BAR_H = 12


def load_csv(name, date_col="DATE", value_col="CLOSE"):
    d = pd.read_csv(f"{DIR}/{name}.csv")
    d = d.rename(columns={date_col: "date", value_col: "val"})[["date", "val"]]
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna().sort_values("date").reset_index(drop=True)


def knowable(d, mode="strict"):
    """Stamp each dated observation with the time it could first be acted on."""
    off = pd.Timedelta(days=1, hours=22) if mode == "strict" else pd.Timedelta(hours=22)
    out = d.copy()
    # The dates carry no zone; the offsets above are quoted in UTC, so they are
    # read as UTC. Both sides of the merge must agree on that.
    at = out["date"] + off
    out["at"] = at.dt.tz_localize("UTC") if at.dt.tz is None else at.dt.tz_convert("UTC")
    return out[["at", "val"]]


def onto_bars(bars_dt, series, mode="strict", tf_hours=BAR_H):
    """Last value knowable at each bar's CLOSE, forward-filled across weekends."""
    close = pd.Series(pd.to_datetime(bars_dt)) + pd.Timedelta(hours=tf_hours)
    k = knowable(series, mode)
    # merge_asof demands identical dtypes down to the resolution, and the panel
    # and the CSV do not agree on it.
    unit = "datetime64[ns, UTC]"
    close = close.astype(unit)
    k["at"] = k["at"].astype(unit)
    m = pd.merge_asof(
        pd.DataFrame({"at": close}).sort_values("at"),
        k.sort_values("at"),
        on="at", direction="backward",
    )
    return m["val"].to_numpy(float)


def dollar():
    """A DXY-weighted dollar index built from the Fed's H.10 daily rates.

    H.10 quotes every one of the six DXY constituents the same way - foreign
    currency units per USD - so the index is a geometric weighted average with
    all-positive exponents and rising means a stronger dollar. The weights are
    the published DXY weights; the constant is omitted because only changes and
    z-scores are ever used.

    H.10 carries the noon rate for day D and is published around 16:15 ET the
    same day, so the `strict` lag above clears it by roughly nine hours on top
    of a full day.
    """
    w = {"Euro": 0.576, "Japan": 0.136, "United Kingdom": 0.119,
         "Canada": 0.091, "Sweden": 0.042, "Switzerland": 0.036}
    d = pd.read_csv(f"{DIR}/fx_raw.csv")
    d.columns = ["date", "country", "rate"]
    d = d[d.country.isin(w)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["rate"] = pd.to_numeric(d["rate"], errors="coerce")
    p = d.pivot_table(index="date", columns="country", values="rate").dropna()
    lv = sum(np.log(p[c]) * wt for c, wt in w.items())
    out = pd.DataFrame({"date": p.index, "val": np.exp(lv).to_numpy()})
    return out.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ features

def z(x, n):
    s = pd.Series(x)
    return ((s - s.rolling(n).mean()) / s.rolling(n).std()).to_numpy(float)


def features(v, bars_per_day=2):
    """Level and change reads on one macro series, on the 12h grid.

    Windows are given in DAYS and converted, so a daily macro series and the
    book's own 12h signals are looking over the same wall-clock span.
    """
    d = bars_per_day
    s = pd.Series(v)
    out = {}
    out["lvl_z60"] = z(v, 60 * d)
    out["lvl_z120"] = z(v, 120 * d)
    for k in (1, 3, 5, 10, 20):
        out[f"chg{k}d"] = z(s.diff(k * d).to_numpy(float), 120 * d)
    out["lchg5d"] = z(np.log(s).diff(5 * d).to_numpy(float), 120 * d)
    return pd.DataFrame(out)
