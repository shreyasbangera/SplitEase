"""
S151 - The point-in-time universe: every perpetual Binance ever listed.

THE DEFECT THIS FIXES
---------------------
S147 to S150 ran on 16 coins that are still listed and liquid in 2026. That is
survivorship selection in the universe itself, and in crypto it is not a rounding
error: LUNA went from a $40bn market cap to zero in four days in May 2022, FTT
followed in November, and neither is in a panel built from what exists today.
Every result so far was measured in a world where coins do not die.

This rebuilds the universe from Binance's own archive, which keeps the history of
delisted contracts. 855 USDT-margined perpetuals ever listed, of which 391 have
history before 2025. A coin enters the universe when it starts trading and enough
history accumulates, and leaves it when its bars stop - which is what delisting
actually looks like from the inside.

WHY THIS SHOULD CUT BOTH WAYS, AND WHICH WAY IT ACTUALLY CUTS
--------------------------------------------------------------
The naive expectation is that survivorship inflates results. For a LONG-ONLY
book that is right. For the factors validated here it is not obvious:

    low volatility   dying coins are violently volatile, so they sit in the
                     SHORT leg. Excluding them removes profitable shorts, and
                     the honest universe should score BETTER.
    carry            a collapsing coin has extreme funding, and again the book
                     is short it. Same direction.
    breakout         a coin that falls 99% is a large sustained downside move,
                     which is what a breakout rule is built to catch.

So including the dead names could help. It could also hurt badly, because a
coin that gaps to zero over a weekend is unexitable and a short in it may not
be collectable at all. Which of those dominates is measured rather than assumed,
and the shorting caveat is quantified in S152 rather than waved at.

The universe is still not complete: it is Binance-only, and a coin that never
listed there is invisible. That is a real limit and it is not claimed otherwise.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, glob, os, zipfile, io, re

D = "/home/user/quant/data"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "tb_base", "tb_quote", "ignore"]


def _read_zip(p):
    try:
        with zipfile.ZipFile(p) as z:
            n = z.namelist()[0]
            raw = z.read(n)
    except Exception:
        return None
    head = raw[:64].decode("utf-8", "ignore")
    hdr = 0 if head[:1].isdigit() else "infer"
    try:
        d = pd.read_csv(io.BytesIO(raw), header=hdr, names=COLS if hdr == 0 else None)
    except Exception:
        return None
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "open_time" not in d.columns:
        return None
    d = d[pd.to_numeric(d["open_time"], errors="coerce").notna()]
    if not len(d):
        return None
    t = pd.to_numeric(d["open_time"])
    unit = "us" if t.max() > 1e14 else ("ms" if t.max() > 1e11 else "s")
    d.index = pd.DatetimeIndex(pd.to_datetime(t.to_numpy(), unit=unit, utc=True)
                               ).tz_localize(None).normalize()
    for c in ("open", "high", "low", "close", "quote_volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["open", "high", "low", "close", "quote_volume"]]


def build(cache=f"{D}/pit_panel.parquet", rebuild=False, min_bars=120,
          min_adv=5e6):
    """Wide daily panels of close / high / low / dollar volume.

    Symbols are filtered as they are read - fewer than `min_bars` bars, or a
    peak 30-day dollar volume below `min_adv`, and the symbol never enters the
    panel at all. Building all 855 columns at full width exhausted memory; this
    keeps only what could ever be tradeable, and stores float32.
    """
    if os.path.exists(cache) and not rebuild:
        w = pd.read_parquet(cache)
        return {k: w.xs(k, axis=1, level=0) for k in ("close", "high", "low", "qv")}
    files = sorted(glob.glob(f"{D}/pit/*.zip"))
    bysym = {}
    for f in files:
        bysym.setdefault(os.path.basename(f).split("__")[0], []).append(f)
    cl, hi, lo, qv = {}, {}, {}, {}
    for s, fs in sorted(bysym.items()):
        parts = [x for x in (_read_zip(f) for f in sorted(fs)) if x is not None]
        if not parts:
            continue
        d = pd.concat(parts).sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if len(d) < min_bars:
            continue
        if float(d["quote_volume"].rolling(30, min_periods=20).median().max()
                 or 0.0) < min_adv:
            continue
        cl[s] = d["close"].astype("float32")
        hi[s] = d["high"].astype("float32")
        lo[s] = d["low"].astype("float32")
        qv[s] = d["quote_volume"].astype("float32")
    ix = pd.date_range(min(v.index.min() for v in cl.values()),
                       max(v.index.max() for v in cl.values()), freq="1D")
    out = {"close": pd.DataFrame(cl).reindex(ix),
           "high": pd.DataFrame(hi).reindex(ix),
           "low": pd.DataFrame(lo).reindex(ix),
           "qv": pd.DataFrame(qv).reindex(ix)}
    pd.concat(out, axis=1).to_parquet(cache)
    return out


MIN_HIST, MIN_ADV = 120, 25e6


def mask(pan, min_hist=MIN_HIST, min_adv=MIN_ADV):
    """Tradeable on day t using only data before t, plus a bar existing on t."""
    px, dv = pan["close"], pan["qv"]
    hist = px.notna().cumsum()
    adv = dv.rolling(30, min_periods=20).median()
    return (hist.shift(1) >= min_hist) & (adv.shift(1) >= min_adv) & px.notna()


if __name__ == "__main__":
    pan = build(rebuild=True)
    px = pan["close"]
    ok = mask(pan)
    print("S151 - the point-in-time universe\n")
    print(f"symbols parsed           {px.shape[1]}")
    print(f"span                     {px.index.min().date()} -> {px.index.max().date()}")
    n = ok.sum(axis=1)
    print(f"tradeable names per day  median {int(n.median())}, min {int(n.min())}, "
          f"max {int(n.max())}")
    print("\nBREADTH BY YEAR")
    for y, v in n.groupby(n.index.year):
        print(f"   {y}: median {int(v.median()):>4} tradeable  "
              f"(min {int(v.min()):>3}, max {int(v.max()):>3})")

    print("\nDEATHS - names that stop trading while listed and liquid")
    last = px.apply(lambda s: s.last_valid_index())
    endix = px.index.max()
    dead = last[last < endix - pd.Timedelta(days=30)]
    wastradeable = ok.any()
    dead = dead[[s for s in dead.index if wastradeable.get(s, False)]]
    print(f"   {len(dead)} of {int(wastradeable.sum())} ever-tradeable names "
          f"stopped trading before {endix.date()}")
    print("\n   the 12 largest final drawdowns among them:")
    rows = []
    for s in dead.index:
        v = px[s].dropna()
        if len(v) < 120:
            continue
        rows.append((s, str(v.index[-1].date()), v.iloc[-1] / v.max() - 1,
                     np.log(v).diff().tail(90).sum()))
    rows.sort(key=lambda r: r[2])
    print(f"      {'symbol':>14}{'last bar':>13}{'from peak':>12}{'last 90d':>10}")
    for s, d, dd, l90 in rows[:12]:
        print(f"      {s:>14}{d:>13}{dd*100:>11.1f}%{(np.exp(l90)-1)*100:>9.1f}%")

    print("\n   LUNA specifically, since it is the canonical case:")
    for s in ("LUNAUSDT", "FTTUSDT", "SRMUSDT", "ANCUSDT"):
        if s in px.columns:
            v = px[s].dropna()
            if len(v):
                tr = ok[s].sum() if s in ok.columns else 0
                print(f"      {s:>10} {str(v.index[0].date())} -> "
                      f"{str(v.index[-1].date())}, {len(v):>4} bars, "
                      f"{int(tr):>4} tradeable, peak-to-last "
                      f"{(v.iloc[-1]/v.max()-1)*100:>7.1f}%")
        else:
            print(f"      {s:>10} not in panel")
    print("\ndone: point-in-time universe")
