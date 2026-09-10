import zipfile, glob, io, os, sys
import pandas as pd, numpy as np

COLS = ["open_time","open","high","low","close","volume","close_time",
        "quote_volume","count","taker_buy_base","taker_buy_quote","ignore"]

def read_dir(d):
    frames = []
    for f in sorted(glob.glob(os.path.join(d, "*.zip"))):
        with zipfile.ZipFile(f) as z:
            nm = z.namelist()[0]
            raw = z.read(nm).decode()
        first = raw.split("\n", 1)[0]
        hdr = 0 if first.lower().startswith("open_time") else None
        df = pd.read_csv(io.StringIO(raw), header=hdr, names=None if hdr == 0 else COLS)
        df = df.iloc[:, :12]
        df.columns = COLS[:df.shape[1]]          # normalise positionally (headers vary by market/era)
        frames.append(df[COLS[:11]])
    df = pd.concat(frames, ignore_index=True)
    # Binance switched open_time from milliseconds to microseconds in 2025.
    ot = df.open_time.astype("int64")
    unit = np.where(ot > 1e15, "us", "ms")
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    m_us = ot > 1e15
    dt[m_us]  = pd.to_datetime(ot[m_us],  unit="us", utc=True)
    dt[~m_us] = pd.to_datetime(ot[~m_us], unit="ms", utc=True)
    df["dt"] = dt
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    return df[["dt","open","high","low","close","volume","quote_volume","count",
               "taker_buy_base","taker_buy_quote"]]

def report(name, df, freq):
    d = df.dt.diff().dropna()
    exp = pd.Timedelta(freq)
    gaps = d[d != exp]
    miss = int(((gaps / exp) - 1).sum()) if len(gaps) else 0
    bad = int((df.high < df[["open","close"]].max(axis=1) - 1e-9).sum() +
              (df.low  > df[["open","close"]].min(axis=1) + 1e-9).sum() +
              (df.high < df.low).sum())
    print(f"{name:<12} n={len(df):>7}  {df.dt.iloc[0]:%Y-%m-%d} -> {df.dt.iloc[-1]:%Y-%m-%d %H:%M}  "
          f"gapruns={len(gaps):<4} missing_bars={miss:<6} ohlc_violations={bad}  "
          f"px {df.low.min():,.0f}-{df.high.max():,.0f}")
    return df

if __name__ == "__main__":
    out = "/home/user/quant/data"
    for name, d, freq in [("spot_1h","raw/spot_1h","1h"), ("fut_1h","raw/fut_1h","1h"),
                          ("spot_15m","raw/spot_15m","15min"), ("fut_15m","raw/fut_15m","15min"),
                          ("fut_5m","raw/fut_5m","5min")]:
        df = read_dir(os.path.join(out, d))
        report(name, df, freq)
        df.to_parquet(os.path.join(out, f"{name}.parquet"))
    # funding
    fr = []
    for f in sorted(glob.glob(os.path.join(out, "raw/funding/*.zip"))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        first = raw.split("\n",1)[0]
        hdr = 0 if not first[0].isdigit() else None
        x = pd.read_csv(io.StringIO(raw), header=hdr,
                        names=None if hdr==0 else ["calc_time","funding_interval_hours","funding_rate"])
        x.columns=[c.strip() for c in x.columns]
        fr.append(x)
    fr = pd.concat(fr, ignore_index=True)
    tcol = [c for c in fr.columns if "time" in c.lower()][0]
    rcol = [c for c in fr.columns if "rate" in c.lower()][0]
    t = fr[tcol].astype("int64")
    m_us = t > 1e15
    dt = pd.Series(pd.NaT, index=fr.index, dtype="datetime64[ns, UTC]")
    dt[m_us]  = pd.to_datetime(t[m_us], unit="us", utc=True)
    dt[~m_us] = pd.to_datetime(t[~m_us], unit="ms", utc=True)
    f2 = pd.DataFrame({"dt": dt, "rate": pd.to_numeric(fr[rcol], errors="coerce")}).dropna()
    f2 = f2.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    f2.to_parquet(os.path.join(out, "funding.parquet"))
    print(f"{'funding':<12} n={len(f2):>7}  {f2.dt.iloc[0]:%Y-%m-%d} -> {f2.dt.iloc[-1]:%Y-%m-%d}  "
          f"mean={f2.rate.mean()*100:.4f}%/8h  ann={(1+f2.rate.mean())**(3*365)-1:.1%}  "
          f"pos_frac={(f2.rate>0).mean():.1%}")
