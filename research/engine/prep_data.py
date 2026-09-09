"""Build clean OHLCV datasets for backtesting.

Sources (all public, fetched over plain https):
  GOLD  : github.com/FeziweMelvin/XAUUSD-Gold-Price  XAU_1h_data.csv
          MetaTrader broker feed for spot gold XAU/USD, 2004-06-11 -> 2025-06-06, 1h bars.
  GOLD2 : github.com/ejtraderLabs/historical-data  XAUUSD{m15,m30,h1,h4,d1}.csv
          Independent MT5 broker feed, 2012-05 -> 2022-03 (prices are x100 ints).
  BTC   : github.com/ff137/bitstamp-btcusd-minute-data
          Bitstamp BTC/USD 1-minute OHLCV, 2012-01-01 -> present.
"""
import os, sys
import numpy as np
import pandas as pd

RAW = "/tmp/claude-0/-home-user-SplitEase/9d81b630-73e8-5554-a848-3dc9a73c8211/scratchpad/data"
OUT = "/tmp/claude-0/-home-user-SplitEase/9d81b630-73e8-5554-a848-3dc9a73c8211/scratchpad/clean"
os.makedirs(OUT, exist_ok=True)


def _finish(df, name):
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[(df.high >= df.low) & (df.low > 0)]
    # basic sanity: high/low must bracket open/close
    df = df[(df.high >= df[["open", "close"]].max(axis=1) - 1e-9) &
            (df.low <= df[["open", "close"]].min(axis=1) + 1e-9)]
    df.to_parquet(f"{OUT}/{name}.parquet")
    print(f"{name:12s} {len(df):>8,} bars  {df.index[0]} -> {df.index[-1]}")
    return df


def resample(df, rule):
    o = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return o.dropna(subset=["open"])


def gold():
    df = pd.read_csv(f"{RAW}/gold_h1_raw.csv", sep=";")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d %H:%M")
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    h1 = _finish(df, "gold_h1")
    _finish(resample(h1, "4h"), "gold_h4")
    _finish(resample(h1, "1D"), "gold_d1")
    return h1


def gold2():
    for tf, name in [("m15", "gold2_m15"), ("h1", "gold2_h1"), ("h4", "gold2_h4")]:
        df = pd.read_csv(f"{RAW}/XAUUSD_{tf}.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={"tick_volume": "volume"})
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float) / 100.0   # feed stores price * 100
        _finish(df[["open", "high", "low", "close", "volume"]], name)


def btc():
    a = pd.read_csv(f"{RAW}/btc_1min.csv.gz")
    b = pd.read_csv(f"{RAW}/btc_1min_latest.csv")
    df = pd.concat([a, b], ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("ts")[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # 2012-2013 liquidity is unusable; start at 2014-01-01
    df = df.loc["2014-01-01":]
    for rule, name in [("15min", "btc_m15"), ("1h", "btc_h1"), ("4h", "btc_h4"), ("1D", "btc_d1")]:
        r = resample(df, rule)
        # drop bars with zero volume (exchange downtime -> flat synthetic candles)
        r = r[r.volume > 0]
        _finish(r, name)


if __name__ == "__main__":
    gold(); gold2(); btc()
