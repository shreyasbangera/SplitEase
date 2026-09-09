import pandas as pd, numpy as np
D="/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/data"
OUT="/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work"

m1 = pd.read_parquet(f"{D}/sherwynjoel_xauusd-historical-data/xauusd_m1_full.parquet")
m1.index = pd.to_datetime(m1.index, utc=True)
m1 = m1.sort_index()
m1 = m1[~m1.index.duplicated(keep='first')]
print(f"M1 bars {len(m1):,}  {m1.index[0]} -> {m1.index[-1]}")

# --- integrity checks ---
bad_ohlc = ((m1.high < m1.low) | (m1.high < m1.open) | (m1.high < m1.close) |
            (m1.low > m1.open) | (m1.low > m1.close)).sum()
nonpos = (m1[['open','high','low','close']] <= 0).any(axis=1).sum()
nans = m1[['open','high','low','close']].isna().any(axis=1).sum()
print(f"invalid OHLC rows: {bad_ohlc}   nonpositive: {nonpos}   NaN: {nans}")

# bar-to-bar jumps (data glitch detector)
r = np.log(m1.close/m1.close.shift(1))
print(f"max |1m log ret|: {np.nanmax(np.abs(r)):.4f}   >2% moves: {(np.abs(r)>0.02).sum()}")
print("largest jumps:")
big = r.abs().nlargest(5)
for t,v in big.items(): print(f"   {t}  {v*100:+.2f}%  close={m1.close.loc[t]:.2f}")

# spread column sanity (points, 3-dec pricing => 1 pt = 0.001)
sp = m1['spread']
print(f"spread pts: median={sp.median():.0f} p90={sp.quantile(.9):.0f} p99={sp.quantile(.99):.0f} max={sp.max():.0f}")
print(f"  -> USD: median={sp.median()*0.001:.3f} p90={sp.quantile(.9)*0.001:.3f} p99={sp.quantile(.99)*0.001:.3f}")
print(f"  spread==0 fraction: {(sp==0).mean():.3f}")

# coverage per year
cov = m1.groupby(m1.index.year).size()
print("bars/year:"); print(cov.to_string())

m1.to_parquet(f"{OUT}/m1_clean.parquet")
print("saved m1_clean.parquet")
