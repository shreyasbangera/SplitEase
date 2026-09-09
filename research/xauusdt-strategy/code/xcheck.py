import pandas as pd, numpy as np
D="/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/data"
W="/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work"
m1 = pd.read_parquet(f"{W}/m1_clean.parquet")

# resample M1 -> M5 to compare with independent Anilk2978 M5 file
m5_mine = m1.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()

a = pd.read_csv(f"{D}/Anilk2978_xauusd-5yr-data/XAUUSDm_M5_.csv", sep='\t')
a.columns=[c.strip('<>').lower() for c in a.columns]
a['time']=pd.to_datetime(a['date']+' '+a['time'], format='%Y.%m.%d %H:%M:%S', utc=True)
a=a.set_index('time').sort_index()
print(f"Anilk M5: {len(a):,}  {a.index[0]} -> {a.index[-1]}")

j = m5_mine.join(a[['open','high','low','close','spread']], how='inner', rsuffix='_b')
print(f"overlapping M5 bars: {len(j):,}")
for c in ['open','high','low','close']:
    d=(j[c]-j[c+'_b']).abs()
    print(f"  {c}: mean abs diff ${d.mean():.4f}  p99 ${d.quantile(.99):.4f}  max ${d.max():.3f}  corr {j[c].corr(j[c+'_b']):.8f}")

# third source
g = pd.read_csv(f"{D}/getdata-finance_xauusd-1m-ohlcv-metals-historical-data/XAUUSD_1m.csv")
g['datetime']=pd.to_datetime(g['datetime'], utc=True); g=g.set_index('datetime').sort_index()
print(f"\ngetdata 1m: {len(g):,}  {g.index[0]} -> {g.index[-1]}")
j2 = m1[['close']].join(g[['close']], how='inner', rsuffix='_g')
print(f"overlapping M1 bars: {len(j2):,}")
if len(j2):
    d=(j2['close']-j2['close_g']).abs()
    print(f"  close: mean abs diff ${d.mean():.4f}  p99 ${d.quantile(.99):.4f}  corr {j2['close'].corr(j2['close_g']):.8f}")
    # correlation of returns is the real test
    r1=np.log(j2['close']).diff(); r2=np.log(j2['close_g']).diff()
    print(f"  1m log-return corr: {r1.corr(r2):.6f}")
