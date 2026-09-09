import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
import numpy as np, pandas as pd, engine
pd.set_option('display.width',200)

IS_END='2022-12-31'
h = engine.bars('1h'); h = h[h.index<=IS_END]
print(f"IS 1h bars {len(h):,}  {h.index[0].date()} -> {h.index[-1].date()}")
r = np.log(h.close/h.close.shift(1))

# ---- hour of day
g = r.groupby(h.index.hour)
tab = pd.DataFrame({'mean_bps':g.mean()*1e4,'std_bps':g.std()*1e4,'n':g.size()})
tab['t'] = tab.mean_bps/ (tab.std_bps/np.sqrt(tab.n))
print("\n=== hourly return (UTC), IS ===")
print(tab.round(2).to_string())

# ---- day of week
g2 = r.groupby(h.index.dayofweek)
t2 = pd.DataFrame({'mean_bps':g2.mean()*1e4,'n':g2.size(),'std':g2.std()*1e4})
t2['t']=t2.mean_bps/(t2['std']/np.sqrt(t2.n))
print("\n=== day of week (0=Mon) ==="); print(t2.round(2).to_string())

# ---- autocorrelation structure: trend vs mean reversion by timeframe
print("\n=== return autocorr (lag1) by timeframe, IS ===")
for tf in ['5min','15min','1h','4h','1D']:
    b = engine.bars(tf); b=b[b.index<=IS_END]
    rr = np.log(b.close/b.close.shift(1)).dropna()
    print(f"  {tf:>6}: n={len(rr):>7}  ac1={rr.autocorr(1):+.4f}  ac2={rr.autocorr(2):+.4f}  ac5={rr.autocorr(5):+.4f}")

# ---- does volatility regime matter? (returns conditioned on prior range)
b15 = engine.bars('15min'); b15=b15[b15.index<=IS_END]
r15 = np.log(b15.close/b15.close.shift(1))
atr = (b15.high-b15.low).rolling(48).mean()
q = pd.qcut(atr.shift(1), 5, labels=False, duplicates='drop')
print("\n=== |15m return| and next-return autocorr by ATR quintile ===")
for k in range(5):
    m = q==k
    sub = r15[m]
    print(f"  Q{k+1}: n={m.sum():>6}  mean|r| {sub.abs().mean()*1e4:6.1f}bps  ac1 {sub.autocorr(1):+.4f}")
