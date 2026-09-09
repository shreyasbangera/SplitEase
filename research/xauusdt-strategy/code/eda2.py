import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
import numpy as np, pandas as pd, engine
IS_END='2022-12-31'

h = engine.bars('1h'); h=h[h.index<=IS_END]
r = np.log(h.close/h.close.shift(1))
d = pd.DataFrame({'r':r,'hr':h.index.hour,'dow':h.index.dayofweek})
print("=== hour 21,22,23 by weekday (bps) — is it just the Sunday gap? ===")
for hh in (21,22,23,0):
    s=d[d.hr==hh]
    piv=s.groupby('dow')['r'].agg(['mean','count','std'])
    piv['bps']=piv['mean']*1e4; piv['t']=piv['bps']/(piv['std']*1e4/np.sqrt(piv['count']))
    print(f"-- hour {hh}")
    print(piv[['bps','count','t']].round(2).to_string())

# ---------- large-move reversal ----------
print("\n\n=== REVERSAL: forward return after a k-bar move, by z-score bucket (15m, IS) ===")
b = engine.bars('15min'); b=b[b.index<=IS_END]
c = b.close
for k in (4, 8, 16):
    rk = np.log(c/c.shift(k))
    sd = rk.rolling(500).std()
    z  = rk/sd
    for m in (4, 8, 16):
        fwd = np.log(c.shift(-m)/c)          # forward, no look-ahead in signal (z uses past only)
        buckets = pd.cut(z, [-99,-3,-2,-1,1,2,3,99])
        gg = fwd.groupby(buckets, observed=True).agg(['mean','count','std'])
        gg['bps']=gg['mean']*1e4; gg['t']=gg['bps']/(gg['std']*1e4/np.sqrt(gg['count']))
        ext = gg.loc[[i for i in gg.index if i.left<=-2 or i.right>=2]]
        s = "  ".join(f"{str(i):>12}:{row.bps:+7.1f}bps(t{row.t:+5.1f},n{int(row['count'])})" for i,row in ext.iterrows())
        print(f" look{k:>2} fwd{m:>2} | {s}")
