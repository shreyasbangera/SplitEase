"""Discrete 1m mean-reversion trades. Net edge AFTER actual per-trade spread+comm+slip.
   Searching for ANY filtered subset where net edge > 0."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
m1=engine.m1()
c=m1['close']; o=m1['open']; spr=m1['spread']*0.001
hr=c.index.hour
ISm=(c.index>=IS_START)&(c.index<=IS_END); OOSm=(c.index>=OOS_START)&(c.index<=OOS_END)
spq=spr.rolling(1440*5).rank(pct=True)

print("Net edge per trade AFTER costs (spread + 0.04bps comm + $0.03/side slip).")
print(f"{'k':>3}{'z':>5}{'hold':>5}{'spread filt':>13}{'hours':>10}{'trades/yr':>10}"
      f"{'gross$':>9}{'cost$':>8}{'NET$':>8}{'t':>7}{'OOSnet$':>9}")
print("-"*95)
res=[]
for k in (3,5,15):
    rk_=np.log(c/c.shift(k)); z=rk_/rk_.rolling(1000).std()
    for zt in (2.5,3.5,4.5):
        sig=((z<-zt)&(z.shift(1)>=-zt)).astype(int)-((z>zt)&(z.shift(1)<=zt)).astype(int)
        for hold in (5,15,30):
            fwd=(c.shift(-hold)-o.shift(-1))
            for spf,spname in [(1.01,'any'),(0.50,'tightest 50%'),(0.25,'tightest 25%')]:
                for hrf,hrname in [(None,'all'),((7,17),'07-17 UTC')]:
                    base=(sig!=0)&fwd.notna()&(spq<=spf)
                    if hrf: base&= (hr>=hrf[0])&(hr<hrf[1])
                    def stat(mask):
                        mm=base&mask
                        n=int(mm.sum())
                        if n<300: return None
                        g=(sig[mm]*fwd[mm])
                        cst=spr[mm]+2*0.03+ (c[mm]*2*0.04/1e4)
                        net=g-cst
                        return n,g.mean(),cst.mean(),net.mean(),net.mean()/(net.std()/np.sqrt(n))
                    a=stat(ISm); b=stat(OOSm)
                    if not a or not b: continue
                    n,gm,cm,nm,t=a
                    yrs=5.68; res.append((nm,t,k,zt,hold,spname,hrname,n/yrs,gm,cm,nm,b[3]))
res.sort(key=lambda r:-r[0])
for nm,t,k,zt,hold,spname,hrname,tpy,gm,cm,_,oosnet in res[:22]:
    flag=' <== NET POSITIVE' if nm>0 and oosnet>0 else ''
    print(f"{k:>3}{zt:>5.1f}{hold:>5}{spname:>13}{hrname:>10}{tpy:>10.0f}"
          f"{gm:>9.3f}{cm:>8.3f}{nm:>8.3f}{t:>7.2f}{oosnet:>9.3f}{flag}")
pos=[r for r in res if r[0]>0 and r[11]>0]
print(f"\nConfigurations with NET-positive edge in BOTH IS and OOS: {len(pos)}")
