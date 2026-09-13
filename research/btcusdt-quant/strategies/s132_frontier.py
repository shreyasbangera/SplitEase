import sys; sys.path.insert(0,"/home/user/quant")
import numpy as np, pandas as pd
import strategies.s110_meta as M, strategies.s87_combined as S87
from strategies.s96_rank import stats_of
from research.robust import bootstrap_dd

M.use_clock(); R = S87.rankings()
print("V7 RISK FRONTIER — what you actually get at each bet size")
print("drawdown probabilities from a 90-day block bootstrap, 4000 paths\n")
print(f"{'risk':>7}{'CAGR':>9}{'Sharpe':>8}{'realDD':>9}{'medianDD':>10}"
      f"{'P(DD>20%)':>11}{'P(DD>30%)':>11}{'P(DD>50%)':>11}{'trades':>8}")
rows=[]
for rk in (0.04,0.06,0.08,0.10,0.12,0.144,0.18,0.22):
    r,pnl = S87.blend(R,3,rk)
    a=np.asarray(r,float); a=a[np.isfinite(a)]
    st=stats_of(a); b=bootstrap_dd(a,n=4000,block=90)
    e=np.cumprod(1+a[None,:]*1.0)
    dd30=float((np.array([1.0])<0).mean())
    # recompute tail probs directly from the bootstrap paths
    rng=np.random.default_rng(0); T=len(a); nb=int(np.ceil(T/90))
    st_=rng.integers(0,max(T-90,1),(4000,nb))
    idx=np.clip((st_[:,:,None]+np.arange(90)[None,None,:]).reshape(4000,-1)[:,:T],0,T-1)
    eq=np.cumprod(1.0+a[idx],axis=1)
    mdd=(eq/np.maximum.accumulate(eq,axis=1)-1).min(axis=1)
    p20,p30,p50=(mdd<-0.20).mean(),(mdd<-0.30).mean(),(mdd<-0.50).mean()
    rows.append((rk,st['cagr']*100,p20,p30))
    print(f"{rk*100:>6.1f}%{st['cagr']*100:>8.1f}%{st['sharpe']:>8.2f}"
          f"{st['dd']*100:>8.1f}%{np.median(mdd)*100:>9.1f}%"
          f"{p20*100:>10.0f}%{p30*100:>10.0f}%{p50*100:>10.0f}%{len(pnl):>8}")

print("\n--- the trade-off, stated plainly ---")
for rk,c,p20,p30 in rows:
    if abs(rk-0.08)<1e-9 or abs(rk-0.144)<1e-9 or abs(rk-0.22)<1e-9:
        print(f"   at {rk*100:.1f}% risk: ~{c:.0f}% a year, but a drawdown worse than "
              f"20% happens on {p20*100:.0f}% of paths and worse than 30% on {p30*100:.0f}%")
