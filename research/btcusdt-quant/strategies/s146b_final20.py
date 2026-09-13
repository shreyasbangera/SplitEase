import sys; sys.path.insert(0,"/home/user/quant")
import numpy as np, pandas as pd
import strategies.s135_more as S135, strategies.s144_blend as S144
import strategies.s146_maxsharpe as S146, strategies.s140_breakout as B
from strategies.s96_rank import at_gate, stats_of

def gate_at(a,target=-0.20,tol=0.01):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not len(a) or a.sum()<=0: return np.nan
    g=at_gate(a,target=target,lo=1e-3,hi=40.0,iters=60)
    return np.nan if (not np.isfinite(g["dd"]) or abs(g["dd"]-target)>tol) else g["cagr"]*100

def honest(a,target=-0.20,nboot=2000,block=90,seed=0,tol=0.005):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not len(a) or a.sum()<=0: return np.nan
    rng=np.random.default_rng(seed); T=len(a); nb=int(np.ceil(T/block))
    st=rng.integers(0,max(T-block,1),(nboot,nb))
    idx=np.clip((st[:,:,None]+np.arange(block)[None,None,:]).reshape(nboot,-1)[:,:T],0,T-1)
    def med(s):
        e=np.cumprod(1.0+a[idx]*s,axis=1)
        return float(np.median((e/np.maximum.accumulate(e,axis=1)-1).min(axis=1)))
    lo,hi=1e-3,40.0
    for _ in range(50):
        m=(lo+hi)/2
        if med(m)<target: hi=m
        else: lo=m
    s=(lo+hi)/2
    return np.nan if abs(med(s)-target)>tol else stats_of(a*s)["cagr"]*100

px,hi,lo,qv,tbq,fd=S135.base()
tr=pd.concat([hi-lo,(hi-px.shift(1)).abs(),(lo-px.shift(1)).abs()],axis=1).max(axis=1)
atr=tr.ewm(span=20,adjust=False).mean(); rng=hi-lo; lr=np.log(px).diff()
SL=S146.all_sleeves(px,hi,lo,rng,atr,lr)

# exclude the crowd180+crowd365 double-count (they correlate +0.94)
CAND={
 "rex144 + crowd365":("rex144","crowd365"),
 "rex144 + vov20_60 + crowd365":("rex144","vov20_60","crowd365"),
 "rex89 + rex144 + vov20_40 + crowd365":("rex89","rex144","vov20_40","crowd365"),
 "rex144 + expos + crowd365":("rex144","expos","crowd365"),
 "rex89+rex144 + vov20_60 + expos + crowd365":("rex89","rex144","vov20_60","expos","crowd365"),
 "rex144 alone (no decaying leg)":("rex144",),
 "rex89+rex144+vov20_60 (price only)":("rex89","rex144","vov20_60"),
}
print("BEST BOOKS AT A 20% DRAWDOWN — one crowding leg only, no double-counting\n")
print(f"{'book':>44}{'Shp':>6}{'gate':>8}{'honest':>9}{'flips':>7}{'PF':>7}{'last2y':>8}{'neg':>5}")
best=None
for tag,c in CAND.items():
    s=sum(SL[x] for x in c)/len(c)
    net,pos=S144.book(px,fd,s,tv=0.30)
    a=net.to_numpy()
    g,h=gate_at(a),honest(a)
    pf,trips=S144.pf_trips(net,pos)
    fl=int((np.abs(np.diff(np.r_[0.0,np.asarray(pos,float)]))>1e-9).sum())
    neg=sum(1 for _,v in B.yearly(net) if v<0)
    print(f"{tag:>44}{stats_of(a)['sharpe']:>6.2f}"
          + (f"{g:>7.1f}%" if np.isfinite(g) else f"{'n/a':>8}")
          + (f"{h:>8.1f}%" if np.isfinite(h) else f"{'n/a':>9}")
          + f"{fl:>7}{pf:>7.2f}{B.last2(net):>7.1f}%{neg:>5}")
    if np.isfinite(h) and (best is None or h>best[0]): best=(h,tag,net,g,fl,pf,neg)

print(f"\nBEST BY HONEST GATE: {best[1]}")
print(f"   gate {best[3]:.1f}%   honest {best[0]:.1f}%   {best[4]} position changes   "
      f"PF {best[5]:.2f}   {best[6]} negative years")
print("   year by year: " + "  ".join(f"{y} {v:+.1f}%" for y,v in B.yearly(best[2])))
print(f"\n300% at a 20% drawdown requires Sharpe 6.58. Best here is "
      f"{stats_of(best[2].to_numpy())['sharpe']:.2f}.")
