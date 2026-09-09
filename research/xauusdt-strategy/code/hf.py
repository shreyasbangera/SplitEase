"""1-minute mean-reversion, continuous & vol-targeted. Signal at bar close, held from NEXT bar.
   Run at zero cost to establish the CEILING independent of execution quality."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

m1=engine.m1()
c=m1['close']; ret=c.pct_change(); spr=m1['spread']*0.001
BPY=1440*252   # minute bars per year (approx, 24/5)

def run(k=5, dead=0.0, comm_bps=0.0, slip=0.0, use_spread=False, tgt=0.30,
        maxlev=10, s=IS_START, e=IS_END, clip=3.0):
    rk_=np.log(c/c.shift(k))
    z=rk_/rk_.rolling(1000).std()
    score=(-z).clip(-clip,clip)/clip                    # fade, normalised to [-1,1]
    rv=ret.rolling(1440*5).std()*np.sqrt(BPY)
    raw=(score*(tgt/rv)).clip(-maxlev,maxlev)
    if dead>0:
        p=raw.values; out=np.empty_like(p); cur=0.0
        for i in range(len(p)):
            if np.isfinite(p[i]) and abs(p[i]-cur)>dead: cur=p[i]
            out[i]=cur
        raw=pd.Series(out,index=raw.index)
    pos=raw.shift(1)
    m=(c.index>=s)&(c.index<=e)
    pos,rr,cc,ss=pos[m],ret[m],c[m],spr[m]
    turn=pos.diff().abs().fillna(pos.abs())
    cost=turn*(comm_bps/1e4)+turn*(((ss/2 if use_spread else 0)+slip)/cc)
    pnl=(pos*rr-cost).fillna(0)
    eq=(1+pnl).cumprod(); yrs=(cc.index[-1]-cc.index[0]).days/365.25
    cagr=(eq.iloc[-1]**(1/yrs)-1) if eq.iloc[-1]>0 else -1
    dd=-(eq/eq.cummax()-1).min()
    sh=pnl.mean()/pnl.std()*np.sqrt(BPY) if pnl.std()>0 else 0
    return dict(cagr=cagr*100,mdd=dd*100,sharpe=sh,turn=float(turn.sum()),
                cost=float(cost.sum())*100)

print("A) ZERO-COST CEILING — what is possible if execution were FREE?  (IS, k=5)")
print(f"{'tgtvol':>8}{'CAGR%':>12}{'maxDD%':>9}{'Sharpe':>9}")
print("-"*40)
for tv in (0.15,0.30,0.60,1.0,2.0):
    r=run(tgt=tv,maxlev=50)
    print(f"{tv:>8.2f}{r['cagr']:>12.1f}{r['mdd']:>9.1f}{r['sharpe']:>9.2f}")

print("\nB) SAME, WITH REAL COSTS (spread + 0.04bps comm + $0.03 slip), deadband to cut turnover")
print(f"{'dead':>6}{'tgtvol':>8}{'CAGR%':>10}{'maxDD%':>9}{'Sharpe':>9}{'turnover':>11}{'costdrag%':>11}")
print("-"*66)
for dead in (0.0,0.5,2.0):
    for tv in (0.30,1.0):
        r=run(tgt=tv,dead=dead,comm_bps=0.04,slip=0.03,use_spread=True,maxlev=50)
        print(f"{dead:>6.1f}{tv:>8.2f}{r['cagr']:>10.1f}{r['mdd']:>9.1f}{r['sharpe']:>9.2f}{r['turn']:>11.0f}{r['cost']:>11.0f}")

print("\nC) OOS check of the zero-cost ceiling")
for tv in (0.30,1.0):
    r=run(tgt=tv,maxlev=50,s=OOS_START,e=OOS_END)
    print(f"  tgt {tv:.2f}  CAGR {r['cagr']:>8.1f}%  maxDD {r['mdd']:>5.1f}%  Sharpe {r['sharpe']:>5.2f}")
