import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
from feat import build
b,X=build('15min'); c=b.close; ret=c.pct_change(); spr=b['spread']*0.001
W=480
def rk(s,sg): return sg*(s.rolling(W).rank(pct=True)-0.5)*2
TREND=(rk(X['mom16'],1)+rk(X['mom48'],1)+rk(X['dev20'],1)+rk(X['dev50'],1)+rk(X['rsi'],1)+rk(X['rngpos'],1))/6
rv=ret.rolling(96*5).std()*np.sqrt(96*252)

def sim(score,tgt,maxlev,rebal,dead,s,e):
    raw=(score*(tgt/rv)).clip(-maxlev,maxlev)
    hold=raw.where(np.arange(len(raw))%rebal==0).ffill()
    if dead>0:
        p=hold.values; out=np.empty_like(p); cur=0.0
        for i in range(len(p)):
            if np.isfinite(p[i]) and abs(p[i]-cur)>dead: cur=p[i]
            out[i]=cur
        hold=pd.Series(out,index=hold.index)
    pos=hold.shift(1); m=(c.index>=s)&(c.index<=e)
    pos,rr,cc,ss=pos[m],ret[m],c[m],spr[m]
    turn=pos.diff().abs().fillna(pos.abs())
    pnl=(pos*rr-turn*(0.04/1e4)-turn*((ss/2+0.03)/cc)).fillna(0)
    eq=(1+pnl).cumprod(); yrs=(cc.index[-1]-cc.index[0]).days/365.25
    cagr=(eq.iloc[-1]**(1/yrs)-1) if eq.iloc[-1]>0 else -1
    dd=-(eq/eq.cummax()-1).min(); sh=pnl.mean()/pnl.std()*np.sqrt(96*252) if pnl.std()>0 else 0
    gp=pnl[pnl>0].sum(); gl=-pnl[pnl<0].sum()
    flips=int((np.sign(pos).diff().fillna(0)!=0).sum())
    return dict(cagr=cagr*100,mdd=dd*100,sharpe=sh,pf=gp/gl if gl>0 else np.inf,flips=flips)

print("Tuning TREND on IS (rebalance / deadband), tgt vol 30%")
print(f"{'rebal':>6}{'dead':>6}{'IS CAGR':>9}{'IS DD':>7}{'IS Sh':>7}  |{'OOS CAGR':>9}{'OOS DD':>7}{'OOS Sh':>7}")
print("-"*70)
best=None
for rb in (96,192,384):
    for dd_ in (0.0,0.5,1.0,2.0):
        a=sim(TREND,0.30,10,rb,dd_,IS_START,IS_END); o=sim(TREND,0.30,10,rb,dd_,OOS_START,OOS_END)
        print(f"{rb:>6}{dd_:>6.1f}{a['cagr']:>9.1f}{a['mdd']:>7.1f}{a['sharpe']:>7.2f}  |"
              f"{o['cagr']:>9.1f}{o['mdd']:>7.1f}{o['sharpe']:>7.2f}")
        if best is None or a['sharpe']>best[0]: best=(a['sharpe'],rb,dd_)
print(f"\nbest IS config: rebal={best[1]} deadband={best[2]}")

rb,dd_=best[1],best[2]
print("\nScaling that config toward the 500% target — what DD does it cost?")
print(f"{'tgtvol':>7}{'FULL CAGR%':>12}{'FULL maxDD%':>13}{'Sharpe':>8}{'PF':>6}{'flips':>7}")
print("-"*55)
for tv in (0.30,0.60,1.20,2.40,4.80):
    f=sim(TREND,tv,50,rb,dd_,IS_START,OOS_END)
    print(f"{tv:>7.2f}{f['cagr']:>12.1f}{f['mdd']:>13.1f}{f['sharpe']:>8.2f}{f['pf']:>6.2f}{f['flips']:>7}")

print("\nFULL-SAMPLE result at the DD-respecting setting (tgt 30%):")
f=sim(TREND,0.30,10,rb,dd_,IS_START,OOS_END)
print(f"  CAGR {f['cagr']:.1f}%   maxDD {f['mdd']:.1f}%   Sharpe {f['sharpe']:.2f}   PF {f['pf']:.2f}   direction changes {f['flips']}")
