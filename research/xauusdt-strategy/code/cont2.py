import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
from feat import build
b,X=build('15min')
c=b.close; ret=c.pct_change(); spr=b['spread']*0.001
W=480
def rk(s,sg): return sg*(s.rolling(W).rank(pct=True)-0.5)*2
TREND=(rk(X['mom16'],1)+rk(X['mom48'],1)+rk(X['dev20'],1)+rk(X['dev50'],1)+rk(X['rsi'],1)+rk(X['rngpos'],1))/6
VOL=rk(X['volq'],-1); MISC=(rk(X['dom'],-1)+rk(X['spq'],1))/2
rv=ret.rolling(96*5).std()*np.sqrt(96*252)

def sim(score,tgt,maxlev,rebal,s,e,comm_bps=0.04,slip=0.03):
    raw=(score*(tgt/rv)).clip(-maxlev,maxlev)
    hold=raw.where(np.arange(len(raw))%rebal==0).ffill()   # only change every `rebal` bars
    pos=hold.shift(1)
    m=(c.index>=s)&(c.index<=e)
    pos,rr,cc,ss=pos[m],ret[m],c[m],spr[m]
    turn=pos.diff().abs().fillna(pos.abs())
    cost=turn*(comm_bps/1e4)+turn*((ss/2+slip)/cc)
    pnl=(pos*rr-cost).fillna(0)
    eq=(1+pnl).cumprod(); yrs=(cc.index[-1]-cc.index[0]).days/365.25
    cagr=(eq.iloc[-1]**(1/yrs)-1) if eq.iloc[-1]>0 else -1
    dd=-(eq/eq.cummax()-1).min(); sh=pnl.mean()/pnl.std()*np.sqrt(96*252) if pnl.std()>0 else 0
    gp=pnl[pnl>0].sum(); gl=-pnl[pnl<0].sum()
    return dict(cagr=cagr*100,mdd=dd*100,sharpe=sh,pf=gp/gl if gl>0 else np.inf,
                turn=float(turn.sum()),costdrag=float(cost.sum())*100,avgpos=float(pos.abs().mean()))

print("Effect of rebalance frequency (TREND factor, tgt 30%, IS) — turnover is the killer")
print(f"{'rebal':>7}{'CAGR%':>9}{'maxDD%':>8}{'Sharpe':>8}{'turnover':>10}{'costdrag%':>11}")
print("-"*54)
for rb in (1,4,16,96,192,480):
    r=sim(TREND,0.30,10,rb,IS_START,IS_END)
    print(f"{rb:>7}{r['cagr']:>9.1f}{r['mdd']:>8.1f}{r['sharpe']:>8.2f}{r['turn']:>10.0f}{r['costdrag']:>11.1f}")

print("\n\nDAILY rebalance (96 bars), factors and combo, IS vs OOS")
print(f"{'model':<8}{'per':<5}{'CAGR%':>9}{'maxDD%':>8}{'Sharpe':>8}{'PF':>6}{'turn':>8}{'cost%':>8}")
print("-"*62)
COMB=(TREND+VOL+MISC)/3
for nm,f in [('TREND',TREND),('VOL',VOL),('MISC',MISC),('COMB',COMB)]:
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        r=sim(f,0.30,10,96,s,e)
        print(f"{nm:<8}{per:<5}{r['cagr']:>9.1f}{r['mdd']:>8.1f}{r['sharpe']:>8.2f}{r['pf']:>6.2f}{r['turn']:>8.0f}{r['costdrag']:>8.1f}")
    print()

print("Benchmark buy&hold for reference")
for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
    m=(c.index>=s)&(c.index<=e); rr=ret[m].fillna(0)
    eq=(1+rr).cumprod(); yrs=(c[m].index[-1]-c[m].index[0]).days/365.25
    print(f"  B&H {per:<4} CAGR {(eq.iloc[-1]**(1/yrs)-1)*100:>7.1f}%  maxDD {-(eq/eq.cummax()-1).min()*100:>5.1f}%  "
          f"Sharpe {rr.mean()/rr.std()*np.sqrt(96*252):>5.2f}")
