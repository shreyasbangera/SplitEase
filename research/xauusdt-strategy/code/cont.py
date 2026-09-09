"""Continuous vol-targeted multi-factor model on 15m bars.
   Position decided at bar close, HELD FROM NEXT BAR OPEN. Deadband limits turnover."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
from feat import build

b,X=build('15min')
c=b.close; ret=c.pct_change(); spr=b['spread']*0.001
W=480
def rk(s,sg): return sg*(s.rolling(W).rank(pct=True)-0.5)*2      # -1..1

TREND=(rk(X['mom16'],1)+rk(X['mom48'],1)+rk(X['dev20'],1)+rk(X['dev50'],1)
       +rk(X['rsi'],1)+rk(X['rngpos'],1))/6
VOL  = rk(X['volq'],-1)
MISC =(rk(X['dom'],-1)+rk(X['spq'],1))/2
FACT={'TREND':TREND,'VOL':VOL,'MISC':MISC}

rv=ret.rolling(96*5).std()*np.sqrt(96*252)      # annualised realised vol, past only

def sim(score,tgt,maxlev,dead,comm_bps,slip,s,e,label):
    m=(c.index>=s)&(c.index<=e)
    raw=(score*(tgt/rv)).clip(-maxlev,maxlev)
    pos=raw.copy()
    # deadband: only move when the target differs enough (cuts turnover)
    p=pos.values; out=np.empty_like(p); cur=0.0
    for i in range(len(p)):
        if np.isfinite(p[i]) and abs(p[i]-cur)>dead: cur=p[i]
        out[i]=cur
    pos=pd.Series(out,index=pos.index).shift(1)          # held from NEXT bar
    pos,rr,cc,ss=pos[m],ret[m],c[m],spr[m]
    turn=pos.diff().abs().fillna(pos.abs())
    cost=turn*(comm_bps/1e4)+turn*((ss/2+slip)/cc)
    pnl=(pos*rr-cost).fillna(0)
    eq=(1+pnl).cumprod(); yrs=(cc.index[-1]-cc.index[0]).days/365.25
    cagr=(eq.iloc[-1]**(1/yrs)-1) if eq.iloc[-1]>0 else -1
    dd=-(eq/eq.cummax()-1).min()
    sh=pnl.mean()/pnl.std()*np.sqrt(96*252) if pnl.std()>0 else 0
    gp=pnl[pnl>0].sum(); gl=-pnl[pnl<0].sum()
    return dict(label=label,cagr=cagr*100,mdd=dd*100,sharpe=sh,pf=gp/gl if gl>0 else np.inf,
                turn=turn.sum(),calmar=cagr/dd if dd>0 else 0)

print("Single factors, tgt vol 30%, deadband 0.5, CFD costs")
print(f"{'factor':<8}{'per':<5}{'CAGR%':>9}{'maxDD%':>8}{'Sharpe':>8}{'PF':>6}{'Calmar':>8}")
print("-"*55)
for fn,f in FACT.items():
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        r=sim(f,0.30,10,0.5,0.04,0.03,s,e,fn)
        print(f"{fn:<8}{per:<5}{r['cagr']:>9.1f}{r['mdd']:>8.1f}{r['sharpe']:>8.2f}{r['pf']:>6.2f}{r['calmar']:>8.2f}")
    print()

print("Combined (equal-weight 3 factors), sweeping target vol")
print(f"{'tgt':<6}{'per':<5}{'CAGR%':>9}{'maxDD%':>8}{'Sharpe':>8}{'PF':>6}{'Calmar':>8}")
print("-"*53)
COMB=(TREND+VOL+MISC)/3
for tgt in (0.30,0.60,1.20,2.40):
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        r=sim(COMB,tgt,20,0.5,0.04,0.03,s,e,'COMB')
        print(f"{tgt:<6.2f}{per:<5}{r['cagr']:>9.1f}{r['mdd']:>8.1f}{r['sharpe']:>8.2f}{r['pf']:>6.2f}{r['calmar']:>8.2f}")
    print()
