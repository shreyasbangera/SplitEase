"""Continuous vol-targeted time-series momentum on daily gold, with costs.
   Signal uses data through day t close; position held from day t+1 open."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

d=engine.bars('1D')
c=d.close; r=c.pct_change()
rv=r.rolling(60).std()*np.sqrt(252)          # realized vol, past only

def sim(lookback, tgt_vol, maxlev, comm_bps_side, slip_usd, start, end, volscale=True, longonly=False):
    mom=np.sign(c-c.shift(lookback))
    if longonly: mom=mom.clip(lower=0)
    lev = (tgt_vol/rv).clip(upper=maxlev) if volscale else pd.Series(maxlev,index=c.index)
    pos = (mom*lev).shift(1)                 # decided at t close -> held over t+1
    m=(c.index>=start)&(c.index<=end)
    pos=pos[m]; rr=r[m]; cc=c[m]
    turn=pos.diff().abs().fillna(pos.abs())
    cost = turn*(comm_bps_side/1e4) + turn*(slip_usd/cc)   # notional-fraction cost
    pnl = (pos*rr - cost).fillna(0)
    eq=(1+pnl).cumprod()
    yrs=(cc.index[-1]-cc.index[0]).days/365.25
    cagr=eq.iloc[-1]**(1/yrs)-1
    dd=(eq/eq.cummax()-1).min()
    sh=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else 0
    ntr=int((np.sign(pos).diff()!=0).sum())
    return dict(cagr=cagr*100,mdd=-dd*100,sharpe=sh,calmar=cagr/abs(dd) if dd else 0,
                flips=ntr,final=eq.iloc[-1])

print("Daily TSMOM, vol-targeted.  IS 2017-04-28..2022-12-31")
print(f"{'lookback':>9}{'tgtvol':>8}{'maxlev':>8}{'regime':>8}{'CAGR%':>9}{'maxDD%':>8}{'Sharpe':>8}{'Calmar':>8}{'flips':>7}")
print("-"*80)
best=[]
for lb in (20,50,100):
    for tv in (0.15,0.30,0.60,1.20,2.40):
        for reg,(cb,sl_) in [("CFD",(0.04,0.06)),("PERP2",(2.0,0.10))]:
            s=sim(lb,tv,20,cb,sl_,IS_START,IS_END)
            print(f"{lb:>9}{tv:>8.2f}{20:>8}{reg:>8}{s['cagr']:>9.1f}{s['mdd']:>8.1f}{s['sharpe']:>8.2f}{s['calmar']:>8.2f}{s['flips']:>7}")
            best.append((lb,tv,reg,s))
    print()

print("\nBest Calmar configs, then pushed to EXTREME leverage to test the 500%/yr target:")
print(f"{'lookback':>9}{'tgtvol':>8}{'maxlev':>8}{'CAGR%':>10}{'maxDD%':>8}{'Sharpe':>8}{'Calmar':>8}")
print("-"*70)
for lb in (50,):
    for tv in (1.2,2.4,4.8,9.6):
        for ml in (5,10,20,50):
            s=sim(lb,tv,ml,0.04,0.06,IS_START,IS_END)
            print(f"{lb:>9}{tv:>8.2f}{ml:>8}{s['cagr']:>10.1f}{s['mdd']:>8.1f}{s['sharpe']:>8.2f}{s['calmar']:>8.2f}")
