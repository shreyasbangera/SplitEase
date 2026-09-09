import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

d=engine.bars('1D'); c=d.close; r=c.pct_change()
def stats(pnl,idx):
    eq=(1+pnl).cumprod(); yrs=(idx[-1]-idx[0]).days/365.25
    cagr=eq.iloc[-1]**(1/yrs)-1 if eq.iloc[-1]>0 else -1
    dd=-(eq/eq.cummax()-1).min()
    return dict(cagr=cagr*100,mdd=dd*100,sharpe=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else 0,
                calmar=(cagr/dd) if dd>0 else 0)

print("="*100); print("BENCHMARK: buy & hold gold")
for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END)),('FULL',(IS_START,OOS_END))]:
    m=(c.index>=s)&(c.index<=e); st=stats(r[m].fillna(0),c[m].index)
    print(f"  {per:<5} CAGR {st['cagr']:>7.1f}%  maxDD {st['mdd']:>5.1f}%  Sharpe {st['sharpe']:>5.2f}  Calmar {st['calmar']:>5.2f}")

print("\n"+"="*100); print("FAMILY F — MULTI-HORIZON TSMOM ENSEMBLE (Moskowitz/Ooi/Pedersen style: 1/3/6/12-month momentum, vol-targeted)")
rv=r.rolling(60).std()*np.sqrt(252)
sig=sum(np.sign(c-c.shift(k)) for k in (21,63,126,252))/4.0
for tv in (0.15,0.30,0.50):
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        lev=(tv/rv).clip(upper=10); pos=(sig*lev).shift(1)
        m=(c.index>=s)&(c.index<=e); pos_,rr,cc=pos[m],r[m],c[m]
        turn=pos_.diff().abs().fillna(pos_.abs())
        pnl=(pos_*rr - turn*(0.04/1e4) - turn*(0.06/cc)).fillna(0)
        st=stats(pnl,cc.index)
        print(f"  tgtvol {tv:.2f} [{per:>3}] CAGR {st['cagr']:>7.1f}%  maxDD {st['mdd']:>5.1f}%  Sharpe {st['sharpe']:>5.2f}  Calmar {st['calmar']:>5.2f}")
    print()

print("="*100); print("FAMILY G — 1h DONCHIAN-20 BREAKOUT, engine backtest, CFD costs (best cost-surviving intraday signal)")
CFD=Costs(comm_bps_side=0.04,slip_usd_side=0.03,stop_slip_usd=0.10)
b=engine.bars('1h'); A=atr(b,32)
dh=b.high.rolling(20).max().shift(1); dl=b.low.rolling(20).min().shift(1)
sg=((b.close>dh).astype(int)-(b.close<dl).astype(int))
for rp in (0.005,0.01,0.02,0.05):
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        m=(b.index>=s)&(b.index<=e)
        res=engine.run(b[m],sg[m].values,A[m].values*2.0,A[m].values*6.0,'1h',costs=CFD,
                       risk=RiskCfg(risk_pct=rp,max_leverage=20),max_hold_bars=72,
                       name=f"G.donch20-1h risk{rp:.1%}")
        show(log(res,per),per)
    print()

print("="*100); print("REQUIRED SHARPE FOR THE TARGET (500%/yr CAGR at <20% maxDD)")
print("""  A strategy compounding at CAGR g with annualised vol s has maxDD that, empirically and
  theoretically, scales with s. Even under the optimistic approximation maxDD ~= 1.0*s:
     maxDD < 20%  =>  s < 0.20
     CAGR  = 500% =>  mu ~= ln(6) = 1.79 continuous  =>  Sharpe = mu/s > 1.79/0.20 ~= 9.0
  Under the more realistic maxDD ~= 2.5*s (typical for a drifting equity curve):
     s < 0.08  =>  required Sharpe > 22
  Best cost-surviving Sharpe measured anywhere in this study: 0.34 (daily TSMOM, CFD costs).""")
