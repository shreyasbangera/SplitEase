import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
d=engine.bars('1D'); c=d.close; r=c.pct_change(); rv=r.rolling(60).std()*np.sqrt(252)
def stats(pnl,idx):
    eq=(1+pnl).cumprod(); yrs=(idx[-1]-idx[0]).days/365.25
    cagr=(eq.iloc[-1]**(1/yrs)-1) if eq.iloc[-1]>0 else -1
    dd=-(eq/eq.cummax()-1).min()
    pos_=pnl[pnl>0].sum(); neg=-pnl[pnl<0].sum()
    return dict(cagr=cagr*100,mdd=dd*100,sharpe=pnl.mean()/pnl.std()*np.sqrt(252),
                calmar=cagr/dd if dd>0 else 0, pf=pos_/neg if neg>0 else np.inf)
sig=sum(np.sign(c-c.shift(k)) for k in (21,63,126,252))/4.0
print("FAMILY F  multi-horizon TSMOM ensemble, vol-target 30%, CFD costs")
for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END)),('FULL',(IS_START,OOS_END))]:
    lev=(0.30/rv).clip(upper=10); pos=(sig*lev).shift(1)
    m=(c.index>=s)&(c.index<=e); p,rr,cc=pos[m],r[m],c[m]
    turn=p.diff().abs().fillna(p.abs())
    pnl=(p*rr-turn*(0.04/1e4)-turn*(0.06/cc)).fillna(0); st=stats(pnl,cc.index)
    flips=int((np.sign(p).diff()!=0).sum())
    print(f"  {per:<5} CAGR {st['cagr']:>7.1f}%  maxDD {st['mdd']:>5.1f}%  Sharpe {st['sharpe']:>5.2f}  "
          f"Calmar {st['calmar']:>5.2f}  PF(daily) {st['pf']:>4.2f}  dir-changes {flips}")

print("\nFAMILY G  1h Donchian-20 breakout, 2ATR stop / 6ATR target, risk 0.5%, CFD costs")
CFD=Costs(comm_bps_side=0.04,slip_usd_side=0.03,stop_slip_usd=0.10)
b=engine.bars('1h'); A=atr(b,32)
dh=b.high.rolling(20).max().shift(1); dl=b.low.rolling(20).min().shift(1)
sg=((b.close>dh).astype(int)-(b.close<dl).astype(int))
for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END)),('FULL',(IS_START,OOS_END))]:
    m=(b.index>=s)&(b.index<=e)
    res=engine.run(b[m],sg[m].values,A[m].values*2.0,A[m].values*6.0,'1h',costs=CFD,
                   risk=RiskCfg(risk_pct=0.005,max_leverage=20),max_hold_bars=72,name="G")
    mm=res.m
    print(f"  {per:<5} trades {mm['trades']:>5}  CAGR {mm['cagr']:>7.1f}%  maxDD {mm['mdd']*100:>5.1f}%  "
          f"PF {mm['pf']:>4.2f}  win {mm['win']*100:>4.1f}%  net {mm['net_pct']:>8.1f}%")

print("\nFAMILY G under CRYPTO-PERP costs (2 bps/side) — the XAUUSDT-native regime")
P2=Costs(comm_bps_side=2.0,slip_usd_side=0.05,stop_slip_usd=0.15)
for per,(s,e) in [('FULL',(IS_START,OOS_END))]:
    m=(b.index>=s)&(b.index<=e)
    res=engine.run(b[m],sg[m].values,A[m].values*2.0,A[m].values*6.0,'1h',costs=P2,
                   risk=RiskCfg(risk_pct=0.005,max_leverage=20),max_hold_bars=72,name="G-perp")
    mm=res.m
    print(f"  {per:<5} trades {mm['trades']:>5}  CAGR {mm['cagr']:>7.1f}%  maxDD {mm['mdd']*100:>5.1f}%  PF {mm['pf']:>4.2f}")
