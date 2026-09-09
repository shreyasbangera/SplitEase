"""Multi-asset diversification test. Same TREND rules as the gold model, applied to 20 instruments.
   Window is short (~8 weeks) so per-asset Sharpe is NOT reliable -- used only to measure the
   CORRELATION STRUCTURE of strategy returns, which is what sets portfolio Sharpe."""
import sys,glob,os; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
import numpy as np, pandas as pd
M="/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/multi"
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))

def load(p):
    d=pd.read_csv(p); d['datetime']=pd.to_datetime(d['datetime'],utc=True)
    d=d.set_index('datetime').sort_index()
    return d.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()

def trend_ret(b):
    c=b.close; pc=c.shift(1)
    tr=pd.concat([b.high-b.low,(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1)
    A=tr.rolling(32).mean()
    def rk(s,sg=1): return sg*(s.rolling(480).rank(pct=True)-0.5)*2
    sc=(rk((c-c.shift(16))/A)+rk((c-c.shift(48))/A)+rk((c-ema(c,20))/A)
        +rk((c-ema(c,50))/A)+rk(rsi(c,14))
        +rk((c-b.low.rolling(96).min())/(b.high.rolling(96).max()-b.low.rolling(96).min()).replace(0,np.nan)))/6
    ret=c.pct_change()
    rv=ret.rolling(96*5).std()*np.sqrt(96*252)
    pos=(sc*(0.30/rv)).clip(-10,10)
    pos=pos.where(np.arange(len(pos))%96==0).ffill().shift(1)   # daily rebalance, next-bar
    turn=pos.diff().abs().fillna(pos.abs())
    return (pos*ret - turn*0.00008).fillna(0)     # ~0.8bp round-turn cost proxy

rets={}
for p in sorted(glob.glob(f"{M}/*.csv")):
    sym=os.path.basename(p)[:-4]
    try:
        b=load(p); r=trend_ret(b)
        if r.abs().sum()>0: rets[sym]=r
    except Exception as ex: print("skip",sym,ex)
R=pd.DataFrame(rets).dropna(how='all')
R=R.loc[:,R.std()>0]
common=R.dropna()
print(f"instruments: {len(R.columns)}   overlapping 15m bars: {len(common):,}"
      f"   window {common.index[0].date()} -> {common.index[-1].date()}")

C=common.corr()
iu=np.triu_indices_from(C.values,1)
rho=C.values[iu]
print(f"\nStrategy-return correlations: mean {rho.mean():+.3f}  median {np.median(rho):+.3f} "
      f" p10 {np.percentile(rho,10):+.3f}  p90 {np.percentile(rho,90):+.3f}")

N=len(C)
# diversification ratio: equal-weight portfolio vol vs average constituent vol
w=np.ones(N)/N
port_var=w@C.values@w
dr=1/np.sqrt(port_var)
print(f"Assets N={N}   equal-weight diversification ratio = {dr:.2f}x")
print(f"  (portfolio vol is {np.sqrt(port_var)*100:.1f}% of a single asset's, for equal risk weights)")

rho_bar=rho.mean()
print(f"\nImplied portfolio Sharpe from the measured correlation structure:")
print(f"  formula  S_port = S_single * sqrt(N / (1 + (N-1)*rho))")
for S in (0.6,0.8,1.0):
    sp=S*np.sqrt(N/(1+(N-1)*rho_bar))
    print(f"  single-asset Sharpe {S:.1f}  ->  portfolio Sharpe {sp:.2f}")

print("\nMax CAGR achievable at maxDD<20%, given portfolio Sharpe (CAGR = Sharpe * sigma):")
print(f"{'S_port':>8}{'sigma@DD1.0x':>14}{'CAGR':>9}{'sigma@DD2.5x':>15}{'CAGR':>9}")
for sp in (1.0,1.5,2.0,3.0):
    print(f"{sp:>8.1f}{0.20:>14.2f}{sp*0.20*100:>8.0f}%{0.08:>15.2f}{sp*0.08*100:>8.0f}%")
print("\nTarget requires 500%.")
