import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

def scan(TF, Ns):
    b=engine.bars(TF); A=atr(b,32)
    b=b[(b.index>=IS_START)&(b.index<=IS_END)]; A=A.reindex(b.index)
    o,h,l,c=b.open,b.high,b.low,b.close
    nxt=o.shift(-1)
    F={N:(c.shift(-N)-nxt)/A for N in Ns}
    Amean=A.mean()
    print(f"\n{'='*118}\n{TF}  |  mean ATR32 = ${Amean:.2f}  |  IS {b.index[0].date()}..{b.index[-1].date()}  n_bars={len(b):,}")
    print(f"{'signal':<40}{'n':>7}  " + "".join(f"{'N'+str(N):>26}" for N in Ns)); print('-'*118)
    def ev(name,sig):
        sig=pd.Series(sig,index=b.index).fillna(0)
        n=int((sig!=0).sum())
        if n<60: return
        cells=[]
        for N in Ns:
            f=F[N]; m=(sig!=0)&f.notna()&A.notna()
            if m.sum()<60: cells.append(f"{'-':>26}"); continue
            x=sig[m]*f[m]
            e=x.mean()-f.mean()*sig[m].mean()
            t=x.mean()/(x.std()/np.sqrt(len(x)))
            cells.append(f"{e:+.3f}atr ${e*Amean:+6.2f} t{t:+5.1f}".rjust(26))
        print(f"  {name:<38}{n:>7}  "+"".join(cells))

    e50,e200=ema(c,50),ema(c,200)
    for k in (5,10,20,50,100):
        ev(f"momentum sign {k}-bar", np.sign(c-c.shift(k)))
    ev("EMA50>EMA200 (long/short)", np.where(e50>e200,1,-1))
    ev("close>EMA200 long-only", (c>e200).astype(int))
    for k in (20,55):
        dh=h.rolling(k).max().shift(1); dl=l.rolling(k).min().shift(1)
        ev(f"donchian{k} breakout", (c>dh).astype(int)-(c<dl).astype(int))
    # vol-scaled momentum (risk parity style)
    for k in (20,50):
        r_=np.log(c/c.shift(k)); z=r_/ (np.log(c/c.shift(1)).rolling(100).std()*np.sqrt(k))
        ev(f"volscaled mom {k} (|z|>0.5)", np.where(z.abs()>0.5,np.sign(z),0))
    dev=(c-ema(c,20))/A
    for zt in (1.5,2.5):
        ev(f"fade dev>|{zt}|atr EMA20", ((dev<-zt)&(dev.shift(1)>=-zt)).astype(int)-((dev>zt)&(dev.shift(1)<=zt)).astype(int))

for TF,Ns in [('1h',(6,24,72)), ('4h',(3,6,30)), ('1D',(1,5,20))]:
    scan(TF,Ns)
