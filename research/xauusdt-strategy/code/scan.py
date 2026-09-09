"""Clean predictive-power scan: signal at bar i (uses data <= close of i),
   forward return measured from bar i+1 OPEN to close of bar i+N, in ATR units."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

TF='15min'
b=engine.bars(TF); A=atr(b,32)
b=b[(b.index>=IS_START)&(b.index<=IS_END)]; A=A.reindex(b.index)
o,h,l,c=b.open,b.high,b.low,b.close
hr=b.index.hour; day=b.index.normalize(); dow=b.index.dayofweek
nxt_o=o.shift(-1)

def fwd(N):
    return (c.shift(-N)-nxt_o)/A     # executable: enter next open, exit N bars later

F={N:fwd(N) for N in (4,8,16,32,96)}

def ev(name,sig,Ns=(4,16,32)):
    sig=pd.Series(sig,index=b.index).fillna(0)
    out=[]
    n=int((sig!=0).sum())
    if n<80: return
    for N in Ns:
        f=F[N]; msk=(sig!=0)&f.notna()&A.notna()
        x=(sig[msk]*f[msk])
        drift=f.mean()*sig[msk].mean()          # UNCONDITIONAL drift x net exposure
        e=x.mean()-drift
        t=x.mean()/(x.std()/np.sqrt(len(x))) if x.std()>0 else 0
        usd=e*A[msk].mean()                      # edge in USD/oz
        out.append(f"N{N:>3}:{e:+.4f}atr(${usd:+.2f}) t{t:+5.2f}")
    print(f"  {name:<44} n={n:>6}  " + "  ".join(out))

print("="*128)
print(f"cost hurdle round-turn: CFD ~$0.26   PERP@2bp ~$1.0-1.5   PERP@5.5bp ~$2.5-4  (gold $1200-2000 in IS)")
print(f"PREDICTIVE SCAN — {TF}, IS {IS_START}..{IS_END}.  edge in ATR units (drift-adjusted), t on signed fwd return")
print("="*128)

print("\n-- momentum / trend --")
e50,e200=ema(c,50),ema(c,200)
ev("trend: close>EMA200 (long only)", (c>e200).astype(int))
ev("trend: EMA50>EMA200 long / else short", np.where(e50>e200,1,-1))
ev("EMA50 cross EMA200", ((e50>e200)&(e50.shift(1)<=e200.shift(1))).astype(int)-((e50<e200)&(e50.shift(1)>=e200.shift(1))).astype(int))
for k in (4,16,48):
    r_=np.log(c/c.shift(k)); ev(f"momentum sign {k}-bar", np.sign(r_))

print("\n-- mean reversion --")
dev=(c-ema(c,48))/A
for z in (1.5,2.0,3.0):
    ev(f"fade dev>|{z}|atr from EMA48", ((dev<-z)&(dev.shift(1)>=-z)).astype(int)-((dev>z)&(dev.shift(1)<=z)).astype(int))
R=rsi(c,14)
ev("RSI14 <30 long / >70 short", ((R<30)&(R.shift(1)>=30)).astype(int)-((R>70)&(R.shift(1)<=70)).astype(int))
ev("RSI14 <20 long / >80 short", ((R<20)&(R.shift(1)>=20)).astype(int)-((R>80)&(R.shift(1)<=80)).astype(int))
for k in (4,16):
    r_=np.log(c/c.shift(k)); z_=r_/r_.rolling(500).std()
    ev(f"fade {k}-bar z>|2|", ((z_<-2)&(z_.shift(1)>=-2)).astype(int)-((z_>2)&(z_.shift(1)<=2)).astype(int))
    ev(f"fade {k}-bar z>|3|", ((z_<-3)&(z_.shift(1)>=-3)).astype(int)-((z_>3)&(z_.shift(1)<=3)).astype(int))

print("\n-- breakout / structure --")
for k in (32,96):
    dh=h.rolling(k).max().shift(1); dl=l.rolling(k).min().shift(1)
    ev(f"donchian{k} breakout", ((c>dh).astype(int)-(c<dl).astype(int)))
    ev(f"donchian{k} FADE",     ((c<dl).astype(int)-(c>dh).astype(int)))
asia=(hr>=0)&(hr<7)
ah=h.where(asia).groupby(day).transform('max'); al=l.where(asia).groupby(day).transform('min')
win=(hr>=7)&(hr<16)
ev("asia-range break (first/day)", (((c>ah)&(c.shift(1)<=ah)&win).astype(int)-((c<al)&(c.shift(1)>=al)&win).astype(int)))
d1=engine.bars('1D')
PDH=pd.Series(d1.high.shift(1).reindex(day).values,index=b.index); PDL=pd.Series(d1.low.shift(1).reindex(day).values,index=b.index)
ev("PDH/PDL sweep-and-reject fade", ((l<PDL)&(c>PDL)).astype(int)-((h>PDH)&(c<PDH)).astype(int))
ev("PDH/PDL clean break follow",    ((c>PDH)&(c.shift(1)<=PDH)).astype(int)-((c<PDL)&(c.shift(1)>=PDL)).astype(int))

print("\n-- volatility regime / compression --")
rng=(h-l); comp=rng.rolling(8).mean()/rng.rolling(96).mean()
ev("vol compression then break up/dn", np.where(comp<0.6,np.sign(c-c.shift(4)),0))
volq=A.rolling(480).rank(pct=True)
ev("HIGH vol regime: fade dev>2atr", np.where(volq>0.7,((dev<-2)&(dev.shift(1)>=-2)).astype(int)-((dev>2)&(dev.shift(1)<=2)).astype(int),0))
ev("LOW vol regime: fade dev>2atr",  np.where(volq<0.3,((dev<-2)&(dev.shift(1)>=-2)).astype(int)-((dev>2)&(dev.shift(1)<=2)).astype(int),0))

print("\n-- time of day / calendar --")
for hh in (1,7,8,12,13,14,22,23):
    ev(f"long at hour {hh} UTC", (hr==hh).astype(int))
ev("long Asia 00-07", ((hr>=0)&(hr<7)).astype(int))
ev("short London 08-12", -(((hr>=8)&(hr<12)).astype(int)))
ev("long 22-24 UTC", (hr>=22).astype(int))
for dd in range(5): ev(f"long dayofweek {dd}", (dow==dd).astype(int))

print("\n-- candle patterns --")
up_=(c>o).astype(bool); ev("3 consecutive up bars -> fade", -(up_&up_.shift(1).fillna(False).astype(bool)&up_.shift(2).fillna(False).astype(bool)&~(up_.shift(3).fillna(False).astype(bool))).astype(int))
ev("3 consecutive dn bars -> fade", (~up_&~up_.shift(1).fillna(False).astype(bool)&~up_.shift(2).fillna(False).astype(bool)).astype(int))
body=(c-o).abs(); ev("large body bar -> continuation", np.where(body>2*A.mean(),np.sign(c-o),0))
ev("large body bar -> fade", np.where(body>2*A.mean(),-np.sign(c-o),0))
