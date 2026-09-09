import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

print("="*112)
print("FAMILY H — CALENDAR / EVENT EFFECTS (NFP=1st Friday, month-end, month-start); daily bars, drift-adjusted")
print("="*112)
d=engine.bars('1D'); c=d.close; r=c.pct_change(); nx=d.open.shift(-1)
fwd1=(c.shift(-1)-nx)/c; fwd5=(c.shift(-5)-nx)/c
dom=c.index.day; dow=c.index.dayofweek; mth=c.index.month
first_fri=(dow==4)&(dom<=7)                       # NFP day
def ev(nm,sig):
    sig=pd.Series(sig,index=c.index).fillna(0).astype(float)
    for lbl,f in [("fwd1d",fwd1),("fwd5d",fwd5)]:
        m=(sig!=0)&f.notna(); 
        if m.sum()<25: continue
        x=sig[m]*f[m]; e=(x.mean()-f.mean()*sig[m].mean())*1e4
        t=x.mean()/(x.std()/np.sqrt(len(x)))
        print(f"  {nm:<40}{lbl}  n={int(m.sum()):>5}  edge {e:+7.1f}bps  t{t:+5.2f}")
ev("long on NFP day (1st Friday)", first_fri.astype(int))
ev("long day AFTER NFP", pd.Series(first_fri,index=c.index).shift(1).fillna(False).astype(int))
ev("long last 3 days of month", (dom>=27).astype(int))
ev("long first 3 days of month", (dom<=3).astype(int))
ev("long September (seasonal)", (mth==9).astype(int))
ev("long January (seasonal)", (mth==1).astype(int))
ev("long mid-month FOMC window d15-21", ((dom>=15)&(dom<=21)).astype(int))

print("\n"+"="*112)
print("FAMILY I — 1-MINUTE MEAN REVERSION FEASIBILITY (does any HF edge exceed the cost floor?)")
print("="*112)
b1=engine.bars('1min'); b1=b1[(b1.index>=IS_START)&(b1.index<=IS_END)]
c1=b1.close; A1=(b1.high-b1.low).rolling(60).mean()
nx1=b1.open.shift(-1)
for k in (5,15,30):
    rk=np.log(c1/c1.shift(k)); z=rk/rk.rolling(1000).std()
    for zt in (2.0,3.0,4.0):
        sig=((z<-zt)&(z.shift(1)>=-zt)).astype(int)-((z>zt)&(z.shift(1)<=zt)).astype(int)
        for N in (5,15,60):
            f=(c1.shift(-N)-nx1)
            m=(sig!=0)&f.notna()
            if m.sum()<200: continue
            x=sig[m]*f[m]
            e=x.mean(); t=x.mean()/(x.std()/np.sqrt(len(x)))
            print(f"  fade {k:>2}m z>|{zt}| hold {N:>2}m  n={int(m.sum()):>6}  gross edge ${e:+.3f}/oz  t{t:+5.2f}   "
                  f"{'** clears CFD cost $0.26' if e>0.26 else ''}")
