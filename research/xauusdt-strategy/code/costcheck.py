import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

# cost regimes
FREE = Costs(comm_bps_side=0, slip_usd_side=0, stop_slip_usd=0, spread_mult=0)
CFD  = Costs(comm_bps_side=0.04, slip_usd_side=0.03, stop_slip_usd=0.10, spread_mult=1.0)  # Exness-like: $3.5/lot/side
PERP2= Costs(comm_bps_side=2.0,  slip_usd_side=0.05, stop_slip_usd=0.15, spread_mult=1.0)  # MEXC-like
PERP5= Costs(comm_bps_side=5.5,  slip_usd_side=0.05, stop_slip_usd=0.15, spread_mult=1.0)  # Bybit-like

b15=engine.bars('15min'); A15=atr(b15,32); day=b15.index.normalize(); hr=b15.index.hour
asia=(hr>=0)&(hr<7)
ah=b15.high.where(asia).groupby(day).transform('max'); al=b15.low.where(asia).groupby(day).transform('min')
win=(hr>=7)&(hr<16)
up=(b15.close>ah)&(b15.close.shift(1)<=ah)&win; dn=(b15.close<al)&(b15.close.shift(1)>=al)&win
fup=up&(up.groupby(day).cumsum()==1)&(dn.groupby(day).cumsum()==0)
fdn=dn&(dn.groupby(day).cumsum()==1)&(up.groupby(day).cumsum()==0)
orb=(fup.astype(int)-fdn.astype(int))

d1=engine.bars('1D'); PDH=pd.Series(d1.high.shift(1).reindex(b15.index.normalize()).values,index=b15.index)
PDL=pd.Series(d1.low.shift(1).reindex(b15.index.normalize()).values,index=b15.index)
sh=(b15.high>PDH)&(b15.close<PDH)&(hr>=7)&(hr<20); sl_=(b15.low<PDL)&(b15.close>PDL)&(hr>=7)&(hr<20)
sweep=(sl_&(sl_.groupby(day).cumsum()==1)).astype(int)-(sh&(sh.groupby(day).cumsum()==1)).astype(int)

dev=(b15.close-ema(b15.close,48))/A15
mr=((dev<-2.0)&(dev.shift(1)>=-2.0)).astype(int)-((dev>2.0)&(dev.shift(1)<=2.0)).astype(int)

m=(b15.index>=IS_START)&(b15.index<=IS_END); bb=b15[m]; aa=A15[m].values
print(f"{'signal':<22}{'regime':<8}{'trades':>7}{'avgR':>9}{'t(R)':>7}{'PF':>7}")
print("-"*62)
for nm,sig,slm,tpm in [("ORB breakout",orb,1.5,3.0),("ORB fade",-orb,1.5,1.5),
                        ("PD sweep fade",sweep,1.5,3.0),("MR z2 fade",mr,1.5,1.5)]:
    ent=sig[m].fillna(0).values.astype(int)
    for rn,cc in [("FREE",FREE),("CFD",CFD),("PERP2",PERP2),("PERP5",PERP5)]:
        r=engine.run(bb,ent,aa*slm,aa*tpm,'15min',costs=cc,
                     risk=RiskCfg(risk_pct=0.001),max_hold_bars=36,name=nm)
        t=r.trades
        tR = t.R.mean()/(t.R.std()/np.sqrt(len(t))) if len(t)>1 else 0
        print(f"{nm:<22}{rn:<8}{len(t):>7}{t.R.mean():>+9.3f}{tR:>+7.2f}{r.m['pf']:>7.2f}")
    print()
