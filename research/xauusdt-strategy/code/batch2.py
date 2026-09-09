import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

b15 = engine.bars('15min'); A15 = atr(b15,32)
day = b15.index.normalize(); hr = b15.index.hour

print("="*130)
print("FAMILY C — FADE THE ASIAN-RANGE BREAK (inverse of B; motivated by negative autocorr = gold mean-reverts)")
print("="*130)
asia=(hr>=0)&(hr<7)
ah=b15.high.where(asia).groupby(day).transform('max')
al=b15.low.where(asia).groupby(day).transform('min')
win=(hr>=7)&(hr<16)
up=(b15.close>ah)&(b15.close.shift(1)<=ah)&win
dn=(b15.close<al)&(b15.close.shift(1)>=al)&win
fup=up&(up.groupby(day).cumsum()==1)&(dn.groupby(day).cumsum()==0)
fdn=dn&(dn.groupby(day).cumsum()==1)&(up.groupby(day).cumsum()==0)
for slm,tpm in [(1.0,1.0),(1.5,1.5),(2.0,1.0),(1.0,2.0)]:
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        m=(b15.index>=s)&(b15.index<=e); bb=b15[m]
        ent=(fdn[m].astype(int)-fup[m].astype(int)).values   # FADE
        r=engine.run(bb,ent,A15[m].values*slm,A15[m].values*tpm,'15min',
                     risk=RiskCfg(risk_pct=0.02),max_hold_bars=36,
                     name=f"C.fadeAsiaBreak sl{slm} tp{tpm}")
        show(log(r,per),per)
    print()

print("="*130)
print("FAMILY D — VOLATILITY-CONDITIONED MEAN REVERSION (fade z-score extremes; EDA: ac1 most negative in top ATR quintile)")
print("="*130)
c=b15.close
dev=(c-ema(c,48))/A15                      # stretch from mean in ATR units
volq=A15.rolling(480).rank(pct=True)        # volatility regime (past only)
for zt,slm,tpm,vmin in [(2.0,1.5,1.5,0.0),(2.5,1.5,1.5,0.0),(2.0,1.5,1.5,0.6),(2.5,2.0,2.0,0.6),(3.0,2.0,2.0,0.0)]:
    sig_l=(dev<-zt)&(dev.shift(1)>=-zt)&(volq>=vmin)
    sig_s=(dev> zt)&(dev.shift(1)<= zt)&(volq>=vmin)
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        m=(b15.index>=s)&(b15.index<=e); bb=b15[m]
        ent=(sig_l[m].astype(int)-sig_s[m].astype(int)).fillna(0).values
        r=engine.run(bb,ent,A15[m].values*slm,A15[m].values*tpm,'15min',
                     risk=RiskCfg(risk_pct=0.02),max_hold_bars=32,
                     name=f"D.MRfade z{zt} sl{slm} tp{tpm} vq>{vmin}")
        show(log(r,per),per)
    print()

print("="*130)
print("FAMILY E — PRIOR-DAY LIQUIDITY SWEEP REVERSAL (sweep PDH/PDL then close back inside -> fade)")
print("="*130)
d1=engine.bars('1D')
pdh=d1.high.shift(1); pdl=d1.low.shift(1)     # completed prior day only
PDH=pdh.reindex(b15.index.normalize()).values
PDL=pdl.reindex(b15.index.normalize()).values
PDH=pd.Series(PDH,index=b15.index); PDL=pd.Series(PDL,index=b15.index)
sweep_hi=(b15.high>PDH)&(b15.close<PDH)&(hr>=7)&(hr<20)
sweep_lo=(b15.low <PDL)&(b15.close>PDL)&(hr>=7)&(hr<20)
fh=sweep_hi&(sweep_hi.groupby(day).cumsum()==1)
fl=sweep_lo&(sweep_lo.groupby(day).cumsum()==1)
for slm,tpm in [(1.0,2.0),(1.5,2.0),(1.0,3.0),(1.5,3.0),(2.0,4.0)]:
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        m=(b15.index>=s)&(b15.index<=e); bb=b15[m]
        ent=(fl[m].astype(int)-fh[m].astype(int)).values
        r=engine.run(bb,ent,A15[m].values*slm,A15[m].values*tpm,'15min',
                     risk=RiskCfg(risk_pct=0.02),max_hold_bars=48,
                     name=f"E.PDsweep sl{slm} tp{tpm}")
        show(log(r,per),per)
    print()
