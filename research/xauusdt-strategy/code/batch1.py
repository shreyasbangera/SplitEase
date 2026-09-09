import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

print("="*130)
print("FAMILY A — INTRADAY SESSION DRIFT  (research: gold Asian-session positive drift; my IS: hours 22-23 UTC t=2.0-3.1 all weekdays)")
print("="*130)

b = engine.bars('1h')
A = atr(b,24)
for hr_in, hold in [(22,2),(22,3),(21,4),(23,1),(22,4)]:
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        bb=slice_(b,s,e); aa=A.reindex(bb.index)
        # decision at close of bar (hr_in-1) -> fill at open of bar hr_in
        ent=np.where(bb.index.hour==(hr_in-1)%24, 1, 0)
        r=engine.run(bb,ent,(aa*3).values,(aa*99).values,'1h',
                     risk=RiskCfg(risk_pct=0.02),max_hold_bars=hold,
                     name=f"A.sessionLong h{hr_in} hold{hold}h")
        show(log(r,per),per)
    print()

print("="*130)
print("FAMILY B — ASIAN-RANGE OPENING BREAKOUT (Zarattini ORB adapted: Asian 00-07 UTC range, break during London/NY)")
print("="*130)

b15 = engine.bars('15min')
A15 = atr(b15,32)
day = b15.index.normalize()
hr  = b15.index.hour
asia = (hr>=0)&(hr<7)
ah = b15.high.where(asia).groupby(day).transform('max')
al = b15.low.where(asia).groupby(day).transform('min')
# only usable AFTER 07:00 -> no look-ahead (range is complete by then)
win = (hr>=7)&(hr<16)
brk_up = (b15.close>ah)&(b15.close.shift(1)<=ah)&win
brk_dn = (b15.close<al)&(b15.close.shift(1)>=al)&win
# one trade per day: first break only
firstup = brk_up & (brk_up.groupby(day).cumsum()==1) & (brk_dn.groupby(day).cumsum()==0)
firstdn = brk_dn & (brk_dn.groupby(day).cumsum()==1) & (brk_up.groupby(day).cumsum()==0)

for slm,tpm in [(1.0,2.0),(1.5,3.0),(1.0,3.0),(2.0,2.0)]:
    for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
        m=(b15.index>=s)&(b15.index<=e); bb=b15[m]
        ent=(firstup[m].astype(int)-firstdn[m].astype(int)).values
        aa=A15[m].values
        r=engine.run(bb,ent,aa*slm,aa*tpm,'15min',risk=RiskCfg(risk_pct=0.02),
                     max_hold_bars=36,name=f"B.asiaORB sl{slm}atr tp{tpm}atr")
        show(log(r,per),per)
    print()
