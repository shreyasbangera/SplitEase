"""Multi-factor composite. Feature signs fixed from IS only; normalisation is past-only
   (rolling percentile), so nothing peeks forward."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
from feat import build

b,X=build('15min')
# signs determined on IS in stability.py
SPECS=[('volq',-1),('mom16',+1),('mom48',+1),('dev20',+1),('dev50',+1),
       ('rsi',+1),('rngpos',+1),('spq',+1),('dom',-1)]
W=480   # 5 trading days of 15m bars

Z=pd.DataFrame(index=X.index)
for ft,sg in SPECS:
    Z[ft]=sg*(X[ft].rolling(W).rank(pct=True)-0.5)     # past-only rank, in [-0.5,0.5]
comp=Z.mean(axis=1)
print("composite corr matrix of factors (IS):")
ISm=(X.index>=IS_START)&(X.index<=IS_END)
print(Z[ISm].corr().round(2).to_string())

print(f"\ncomposite: mean {comp.mean():.4f} std {comp.std():.4f}")
print("\nForward edge (USD/oz) by composite decile, IS vs OOS:")
for N in (32,96):
    for lbl,m in [('IS',ISm),('OOS',(X.index>=OOS_START)&(X.index<=OOS_END))]:
        cq=pd.qcut(comp[m],10,labels=False,duplicates='drop'); f_=X[f'fwd{N}'][m]
        g=f_.groupby(cq).mean()
        print(f"  N{N:>3} {lbl:<4} " + " ".join(f"{v:+6.2f}" for v in g.values) +
              f"   | top-bot {g.iloc[-1]-g.iloc[0]:+.2f}")

# ---- trade it through the engine ----
print("\n" + "="*118)
print("FAMILY J — MULTI-FACTOR COMPOSITE, traded via engine (15m bars, 24h hold, CFD costs)")
print("="*118)
CFD=Costs(comm_bps_side=0.04,slip_usd_side=0.03,stop_slip_usd=0.10)
P2 =Costs(comm_bps_side=2.0, slip_usd_side=0.05,stop_slip_usd=0.15)
A=atr(b,32)
for thr in (0.10,0.15,0.20):
    for rp in (0.01,0.02):
        for per,(s,e) in [('IS',(IS_START,IS_END)),('OOS',(OOS_START,OOS_END))]:
            m=(b.index>=s)&(b.index<=e)
            sig=((comp[m]>thr).astype(int)-(comp[m]<-thr).astype(int)).values
            res=engine.run(b[m],sig,A[m].values*3.0,A[m].values*6.0,'15min',costs=CFD,
                           risk=RiskCfg(risk_pct=rp,max_leverage=20),max_hold_bars=96,
                           name=f"J.factor thr{thr} risk{rp:.0%}")
            show(log(res,per),per)
        print()
