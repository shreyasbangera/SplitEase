"""Gradient-boosted trees on the feature set. Trained ONLY on IS, evaluated on untouched OOS.
   Target = forward return in USD/oz (what actually has to beat the cost floor)."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
from feat import build
from sklearn.ensemble import HistGradientBoostingRegressor

b,X=build('15min')
feats=[f for f in X.columns if not f.startswith('fwd') and f!='atr_usd']
HOR=32   # 8 hours
D=X.dropna(subset=feats+[f'fwd{HOR}']).copy()
ISm=(D.index>=IS_START)&(D.index<=IS_END); OOSm=(D.index>=OOS_START)&(D.index<=OOS_END)
Xi,yi=D.loc[ISm,feats],D.loc[ISm,f'fwd{HOR}']
Xo,yo=D.loc[OOSm,feats],D.loc[OOSm,f'fwd{HOR}']
print(f"train {len(Xi):,}  test {len(Xo):,}  target=fwd{HOR} (USD/oz)")

# purge the last HOR bars of train so train/test targets cannot overlap
Xi,yi=Xi.iloc[:-HOR],yi.iloc[:-HOR]

for depth,leaf in [(3,200),(5,100)]:
    mdl=HistGradientBoostingRegressor(max_depth=depth,min_samples_leaf=leaf,
        learning_rate=0.05,max_iter=300,l2_regularization=1.0,random_state=0)
    mdl.fit(Xi,yi)
    pi,po=mdl.predict(Xi),mdl.predict(Xo)
    print(f"\n--- depth {depth}, leaf {leaf} ---")
    for lbl,p,y in [('IS(fit)',pi,yi),('OOS',po,yo)]:
        q=pd.qcut(pd.Series(p),10,labels=False,duplicates='drop')
        g=pd.Series(y.values).groupby(q).mean()
        ic=np.corrcoef(p,y)[0,1]
        print(f"  {lbl:<8} IC={ic:+.4f}  decile means: "+" ".join(f"{v:+5.2f}" for v in g.values))
        print(f"  {'':8} top-bot spread ${g.iloc[-1]-g.iloc[0]:+.2f}/oz   "
              f"(cost floor ~$0.30 round trip)")
    # tradeable check: long top decile, short bottom, net of cost
    qo=pd.qcut(pd.Series(po),10,labels=False,duplicates='drop')
    longs=yo.values[qo==qo.max()]; shorts=yo.values[qo==0]
    cost=0.30
    net=np.concatenate([longs-cost, -shorts-cost])
    t=net.mean()/(net.std()/np.sqrt(len(net)))
    print(f"  OOS tradeable: {len(net):,} trades, net ${net.mean():+.3f}/oz per trade, t={t:+.2f}"
          f"  {'PROFITABLE' if net.mean()>0 else 'unprofitable'}")
