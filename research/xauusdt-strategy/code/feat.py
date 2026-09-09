"""Feature matrix + systematic conditional-edge search. All features use completed bars only."""
import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *

def build(TF='15min'):
    b=engine.bars(TF); A=atr(b,32)
    o,h,l,c,v,sp=b.open,b.high,b.low,b.close,b.tick_volume,b['spread']*0.001
    X=pd.DataFrame(index=b.index)
    X['hour']=b.index.hour; X['dow']=b.index.dayofweek; X['dom']=b.index.day
    X['atr']=A; X['volq']=A.rolling(480).rank(pct=True)
    X['spread']=sp; X['spq']=sp.rolling(480).rank(pct=True)
    X['volume_r']=v/v.rolling(96).mean()
    for k in (4,16,48,192):
        X[f'mom{k}']=(c-c.shift(k))/A
    for k in (20,50,200):
        X[f'dev{k}']=(c-ema(c,k))/A
    X['rsi']=rsi(c,14)
    hh=h.rolling(96).max(); ll=l.rolling(96).min()
    X['rngpos']=(c-ll)/(hh-ll).replace(0,np.nan)
    X['body']=(c-o)/A
    X['wick_up']=(h-np.maximum(c,o))/A; X['wick_dn']=(np.minimum(c,o)-l)/A
    X['barret']=(c-o)/A
    # forward targets: enter next bar OPEN, exit N bars later. USD per oz.
    nx=o.shift(-1)
    for N in (4,8,32,96):
        X[f'fwd{N}']=(c.shift(-N)-nx)
    X['atr_usd']=A
    return b,X

if __name__=='__main__':
    b,X=build('15min')
    ISm=(X.index>=IS_START)&(X.index<=IS_END)
    Xi=X[ISm]
    feats=[f for f in X.columns if not f.startswith('fwd') and f!='atr_usd']
    print(f"IS rows {len(Xi):,}  features {len(feats)}")
    print("\nSINGLE-FEATURE DECILE SCAN — directional edge (long-minus-short) in USD/oz")
    print(f"{'feature':<12}{'N':>5}  {'top-decile edge':>18}{'bot-decile edge':>18}{'spread(top-bot)':>18}{'t':>7}")
    print("-"*90)
    rows=[]
    for N in (8,32,96):
        f_=Xi[f'fwd{N}']
        for ft in feats:
            x=Xi[ft]
            if x.nunique()<10: continue
            try: q=pd.qcut(x,10,labels=False,duplicates='drop')
            except Exception: continue
            g=f_.groupby(q)
            top=g.get_group(q.max()) if q.max() in g.groups else None
            bot=g.get_group(0) if 0 in g.groups else None
            if top is None or bot is None or len(top)<300 or len(bot)<300: continue
            d=top.mean()-bot.mean()
            se=np.sqrt(top.var()/len(top)+bot.var()/len(bot))
            t=d/se if se>0 else 0
            rows.append((abs(t),ft,N,top.mean(),bot.mean(),d,t))
    rows.sort(reverse=True)
    for _,ft,N,tm,bm,d,t in rows[:25]:
        print(f"{ft:<12}{N:>5}  {tm:>+18.3f}{bm:>+18.3f}{d:>+18.3f}{t:>7.2f}")
