import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
from lab import *
from feat import build

b,X=build('15min')
ISm=(X.index>=IS_START)&(X.index<=IS_END); OOSm=(X.index>=OOS_START)&(X.index<=OOS_END)
feats=[f for f in X.columns if not f.startswith('fwd') and f!='atr_usd']

def dec_edge(D,ft,N):
    x=D[ft]; f_=D[f'fwd{N}']
    try: q=pd.qcut(x,10,labels=False,duplicates='drop')
    except Exception: return None
    if q.max() is np.nan or q.max()<9: return None
    top=f_[q==q.max()]; bot=f_[q==0]
    if len(top)<200 or len(bot)<200: return None
    d=top.mean()-bot.mean(); se=np.sqrt(top.var()/len(top)+bot.var()/len(bot))
    # normalise by the ATR of that period so IS/OOS are comparable across price levels
    return d, (d/se if se>0 else 0), d/D['atr_usd'].mean()

print("IS -> OOS STABILITY OF THE DECILE EDGE  (edge normalised by mean ATR so periods are comparable)")
print(f"{'feature':<12}{'N':>4} | {'IS edge/atr':>12}{'IS t':>7} | {'OOS edge/atr':>13}{'OOS t':>7} | {'stable?':>8}")
print("-"*80)
out=[]
for N in (8,32,96):
    for ft in feats:
        a=dec_edge(X[ISm],ft,N); o=dec_edge(X[OOSm],ft,N)
        if not a or not o: continue
        stable = (np.sign(a[2])==np.sign(o[2])) and abs(a[1])>3 and abs(o[1])>2
        out.append((abs(a[1]),ft,N,a,o,stable))
out.sort(reverse=True)
for _,ft,N,a,o,st in out[:28]:
    print(f"{ft:<12}{N:>4} | {a[2]:>+12.3f}{a[1]:>7.1f} | {o[2]:>+13.3f}{o[1]:>7.1f} | {'YES' if st else '-':>8}")

print("\n\nSURVIVORS (same sign IS & OOS, |t_IS|>3, |t_OOS|>2):")
surv=[(ft,N,a,o) for _,ft,N,a,o,st in out if st]
for ft,N,a,o in surv: print(f"  {ft:<12} N{N:<4} IS {a[2]:+.3f}atr (t{a[1]:.1f})   OOS {o[2]:+.3f}atr (t{o[1]:.1f})")
if not surv: print("  (none)")
