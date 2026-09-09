import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
import numpy as np, pandas as pd, engine, json, os
from engine import Costs, RiskCfg

IS_START,IS_END   = '2017-04-28','2022-12-31'
OOS_START,OOS_END = '2023-01-01','2026-12-31'
LOG = "/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work/results.jsonl"

def slice_(b,a,z): return b[(b.index>=a)&(b.index<=z)]

def atr(b,n=14):
    pc=b.close.shift(1)
    tr=pd.concat([b.high-b.low,(b.high-pc).abs(),(b.low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))

def log(res, tag, extra=None):
    m=dict(res.m); m['name']=res.name; m['tf']=res.tf; m['tag']=tag
    m.update(extra or {})
    m={k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in m.items()}
    with open(LOG,'a') as f: f.write(json.dumps(m,default=str)+"\n")
    return res

def show(res, tag=''):
    print(f"  [{tag:>3}] {res.line()}")
    return res
