import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work')
import numpy as np, pandas as pd, engine
from engine import Costs, RiskCfg

b = engine.bars('15min')
print("spread by year (USD):"); print((b['spread']*0.001).groupby(b.index.year).median().to_string())

atr = (b.high-b.low).rolling(14).mean().values
rng = np.random.default_rng(7)

print("\n--- NULL TEST: random entries, symmetric 1.5R target, real costs ---")
for seed in range(3):
    rng = np.random.default_rng(seed)
    ent = np.where(rng.random(len(b)) < 0.01, rng.choice([-1,1],len(b)), 0)
    r = engine.run(b, ent, atr*1.5, atr*2.25, '15min',
                   costs=Costs(), risk=RiskCfg(risk_pct=0.005),
                   max_hold_bars=100, name=f"random_seed{seed}")
    m=r.m
    print(f"  seed{seed}: trades {m['trades']:5d}  avg_R {m['avg_R']:+.4f}  "
          f"expectancy ${m['expectancy']:+.2f}  PF {m['pf']:.3f}  net {m['net_pct']:+.1f}%")

print("\n--- ZERO-COST control (should be ~0.000 avg R, not positive) ---")
free = Costs(comm_bps_side=0, slip_usd_side=0, stop_slip_usd=0, spread_mult=0)
for seed in range(3):
    rng = np.random.default_rng(seed)
    ent = np.where(rng.random(len(b)) < 0.01, rng.choice([-1,1],len(b)), 0)
    r = engine.run(b, ent, atr*1.5, atr*2.25, '15min', costs=free,
                   risk=RiskCfg(risk_pct=0.005), max_hold_bars=100, name="free")
    m=r.m
    print(f"  seed{seed}: trades {m['trades']:5d}  avg_R {m['avg_R']:+.4f}  PF {m['pf']:.3f}")
