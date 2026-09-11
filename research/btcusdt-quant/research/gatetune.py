"""Find the largest risk whose realised drawdown stays STRICTLY under the 20% gate.

The registry's bisection stops within a tolerance and can land a shade over, which
is not good enough for a gate the brief states as a hard limit.
"""
import sys; sys.path.insert(0, "/home/user/quant")
from strategies.registry import v1, v3, v5, v6, measure

CASES = (("V6 top-5 gated", v6, (0.110, 0.113)),
         ("V5 top-5 blend", v5, (0.094, 0.096)),
         ("V3 12m plain",   v3, (0.108, 0.110)),
         ("V1 fixed config", v1, (0.118, 0.120)))

for tag, fn, risks in CASES:
    for r in risks:
        m = measure(*fn(r))
        ok = abs(m["dd"]) < 0.200
        print(f"{tag:<16} risk {r*100:5.2f}%  CAGR {m['cagr']*100:7.1f}%  "
              f"DD {m['dd']*100:6.2f}%  {'OK ' if ok else 'over'}  PF {m['pf']:5.2f}  "
              f"N {m['n']:5d}  Shp {m['sharpe']:5.2f}  Clm {m['calmar']:5.2f}  "
              f"P>20 {m['p20']*100:3.0f}%", flush=True)
        if ok:
            print("      yearly " + "  ".join(f"{y}:{x*100:+.0f}%"
                  for y, x in m["yearly"].items()), flush=True)
print("done: gate tune")
