"""
S105b - Is the account cap's +8.7 points a small real effect or a sweep artefact?

S105's two families each traced a hump: notional caps read +0.3, -0.8, +4.4,
-6.3, -20.2, -50.1 and risk caps +1.4, +8.7, -3.6, -28.9. The best cell is
already inside S96c's selection-noise band (sd about 24 points), so it does not
clear the bar for a candidate and nothing will be adopted from it either way.

It is still worth knowing which it is. A cap that genuinely trims a fat tail
should help in BOTH halves of the sample, because the tail is a property of the
sizing rule and not of a particular market. A cap that is a sweep artefact will
carry its whole gain in one half - most likely the half containing the single
2024-07 episode that motivated it, which is the trap this file exists to catch.

The split is at the midpoint of the tradeable record, chosen for no other reason
than that it is the midpoint, and each half is scaled to the -20% gate
separately so the two halves are judged on shape rather than on which one
happened to trend.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s87_combined as S87
from strategies.s105_agglev import replay
from strategies.s96_rank import at_gate, stats_of

VARIANTS = [("baseline", np.inf, "notional"),
            ("notional <= 2.5x", 2.5, "notional"),
            ("notional <= 2.0x", 2.0, "notional"),
            ("risk <= 16%", 0.16, "risk"),
            ("risk <= 12%", 0.12, "risk"),
            ("risk <= 10%", 0.10, "risk")]


if __name__ == "__main__":
    R = S87.rankings()
    print("the account cap, measured separately on each half of the record\n")
    print(f"{'variant':>20}{'full':>10}{'1st half':>11}{'2nd half':>11}"
          f"{'worse half':>12}{'DD 1st':>9}{'DD 2nd':>9}")
    base = {}
    for tag, cap, mode in VARIANTS:
        r, pl, b = replay(R, cap, mode)
        a = np.asarray(r, float)
        h = len(a) // 2
        full, h1, h2 = at_gate(a), at_gate(a[:h]), at_gate(a[h:])
        s1, s2 = stats_of(a[:h]), stats_of(a[h:])
        if not base:
            base = dict(full=full["cagr"], h1=h1["cagr"], h2=h2["cagr"])
        print(f"{tag:>20}{full['cagr']*100:9.1f}%{h1['cagr']*100:10.1f}%"
              f"{h2['cagr']*100:10.1f}%"
              f"{min(h1['cagr'] - base['h1'], h2['cagr'] - base['h2'])*100:+11.1f}"
              f"{s1['dd']*100:8.1f}%{s2['dd']*100:8.1f}%", flush=True)
    print(f"\nsplit at day {len(np.asarray(r, float))//2} of "
          f"{len(np.asarray(r, float))}")
    print("\ndone: account cap, split")
