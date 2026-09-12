"""
S102c - What a second sleeve has to be worth, stated as a number.

S102b showed every horizon blend losing at the gate. That is the answer to the
question asked, but it is the weaker form of it, because it only tested equal
weights and only tested four sleeves. Two things are missing.

    1. An ORACLE WEIGHT. If even the best possible mix - chosen with full
       hindsight over the whole sample, which is not tradable - fails to beat
       12h alone, the channel is closed rather than merely badly weighted. Same
       device as S96b's oracle ceiling on the ranking.

    2. A REUSABLE BAR. At the -20% gate the comparison is a Calmar comparison,
       and for well-behaved series a Sharpe comparison. Combining two streams at
       correlation rho and Sharpes S1, S2 with vol weights beats S1 alone only if

               S2  >  S1 * (sqrt(2 + 2*rho) - 1)          [equal vol weights]

       That converts "blending did not help" into a threshold any candidate
       sleeve can be held to BEFORE it is built, and it explains the whole run
       of blend failures in this log at once: S42's phases correlated 0.70-0.82
       and would have needed Sharpe ~1.9; S98's disjoint books correlated 0.068
       and needed only ~0.9, and still failed - because their edge was gone, not
       because of their correlation.

Both are computed from the same sleeves S102b measured, so nothing new is fitted
and the two files cannot disagree.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s69_calsel as S69
import strategies.s102_horizon as S102
from strategies.s102b_blend import use, HORIZONS
from research.harness import OOS_END
from strategies.s96_rank import at_gate, stats_of

CFG = (2.0, 3.0, 2.0, 21)
RISK = 0.08


def series(risk):
    out = {}
    for tf in HORIZONS:
        use(tf)
        out[tf] = S69.daily(S69.sim(CFG, S102.FULL_START, OOS_END, risk))
    return pd.DataFrame(out).fillna(0.0)


def required_sharpe(s1, rho):
    """Sharpe a second, equal-vol sleeve needs before it is worth holding."""
    return s1 * (np.sqrt(2 + 2 * rho) - 1)


if __name__ == "__main__":
    D = series(RISK / 2)
    sh = {t: stats_of(D[t].to_numpy())["sharpe"] for t in HORIZONS}
    C = D.corr()

    print(f"config exp {CFG[0]}  {CFG[1]}ATR x{CFG[2]}R  {CFG[3]}d, each sleeve at "
          f"{RISK/2*100:.0f}% risk\n")
    print("1. the admission bar, applied to the horizons that exist")
    s1 = sh["12h"]
    print(f"   12h sleeve Sharpe {s1:.2f}\n")
    print(f"{'sleeve':>10}{'Sharpe':>9}{'rho vs 12h':>12}{'needs':>9}{'shortfall':>11}")
    for t in HORIZONS:
        if t == "12h":
            continue
        need = required_sharpe(s1, C.loc[t, "12h"])
        print(f"{t:>10}{sh[t]:9.2f}{C.loc[t, '12h']:12.3f}{need:9.2f}"
              f"{sh[t] - need:+11.2f}")

    print("\n2. the bar as a function of correlation, for any future sleeve")
    print(f"{'rho':>8}{'Sharpe needed':>16}{'as % of 12h':>14}")
    for rho in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        need = required_sharpe(s1, rho)
        print(f"{rho:8.1f}{need:16.2f}{need/s1*100:13.0f}%")

    print("\n3. oracle weights - the best mix with full hindsight, which is not tradable")
    print(f"{'pair':>16}{'best w on 2nd':>15}{'Shp':>7}{'Clm':>7}{'at -20%':>10}{'vs 12h':>9}")
    base = at_gate(D["12h"].to_numpy())["cagr"] * 100
    print(f"{'12h alone':>16}{'-':>15}{sh['12h']:7.2f}"
          f"{stats_of(D['12h'].to_numpy())['calmar']:7.2f}{base:10.1f}%{'':>9}")
    ws = np.linspace(0.0, 1.0, 51)
    for t in HORIZONS:
        if t == "12h":
            continue
        best = None
        for w in ws:
            r = ((1 - w) * D["12h"] + w * D[t]).to_numpy()
            g = at_gate(r)
            if best is None or g["cagr"] > best[1]["cagr"]:
                best = (w, g, stats_of(r))
        w, g, s = best
        print(f"{'12h + ' + t:>16}{w:15.2f}{s['sharpe']:7.2f}{s['calmar']:7.2f}"
              f"{g['cagr']*100:10.1f}%{g['cagr']*100 - base:+9.1f}")

    # all three alternatives at once, still with hindsight on the weights
    best = None
    for w6 in np.linspace(0, 0.6, 13):
        for w8 in np.linspace(0, 0.6, 13):
            for w24 in np.linspace(0, 0.6, 13):
                if w6 + w8 + w24 > 0.9:
                    continue
                r = ((1 - w6 - w8 - w24) * D["12h"] + w6 * D["6h"]
                     + w8 * D["8h"] + w24 * D["24h"]).to_numpy()
                g = at_gate(r)
                if best is None or g["cagr"] > best[-1]["cagr"]:
                    best = (w6, w8, w24, g)
    w6, w8, w24, g = best
    print(f"{'all four':>16}{f'{w6:.2f}/{w8:.2f}/{w24:.2f}':>15}{'':>7}{'':>7}"
          f"{g['cagr']*100:10.1f}%{g['cagr']*100 - base:+9.1f}")
    print("\ndone: horizon bar")
