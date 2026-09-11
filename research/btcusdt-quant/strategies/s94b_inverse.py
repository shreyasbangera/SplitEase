"""
S94b - Pushing the one overlay that worked until it breaks.

S94 found every de-risking overlay loses at the gate, as S34 predicted, and
found the INVERSE - sizing UP as the drawdown deepens - gains: 176.4% against a
same-method baseline of 168.8%, Sharpe 2.22 against 2.15.

A positive result from a grid I chose is not a finding until it has been pushed
until it breaks. Three things decide whether this is real:

1. WHERE DOES IT TURN OVER? In S94 the gain was monotone in aggressiveness -
   the most aggressive setting on the grid was the best one. An optimum sitting
   on the edge of the menu is not an optimum, it is a direction. If it keeps
   improving as the up-sizing gets more extreme, with no turnover inside sane
   bounds, then the "gain" is just leverage applied at the worst moment being
   rewarded by a sample in which every drawdown happened to recover. That is
   the martingale's signature, not an edge.

2. DOES IT SURVIVE A SPLIT? The overlay is fitted on the whole sample. Halved,
   it either holds in both halves or it is one episode.

3. WHAT SIZE DOES IT DEMAND? This is the question a backtest answers badly and
   an exchange answers immediately. A martingale that needs 2x the nominal
   position at the bottom of the worst drawdown is a margin call, not a
   strategy, and the engine's 10x leverage cap is not modelled by scaling daily
   returns. Peak multiplier is reported for every row.

The baseline for all of it is 168.8% - S94's own no-overlay number under the
same scaling method, NOT the 179.0% the engine produces directly. Comparing an
overlay measured one way against a baseline measured another is how a 6%
methodology gap becomes a discovery.
"""
import sys
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

from strategies.s94_shape import v7_daily, stats, at_gate, none


def inverse(soft, hard, peak):
    """Size UP as the drawdown deepens: 1x until `soft`, `peak` x at `hard`.

    Returns the scaled series and the largest multiplier it ever demanded.
    """
    def f(r, want_peak=False):
        r = np.asarray(r, float)
        out = np.empty_like(r)
        eq = hwm = 1.0
        mx = 1.0
        for i, x in enumerate(r):
            dd = eq / hwm - 1.0                        # through yesterday only
            if dd < -soft:
                frac = min(max((-dd - soft) / (hard - soft), 0.0), 1.0)
                m = 1.0 + (peak - 1.0) * frac
            else:
                m = 1.0
            mx = max(mx, m)
            out[i] = m * x
            eq *= 1.0 + out[i]
            hwm = max(hwm, eq)
        return (out, mx) if want_peak else out
    return f


def peak_of(f, r, scale):
    return f(np.asarray(r, float) * scale, want_peak=True)[1]


def row(tag, f, r, base=None):
    m = at_gate(f, r)
    pk = peak_of(f, r, m["scale"]) if f is not none else 1.0
    d = "" if base is None else f"  {(m['cagr']-base)*100:+6.1f}pt"
    print(f"{tag:>30} | CAGR {m['cagr']*100:7.1f}%  DD {m['dd']*100:6.1f}%"
          f"  Shp {m['sharpe']:5.2f}  Clm {m['calmar']:5.2f}"
          f"  peak size {pk:4.2f}x{d}", flush=True)
    return m


if __name__ == "__main__":
    r8 = v7_daily(0.08)
    base = at_gate(none, r8)["cagr"]
    print(f"V7 daily returns, {len(r8)} days, "
          f"{r8.index[0].date()} -> {r8.index[-1].date()}")
    print(f"same-method baseline at the -20% gate: {base*100:.1f}%\n")

    print("1. HOW FAR DOES IT GO BEFORE IT TURNS OVER?")
    print("   soft/hard fixed at 5/20; the peak multiplier is the lever.\n")
    row("no overlay", none, r8, base)
    for peak in (1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
        row(f"inverse 5/20 peak {peak:.2f}x", inverse(0.05, 0.20, peak), r8, base)

    print("\n   and with the ramp starting immediately (soft=0):\n")
    for peak in (1.5, 2.0, 3.0, 6.0):
        row(f"inverse 0/20 peak {peak:.2f}x", inverse(0.0, 0.20, peak), r8, base)

    print("\n2. DOES IT SURVIVE A SPLIT?")
    mid = len(r8) // 2
    for lab, seg in (("first half", r8.iloc[:mid]), ("second half", r8.iloc[mid:])):
        b = at_gate(none, seg)["cagr"]
        print(f"\n   {lab}: {seg.index[0].date()} -> {seg.index[-1].date()}, "
              f"baseline {b*100:.1f}%\n")
        row("no overlay", none, seg, b)
        for peak in (1.5, 2.0, 3.0):
            row(f"inverse 5/20 peak {peak:.2f}x", inverse(0.05, 0.20, peak), seg, b)

    print("\ndone: inverse")
