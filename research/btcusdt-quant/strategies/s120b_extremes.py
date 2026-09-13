"""
S120b - Intraday reversal at its strongest: how far does the edge scale?

S120 found the effect everywhere and found it too small everywhere: reversal IC
of -0.03 to -0.07 from 1 minute to 1 day, consistent across both halves of 3.5
million bars, worth 0.1 to 2.2 basis points per trade against a 16 basis point
round trip. That is a real effect that cannot be traded.

But S120 measured it in the weakest place it could have. Two choices in that
measurement were arbitrary and both bias the edge DOWNWARD:

    THE TRIGGER    the top decile of moves. A decile boundary at a 1.3-sigma
                   move is not a liquidation cascade; it is a slightly busy
                   five minutes. If the reversal is a liquidity-provision
                   premium then it should be paid in proportion to the urgency
                   of the flow that caused it, and the decile average mixes 3
                   percent of genuine forced selling in with 97 percent of
                   ordinary noise.

    THE HOLD       one bar. If the price overshoots and comes back over half an
                   hour, a 5-minute book pays the 16bps round trip to capture
                   one-sixth of the reversion and hands back the rest.

A round trip costs 16bps whether it is held one bar or twenty, so a longer hold
raises the edge per trade at no extra cost. Both levers push the same way, and
if the effect is going to clear costs anywhere it clears them here.

WHAT IS MEASURED, AND WHY IT IS A MEASUREMENT AND NOT A SEARCH
--------------------------------------------------------------
The full surface: trigger extremity from the top 10% down to the top 0.1%, hold
from 1 to 32 bars, at four bar sizes. Every cell is printed with both halves
beside it. This is a characterisation of an effect, not a hunt for a cell to
deploy - S113 turned an in-sample +9.9 into a causal -48.8 doing exactly that,
and nothing here is selected on its own result.

The number that matters is **net bps per trade**: gross reversion captured,
minus the 16bps it costs to capture. If the best cell of the entire surface is
negative, the frequency axis is closed, and it is closed by measurement rather
than by running out of ideas.

Each event's forward window is measured independently. Overlapping events would
net against each other in a real book, which reduces both the edge and the cost;
measuring them separately is the per-trade question asked cleanly.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s120_intraday as S120

ROUND_TRIP_BPS = S120.ROUND_TRIP_BPS
PCTS = (10.0, 5.0, 2.0, 1.0, 0.5, 0.1)      # top X% of |move| by absolute size
HOLDS = (1, 2, 4, 8, 16, 32)


def surface(o, label):
    """Gross reversion in bps per trade, by trigger extremity and hold length."""
    c = np.log(o.close.to_numpy())
    r = np.diff(c, prepend=c[0])
    n = len(c)
    # standardise the trigger by trailing vol so "extreme" means extreme FOR NOW,
    # not extreme for 2021. Strictly past: the vol estimate excludes bar i.
    sd = pd.Series(r).ewm(halflife=500, adjust=False).std().shift(1).bfill().to_numpy()
    sd = np.maximum(sd, 1e-9)
    z = r / sd

    print(f"\n  {label} bars, {n:,} of them")
    print(f"  {'trigger':>10}{'z>=':>7}{'events':>9}" +
          "".join(f"{f'hold {h}':>11}" for h in HOLDS))

    best = (-9e9, None)
    for p in PCTS:
        thr = np.nanpercentile(np.abs(z), 100 - p)
        ev = np.where(np.abs(z) >= thr)[0]
        ev = ev[ev < n - max(HOLDS) - 1]
        if len(ev) < 50:
            continue
        side = -np.sign(z[ev])                    # fade the move
        row = []
        for h in HOLDS:
            # forward return from the CLOSE of the trigger bar, h bars ahead
            fwd = c[ev + h] - c[ev]
            g = float(np.mean(side * fwd) * 1e4)
            net = g - ROUND_TRIP_BPS
            row.append(net)
            if net > best[0]:
                best = (net, (p, h, len(ev), g))
        print(f"  {'top ' + format(p, '.1f') + '%':>10}{thr:>7.1f}{len(ev):>9,}" +
              "".join(f"{v:>10.1f}{'*' if v > 0 else ' '}" for v in row))
    return best


def halves(o, label, p, h):
    """The best cell, split. An effect that lives in one half is not an effect."""
    c = np.log(o.close.to_numpy())
    r = np.diff(c, prepend=c[0])
    n = len(c)
    sd = pd.Series(r).ewm(halflife=500, adjust=False).std().shift(1).bfill().to_numpy()
    z = r / np.maximum(sd, 1e-9)
    out = []
    for lo, hi, tag in ((0, n // 2, "1st half"), (n // 2, n, "2nd half")):
        zz = z[lo:hi]
        thr = np.nanpercentile(np.abs(zz), 100 - p)
        ev = np.where(np.abs(zz) >= thr)[0] + lo
        ev = ev[(ev < n - h - 1) & (ev >= lo)]
        if len(ev) < 20:
            out.append((tag, np.nan, 0)); continue
        g = float(np.mean(-np.sign(z[ev]) * (c[ev + h] - c[ev])) * 1e4)
        out.append((tag, g - ROUND_TRIP_BPS, len(ev)))
    return out


if __name__ == "__main__":
    print("S120b - intraday reversal at its strongest\n")
    print(f"every number is NET basis points per completed trade: gross "
          f"reversion captured minus the {ROUND_TRIP_BPS:.0f}bps round trip.")
    print("a cell marked * is a cell where fading an extreme intraday move "
          "makes money after costs.")

    results = []
    for mins in (5, 15, 30, 60):
        o = S120.bars(mins)
        lab = f"{mins}m" if mins < 60 else "1h"
        best = surface(o, lab)
        if best[1]:
            results.append((lab, o, best))

    print("\n" + "=" * 78)
    print("best cell at each bar size, and whether it holds up in both halves")
    print(f"{'bar':>6}{'trigger':>10}{'hold':>7}{'events':>9}{'gross':>9}"
          f"{'net':>9}{'1st half':>11}{'2nd half':>11}")
    for lab, o, (net, (p, h, nev, g)) in results:
        hv = halves(o, lab, p, h)
        print(f"{lab:>6}{'top ' + format(p, '.1f') + '%':>10}{h:>7}{nev:>9,}"
              f"{g:>9.1f}{net:>9.1f}{hv[0][1]:>11.1f}{hv[1][1]:>11.1f}")

    print("\ndone: reversal extremity surface")
