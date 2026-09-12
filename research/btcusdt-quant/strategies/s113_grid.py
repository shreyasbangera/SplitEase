"""
S113 - Is the configuration grid itself the binding constraint?

S96b bounded config selection at 268.9% with perfect foresight, and that closed
the channel. But the bound is over a FIXED MENU of 200 configurations, and it
says nothing about whether the menu is the right one. Widening the menu raises
the oracle ceiling by construction; it is the one axis the oracle argument does
not cover.

There is direct evidence the menu is too narrow. Over 18 quarters and three
sleeves, the selection's modal pick sits **on a grid boundary on four of the
five axes**:

    exponent  3.0 chosen 27 of 54   <- the MAXIMUM offered
    stop      2.5 chosen 37 of 54   <- the MINIMUM offered
    hold       14 chosen 37 of 54   <- the MINIMUM offered
    RR        3.0 chosen 32 of 54   <- the MAXIMUM offered
    gate      100 chosen 21 of 54   (interior)

That is the signature of an optimiser pressing against a fence.

One of those fences is already known to be real. **S78 extended the exponent to
6.0**: the selection reaches for it, profit factor rises to 3.72, and Sharpe and
Calmar both FALL (6.58 -> 5.22). Concentrating into fewer, larger bets keeps
improving the win/loss ratio long after it stops improving risk-adjusted return.
So exponent stays capped at 3.0 and is not re-opened here.

**The other three have never been moved.** Stop has only ever been offered
{2.5, 3.0}, hold only {14, 21}, reward:risk only {2.0, 3.0}. S71 swept the
betting parameters (the |net| cap, the flat-exit threshold, the threshold scale)
and S94 swept the payoff shape at portfolio level, but the menu the quarterly
selection draws from has never been widened on these axes.

WHAT THIS FILE IS AND IS NOT
----------------------------
This is a full-sample, single-configuration SCREEN. It is in-sample by
construction and can only answer "is the fence binding?" - never "what should be
adopted". Nothing here is adopted. If a fence turns out to be binding, the
widened axis goes into the quarterly grid and has to be chosen causally, quarter
by quarter, exactly as S85 did for the trend gate.

The prior is not favourable, and it is worth stating first so the result cannot
be read after the fact. S104 tightened exits and the book collapsed: a 1-ATR
trailing stop took Calmar from 7.00 to 1.79, and every exit intervention
converged on "do nothing" from below. A tighter FIXED stop is a milder version of
the same thing. Against that, only 1.9% of V7's trades currently exit on the
stop and 93.2% exit on signal, so the stop is nearly inert - which cuts both
ways: an inert parameter cannot be the constraint, but it also cannot cost much
to tighten.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
from research.harness import OOS_END
import strategies.s45_single as S45
import strategies.s69_calsel as S69
import strategies.s84_gate as S84
import strategies.s110_meta as M
from strategies.s96_rank import at_gate, stats_of

RISK = 0.08
# The modal pick, so each axis is swept around the configuration the selection
# actually favours rather than around an arbitrary centre.
BASE = dict(p=3.0, stp=2.5, rr=3.0, hold=14, span=100, mode="s")

AXES = {
    "stop (ATR)":  ("stp", [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
    "reward:risk": ("rr", [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]),
    "hold (days)": ("hold", [3, 5, 7, 10, 14, 21, 35]),
}
CURRENT = {"stp": [2.5, 3.0], "rr": [2.0, 3.0], "hold": [14, 21]}


def sim(cfg):
    p, stp, rr, hold = cfg["p"], cfg["stp"], cfg["rr"], cfg["hold"]
    c = S69.ctx(); u0 = np.nan_to_num(S69.shape(p)); u = u0; a = c["a"]
    if cfg["span"]:
        up = S84.trend(cfg["span"])
        block = np.zeros(len(u), bool)
        if "s" in cfg["mode"]: block |= (u < 0) & up
        if "l" in cfg["mode"]: block |= (u > 0) & ~up
        u = np.where(block, 0.0, u)
    arr = dict(entry=u, stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u0) <= 0.0).astype(float))
    from research.harness import backtest
    return backtest(c["g"], arr, "12h", start=S45.FULL_START, end=OOS_END,
                    risk=RISK, max_lev=10.0, max_bars_h=hold * 24)


if __name__ == "__main__":
    M.use_clock()
    print("full-sample screen, one configuration at a time, risk 8%.")
    print("IN-SAMPLE by construction: this asks whether the fence binds, nothing more.\n")
    base_m = sim(BASE)
    base_g = at_gate(np.asarray(S69.daily(base_m), float))["cagr"] * 100
    print(f"centre of the sweep = the modal pick: exp {BASE['p']} "
          f"{BASE['stp']}ATR x{BASE['rr']}R {BASE['hold']}d gate EMA{BASE['span']}")
    print(f"  -> CAGR {base_m['cagr']*100:.1f}%  DD {base_m['max_dd']*100:.1f}%  "
          f"Calmar {base_m['calmar']:.2f}  at -20% {base_g:.1f}%\n")

    for label, (key, values) in AXES.items():
        print(f"{label}")
        print(f"{'value':>9}{'in grid':>9}{'CAGR':>9}{'MaxDD':>8}{'PF':>6}{'N':>6}"
              f"{'stops':>7}{'Shp':>6}{'Clm':>6}{'at -20%':>10}{'vs centre':>11}")
        for v in values:
            cfg = dict(BASE); cfg[key] = v
            m = sim(cfg)
            td = m["trades_df"]
            stops = (td.reason == 1).mean() * 100 if td is not None and len(td) else np.nan
            g = at_gate(np.asarray(S69.daily(m), float))["cagr"] * 100
            mark = "yes" if v in CURRENT[key] else "NEW"
            print(f"{v:>9}{mark:>9}{m['cagr']*100:8.1f}%{m['max_dd']*100:7.1f}%"
                  f"{m['profit_factor']:6.2f}{m['trades']:6d}{stops:6.1f}%"
                  f"{m['sharpe']:6.2f}{m['calmar']:6.2f}{g:9.1f}%{g-base_g:+11.1f}",
                  flush=True)
        print()
    print("done: grid edges")
