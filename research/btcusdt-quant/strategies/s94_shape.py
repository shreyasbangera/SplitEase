"""
S94 - Can the SHAPE of V7's equity curve be changed? Portfolio overlays, re-tested.

WHY ASK AGAIN
-------------
The risk dial settled that leverage is a pure scale knob on this book: profit
factor 3.18 +/- 0.05 and Sharpe 2.15 +/- 0.03 across a six-fold range of size,
CAGR and drawdown moving together. So the 20% drawdown gate binds at 14.4% risk
and 179.0% CAGR, and the brief's 300% is out of reach by leverage alone.

The only way past that is to change the SHAPE - to cut the deep part of the
drawdown without giving up the same fraction of return. Then the gate binds at a
larger base size and the return at the gate rises.

S34 tested exactly this and recorded a clean negative: volatility targeting was
Calmar-neutral, the high-water-mark throttle "actively destroys the book", and
the conclusion was "do not de-risk into drawdowns on a mean-reverting-equity
strategy". But S34 ran on the S32 multi-instrument portfolio - PF 1.27, Sharpe
1.34 - which is not this book. V7 is single-instrument, conviction-sized,
quarterly-selected, PF 3.18, Sharpe 2.15. A conclusion about the shape of one
equity curve does not transfer to another by assertion.

THE PRIOR IS STILL NEGATIVE, AND NOW FOR A MEASURED REASON
----------------------------------------------------------
Two facts about V7 specifically, both already in the log, predict the throttle
fails here too:

  * 72% of days are spent in drawdown, and the worst episode is 129 days of
    grind rather than a shock. A high-water-mark throttle would therefore be
    throttling most of the time.
  * the top 10% of trades contribute 130% of net profit - the other 90% lose
    money together. De-risking in a drawdown means being small precisely when
    the rare payers arrive, and episode 2 in the anatomy recovered 11.3% in
    TWO DAYS.

So this is run expecting a negative. It is worth running anyway because the
lever is the last structural one open, and because the prior generates a new
idea worth testing in the same breath (below).

THE INVERSE, WHICH S34 DID NOT TRY
----------------------------------
S34's own explanation for the failure - "cutting size in a drawdown means being
small through the recovery, and for a book whose losses are not serially
correlated that is a pure tax" - implies the opposite may pay: size UP in
drawdown. That is a martingale and dangerous in practice; it is included here to
measure the mechanism, not to recommend it, and a positive result would be
reported with that caveat attached.

METHOD
------
Every overlay is applied to V7's own daily returns, causally: the multiplier for
day t is computed from data through day t-1 only. This is valid because the book
is linear in risk - trade P&L scales with the risk budget, which the risk dial
confirms - so scaling a daily return by m is the same as having traded that day
at m x the base size. The linearity is CHECKED rather than assumed, below.

Each overlay is then compared AT THE GATE: bisect the global scale that puts
realised max drawdown on -20%, and read CAGR there. That is the only comparison
that means anything, because any overlay can be made to look good by quietly
changing the size it runs at.

Baseline to beat: 179.0% CAGR at -20.0%.
"""
import sys, os, json
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

CACHE = "/home/user/quant/results/s94_v7_daily.json"


# ----------------------------------------------------------------- the book

def v7_daily(risk):
    """V7's daily returns at a base risk, cached - each call is ~3 minutes."""
    key = f"{risk:.4f}"
    store = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if key not in store:
        from strategies.registry import v7
        r, _ = v7(risk)
        store[key] = dict(idx=[str(d.date()) for d in r.index],
                          val=[float(x) for x in r.to_numpy()])
        json.dump(store, open(CACHE, "w"))
    d = store[key]
    return pd.Series(d["val"], index=pd.to_datetime(d["idx"]))


# ----------------------------------------------------------------- measuring

def curve(r):
    return np.cumprod(1.0 + np.asarray(r, float))


def stats(r):
    r = np.asarray(r, float)
    e = curve(r)
    yrs = len(r) / 365.25
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    cagr = float(e[-1] ** (1 / yrs) - 1)
    sd = r.std()
    return dict(cagr=cagr, dd=dd,
                sharpe=float(r.mean() / sd * np.sqrt(365.25)) if sd else float("nan"),
                calmar=cagr / abs(dd) if dd else float("nan"))


def at_gate(apply_overlay, r, target=-0.20, lo=0.2, hi=6.0, iters=40):
    """Scale the book until realised max drawdown sits on the gate.

    Bisects the SCALE, not the overlay's knobs, so every row below is quoted at
    the same realised risk and the only thing that differs is the shape.
    """
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = stats(apply_overlay(r * mid))
        m["scale"] = mid
        if best is None or abs(m["dd"] - target) < abs(best["dd"] - target):
            best = m
        if abs(m["dd"] - target) < 1e-4:
            break
        if m["dd"] < target:
            hi = mid
        else:
            lo = mid
    return best


# ----------------------------------------------------------------- overlays

def none(r):
    return np.asarray(r, float)


def hwm_throttle(soft, hard, floor, invert=False):
    """Size by how deep the drawdown is, measured at yesterday's close.

    Full size while shallower than `soft`, tapering linearly to `floor` x at
    `hard`. invert=True runs the same ramp upwards instead - a martingale,
    measured for the mechanism rather than recommended.
    """
    def f(r):
        r = np.asarray(r, float)
        out = np.empty_like(r)
        eq = 1.0
        hwm = 1.0
        for i, x in enumerate(r):
            dd = eq / hwm - 1.0                       # through yesterday only
            if dd < -soft:
                frac = (hard + dd) / (hard - soft)    # 1 at soft, 0 at hard
                frac = min(max(frac, 0.0), 1.0)
                m = floor + (1.0 - floor) * frac
                if invert:
                    m = 1.0 + (1.0 - floor) * (1.0 - frac)
            else:
                m = 1.0
            out[i] = m * x
            eq *= 1.0 + out[i]
            hwm = max(hwm, eq)
        return out
    return f


def vol_target(annual, window=30, cap=3.0):
    """Scale to a constant annualised volatility, from trailing realised vol."""
    def f(r):
        s = pd.Series(np.asarray(r, float))
        rv = s.rolling(window).std().shift(1) * np.sqrt(365.25)
        m = (annual / rv).clip(upper=cap).fillna(1.0).to_numpy()
        return s.to_numpy() * m
    return f


# ----------------------------------------------------------------- run

def line(tag, m):
    print(f"{tag:>34} | CAGR {m['cagr']*100:7.1f}%  DD {m['dd']*100:6.1f}%"
          f"  Shp {m['sharpe']:5.2f}  Clm {m['calmar']:5.2f}  scale {m['scale']:.2f}",
          flush=True)


if __name__ == "__main__":
    print("loading V7 daily returns (cached after the first run)", flush=True)
    r8 = v7_daily(0.08)
    print(f"  {len(r8)} days, {r8.index[0].date()} -> {r8.index[-1].date()}\n", flush=True)

    # --- the assumption everything below rests on -------------------------
    print("LINEARITY CHECK - is scaling the daily return the same as sizing up?")
    r12 = v7_daily(0.12)
    scaled = r8 * 1.5
    a, b = stats(scaled), stats(r12)
    print(f"  8% x 1.5   CAGR {a['cagr']*100:7.1f}%  DD {a['dd']*100:6.1f}%", flush=True)
    print(f"  12% actual CAGR {b['cagr']*100:7.1f}%  DD {b['dd']*100:6.1f}%", flush=True)
    print(f"  correlation of daily returns {np.corrcoef(scaled, r12)[0,1]:.4f}", flush=True)
    print(f"  mean abs difference {np.abs(scaled - r12).mean():.6f} on a daily "
          f"sd of {r12.std():.4f}\n", flush=True)

    # --- everything at the gate ------------------------------------------
    print("AT THE -20% GATE - scale bisected so realised drawdown matches\n")
    base = at_gate(none, r8)
    line("no overlay (V7 as deployed)", base)
    print()

    for annual in (0.20, 0.30, 0.40, 0.50, 0.70):
        line(f"vol target {annual*100:.0f}%", at_gate(vol_target(annual), r8))
    print()

    for soft, hard, floor in [(0.05, 0.20, 0.0), (0.05, 0.15, 0.25),
                              (0.08, 0.22, 0.0), (0.08, 0.20, 0.25),
                              (0.10, 0.25, 0.0), (0.10, 0.20, 0.50),
                              (0.12, 0.30, 0.0), (0.15, 0.30, 0.25)]:
        line(f"throttle {soft*100:.0f}/{hard*100:.0f} floor {floor:.2f}",
             at_gate(hwm_throttle(soft, hard, floor), r8))
    print()

    for soft, hard, floor in [(0.05, 0.20, 0.5), (0.08, 0.22, 0.5),
                              (0.10, 0.25, 0.5), (0.10, 0.25, 0.0)]:
        line(f"INVERSE {soft*100:.0f}/{hard*100:.0f} floor {floor:.2f}",
             at_gate(hwm_throttle(soft, hard, floor, invert=True), r8))

    print(f"\nbaseline to beat: 179.0% at -20.0%")
    print("done: shape overlays")
