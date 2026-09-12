"""
S102 - The other half of the sampling question: BAR LENGTH.

S101 found the most fragile thing in this study. The book decides on 12h bars
at 00:00/12:00 UTC, and shifting that PHASE by four hours takes the gate result
from 168.8% to 16.4%. The only defence offered was structural: 00:00 UTC is
crypto's daily anchor, funding settles at 00/08/16, and the two
funding-settlement phases were the two best of six.

That defence makes a testable prediction, and it is not about phase.

    If the edge lives on the canonical UTC grid, it should survive a change of
    HORIZON that keeps the anchor. 8h bars at 00:00 are the funding grid
    exactly. 24h bars at 00:00 are the daily candle exactly. Both are more
    canonical than 12h, which is not a native crypto interval at all - it is
    what pandas resamples to by default.

    If instead the number is a property of one particular sampling of the data,
    then 12h will spike and its neighbours will collapse, exactly as the phases
    did - and the structural story dies, because 8h and 24h are anchored at
    00:00 too.

There is also a constructive reason to run this, and it is the stronger one.
S96b established an ORACLE CEILING of 268.9% over the existing 200-config grid:
with perfect foresight of next-quarter Sharpe, no selection rule whatever
reaches 300%. The target is therefore unreachable without enlarging the space
the configs are drawn from. Horizon is the one axis of the data-generating grid
that has never been varied. If a second horizon is independently alive, a
two-horizon book is a genuinely new object rather than another way of ranking
the same 200 cells - and S42's finding that phase-shifted books correlate
0.70-0.82 does not apply, because those shared a bar length.

WHAT IS HELD CONSTANT, AND WHY IT MATTERS
-----------------------------------------
A naive horizon sweep measures four things at once and cannot separate them.
Every bar-counted parameter in the pipeline is implicitly a wall-clock
parameter, so all of them are rescaled by k = 12/h:

    flow        z-window 480 bars   (240 days)
    cmpx        6-bar log difference, z-window 120 bars
    fundz       z-window 240 bars   (120 days)
    ATR         14 bars
    trend gate  EMA spans 100/150/200/300 bars

btcdom is already computed on a fixed hourly grid upstream and needs no
rescaling; posn is built on the 4h panel ONCE, at its native scaling, and only
SAMPLED onto each horizon - so it is the identical wall-clock signal every time.

The ATR needs a second correction that the others do not. Even over the same
wall-clock window, the average true range of a 6h bar is smaller than that of a
12h bar, so 3xATR would be a tighter stop in price terms and "shorter horizon"
would be confounded with "tighter stop" - while the stop multiple is drawn from
{2.5, 3.0} and cannot compensate. ATR is therefore expressed in 12h-equivalent
units by multiplying by sqrt(12/h), a fixed constant fitted to nothing. The
empirical ratio is printed alongside so the assumption can be checked rather
than trusted.

Hold is already in hours and needs nothing. Fees are NOT normalised: a shorter
horizon really does trade more often and really does pay for it.
"""
import sys, os, json, math; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
import research.features as FE
from research.harness import panel, backtest, OOS_END
from engine.indicators import atr as _atr
from research.runner import zs as _zs
import strategies.s07_smart as s07
import strategies.s69_calsel as S69
import strategies.s84_gate as S84
import strategies.s87_combined as S87
import strategies.s38_orth as s38
import strategies.s36_ivmom as s36
from strategies.s96_rank import at_gate, stats_of, quarters

D = "/home/user/quant/data"
FULL_START = "2021-03-01"
HORIZONS = ("6h", "8h", "12h", "24h")
_K = {"k": 1.0}
_POSN = {}


def hours(tf):
    return pd.Timedelta(tf) / pd.Timedelta("1h")


def B(n):
    """n bars at 12h, expressed in bars at the horizon currently set."""
    return max(3, int(round(n * _K["k"])))


# --- rescale every z-window in the imported pipeline ------------------------
# research.features does `from engine.indicators import *`, so patching the name
# in that module is what reaches add_funding's zscore(cur, 240).
_zscore0 = FE.zscore


def _zscore_scaled(x, n, *a, **kw):
    return _zscore0(x, B(n), *a, **kw)


FE.zscore = _zscore_scaled

# The trend gate's EMA spans are bar counts too. GATES is {100,150,200,300} on
# the 12h grid, i.e. 50/75/100/150 days; the gate menu must mean the same number
# of DAYS at every horizon or the comparison is between different gates.
_trend0 = S84.trend


def _trend_scaled(span):
    return _trend0(B(span))


S84.trend = _trend_scaled


def utc(s):
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def posn4h():
    """Trader positioning on its NATIVE 4h grid, built once and never rescaled.

    This is the one signal whose own construction is independent of the decision
    horizon: it is the same wall-clock series every time, and each horizon only
    samples it at different moments. Building it under the rescaled z-windows
    would make it a different signal at every horizon and confound the test.
    """
    if not _POSN:
        _K["k"] = 1.0
        _, f4 = panel("4h")
        _POSN["s"] = pd.DataFrame({"dt": utc(f4.dt), "posn": s07.composite(f4)})
    return _POSN["s"]


def set_horizon(tf):
    """Point the whole pipeline at a new bar length and drop every stale cache."""
    _K["k"] = 12.0 / hours(tf)
    H._cache.pop(tf, None)
    H._ctxc.clear()
    s38._P.clear(); s36._C.clear(); S69._C.clear(); S84._T.clear()


def grid_h(tf, start=FULL_START):
    """The V7 decision panel at bar length `tf`, anchored at 00:00 UTC.

    Rebuilt here rather than reusing S45.grid because that function hard-codes
    the 12h bar counts it was written for, and because V7 needs five of its
    columns - flow, cmpx, btcdom, fundz, posn - so the unused panels (implied
    vol, term, breadth, ethrel) are pure cost.
    """
    set_horizon(tf)
    p = posn4h()
    set_horizon(tf)                       # posn4h() resets k while it builds
    fut, f = panel(tf)                    # fund_z inside is already rescaled
    f = f.copy(); f["dt"] = utc(f.dt)

    # ATR over the same wall-clock span, expressed in 12h-equivalent price
    # units. `f` carries no OHLC - only derived features - so the ranges come
    # from the aggregated bars, which are row-aligned with it by construction.
    a_raw = _atr(fut.high.to_numpy(float), fut.low.to_numpy(float),
                 fut.close.to_numpy(float), B(14))
    f["atr_raw"] = a_raw
    f["atr14"] = a_raw * math.sqrt(12.0 / hours(tf))

    # coin-margined price gap -> cmpx
    cm = pd.read_parquet(f"{D}/cm_1h.parquet")
    cm["dt"] = utc(cm.dt)
    cm = cm.set_index("dt")["close"].resample(tf).last().rename("cm_px")
    f = f.merge(cm.reset_index(), on="dt", how="left")
    f["cm_px"] = f.cm_px.ffill(limit=B(6))

    # BTC dominance: already z-scored upstream on a fixed hourly grid
    b = pd.read_parquet(f"{D}/breadth.parquet")
    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.to_datetime(b.index, utc=True)
    b.index = b.index.tz_convert("UTC") if b.index.tz else b.index.tz_localize("UTC")
    b = b[["btc_dom_z"]].resample(tf).last()
    b.index.name = "dt"
    f = f.merge(b.reset_index(), on="dt", how="left")

    pr = p.set_index("dt").resample(tf).last().reset_index()   # last CLOSED 4h bar
    f = f.merge(pr, on="dt", how="left")

    # cmpx is z-scored on the WHOLE panel before the window is cut, as the
    # 12h book does. Scoring it after the cut would spend the first 120 bars of
    # the tradeable record warming up a window that the data can already fill,
    # and would spend a different number of them at every horizon.
    f["s_cmpx"] = _zs(pd.Series(np.log(f.cm_px / f.close)).diff(B(6)).to_numpy(), B(120))

    g = f[f.dt >= start].reset_index(drop=True)
    g["s_flow"] = _zs(g.ofi6_res.to_numpy(float), B(480))
    g["s_btcdom"] = g.btc_dom_z
    g["s_fundz"] = -g.fund_z
    g["s_posn"] = g.posn
    return g


# --- the V7 composite, unchanged in form -----------------------------------
THR = {"flow": 1.0, "cmpx": 1.0, "btcdom": 1.0, "fundz": 1.0, "posn": 0.7}
LONG = ["flow", "cmpx", "btcdom", "fundz", "posn"]
CAP = 2.0


def unit(g, name):
    z = g[f"s_{name}"].to_numpy(float); t = THR[name]
    e = np.where(z > t, 1.0, np.where(z < -t, -1.0, 0.0))
    return np.nan_to_num(e * np.clip(np.abs(z) / t, 1.0, CAP))


def composite(g, names=LONG):
    U = np.column_stack([unit(g, n) for n in names])
    return U @ (np.ones(len(names)) / len(names))


def install(g):
    """Make this panel the one S69/S84/S87 see, so the whole V7 stack replays."""
    v = composite(g); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    S84._T.clear()
    H._ctxc.clear()
    return v


def daily(m):
    return S69.daily(m)


# --- stage A: is the edge there at all, before paying for the full stack ----
def screen(tf):
    g = grid_h(tf)
    install(g)
    out = {"tf": tf, "bars": len(g),
           "atr_raw": float(np.nanmedian(g.atr_raw)),
           "atr12": float(np.nanmedian(g.atr14))}
    rows = []
    for p, stp, rr, hold in ((1.0, 3.0, 2.0, 21), (2.0, 3.0, 2.0, 21),
                             (2.0, 2.5, 3.0, 14)):
        m = S69.sim((p, stp, rr, hold), FULL_START, OOS_END, 0.08)
        r = daily(m)
        rows.append((p, stp, rr, hold, m, np.asarray(r, float)))
    out["rows"] = rows
    return out


HDRA = (f"{'horizon':>9}{'bars':>7}{'exp':>5}{'cfg':>12}{'CAGR':>9}{'MaxDD':>8}"
        f"{'PF':>6}{'N':>6}{'Shp':>6}{'Clm':>6}{'at -20%':>10}")


def stage_a():
    print("STAGE A - one configuration, no selection, risk 8%, 2021-03 -> 2026-08")
    print("Three fixed configs so the comparison cannot be a selection artefact.\n")
    print(HDRA)
    ref = None
    keep = {}
    for tf in HORIZONS:
        o = screen(tf)
        if tf == "12h":
            ref = o["atr_raw"]
        keep[tf] = o
        for (p, stp, rr, hold, m, r) in o["rows"]:
            gt = at_gate(r)
            print(f"{tf:>9}{o['bars']:7d}{p:5.1f}{f'{stp}ATR x{rr}R {hold}d':>12}"
                  f"{m['cagr']*100:8.1f}%{m['max_dd']*100:7.1f}%"
                  f"{m['profit_factor']:6.2f}{m['trades']:6d}{m['sharpe']:6.2f}"
                  f"{m['calmar']:6.2f}{gt['cagr']*100:9.1f}%", flush=True)
        print()
    print("ATR normalisation check - sqrt(12/h) against the measured ratio:")
    for tf in HORIZONS:
        pred = math.sqrt(12.0 / hours(tf))
        emp = ref / keep[tf]["atr_raw"] if keep[tf]["atr_raw"] else float("nan")
        print(f"{tf:>9}   median raw ATR {keep[tf]['atr_raw']:9.1f}   "
              f"sqrt rule {pred:5.3f}   measured {emp:5.3f}   "
              f"error {(pred/emp-1)*100:+5.1f}%")
    return keep


# --- stage B: the full V7 stack at one horizon -----------------------------
def rankings_resumable(tf):
    """S87.rankings with a checkpoint after every quarter.

    Same reason as S101: the upstream version writes its JSON only after the
    whole 18-quarter loop, so a container reclaim mid-run loses all of it.
    """
    path = f"/home/user/quant/results/ranks_h{tf}.json"
    done = []
    if os.path.exists(path):
        done = [(s, e, [tuple(c) for c in cs]) for s, e, cs in json.load(open(path))]
    Q = quarters()
    if len(done) >= len(Q):
        return done
    for s, e in Q[len(done):]:
        tr0 = str((pd.Timestamp(s, tz="UTC") - pd.DateOffset(months=12)).date())
        sc = sorted(((S69.calmar_of(S87.sim(cfg, tr0, s, 0.10)), cfg) for cfg in S87.GRID),
                    key=lambda x: -x[0])
        done.append((s, e, [tuple(c) for c in (c for _, c in sc)]))
        json.dump([[a, b, [list(c) for c in cs]] for a, b, cs in done], open(path, "w"))
        print(f"    {s[:7]}  best Calmar {sc[0][0]:6.2f}   "
              f"[{len(done)}/{len(Q)}]", flush=True)
    return done


def full(tf, k=3, risk=0.08):
    g = grid_h(tf)
    install(g)
    R = rankings_resumable(tf)
    r, pnl = S87.blend(R, k, risk)
    return np.asarray(r, float), r, pnl


def stage_b(tfs):
    print(f"\nSTAGE B - the full V7 stack (200 configs, quarterly Calmar, top-3 "
          f"at risk/3) rebuilt at each horizon\n")
    print(f"{'horizon':>9}{'CAGR':>9}{'MaxDD':>8}{'Shp':>7}{'Clm':>7}"
          f"{'trades':>8}{'at -20%':>10}")
    series = {}
    for tf in tfs:
        a, r, pnl = full(tf)
        series[tf] = r
        s, gt = stats_of(a), at_gate(a)
        print(f"{tf:>9}{s['cagr']*100:8.1f}%{s['dd']*100:7.1f}%{s['sharpe']:7.2f}"
              f"{s['calmar']:7.2f}{len(pnl):8d}{gt['cagr']*100:9.1f}%", flush=True)
    return series


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "a"
    if what == "a":
        stage_a()
    else:
        stage_b(sys.argv[2:] or list(HORIZONS))
    print("\ndone: horizon")
