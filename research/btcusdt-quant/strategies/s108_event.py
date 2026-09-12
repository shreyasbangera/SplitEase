"""
S108 - V7's five signals on bars that have no clock.

S101 and S102 are the two most serious findings in this study and they say the
same thing from different directions. The book's 168.8% lives in one cell of a
CLOCK grid:

                      04:00 phase   16.4%          6h    43.8%
    12h at 00:00 UTC  168.8%                8h    29.5%
                      08:00 phase   46.3%         24h    10.1%

Every neighbour in phase and every neighbour in length is 2.4x to 10x worse, the
funding grid (8h at 00:00) and the daily candle (24h at 00:00) among them - so
the "00:00 is crypto's anchor" defence is dead, and what remains is a headline
resting on a sampling choice nobody deliberately made.

A dollar bar closes when a fixed amount of value has changed hands. It has no
phase, because there is no origin to shift, and no length, because duration is
whatever the market takes. **The question S101 asked cannot be posed of it.** So
this is not one more cell in the grid; it is a test of whether the edge needs a
grid at all.

Three outcomes, and all three are worth having:

    the edge survives   the fragility is answered rather than argued about, and
                        the deployed book gains a defence it does not currently
                        have
    the edge dies       the clock is load-bearing, which is a far more damning
                        version of S101 than S101 itself - it would mean the
                        signals measure something about times of day rather than
                        about the market
    the edge changes    a stream genuinely different from the clock book, which
                        is the one thing that could clear S102c's admission bar
                        (Sharpe 1.79 at the correlation the horizons showed)

CALIBRATION
-----------
The threshold tracks a lagged 30-day mean of daily dollar volume and targets two
bars a day, which lands at 4,979 bars against the 12h clock's 4,870 and within
10% of its bar count in every year of the sample. So trade counts, cost drag and
z-score windows are all comparable, and any difference is the SAMPLING rather
than the frequency.

Bar-counted lookbacks are kept as bar counts - 480 for flow, 120 for cmpx, 240
for funding, 14 for ATR - deliberately. Converting them to wall clock would undo
the point: in volume time a window of 480 bars is 480 equal parcels of traded
value, which is the quantity these z-scores were always trying to approximate.

The engine had to be corrected first. `align_to_exec` inferred a bar's close as
its open plus the MEDIAN bar spacing, and `_ctx` assigned execution bars to
decision bars by integer division on the same median. Both are exact on an
evenly spaced grid and both drift - in the look-ahead direction, on any bar
longer than the median - on one that is not. They now read the boundaries
directly. V7 on the 12h grid reproduces bit-for-bit after the change (168.8194%
at the gate, 1,697 trades), so nothing earlier in this log moved.
"""
import sys, math; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import engine.data as ED
import research.harness as H
import research.eventbars as EB
from research.harness import panel, OOS_END
from research.runner import zs
import strategies.s07_smart as s07
import strategies.s69_calsel as S69
import strategies.s84_gate as S84
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"
FULL_START = "2021-03-01"
SENTINEL = "EVENTBARS"
_ST = {}


def utc(s):
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def _np_dt(x):
    """Naive UTC datetime64[ns], so searchsorted compares numbers not objects."""
    s = pd.Series(pd.to_datetime(x, utc=True))
    return s.dt.tz_localize(None).to_numpy("datetime64[ns]")


def asof_within(src_dt, src_val, opens):
    """The last source value strictly BEFORE each bar's close.

    Bars are contiguous and labelled by open, so bar t closes when bar t+1
    opens. `resample(tf).last()` on a clock grid takes the last value inside the
    bucket and never the one stamped at the close; this reproduces that exactly
    on boundaries that are not evenly spaced.
    """
    o = _np_dt(opens)
    s = _np_dt(src_dt)
    closes = np.r_[o[1:], o[-1] + (o[-1] - o[-2])]
    i = np.searchsorted(s, closes, side="left") - 1
    out = np.full(len(o), np.nan)
    ok = i >= 0
    out[ok] = np.asarray(src_val, float)[i[ok]]
    return out


def install_bars(per_day=2.0, mode="adaptive", thr=None):
    """Make `panel(SENTINEL)` return a panel built on event bars.

    `agg` is patched rather than `panel` reimplemented: the feature build, the
    basis, the funding merge, the positioning merge and the expanding-window OFI
    residual are all intricate and all already correct, and a second copy of
    them is a second place for look-ahead to live.
    """
    f5 = ED.load("fut_5m")
    s5 = ED.load("spot_15m")
    bars = EB.build_bars(f5.assign(dt=utc(f5.dt)), mode, per_day=per_day, thr=thr)
    spot = EB.to_edges(s5.assign(dt=utc(s5.dt)), bars.dt.to_numpy())
    _ST.update(bars=bars, spot=spot)

    if not _ST.get("patched"):
        orig = ED.agg

        def agg(df, rule):
            if rule != SENTINEL:
                return orig(df, rule)
            return _ST["bars"] if df is f5 else _ST["spot"]

        ED.agg = agg
        H.agg = agg
        _ST["patched"] = True
    H._cache.pop(SENTINEL, None)
    H._ctxc.clear()
    S69._C.clear(); S84._T.clear()
    return bars


THR = {"flow": 1.0, "cmpx": 1.0, "btcdom": 1.0, "fundz": 1.0, "posn": 0.7}
LONG = ["flow", "cmpx", "btcdom", "fundz", "posn"]
CAP = 2.0


def grid_ev(start=FULL_START):
    """V7's decision panel on event bars. Mirrors S102's grid_h column for column."""
    fut, f = panel(SENTINEL)
    f = f.copy(); f["dt"] = utc(f.dt)
    opens = f.dt.to_numpy()          # atr14 already comes from build() on these bars

    cm = pd.read_parquet(f"{D}/cm_1h.parquet")
    cm["dt"] = utc(cm.dt)
    f["cm_px"] = asof_within(cm.dt.to_numpy(), cm.close.to_numpy(float), opens)

    b = pd.read_parquet(f"{D}/breadth.parquet")
    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.to_datetime(b.index, utc=True)
    b.index = b.index.tz_convert("UTC") if b.index.tz else b.index.tz_localize("UTC")
    f["btc_dom_z"] = asof_within(b.index.to_numpy(),
                                 b.btc_dom_z.to_numpy(float), opens)

    _, f4 = panel("4h")
    p4 = pd.DataFrame({"dt": utc(f4.dt), "posn": s07.composite(f4)})
    f["posn"] = asof_within(p4.dt.to_numpy(), p4.posn.to_numpy(float), opens)

    f["s_cmpx"] = zs(pd.Series(np.log(f.cm_px / f.close)).diff(6).to_numpy(), 120)
    g = f[f.dt >= start].reset_index(drop=True)
    g["s_flow"] = zs(g.ofi6_res.to_numpy(float), 480)
    g["s_btcdom"] = g.btc_dom_z
    g["s_fundz"] = -g.fund_z
    g["s_posn"] = g.posn
    return g


def unit(g, name):
    z = g[f"s_{name}"].to_numpy(float); t = THR[name]
    e = np.where(z > t, 1.0, np.where(z < -t, -1.0, 0.0))
    return np.nan_to_num(e * np.clip(np.abs(z) / t, 1.0, CAP))


def composite(g, names=LONG):
    U = np.column_stack([unit(g, n) for n in names])
    return U @ (np.ones(len(names)) / len(names))


def install(g):
    v = composite(g); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    S84._T.clear(); H._ctxc.clear()
    return v


CFGS = ((1.0, 3.0, 2.0, 21), (2.0, 3.0, 2.0, 21), (2.0, 2.5, 3.0, 14))


if __name__ == "__main__":
    print("V7's five signals, same thresholds, same shape, on event bars.\n")
    print(f"{'bars':>22}{'n':>7}{'exp':>5}{'cfg':>13}{'CAGR':>9}{'MaxDD':>8}"
          f"{'PF':>6}{'N':>6}{'Shp':>6}{'Clm':>6}{'at -20%':>10}")
    f5 = ED.load("fut_5m")
    fixed_thr = float(f5.quote_volume.sum() / (len(f5) / 144) / 2.0)
    runs = [("adaptive 2/day", dict(per_day=2.0)),
            ("adaptive 4/day", dict(per_day=4.0)),
            ("fixed threshold", dict(mode="fixed", thr=fixed_thr))]
    for lab, kw in runs:
        bars = install_bars(**kw)
        g = grid_ev()
        install(g)
        for cfg in CFGS:
            m = S69.sim(cfg, FULL_START, OOS_END, 0.08)
            r = np.asarray(S69.daily(m), float)
            gt = at_gate(r)
            print(f"{lab:>22}{len(g):7d}{cfg[0]:5.1f}"
                  f"{f'{cfg[1]}ATR x{cfg[2]}R {cfg[3]}d':>13}"
                  f"{m['cagr']*100:8.1f}%{m['max_dd']*100:7.1f}%"
                  f"{m['profit_factor']:6.2f}{m['trades']:6d}{m['sharpe']:6.2f}"
                  f"{m['calmar']:6.2f}{gt['cagr']*100:9.1f}%", flush=True)
        print(EB.describe(bars, lab) + "\n", flush=True)
    print("12h clock reference, same three configs: 98.8% / 116.9% / 113.0% at the gate")
    print("\ndone: event bars")
