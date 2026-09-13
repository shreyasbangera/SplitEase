"""
S138 - Push the frequency result, and check it is not a cost artefact.

S137 found the standalone crowding book improves monotonically as its decisions
get faster:

    daily        Sharpe 0.77    7.5% at the gate
    12-hourly    Sharpe 1.08   11.1%
    8-hourly     Sharpe 1.12   16.5%

That is the strongest new lead in a while, and it has an obvious mechanism:
positioning data publishes every five minutes and funding settles every eight
hours, so a daily book is reading a fast series on a slow clock and throwing away
most of its information. It also has an obvious failure mode, which is why this
file exists rather than a headline.

THE THREE WAYS A FREQUENCY RESULT LIES
---------------------------------------
  COSTS        Turnover rises roughly with the square root of frequency. At
               8bps a round trip a faster book can look better simply because
               the backtest is not charging it enough. **The cost sweep is the
               experiment**: 8, 16, 30 and 50 basis points per round trip. If
               the ranking survives 50bps it is real; if it inverts by 16bps it
               was never there.

  THE CLOCK    S101 found V7's 12-hour result was partly a phase artefact - the
               same strategy on a different hour of the day behaved differently.
               Every offset of the 8-hour and 4-hour grids is run here, so a
               lucky phase cannot masquerade as an edge.

  UNITS        S119's weekly book read 349.8% because a sub-daily series was
               handed to a scorer that assumes daily bars. Every book here is
               summed onto a calendar-day index before anything scores it.

Then, if it survives: combine the best frequency with the exposure-timing sleeve,
since S136's crowd + exposure pair at 29.1% is the best non-V7 book in the study
and it was built on the *daily* crowding leg.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s118_crowd as C
import strategies.s135_more as S135
import strategies.s136_ensemble as S136
from strategies.s96_rank import stats_of

gate, score, HDR = S135.gate, S136.score, S136.HDR
D = "/home/user/quant/data"


def panel(rule, bars_yr, offset=None):
    """Crowding features on an arbitrary bar, optionally phase-shifted."""
    kw = {"offset": offset} if offset else {}
    d = pd.read_parquet(f"{D}/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True); d = d.set_index("dt")
    px = d["close"].resample(rule, **kw).last().dropna()
    f = pd.read_parquet(f"{D}/funding.parquet")
    f["dt"] = pd.to_datetime(f.dt, utc=True)
    fd = f.set_index("dt")["rate"].resample(rule, **kw).sum().reindex(px.index).fillna(0.0)
    m = pd.read_parquet(f"{D}/metrics_1h.parquet")
    m["dt"] = pd.to_datetime(m.dt, utc=True); m = m.set_index("dt")
    F = pd.DataFrame(index=px.index)
    F["tt_pos"] = m["tt_pos"].resample(rule, **kw).last().reindex(px.index)
    F["tt_acct"] = m["tt_acct"].resample(rule, **kw).last().reindex(px.index)
    F["retail"] = m["retail_acct"].resample(rule, **kw).last().reindex(px.index)
    F["tt_vs_retail"] = np.log(F.tt_pos / F.retail.clip(lower=1e-6))
    F["taker"] = m["taker_ratio"].resample(rule, **kw).last().reindex(px.index)
    F["oi_chg"] = (m["oi"].resample(rule, **kw).last().reindex(px.index)
                   .pct_change(max(1, int(7 * bars_yr / 365.25))))
    F["funding"] = fd
    return px, fd, F


def csig(F, bars_yr, zdays=365):
    cols = [c for c in C.SIGNS if c in F.columns]
    win = max(20, int(zdays * bars_yr / 365.25))
    parts = []
    for c in cols:
        x = F[c].astype(float)
        z = (x - x.rolling(win, min_periods=win // 3).mean()) / \
            (x.rolling(win, min_periods=win // 3).std() + 1e-12)
        parts.append(np.clip(z * C.SIGNS[c], -2, 2))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).shift(1).fillna(0.0)


def book(px, fd, sig, bars_yr, tv=0.30, rt_bps=16.0, max_lev=3.0, band=0.10,
         extra=None):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = np.maximum(r.ewm(halflife=max(4, int(32 * bars_yr / 365.25)), adjust=False)
                    .std().shift(1).bfill().to_numpy() * np.sqrt(bars_yr), 0.05)
    s = np.asarray(sig, float)
    if extra is not None:
        s = (s + np.asarray(extra, float)) / 2.0
    want = np.clip(s * tv / rv, -max_lev, max_lev)
    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * px.pct_change().fillna(0.0).to_numpy() \
        - pos * fd.to_numpy() - turn * (rt_bps / 2.0) / 1e4
    n = pd.Series(net, index=px.index)
    nd = n.groupby(n.index.normalize()).sum()          # DAILY clock before scoring
    p = pd.Series(pos, index=px.index)
    pdd = p.groupby(p.index.normalize()).last()
    return nd, pdd.to_numpy(), float(turn.sum())


GRID = (("1h", 8766.0, "hourly"), ("2h", 4383.0, "2-hourly"),
        ("4h", 2191.5, "4-hourly"), ("8h", 1095.75, "8-hourly"),
        ("12h", 730.5, "12-hourly"), ("1D", 365.25, "daily"))

if __name__ == "__main__":
    print("S138 - is the frequency result real, or is it undercharged?\n")
    print("1. COST SWEEP — every frequency at four round-trip costs")
    print(f"   {'frequency':>12}{'turnover':>11}" +
          "".join(f"{'rt ' + str(int(c)) + 'bps':>11}" for c in (8, 16, 30, 50)))
    cache, best = {}, (-9e9, None)
    for rule, byr, lab in GRID:
        px, fd, F = panel(rule, byr)
        sg = csig(F, byr)
        cache[rule] = (px, fd, sg, byr)
        row, tn = [], None
        for rt in (8.0, 16.0, 30.0, 50.0):
            nd, pdd, turn = book(px, fd, sg, byr, rt_bps=rt)
            tn = turn if tn is None else tn
            g = gate(nd.to_numpy())
            row.append(g)
            if rt == 16.0 and np.isfinite(g) and g > best[0]:
                best = (g, rule)
        print(f"   {lab:>12}{tn:>11.0f}" +
              "".join((f"{v:>10.1f}%" if np.isfinite(v) else f"{'n/a':>11}")
                      for v in row))

    print("\n2. CLOCK PHASE — every offset of the 8h and 4h grids (S101's trap)")
    print(f"   {'grid':>12}{'offset':>9}{'Sharpe':>9}{'at -20%':>10}")
    for rule, byr, n_off, step in (("8h", 1095.75, 8, 1), ("4h", 2191.5, 4, 1)):
        vals = []
        for off in range(0, n_off, step):
            px, fd, F = panel(rule, byr, offset=f"{off}h" if off else None)
            nd, pdd, _ = book(px, fd, csig(F, byr), byr, rt_bps=16.0)
            g = gate(nd.to_numpy())
            vals.append(g)
            print(f"   {rule:>12}{str(off) + 'h':>9}"
                  f"{stats_of(nd.to_numpy())['sharpe']:>9.2f}"
                  + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}"))
        v = np.array([x for x in vals if np.isfinite(x)])
        if len(v):
            print(f"   -> {rule} across phases: median {np.median(v):.1f}%, "
                  f"min {v.min():.1f}%, max {v.max():.1f}%, sd {v.std(ddof=1):.1f}\n")

    print("3. BEST FREQUENCY + EXPOSURE TIMING (S136's pair used the DAILY leg)")
    print(HDR)
    rule = best[1] or "8h"
    px, fd, sg, byr = cache[rule]
    lp = np.log(px)
    span = lambda d: max(2, int(d * byr / 365.25))
    agree3 = (sum(np.sign(lp.ewm(span=span(a), adjust=False).mean()
                          - lp.ewm(span=span(b), adjust=False).mean())
                  for a, b in ((8, 32), (16, 64), (32, 128))) / 3)
    ex = agree3.clip(0, 1).shift(1).fillna(0.0)
    for tag, extra in ((f"crowding {rule} alone", None),
                       (f"crowding {rule} + exposure", ex)):
        for rt in (16.0, 30.0):
            nd, pdd, _ = book(px, fd, sg, byr, rt_bps=rt, extra=extra)
            score(f"{tag}, rt {rt:.0f}bps", nd, pdd)
    print("\ndone: S138")
