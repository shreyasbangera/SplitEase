"""
S119 - Developing the crowding book. Iteration, not a one-pass demo.

S118 was a first draft: nine features I signed by reasoning, averaged flat,
z-scored on a window I picked, sized to a vol target I picked, rebalanced daily
because daily was convenient. Sharpe 1.18, 28.9% at the -20% gate. That is what
one pass buys.

This develops it. Three things in that draft were never chosen, only defaulted,
and each is a real lever:

    SELECTIVITY   the book is in the market every single day. A 0.15-sigma read
                  and a 1.5-sigma read both get a position, differing only in
                  size. Trading only when the signal is worth trading is basic
                  risk management, and it is the single biggest lever on
                  drawdown available to any book.

    HORIZON       daily was convenient, not chosen. Crowding data publishes
                  every 5 minutes and funding settles every 8 hours; there is no
                  reason the decision has to be daily, and the 2-day and 4-day
                  horizons are equally untested.

    VOL WINDOW    a 32-day halflife on the vol estimate, and a 180-day
                  standardisation window, both picked out of the air.

Swept together rather than one at a time, because they interact: a selective
book trades less, so it tolerates a faster horizon; a faster horizon needs a
shorter vol window.

Judged against the brief - 300% net, drawdown under 20%, 100+ completed trades,
profit factor above 1.10 - and against ITS OWN previous iteration. Not against
anything else.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s118_crowd as S
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

FEE_BPS, SLIP_BPS = 5.0, 3.0


def build(freq="1D"):
    """The panel at an arbitrary decision frequency, not just daily.

    Trimmed to where the features actually exist. The first version returned the
    raw panel from 2021-01, where most columns are still NaN, so every variant
    was measured partly over a period the signal could not be computed in - the
    baseline read Sharpe 0.65 against the 1.18 S118 measured on the same book.
    """
    px, fd, F = S.panel()
    cols = list(S.SIGNS)
    cols = [c for c in cols if c in F.columns]
    start = F[cols].dropna(thresh=len(cols) - 1).index.min()
    px, fd, F = px[px.index >= start], fd[fd.index >= start], F[F.index >= start]
    if freq == "1D":
        return px, fd, F
    r = lambda s, how="last": getattr(s.resample(freq), how)()
    px2 = r(px).dropna()
    fd2 = r(fd, "sum").reindex(px2.index).fillna(0.0)
    F2 = pd.DataFrame({c: r(F[c]) for c in F.columns}).reindex(px2.index)
    return px2, fd2, F2


def signal(F, cols, zwin):
    parts = []
    for c in cols:
        x = F[c].astype(float)
        z = (x - x.rolling(zwin, min_periods=max(20, zwin // 3)).mean()) / \
            (x.rolling(zwin, min_periods=max(20, zwin // 3)).std() + 1e-12)
        parts.append(np.clip(z * S.SIGNS[c], -S.CLIP, S.CLIP))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).shift(1).fillna(0.0)


def run(px, fd, sig, target_vol, thr, vol_hl, bars_per_year, max_lev=3.0, band=0.10):
    """Selective, vol-targeted. `thr` is the dead band on the SIGNAL: below it
    the book holds nothing at all rather than a small position."""
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = (r.ewm(halflife=vol_hl, adjust=False).std().shift(1).bfill().to_numpy()
          * np.sqrt(bars_per_year))
    rv = np.maximum(rv, 0.05)
    s = sig.to_numpy().copy()
    s[np.abs(s) < thr] = 0.0                       # <- the selectivity lever
    want = np.clip(s * target_vol / rv, -max_lev, max_lev)

    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur

    cost = (FEE_BPS + SLIP_BPS) / 1e4
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * px.pct_change().fillna(0.0).to_numpy() - pos * fd.to_numpy() - turn * cost
    # a completed trade = a round trip through flat
    live = pos != 0
    trades = int(((~live[:-1]) & live[1:]).sum() + (1 if live[0] else 0))
    return pd.Series(net, index=px.index), trades, pos


def pf_of(net, pos):
    """Profit factor over completed trades, not over days."""
    live = pos != 0
    pnl, acc, on = [], 0.0, False
    for i in range(len(net)):
        if live[i]:
            acc += net.iloc[i]; on = True
        elif on:
            pnl.append(acc); acc, on = 0.0, False
    if on:
        pnl.append(acc)
    p = np.array(pnl)
    if not len(p):
        return np.nan
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    return g / l if l > 0 else np.inf


def to_daily(net):
    """Re-express any decision frequency on a DAILY index before it is scored.

    `stats_of` and `at_gate` compute years as len(returns)/365.25 and annualise
    by sqrt(365.25) - both assume a daily series. Handed 250 weekly bars they
    read 4.8 years as 0.68, which inflates CAGR enormously and Sharpe by
    sqrt(7). The first version of this file did exactly that and produced a
    weekly book reading 349.8% at the gate with a first half of 4696%: not a
    strategy, a units error. Spreading each bar's return across the days it
    actually spans puts every variant on one clock.
    """
    s = pd.Series(np.asarray(net, float), index=pd.to_datetime(net.index))
    idx = pd.date_range(s.index.min(), s.index.max(), freq="1D")
    out = pd.Series(0.0, index=idx)
    pos = np.searchsorted(idx.to_numpy(), s.index.to_numpy())
    out.iloc[np.clip(pos, 0, len(idx) - 1)] = s.to_numpy()
    return out


def show(tag, net, trades, pos, bpy):
    d = to_daily(net)
    a = np.asarray(d, float); a = a[np.isfinite(a)]
    st = stats_of(a); g = at_gate(a); b = bootstrap_dd(a, n=2000, block=90)
    h = len(a) // 2
    g1, g2 = at_gate(a[:h])["cagr"] * 100, at_gate(a[h:])["cagr"] * 100
    pf = pf_of(net, pos)
    expo = (pos != 0).mean() * 100
    print(f"{tag:>30}{st['sharpe']:7.2f}{st['calmar']:7.2f}{st['dd']*100:8.1f}%"
          f"{b['dd_median']*100:8.1f}%{trades:7d}{pf:7.2f}{expo:6.0f}%"
          f"{g['cagr']*100:9.1f}%{g1:8.1f}%{g2:8.1f}%", flush=True)
    return g["cagr"] * 100


HDR = (f"{'variant':>30}{'Shp':>7}{'Clm':>7}{'realDD':>8}{'medDD':>8}"
       f"{'trades':>7}{'PF':>7}{'expo':>6}{'at -20%':>9}{'1st h':>8}{'2nd h':>8}")

if __name__ == "__main__":
    print("developing the crowding book. baseline is S118's first draft.\n")
    cols = list(S.SIGNS)
    print(HDR)

    # --- where it started
    px, fd, F = build("1D")
    cols = [c for c in cols if c in F.columns]
    sig = signal(F, cols, 180)
    n0, t0, p0 = run(px, fd, sig, 0.20, 0.0, 32, 365.25)
    base = show("S118 draft (daily, no thr)", n0, t0, p0, 365.25)

    print("\n--- lever 1: selectivity. hold nothing below a signal threshold")
    for thr in (0.10, 0.20, 0.30, 0.40, 0.50):
        n, t, p = run(px, fd, sig, 0.20, thr, 32, 365.25)
        show(f"daily, threshold {thr:.2f}", n, t, p, 365.25)

    print("\n--- lever 2: decision horizon, at the best threshold so far")
    for freq, bpy, lab in (("1D", 365.25, "daily"), ("2D", 182.6, "2-day"),
                           ("4D", 91.3, "4-day"), ("7D", 52.2, "weekly")):
        p2, f2, F2 = build(freq)
        scale = 365.25 / bpy
        s2 = signal(F2, cols, max(20, int(180 / scale)))
        n, t, p = run(p2, f2, s2, 0.20, 0.30, max(6, int(32 / scale)), bpy)
        show(f"{lab}, threshold 0.30", n, t, p, bpy)

    print("\n--- lever 3: vol window and standardisation window, daily")
    for zw, vh in ((90, 16), (90, 32), (180, 16), (180, 64), (365, 32)):
        s3 = signal(F, cols, zw)
        n, t, p = run(px, fd, s3, 0.20, 0.30, vh, 365.25)
        show(f"zwin {zw}d, vol hl {vh}d", n, t, p, 365.25)

    print("\n--- weekly looked different. sweep the threshold there properly")
    pw, fw, Fw = build("7D")
    for thr in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
        sw = signal(Fw, cols, 26)
        n, t, p = run(pw, fw, sw, 0.20, thr, 5, 52.2)
        show(f"weekly, threshold {thr:.2f}", n, t, p, 52.2)

    print("\n--- the risk dial on the best shape found")
    for tv in (0.20, 0.40, 0.60, 0.90, 1.20):
        n, t, p = run(px, fd, sig, tv, 0.30, 32, 365.25, max_lev=6.0)
        show(f"target vol {tv*100:.0f}%", n, t, p, 365.25)
    print("\ndone: development pass 1")
