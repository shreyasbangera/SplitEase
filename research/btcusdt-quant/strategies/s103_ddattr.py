"""
S103 - Attributing what is LEFT of the drawdown, the way S84 attributed what was there.

Exactly one thing in this study has ever improved Calmar without costing more
than it gave: the trend gate. It was not found by searching a grid. It was found
by opening the single worst drawdown episode and reading what was inside it -
50 trades, 29 of them shorts into a melt-up, 89% of the loss - and then building
a rule against that cause and selecting it causally, quarter by quarter.

That was the PRE-gate book. V7 now carries the gate, and the brief needs Calmar
8.95 -> 15 to put 300% inside a -20% limit. If the remaining drawdowns share a
second cause that is knowable in advance, the same method applies a second time.
If they do not - if what is left is scattered and unattributable - then the
denominator is irreducible on this book and the gate was the only one there was.

This file only MEASURES. Nothing is adopted here. Anything that looks like a
cause has to go into the quarterly grid and be chosen causally before it counts,
which is what S85 did for the gate and what S84 was careful not to do.

The candidate conditions are all recorded at the START of each episode, from
data available then, so that "this episode was predictable" means predictable
rather than merely describable.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s45_single as S45
import strategies.s46_net as S46
from strategies.s84_gate import sim_gate

RISK, K = 0.08, 3
TOPN = 6
REASON = {1: "stop", 2: "target", 3: "signal", 4: "eod"}


def replay():
    """V7 exactly as deployed, but keeping the trade frames blend() throws away.

    Also records each trade's LEVERAGE against its own sleeve's equity at entry.
    The three sleeves are simulated as three accounts and only their daily
    returns are summed, so no single number in the blend ever shows what the one
    real account is actually holding - which turns out to be the whole
    explanation of the book's worst drawdown.
    """
    R = S87.rankings()
    segs, trades = [], []
    for s, e, cfgs in R:
        rs = []
        for rank, cfg in enumerate(cfgs[:K]):
            m = S87.sim(cfg, s, e, RISK / K)
            rs.append(S69.daily(m))
            td = m["trades_df"]
            if td is not None and len(td):
                td = td.copy()
                td["quarter"] = s; td["rank"] = rank; td["cfg"] = str(cfg)
                td["gate"] = cfg[4]
                eq = pd.Series(m["equity"], index=pd.to_datetime(m["dt"]))
                ent = pd.to_datetime(td.entry_dt, utc=True)
                td["eq_at_entry"] = [float(eq.asof(t)) for t in ent]
                td["lev"] = (td.qty.abs() * td.entry) / td.eq_at_entry
                trades.append(td)
        segs.append(pd.DataFrame({j: r for j, r in enumerate(rs)}).fillna(0.0).sum(axis=1))
    daily = pd.concat(segs)
    T = pd.concat(trades, ignore_index=True)
    T["entry_dt"] = pd.to_datetime(T.entry_dt, utc=True)
    T["exit_dt"] = pd.to_datetime(T.exit_dt, utc=True)
    return daily, T


def episodes(daily, topn=TOPN):
    """The deepest peak-to-trough episodes, non-overlapping, deepest first."""
    eq = np.cumprod(1 + daily.to_numpy())
    idx = daily.index
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    out, mask = [], np.zeros(len(eq), bool)
    for _ in range(topn):
        d = np.where(mask, 0.0, dd)
        t = int(np.argmin(d))
        if d[t] >= -1e-9:
            break
        p = t
        while p > 0 and eq[p] < peak[t]:
            p -= 1
        r = t
        while r < len(eq) - 1 and eq[r] < eq[p]:
            r += 1
        out.append(dict(depth=float(dd[t]), peak=idx[p], trough=idx[t], rec=idx[r],
                        days=int((idx[t] - idx[p]).days),
                        healdays=int((idx[r] - idx[t]).days)))
        mask[p:r + 1] = True
    return out


def context():
    """Causal market state on the 12h decision grid: trend, vol, funding, conviction."""
    g = S45.grid(S45.FULL_START).copy()
    v = S45.composite(g, S46.LONG)
    c = pd.Series(g.close.to_numpy(float))
    g["ema200"] = c.ewm(span=200, adjust=False).mean().to_numpy()
    g["above"] = (c.to_numpy() > g.ema200.to_numpy())
    g["vol"] = (g.atr14.to_numpy(float) / g.close.to_numpy(float))
    g["volrank"] = pd.Series(g.vol).rolling(360).rank(pct=True).to_numpy()
    g["conv"] = np.abs(v)
    g["net"] = v
    g["fwd30"] = (c.shift(-60) / c - 1.0).to_numpy()      # 60 bars = 30 days AHEAD
    return g.set_index("dt")


def btc_daily():
    """BTC marked on the SAME clock the equity curve is marked on.

    The 12h decision grid closes at 00:00 and 12:00; daily equity marks are the
    last execution bar of the calendar day. Reading the move off the decision
    grid at a daily timestamp compares 00:00 closes against 23:45 marks, which
    made the book's worst drawdown look like a 12.3% loss on a 0.7% market move.
    It was 4.1%, at 3x leverage.
    """
    from engine.data import load
    d = load("fut_15m").copy()
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    return d.set_index("dt")["close"].resample("1D").last().dropna()


def agg_leverage(T, index):
    """Aggregate leverage of the ONE real account, day by day.

    The blend is simulated as K independent accounts whose daily returns are
    summed, so nothing in it ever reports what the single netted account holds.
    All K sleeves read the same composite and the conviction exponent cannot
    change a sign, so they are always on the same side or flat: the account's
    leverage is the SUM of the sleeves', and no number in the study had measured
    it.
    """
    lev = pd.Series(0.0, index=index)
    for _, t in T.iterrows():
        m = (index >= t.entry_dt.normalize()) & (index <= t.exit_dt.normalize())
        lev[m] += t.lev
    return lev


def window(T, a, b):
    """Trades whose HOLDING PERIOD overlaps the episode, not just those that exit in it.

    Selecting on exit_dt alone attributes nothing to a short, sharp episode: the
    book's deepest drawdown is three days long and no trade closed inside it, so
    the loss is entirely mark-to-market on positions opened before it and closed
    after. Overlap counts those, at the cost of double-counting a trade that
    straddles two episodes - which is the right way round for asking what the
    book was HOLDING while it lost.
    """
    return T[(T.entry_dt <= b) & (T.exit_dt >= a)]


def state_at(G, ts):
    """Last decision bar at or before `ts`. G.asof would do this, but pandas 3
    implements it by writing NaN into the frame and that fails on a bool column."""
    sub = G.loc[:ts]
    return sub.iloc[-1] if len(sub) else G.iloc[0]


if __name__ == "__main__":
    daily, T = replay()
    eq = np.cumprod(1 + daily.to_numpy())
    print(f"V7 replayed: {len(T)} trades, {len(daily)} days, "
          f"terminal {eq[-1]:.2f}x, max DD {(eq/np.maximum.accumulate(eq)-1).min()*100:.1f}%\n")

    G = context()
    P = btc_daily()
    LEV = agg_leverage(T, daily.index)
    E = episodes(daily)
    tot = T.pnl.sum()

    print(f"aggregate leverage of the ONE netted account (sum over the {K} sleeves)")
    print(f"  median {LEV[LEV > 0].median():.2f}x   90th pct "
          f"{LEV[LEV > 0].quantile(0.9):.2f}x   99th {LEV[LEV > 0].quantile(0.99):.2f}x"
          f"   max {LEV.max():.2f}x   flat on {(LEV == 0).mean()*100:.0f}% of days")
    print(f"  per-sleeve max {T.lev.max():.2f}x against a {10.0:.0f}x cap that is "
          f"applied PER SLEEVE, so the account's effective cap is {10.0*K:.0f}x\n")

    print("the deepest episodes that remain AFTER the trend gate")
    print("(trades = positions HELD during the window, so a 3-day episode that "
          "closes nothing is still attributable)\n")
    print(f"{'#':>2}{'peak':>13}{'trough':>13}{'depth':>8}{'days':>6}{'heal':>6}"
          f"{'trades':>8}{'P&L':>10}{'longs':>9}{'shorts':>9}{'BTC':>8}"
          f"{'lev':>7}{'BTCxlev':>9}{'worst day':>11}{'down days':>11}")
    rows = []
    for i, e in enumerate(E, 1):
        w = window(T, e["peak"], e["trough"])
        L = w[w.side > 0]; S = w[w.side < 0]
        btc = (float(P.asof(e["trough"])) / float(P.asof(e["peak"])) - 1) * 100
        seg = daily[(daily.index > e["peak"]) & (daily.index <= e["trough"])]
        lv = LEV[(LEV.index >= e["peak"]) & (LEV.index <= e["trough"])]
        lev = float(lv.max()) if len(lv) else float("nan")
        rows.append((e, w, L, S, btc))
        print(f"{i:2d}{str(e['peak'].date()):>13}{str(e['trough'].date()):>13}"
              f"{e['depth']*100:7.1f}%{e['days']:6d}{e['healdays']:6d}{len(w):8d}"
              f"{w.pnl.sum():10.0f}{L.pnl.sum():9.0f}{S.pnl.sum():9.0f}{btc:7.1f}%"
              f"{lev:6.2f}x{btc*lev:8.1f}%{seg.min()*100:10.1f}%"
              f"{int((seg < 0).sum()):6d}/{len(seg):<4d}")
    print(f"\n   total P&L over the whole record {tot:,.0f}; the {len(E)} episodes "
          f"above cost {sum(r[1].pnl.sum() for r in rows):,.0f}")

    print("\nwhat was KNOWABLE at the start of each episode (state at the peak bar)")
    print(f"{'#':>2}{'above EMA200':>14}{'vol pct':>9}{'fund_z':>8}{'net':>7}"
          f"{'|conv|':>8}{'exit mix (stop/tp/sig/eod)':>28}")
    for i, (e, w, L, S) in enumerate([(r[0], r[1], r[2], r[3]) for r in rows], 1):
        row = state_at(G, e["peak"])
        rc = w.reason.value_counts()
        mix = "/".join(str(int(rc.get(k, 0))) for k in (1, 2, 3, 4))
        print(f"{i:2d}{str(bool(row['above'])):>14}{row['volrank']:9.2f}"
              f"{row['fund_z']:8.2f}{row['net']:7.2f}{row['conv']:8.2f}{mix:>28}")

    print("\nDECOMPOSITION - how much of each drawdown was actually LOST, in dollars")
    print("(equity change over the window, against the P&L of trades that CLOSED inside it;")
    print(" the gap is mark-to-market on positions still open at the trough)")
    eqs = pd.Series(eq, index=daily.index)
    print(f"{'#':>2}{'equity at peak':>16}{'at trough':>12}{'d equity':>11}"
          f"{'realised':>11}{'MTM gap':>10}{'closed':>8}")
    for i, (e, w, *_ ) in enumerate(rows, 1):
        e0 = float(eqs.asof(e["peak"])); e1 = float(eqs.asof(e["trough"]))
        de = (e1 - e0) * 10_000.0                  # the engine starts at 10,000
        cl = T[(T.exit_dt > e["peak"]) & (T.exit_dt <= e["trough"])]
        print(f"{i:2d}{e0*10_000:16,.0f}{e1*10_000:12,.0f}{de:11,.0f}"
              f"{cl.pnl.sum():11,.0f}{de - cl.pnl.sum():10,.0f}{len(cl):8d}")

    print("\nthe trades held through the single deepest episode")
    w1 = window(T, E[0]["peak"], E[0]["trough"])
    for _, t in w1.sort_values("entry_dt").iterrows():
        print(f"   {str(t.entry_dt)[:16]} -> {str(t.exit_dt)[:16]}  "
              f"{'LONG ' if t.side > 0 else 'SHORT'}  qty {t.qty:7.3f}  "
              f"P&L {t.pnl:9.0f}  "
              f"{REASON.get(int(t.reason), '?')}")
    seg = daily[(daily.index > E[0]['peak']) & (daily.index <= E[0]['trough'])]
    print("   daily path " + " ".join(f"{v*100:+.1f}%" for v in seg))

    print("\nis the loss concentrated by SIDE, as it was before the gate?")
    for i, (e, w, L, S, _) in enumerate(rows, 1):
        n = w.pnl.sum()
        share = (S.pnl.sum() / n * 100) if n < 0 else float("nan")
        print(f"{i:2d}  longs {len(L):4d} trades {L.pnl.sum():9.0f}   "
              f"shorts {len(S):4d} trades {S.pnl.sum():9.0f}   "
              f"shorts are {share:5.0f}% of the episode loss")

    print("\nand by SELECTED CONFIG - is one quarter's pick doing the damage?")
    for i, (e, w, *_ ) in enumerate(rows, 1):
        q = w.groupby("quarter").pnl.sum().sort_values()
        gates = w.groupby("gate").pnl.sum().sort_values()
        print(f"{i:2d}  worst quarter {q.index[0] if len(q) else '-'} "
              f"{q.iloc[0] if len(q) else 0:8.0f}   "
              f"gate P&L " + " ".join(f"{int(k) or 'none'}:{v:.0f}" for k, v in gates.items()))

    print("\nBASE RATE - the same states measured over the whole record, for comparison")
    d12 = G.reindex(G.index)
    up = d12["above"].to_numpy(bool)
    print(f"  bars above EMA200      {up.mean()*100:5.1f}%")
    print(f"  median vol percentile  {np.nanmedian(d12['volrank']):5.2f}")
    print(f"  median fund_z          {np.nanmedian(d12['fund_z']):5.2f}")
    allw = T
    print(f"  whole record: longs {int((allw.side>0).sum()):5d} "
          f"P&L {allw[allw.side>0].pnl.sum():10.0f}   "
          f"shorts {int((allw.side<0).sum()):5d} P&L {allw[allw.side<0].pnl.sum():10.0f}")
    print("\ndone: drawdown attribution")
