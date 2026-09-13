"""
S121 - The scorecard. Every standalone book, one table, the brief's own criteria.

The brief, in the form it was actually written:

    1. net yearly profit after fees and slippage greater than 300%
    2. at least 100 completed trades
    3. profit factor greater than 1.10
    4. maximum drawdown less than 20%
    5. realistic risk management
    6. no look-ahead bias

Criteria 1 and 4 are one test, not two, because both are functions of position
size and size is a free dial. Halve the bet and both the return and the drawdown
roughly halve. The joint question is the only one with a fixed answer:

    **is there ANY size at which this book returns 300% a year AND stays inside
    a 20% drawdown?**

which is exactly what `at_gate` computes - scale the returns until the drawdown
sits on -20%, and read the CAGR there. A book reading 27% at the gate does not
fail by being sized wrong; it fails at every size.

THE HONEST GATE
---------------
S112 established that scaling to the *realised* drawdown of one path overstates
what a book can carry. V7's worst drawdown is a three-day mark-to-market
excursion on a position that went on to make money - reorder the same trades and
it lands somewhere else. At 8% risk its realised drawdown is -12.3% while the
bootstrap median is -18.1%.

So every book here is measured twice: at the realised-path gate, which is what
the rest of this log quotes and is comparable with it, and at the **honest
gate** - the size at which the *median* bootstrapped path draws down 20%. The
second number is the one a person sizing real money should read. Applying it to
one book and not the others would be the kind of asymmetry this log exists to
catch, so it is applied to all of them, the deployed book included.

WHAT IS RANKED, AND ON WHAT
---------------------------
Ranked on the brief, not against each other and not against any one book. Each
row is a different mechanism, built standalone and judged standalone:

    buy & hold      the free alternative. Any book that cannot beat holding the
                    asset has no reason to exist, and it is charged funding
                    because a perp position pays it.
    trend  S117     price only, vol-targeted, no stops. 6.7 years.
    crowding S119   nine published positioning series, equal-weighted.
    carry  S114     long quarterly against short perp, futures only.
    cascade S120c   fading 4-sigma hourly moves, at 10bps slippage.
    V7              the deployed book. One row, the same six tests as the rest.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from strategies.s96_rank import stats_of

BLOCK = 90
NBOOT = 1500


def paths(r, n=NBOOT, block=BLOCK, seed=0):
    """Block-resample the return series ONCE, returning an (n, T) index matrix.

    The same resampled paths are then re-used at every trial size, so bisecting
    for the size at which the median path draws down 20% costs one bootstrap
    rather than one per bisection step.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    T = len(r)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, max(T - block, 1), (n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    return r, np.clip(idx, 0, T - 1)


def median_dd(r, idx, scale):
    e = np.cumprod(1.0 + r[idx] * scale, axis=1)
    peak = np.maximum.accumulate(e, axis=1)
    return float(np.median((e / peak - 1.0).min(axis=1)))


def real_dd(r, scale):
    e = np.cumprod(1.0 + np.asarray(r, float) * scale)
    return float((e / np.maximum.accumulate(e) - 1.0).min())


def gate(r, how, target=-0.20, lo=0.01, hi=8.0, iters=44):
    """Size at which drawdown sits on the target, and the CAGR there."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if not len(r) or np.allclose(r, 0):
        return 0.0, 0.0
    idx = None
    if how == "median":
        r, idx = paths(r)
    f = (lambda s: median_dd(r, idx, s)) if how == "median" else (lambda s: real_dd(r, s))
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < target:            # too deep -> smaller size
            hi = mid
        else:
            lo = mid
    s = (lo + hi) / 2
    return s, stats_of(r * s)["cagr"]


def trades_pf(pnl):
    """Completed-trade count and profit factor from a list of per-trade P&L."""
    p = np.asarray([x for x in pnl if np.isfinite(x)], float)
    if not len(p):
        return 0, np.nan
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    return len(p), (g / l if l > 0 else np.inf)


def from_position(net, pos):
    """Per-trade P&L for a continuously-sized book: a trade is a round trip
    through flat, or a reversal of sign. Accumulates the net return while the
    sign is held, and closes the trade when it changes."""
    net = np.asarray(net, float); pos = np.asarray(pos, float)
    out, acc, cur = [], 0.0, 0.0
    for i in range(len(net)):
        s = np.sign(pos[i])
        if s != cur:
            if cur != 0.0:
                out.append(acc)
            acc, cur = 0.0, s
        if cur != 0.0:
            acc += net[i]
    if cur != 0.0:
        out.append(acc)
    return out


ROWS = []


def add(name, mech, r, pnl, years, note):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    st = stats_of(r)
    _, g_real = gate(r, "real")
    _, g_med = gate(r, "median")
    n, pf = trades_pf(pnl)
    ROWS.append(dict(name=name, mech=mech, sharpe=st["sharpe"], years=years,
                     n=n, pf=pf, real=g_real * 100, honest=g_med * 100, note=note))
    print(f"  {name:<22}{st['sharpe']:>7.2f}{n:>8}{pf:>7.2f}"
          f"{g_real*100:>10.1f}%{g_med*100:>10.1f}%", flush=True)


if __name__ == "__main__":
    print("S121 - every standalone book against the brief\n")
    print(f"  {'book':<22}{'Sharpe':>7}{'trades':>8}{'PF':>7}"
          f"{'gate':>11}{'honest':>11}")

    # ---- buy & hold, the free alternative -------------------------------
    import strategies.s117_mf as S117
    px, fd = S117.data()
    bh = px.pct_change().fillna(0.0).to_numpy() - fd.to_numpy()
    yrs = len(px) / 365.25
    add("buy & hold", "none", bh, [float(np.prod(1 + bh) - 1)], yrs,
        "one position, never closed")

    # ---- S117 trend -----------------------------------------------------
    net, turns, pos = S117.run(px, fd, "longshort", 0.40)
    add("trend (S117)", "price", net.to_numpy(),
        from_position(net.to_numpy(), pos), yrs, "vol-targeted, no stops")

    # ---- S119 crowding --------------------------------------------------
    import strategies.s119_dev as S119
    cpx, cfd, F = S119.build("1D")
    cols = [c for c in S119.S.SIGNS if c in F.columns]
    sig = S119.signal(F, cols, 365)
    cnet, ctr, cpos = S119.run(cpx, cfd, sig, 0.20, 0.30, 32, 365.25)
    add("crowding (S119)", "positioning", cnet.to_numpy(),
        from_position(cnet.to_numpy(), cpos), len(cpx) / 365.25,
        "equal-weight, selective")

    # ---- S120c cascade, at realistic slippage ---------------------------
    import strategies.s120c_cascade as S120c
    o = S120c.bars(60)
    xnet, XT = S120c.book(o, 4.0, 4, 10.0)
    xd = S120c.to_daily(xnet)
    add("cascade (S120c)", "liquidity", xd.to_numpy(),
        list(XT.pnl) if len(XT) else [], len(xd) / 365.25,
        "10bps slippage into a cascade")

    # ---- V7, the deployed book, same six tests --------------------------
    import strategies.s110_meta as M
    import strategies.s87_combined as S87
    M.use_clock()
    R = S87.rankings()
    vr, vpnl = S87.blend(R, 3, 0.144)
    vr = np.asarray(vr, float)
    add("V7 (deployed)", "positioning", vr,
        [float(t) for t in (vpnl.pnl if hasattr(vpnl, "pnl") else vpnl)],
        len(vr) / 365.25, "quarterly-selected blend")

    # ---- the verdict ----------------------------------------------------
    D = pd.DataFrame(ROWS).sort_values("honest", ascending=False)
    print("\n" + "=" * 74)
    print("RANKED, and tested against the brief. 'pass' needs ALL of:")
    print("300% at a 20% drawdown, 100+ completed trades, profit factor > 1.10\n")
    print(f"  {'#':<3}{'book':<22}{'300%?':>8}{'100+?':>8}{'PF>1.1?':>9}"
          f"{'verdict':>10}{'shortfall':>12}")
    for i, (_, x) in enumerate(D.iterrows(), 1):
        c1 = x.honest >= 300.0
        c2 = x.n >= 100
        c3 = np.isfinite(x.pf) and x.pf > 1.10
        ok = c1 and c2 and c3
        short = f"{300.0 / x.honest:.0f}x" if x.honest > 0 else "n/a"
        print(f"  {i:<3}{x['name']:<22}{'YES' if c1 else 'no':>8}"
              f"{'YES' if c2 else 'no':>8}{'YES' if c3 else 'no':>9}"
              f"{'PASS' if ok else 'FAIL':>10}{short:>12}")
    D.to_csv("/home/user/quant/out/s121_scorecard.csv", index=False)
    print("\ndone: scorecard")
