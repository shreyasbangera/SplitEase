"""
S135 - Five more strategies. Two of them attack failures found in S133.

  1  MULTI-HORIZON BREAKOUT ENSEMBLE
     S133 found the Donchian book is the best purely price-based strategy here
     (13.1%, profit factor 2.69) and that it fails the brief on TRADE COUNT - 32
     completed trades, because a breakout book is flat most of the time by
     design. Running several horizons side by side is the direct fix: it
     multiplies the trade count and diversifies the breakout length, which is
     the parameter a single-horizon book is most exposed to.

  2  VOLATILITY-REGIME SWITCHING
     Every book here applies ONE rule in every environment. The oldest finding in
     market microstructure is that trends persist in some regimes and revert in
     others. This estimates the regime from trailing realised volatility - a
     tercile boundary set on past data only - and runs trend in one and reversion
     in the other, letting the data say which way round.

  3  ORDER-FLOW IMBALANCE, STANDALONE
     Aggressive buy share was one of nine inputs in S118's average and one column
     in S120's intraday diagnostic. It has never been a book. Flow is a different
     object from positioning: it is what traders DID, not what they hold.

  4  EXPOSURE TIMING, LONG/FLAT ONLY
     A different question from every other book here. Not "which way" but "how
     much" - always long or flat, never short, timing participation only. BTC
     has a large positive drift and a brutal drawdown profile; capturing the
     first while cutting the second is a real strategy and the brief's drawdown
     constraint is exactly what it targets.

  5  BREAKOUT + CROWDING
     The two best non-price-trend mechanisms in this study, combined. Price
     structure and positioning are different information; breakout says the
     market has left its range, crowding says who is on the wrong side of it.
     Requiring agreement should raise conviction; using one to gate the other
     should raise selectivity. Both are tried.

Each judged against the brief alone: 300% a year at a 20% drawdown, 100+
completed trades, profit factor above 1.10. Costs 5bps fee + 3bps slippage on
turnover, settled funding charged, everything lagged a full bar.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from strategies.s96_rank import at_gate, stats_of
from research.robust import bootstrap_dd

FEE, SLIP = 5.0, 3.0
D = "/home/user/quant/data"


def base():
    d = pd.read_parquet(f"{D}/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True); d = d.set_index("dt")
    px = d["close"].resample("1D").last().dropna()
    hi = d["high"].resample("1D").max().reindex(px.index)
    lo = d["low"].resample("1D").min().reindex(px.index)
    qv = d["quote_volume"].resample("1D").sum().reindex(px.index)
    tbq = d["taker_buy_quote"].resample("1D").sum().reindex(px.index)
    f = pd.read_parquet(f"{D}/funding.parquet")
    f["dt"] = pd.to_datetime(f.dt, utc=True)
    fd = f.set_index("dt")["rate"].resample("1D").sum().reindex(px.index).fillna(0.0)
    return px, hi, lo, qv, tbq, fd


def gate(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


def run(px, fd, sig, tv, vol_hl=32, max_lev=3.0, band=0.10):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = np.maximum(r.ewm(halflife=vol_hl, adjust=False).std()
                    .shift(1).bfill().to_numpy() * np.sqrt(365.25), 0.05)
    s = np.asarray(sig, float)
    want = np.clip(s * tv / rv, -max_lev, max_lev)
    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * px.pct_change().fillna(0.0).to_numpy() \
        - pos * fd.to_numpy() - turn * (FEE + SLIP) / 1e4
    return pd.Series(net, index=px.index), pos


def score(tag, net, pos):
    a = np.asarray(net, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        print(f"   {tag:>32}   no position taken"); return np.nan
    st = stats_of(a); g = gate(a)
    b = bootstrap_dd(a, n=1200, block=90)
    live = np.asarray(pos, float) != 0
    trades = int(((~live[:-1]) & live[1:]).sum() + (1 if live[0] else 0))
    pnl, acc, on = [], 0.0, False
    for i in range(len(a)):
        if live[i]:
            acc += a[i]; on = True
        elif on:
            pnl.append(acc); acc, on = 0.0, False
    if on:
        pnl.append(acc)
    p = np.array(pnl) if pnl else np.array([0.0])
    pf = (p[p > 0].sum() / -p[p < 0].sum()) if (p < 0).any() else np.inf
    h = len(a) // 2
    g1, g2 = gate(a[:h]), gate(a[h:])
    gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
    s1 = "n/a" if not np.isfinite(g1) else f"{g1:.0f}%"
    s2 = "n/a" if not np.isfinite(g2) else f"{g2:.0f}%"
    ok = (np.isfinite(g) and g >= 300) and trades >= 100 and pf > 1.10
    print(f"   {tag:>32}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%"
          f"{b['dd_median']*100:>8.1f}%{trades:>7}{pf:>7.2f}{gs:>9}{s1:>7}{s2:>7}"
          f"{'  PASS' if ok else ''}")
    return g


HDR = (f"   {'variant':>32}{'Shp':>7}{'realDD':>8}{'medDD':>8}{'trd':>7}{'PF':>7}"
       f"{'at -20%':>9}{'1st':>7}{'2nd':>7}")


def donchian(px, hi, lo, N, k, atr):
    up = hi.rolling(N).max().shift(1).to_numpy()
    dn = lo.rolling(N).min().shift(1).to_numpy()
    c, a = px.to_numpy(), atr.to_numpy()
    s = np.zeros(len(px)); cur = 0.0; peak = 0.0
    for i in range(len(px)):
        if not np.isfinite(up[i]) or not np.isfinite(a[i]):
            continue
        if cur == 0.0:
            if c[i] > up[i]: cur, peak = 1.0, c[i]
            elif c[i] < dn[i]: cur, peak = -1.0, c[i]
        elif cur > 0:
            peak = max(peak, c[i])
            if c[i] < peak - k * a[i]: cur = 0.0
        else:
            peak = min(peak, c[i])
            if c[i] > peak + k * a[i]: cur = 0.0
        s[i] = cur
    return pd.Series(s, index=px.index).shift(1).fillna(0.0)


if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = base()
    lp = np.log(px); r = lp.diff()
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    print("S135 - five more strategies, each judged against the brief alone")
    print(f"BTCUSDT perp daily, {px.index.min().date()} -> {px.index.max().date()}\n")

    # ---- 1. multi-horizon breakout ensemble
    print("1. MULTI-HORIZON BREAKOUT ENSEMBLE — fixing S133's trade-count failure")
    print(HDR)
    legs = {N: donchian(px, hi, lo, N, 3.0, atr) for N in (10, 20, 34, 55, 89)}
    for use in ((20, 55), (10, 20, 34, 55, 89)):
        ens = sum(legs[N] for N in use) / len(use)
        for tv in (0.30, 0.50):
            score(f"ensemble {len(use)} horizons, tgt {tv*100:.0f}%",
                  *run(px, fd, ens, tv))

    # ---- 2. volatility-regime switching
    print("\n2. VOLATILITY-REGIME SWITCHING — different rule per regime")
    print(HDR)
    rv60 = r.rolling(60).std()
    q = rv60.rolling(500, min_periods=200).quantile(0.5).shift(1)   # causal split
    high = (rv60.shift(1) > q)
    trend = np.sign(lp.ewm(span=32, adjust=False).mean()
                    - lp.ewm(span=128, adjust=False).mean()).shift(1).fillna(0.0)
    rev = (-((lp.diff(10) - lp.diff(10).rolling(180).mean())
             / (lp.diff(10).rolling(180).std() + 1e-12))).clip(-2, 2).shift(1).fillna(0.0)
    for tag, s in (("trend in HIGH vol, revert in LOW",
                    trend.where(high, 0.0).fillna(0.0) + rev.where(~high, 0.0).fillna(0.0)),
                   ("revert in HIGH vol, trend in LOW",
                    rev.where(high, 0.0).fillna(0.0) + trend.where(~high, 0.0).fillna(0.0)),
                   ("trend ONLY in low vol",
                    trend.where(~high, 0.0).fillna(0.0)),
                   ("trend ONLY in high vol",
                    trend.where(high, 0.0).fillna(0.0))):
        score(tag, *run(px, fd, s, 0.30))

    # ---- 3. order-flow imbalance standalone
    print("\n3. ORDER-FLOW IMBALANCE — what traders did, not what they hold")
    print(HDR)
    imb = (tbq / qv.replace(0, np.nan) - 0.5)
    for w in (20, 60):
        zi = ((imb - imb.rolling(w, min_periods=10).mean())
              / (imb.rolling(w, min_periods=10).std() + 1e-12)).clip(-2, 2).shift(1).fillna(0.0)
        for sgn, lab in ((1.0, "follow"), (-1.0, "fade")):
            score(f"flow {lab}, {w}d window", *run(px, fd, sgn * zi, 0.30))

    # ---- 4. exposure timing, long/flat only
    print("\n4. EXPOSURE TIMING — long or flat, never short. 'how much', not 'which way'")
    print(HDR)
    score("buy & hold (reference)", *run(px, fd, pd.Series(1.0, index=px.index), 0.30))
    dd = (px / px.cummax() - 1.0).shift(1)
    for tag, s in (("above 200d MA only",
                    (px > px.rolling(200).mean()).astype(float).shift(1).fillna(0.0)),
                   ("above 200d MA and not in >15% DD",
                    ((px > px.rolling(200).mean()) & (dd > -0.15)).astype(float).shift(1).fillna(0.0)),
                   ("trend agreement 3 speeds",
                    (sum(np.sign(lp.ewm(span=f, adjust=False).mean()
                                 - lp.ewm(span=s2, adjust=False).mean())
                         for f, s2 in ((8,32),(16,64),(32,128))) / 3).clip(0, 1).shift(1).fillna(0.0))):
        score(tag, *run(px, fd, s, 0.30))

    # ---- 5. breakout + crowding
    print("\n5. BREAKOUT + CROWDING — price structure and positioning together")
    print(HDR)
    try:
        import strategies.s119_dev as S119
        import strategies.s118_crowd as C
        cpx, cfd, F = S119.build("1D")
        cols = [c for c in C.SIGNS if c in F.columns]
        cs = S119.signal(F, cols, 365).reindex(px.index).fillna(0.0)
        bo = legs[55].reindex(px.index).fillna(0.0)
        score("crowding alone (reference)", *run(px, fd, cs, 0.30))
        score("breakout alone (reference)", *run(px, fd, bo, 0.30))
        score("SUM of the two", *run(px, fd, (cs + bo) / 2.0, 0.30))
        agree = np.where(np.sign(cs) == np.sign(bo), (cs + bo) / 2.0, 0.0)
        score("only when they AGREE", *run(px, fd, pd.Series(agree, index=px.index), 0.30))
        score("breakout GATED by crowding sign",
              *run(px, fd, bo * (np.sign(cs) == np.sign(bo)).astype(float), 0.30))
        score("crowding GATED by breakout sign",
              *run(px, fd, cs * (np.sign(cs) == np.sign(bo)).astype(float), 0.30))
    except Exception as e:
        print("   (crowding unavailable:", e, ")")

    print("\ndone: S135")
