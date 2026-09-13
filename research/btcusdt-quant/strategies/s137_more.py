"""
S137 - Four more. Leverage flush, basis shape, faster crowding, risk parity.

  1  LEVERAGE BUILD vs FLUSH  (the classic crypto setup never built here)
     Price and open interest together are a state variable neither is alone:

         price UP   + OI UP    new longs entering - leverage building, fragile
         price UP   + OI DOWN  shorts covering - a squeeze, not real demand
         price DOWN + OI UP    new shorts entering
         price DOWN + OI DOWN  longs being liquidated - a FLUSH

     The flush quadrant is the interesting one: forced selling exhausts itself
     and historically marks lows. S118 used a 7-day OI change as one of nine
     averaged features, which cannot express a quadrant at all - the interaction
     is the whole idea and an additive average destroys it.

  2  BASIS TERM STRUCTURE AS A DIRECTIONAL SIGNAL
     S114 tested the quarterly basis as a CARRY - money earned holding the
     spread - and found 0.2% a year. This is a different question: the SHAPE of
     the curve as a sentiment reading. Steep contango means the market is paying
     up for forward exposure; backwardation means it will not. That is
     information about positioning appetite regardless of whether the spread
     itself is harvestable.

  3  FASTER CROWDING
     The crowding book decides daily on a 365-day standardisation. Positioning
     data publishes every five minutes and funding settles every eight hours.
     S102 swept horizons for a different architecture; the standalone crowding
     book has only ever been run daily.

  4  RISK PARITY ACROSS SLEEVES
     S136's ensemble weights each sleeve's SIGNAL equally, which means the
     sleeve with the most volatile signal dominates the risk. Scaling each to an
     equal volatility contribution is the standard fix and is not a fitted
     weighting - it uses no information about returns, only about variance, so
     S106b's finding against informed weighting does not apply.

Judged against the brief alone.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s135_more as S135
import strategies.s136_ensemble as S136
from strategies.s96_rank import stats_of

gate, run, score, HDR = S135.gate, S135.run, S136.score, S136.HDR
D = "/home/user/quant/data"


def oi_daily(index):
    m = pd.read_parquet(f"{D}/metrics_1h.parquet")
    m["dt"] = pd.to_datetime(m.dt, utc=True)
    return m.set_index("dt")["oi"].resample("1D").last().reindex(index)


def basis_daily(index):
    q = pd.read_parquet(f"{D}/quarterly_front.parquet")
    q["dt"] = pd.to_datetime(q.dt, utc=True)
    q = q[(q.dte > 2) & (q.basis.abs() < 0.25)]
    return q.set_index("dt")["ann"].resample("1D").last().reindex(index)


if __name__ == "__main__":
    px, hi, lo, qv, tbq, fd = S135.base()
    lp = np.log(px)
    print("S137 - four more strategies\n")

    # ---------------------------------------------------- 1. leverage quadrant
    oi = oi_daily(px.index)
    dp = lp.diff(5)
    doi = np.log(oi.clip(lower=1e-9)).diff(5)
    print("1. LEVERAGE BUILD vs FLUSH — price x open interest as a quadrant")
    q = pd.Series("", index=px.index)
    q[(dp > 0) & (doi > 0)] = "build"
    q[(dp > 0) & (doi < 0)] = "squeeze"
    q[(dp < 0) & (doi > 0)] = "newshort"
    q[(dp < 0) & (doi < 0)] = "flush"
    fwd = lp.diff(10).shift(-10)
    print(f"   {'quadrant':>10}{'days':>8}{'mean fwd 10d':>15}{'1st half':>11}{'2nd half':>11}")
    h = len(px) // 2
    for name in ("build", "squeeze", "newshort", "flush"):
        m = (q == name).shift(1).fillna(False)
        a = fwd[m]
        m1 = m.copy(); m1.iloc[h:] = False
        m2 = m.copy(); m2.iloc[:h] = False
        print(f"   {name:>10}{m.sum():>8}{a.mean()*100:>14.2f}%"
              f"{fwd[m1].mean()*100:>10.2f}%{fwd[m2].mean()*100:>10.2f}%")
    print(HDR)
    for tag, s in (("long FLUSH only",
                    (q == "flush").astype(float).shift(1).fillna(0.0)),
                   ("long flush, short build",
                    ((q == "flush").astype(float)
                     - (q == "build").astype(float)).shift(1).fillna(0.0)),
                   ("long flush + squeeze",
                    ((q == "flush") | (q == "squeeze")).astype(float).shift(1).fillna(0.0)),
                   ("short BUILD only",
                    (-(q == "build").astype(float)).shift(1).fillna(0.0))):
        score(tag, *run(px, fd, s, 0.30))

    # ------------------------------------------------------- 2. basis shape
    print("\n2. BASIS TERM STRUCTURE — curve shape as sentiment, not as carry")
    bs = basis_daily(px.index)
    print(f"   annualised basis: mean {bs.mean()*100:+.1f}%, "
          f"negative on {100*(bs<0).mean():.0f}% of days, "
          f"coverage {bs.notna().mean()*100:.0f}%")
    print(HDR)
    for w in (60, 180):
        z = ((bs - bs.rolling(w, min_periods=20).mean())
             / (bs.rolling(w, min_periods=20).std() + 1e-12)).clip(-2, 2).shift(1).fillna(0.0)
        for sgn, lab in ((-1.0, "fade rich basis"), (1.0, "follow basis")):
            score(f"{lab}, {w}d window", *run(px, fd, sgn * z, 0.30))

    # ----------------------------------------------------- 3. faster crowding
    print("\n3. FASTER CROWDING — the standalone book has only ever run daily")
    import strategies.s118_crowd as C
    m = pd.read_parquet(f"{D}/metrics_1h.parquet")
    m["dt"] = pd.to_datetime(m.dt, utc=True); m = m.set_index("dt")
    d1h = pd.read_parquet(f"{D}/fut_1h.parquet")
    d1h["dt"] = pd.to_datetime(d1h.dt, utc=True); d1h = d1h.set_index("dt")
    f = pd.read_parquet(f"{D}/funding.parquet")
    f["dt"] = pd.to_datetime(f.dt, utc=True); fr = f.set_index("dt")["rate"]
    print(HDR)
    for rule, bars_yr, lab in (("8h", 1095.75, "8-hourly"), ("12h", 730.5, "12-hourly"),
                               ("1D", 365.25, "daily")):
        p2 = d1h["close"].resample(rule).last().dropna()
        f2 = fr.resample(rule).sum().reindex(p2.index).fillna(0.0)
        F2 = pd.DataFrame(index=p2.index)
        F2["tt_pos"] = m["tt_pos"].resample(rule).last().reindex(p2.index)
        F2["tt_acct"] = m["tt_acct"].resample(rule).last().reindex(p2.index)
        F2["retail"] = m["retail_acct"].resample(rule).last().reindex(p2.index)
        F2["tt_vs_retail"] = np.log(F2.tt_pos / F2.retail.clip(lower=1e-6))
        F2["taker"] = m["taker_ratio"].resample(rule).last().reindex(p2.index)
        F2["oi_chg"] = m["oi"].resample(rule).last().reindex(p2.index).pct_change(int(7*bars_yr/365.25))
        F2["funding"] = f2
        cols = [c for c in C.SIGNS if c in F2.columns]
        win = int(365 * bars_yr / 365.25)
        parts = []
        for c in cols:
            x = F2[c].astype(float)
            zz = (x - x.rolling(win, min_periods=win//3).mean()) / \
                 (x.rolling(win, min_periods=win//3).std() + 1e-12)
            parts.append(np.clip(zz * C.SIGNS[c], -2, 2))
        sg = pd.concat(parts, axis=1).mean(axis=1, skipna=True).shift(1).fillna(0.0)
        r2 = np.log(p2/p2.shift(1)).fillna(0.0)
        rv = np.maximum(r2.ewm(halflife=int(32*bars_yr/365.25), adjust=False).std()
                        .shift(1).bfill().to_numpy()*np.sqrt(bars_yr), 0.05)
        want = np.clip(sg.to_numpy()*0.30/rv, -3.0, 3.0)
        pos, cur = np.zeros(len(p2)), 0.0
        for i in range(len(p2)):
            if abs(want[i]-cur) > 0.10: cur = want[i]
            pos[i] = cur
        turn = np.abs(np.diff(np.r_[0.0, pos]))
        net = pos*p2.pct_change().fillna(0.0).to_numpy() - pos*f2.to_numpy() \
              - turn*(S135.FEE+S135.SLIP)/1e4
        nd = pd.Series(net, index=p2.index)
        nd = nd.groupby(nd.index.normalize()).sum()       # onto a DAILY clock
        pd_ = pd.Series(pos, index=p2.index)
        pdd = pd_.groupby(pd_.index.normalize()).last()
        score(f"crowding, {lab} decisions", nd, pdd.to_numpy())

    # ------------------------------------------------------- 4. risk parity
    print("\n4. RISK PARITY ACROSS SLEEVES — equal risk, not equal signal")
    import strategies.s119_dev as S119
    cpx, cfd, F = S119.build("1D")
    cols = [c for c in C.SIGNS if c in F.columns]
    tr = pd.concat([hi-lo, (hi-px.shift(1)).abs(), (lo-px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    agree3 = (sum(np.sign(lp.ewm(span=a, adjust=False).mean()
                          - lp.ewm(span=b, adjust=False).mean())
                  for a, b in ((8,32),(16,64),(32,128)))/3)
    SL = {"crowd": S119.signal(F, cols, 365).reindex(px.index).fillna(0.0),
          "expos": agree3.clip(0,1).shift(1).fillna(0.0),
          "break": S135.donchian(px, hi, lo, 55, 3.0, atr)}
    print(HDR)
    # per-sleeve realised vol on a TRAILING window, so the weight is causal
    nets = {k: run(px, fd, v, 0.30)[0] for k, v in SL.items()}
    for combo in (("crowd","expos"), ("crowd","expos","break")):
        w = {}
        for k in combo:
            sd = nets[k].rolling(120, min_periods=60).std().shift(1)
            w[k] = (1.0/(sd+1e-12))
        tot = sum(w.values())
        s = sum(SL[k]*(w[k]/tot) for k in combo).fillna(0.0)
        score(f"risk parity: {' + '.join(combo)}", *run(px, fd, s, 0.30))
        se = sum(SL[k] for k in combo)/len(combo)
        score(f"equal signal: {' + '.join(combo)}", *run(px, fd, se, 0.30))
    print("\ndone: S137")
