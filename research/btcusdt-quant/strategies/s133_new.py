"""
S133 - Three strategies this study has never built. Judged on their own.

No benchmarking against anything else in this log. Each is scored against the
brief and nothing else: 300% a year, drawdown under 20%, 100+ completed trades,
profit factor above 1.10, realistic risk management, no look-ahead.

WHY THESE THREE
---------------
Everything built in this study so far forecasts DIRECTION from a state variable -
crowding, price trend, macro, on-chain. These three do not work that way.

  1  FUNDING HARVEST
     The perpetual pays a cash flow every eight hours. When funding is deeply
     negative, shorts pay longs: you are PAID to hold the long. This is not a
     forecast, it is a coupon, and it is the only mechanical cash flow available
     on a single perpetual. S118 used funding as one of nine features inside a
     direction model, which is a different thing entirely - it has never been
     traded as the carry it actually is.

  2  SEASONALITY
     Completely untested in this log, and the cheapest thing here to check.
     Crypto trades continuously across sessions with structurally different
     participants - Asian hours, European hours, US hours, and a weekend with no
     traditional-finance hedging available. Day-of-week and hour-of-day effects
     are plausible on those grounds and are pure calendar, so they cannot
     correlate with any state-variable model by construction.
     Estimated on a TRAILING window only and walked forward, because a seasonal
     pattern fitted on the whole sample is the textbook way to manufacture one.

  3  BREAKOUT / RANGE EXPANSION
     The oldest systematic strategy there is, and this study went straight to
     EMA crossovers without trying it. A breakout book has a different payoff
     shape from a moving-average book: it is flat inside the range and only
     takes risk when the market leaves it, so it holds a position far less of
     the time.

Every one is charged 5bps fee, 3bps slippage and settled funding, sized off
trailing volatility, and lagged so the decision on bar t uses only bars < t.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from strategies.s96_rank import at_gate, stats_of
from research.robust import bootstrap_dd

FEE, SLIP = 5.0, 3.0
D = "/home/user/quant/data"


def data():
    d = pd.read_parquet(f"{D}/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    d = d.set_index("dt")
    f = pd.read_parquet(f"{D}/funding.parquet")
    f["dt"] = pd.to_datetime(f.dt, utc=True)
    fr = f.set_index("dt")["rate"]
    return d, fr


def gate(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


def score(tag, net, pos):
    a = np.asarray(net, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        print(f"   {tag:>30}{'no position taken':>44}")
        return
    st = stats_of(a); g = gate(a)
    b = bootstrap_dd(a, n=1500, block=90)
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
    print(f"   {tag:>30}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%"
          f"{b['dd_median']*100:>8.1f}%{trades:>7}{pf:>7.2f}{(live.mean()*100):>6.0f}%"
          f"{gs:>9}{s1:>8}{s2:>8}")


HDR = (f"   {'variant':>30}{'Shp':>7}{'realDD':>8}{'medDD':>8}{'trades':>7}"
       f"{'PF':>7}{'expo':>6}{'at -20%':>9}{'1st h':>8}{'2nd h':>8}")


def run_daily(px, fd, sig, tv, vol_hl=32, max_lev=3.0, band=0.10):
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


# ---------------------------------------------------------------- 1. FUNDING
def funding_harvest(d, fr):
    """Long when the perpetual PAYS you to be long; short when it pays to short.

    The signal is the funding rate itself, standardised on a trailing window and
    sign-flipped: deeply negative funding means shorts are paying longs, so the
    coupon is collected by going long. Strictly past - the rate used on day t is
    the one already settled before t.
    """
    px = d["close"].resample("1D").last().dropna()
    fd = fr.resample("1D").sum().reindex(px.index).fillna(0.0)
    print("\n1. FUNDING HARVEST — collecting the coupon, not forecasting price")
    print(f"   funding: mean {fd.mean()*365.25*100:+.1f}%/yr to a long, "
          f"negative on {100*(fd<0).mean():.0f}% of days")
    print(HDR)
    for win in (30, 90):
        z = ((fd - fd.rolling(win, min_periods=20).mean())
             / (fd.rolling(win, min_periods=20).std() + 1e-12)).shift(1).fillna(0.0)
        for thr in (0.0, 1.0, 1.5):
            s = (-z).clip(-2, 2)
            s = s.where(s.abs() >= thr, 0.0)
            net, pos = run_daily(px, fd, s, 0.30)
            score(f"win {win}d, |z|>={thr:.1f}", net, pos)


# ------------------------------------------------------------ 2. SEASONALITY
def seasonality(d, fr):
    """Day-of-week and hour-of-day, estimated on a TRAILING window only.

    At each point the expected return for this calendar slot is the mean of that
    slot over the previous `win` days, using nothing at or after the decision
    bar. A seasonal pattern fitted on the whole sample would be guaranteed to
    look profitable and mean nothing.
    """
    px = d["close"].resample("1D").last().dropna()
    fd = fr.resample("1D").sum().reindex(px.index).fillna(0.0)
    r = np.log(px / px.shift(1)).fillna(0.0)
    print("\n2. SEASONALITY — calendar only, trailing estimate, walked forward")
    dow = px.index.dayofweek
    full = pd.Series(r.to_numpy()).groupby(dow).mean() * 1e4
    print("   full-sample day-of-week mean return (bps) — "
          "DESCRIPTIVE ONLY, not tradeable:")
    print("   " + "  ".join(f"{['Mo','Tu','We','Th','Fr','Sa','Su'][i]} {full[i]:+.1f}"
                            for i in range(7)))
    print(HDR)
    for win in (180, 365, 730):
        sig = np.zeros(len(px))
        rv = r.to_numpy()
        for i in range(len(px)):
            lo = max(0, i - win)
            if i - lo < 60:
                continue
            m = dow[lo:i] == dow[i]
            if m.sum() < 8:
                continue
            sig[i] = np.sign(rv[lo:i][m].mean())
        s = pd.Series(sig, index=px.index)
        for tv in (0.20, 0.40):
            net, pos = run_daily(px, fd, s, tv)
            score(f"dow, trailing {win}d, tgt {tv*100:.0f}%", net, pos)


# --------------------------------------------------------------- 3. BREAKOUT
def breakout(d, fr):
    """Donchian channel breakout with an ATR trailing exit. Flat inside the range."""
    px = d["close"].resample("1D").last().dropna()
    hi = d["high"].resample("1D").max().reindex(px.index)
    lo = d["low"].resample("1D").min().reindex(px.index)
    fd = fr.resample("1D").sum().reindex(px.index).fillna(0.0)
    tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False).mean()
    print("\n3. BREAKOUT — flat inside the range, risk only on expansion")
    print(HDR)
    c = px.to_numpy(); a = atr.to_numpy()
    for N in (20, 55, 100):
        up = hi.rolling(N).max().shift(1).to_numpy()
        dn = lo.rolling(N).min().shift(1).to_numpy()
        for k in (2.0, 3.0):
            s = np.zeros(len(px)); cur = 0.0; peak = 0.0
            for i in range(len(px)):
                if not np.isfinite(up[i]) or not np.isfinite(a[i]):
                    s[i] = 0.0; continue
                if cur == 0.0:
                    if c[i] > up[i]:
                        cur, peak = 1.0, c[i]
                    elif c[i] < dn[i]:
                        cur, peak = -1.0, c[i]
                elif cur > 0:
                    peak = max(peak, c[i])
                    if c[i] < peak - k * a[i]:
                        cur = 0.0
                else:
                    peak = min(peak, c[i])
                    if c[i] > peak + k * a[i]:
                        cur = 0.0
                s[i] = cur
            sig = pd.Series(s, index=px.index).shift(1).fillna(0.0)
            net, pos = run_daily(px, fd, sig, 0.30)
            score(f"Donchian {N}d, {k:.0f}xATR exit", net, pos)


if __name__ == "__main__":
    d, fr = data()
    print("S133 - three strategies never built in this study")
    print(f"BTCUSDT perp, {d.index.min().date()} -> {d.index.max().date()}")
    print("each judged against the brief alone: 300% at a 20% drawdown, "
          "100+ trades, PF > 1.10\n")
    funding_harvest(d, fr)
    seasonality(d, fr)
    breakout(d, fr)
    print("\ndone: S133")
