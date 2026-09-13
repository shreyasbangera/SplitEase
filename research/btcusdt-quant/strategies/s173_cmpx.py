"""
S173 - Is cmpx still alive?

WHY THIS SIGNAL SPECIFICALLY
-----------------------------
S172's leave-one-out put cmpx at the top of V7's dependency list: removing it
drops the book from 117.4% to 42.7% at a 20% drawdown, a 64% cut and the largest
of the five by a wide margin. It is the load-bearing wall.

And it is built on the most perishable input in the stack:

    f_cmpx = z( 3-day change in log(coin-margined perp / USDT-margined perp) )

That is the price gap between BTCUSD_PERP (collateralised in BTC, held mostly by
people who already own coin) and BTCUSDT (collateralised in stablecoin). The
signal reads divergence between those two populations. Coin-margined open
interest has been shrinking relative to USDT-margined for years, which is exactly
the way a signal like this dies: not with a bang, but with the contract it
measures becoming a backwater.

So this asks three separate questions that are easy to conflate:

  1  IS THE DATA STILL THERE? Coverage and staleness of cm_1h and cm_funding,
     year by year. A signal computed off forward-filled stale prices will look
     alive on a chart and be dead in the book.
  2  IS THE SIGNAL STILL PREDICTIVE? Rolling and per-year information
     coefficient against forward returns, with the other four signals measured
     the same way for contrast - the question is whether cmpx is decaying
     faster than its neighbours or whether everything is.
  3  DOES IT STILL PAY? A single-signal book on cmpx alone, year by year,
     since an IC that survives while the money does not is the trap S157 caught.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s45_single as S
import strategies.s46_net as S46
from research.harness import backtest, OOS_END
from strategies.s69_calsel import daily
from strategies.s96_rank import stats_of

D = "/home/user/quant/data"
SIGS = ["flow", "cmpx", "btcdom", "fundz", "posn"]


def ic_series(x, y, win):
    """Rolling Spearman IC of a signal against a forward return."""
    xr = pd.Series(x).rolling(win).apply(
        lambda v: 0.0, raw=True)                       # placeholder, replaced below
    out = np.full(len(x), np.nan)
    xs = pd.Series(x); ys = pd.Series(y)
    for i in range(win, len(x)):
        a = xs.iloc[i - win:i]; b = ys.iloc[i - win:i]
        m = a.notna() & b.notna()
        if m.sum() < win // 2:
            continue
        out[i] = a[m].corr(b[m], method="spearman")
    return pd.Series(out)


def block_ic(x, y, k=12):
    """IC with a t-stat deflated for overlap in a k-bar forward return."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 60:
        return np.nan, np.nan, int(m.sum())
    a = pd.Series(x[m]); b = pd.Series(y[m])
    ic = a.corr(b, method="spearman")
    n_eff = max(m.sum() / k, 2)
    # standard error of a Spearman correlation
    se = 1.0 / np.sqrt(max(n_eff - 3, 1))
    return float(ic), float(ic / se), int(m.sum())


if __name__ == "__main__":
    print("S173 - is cmpx still alive?\n")

    # ---------- 1. data health -------------------------------------------
    print("1. THE DATA FEED - coin-margined price and funding")
    cm = pd.read_parquet(f"{D}/cm_1h.parquet")
    cm["dt"] = pd.to_datetime(cm.dt, utc=True)
    cm = cm.set_index("dt").sort_index()
    cf = pd.read_parquet(f"{D}/cm_funding.parquet")
    cf["dt"] = pd.to_datetime(cf.dt, utc=True)
    cf = cf.set_index("dt").sort_index()
    print(f"   cm_1h      {cm.index.min().date()} -> {cm.index.max().date()}, "
          f"{len(cm)} rows, cols {list(cm.columns)[:6]}")
    print(f"   cm_funding {cf.index.min().date()} -> {cf.index.max().date()}, "
          f"{len(cf)} rows")
    print(f"\n   {'year':>6}{'cm bars':>10}{'expected':>10}{'coverage':>10}"
          f"{'median |ret|':>14}{'zero-move bars':>16}")
    for y, v in cm.groupby(cm.index.year):
        exp = 366 * 24 if y % 4 == 0 else 365 * 24
        if y == cm.index.max().year:
            exp = int((cm.index.max() - pd.Timestamp(f"{y}-01-01", tz="UTC")).total_seconds() / 3600)
        r = v["close"].pct_change()
        zero = float((r.abs() < 1e-9).mean() * 100)
        print(f"   {y:>6}{len(v):>10}{exp:>10}{len(v)/max(exp,1)*100:>9.0f}%"
              f"{r.abs().median()*1e4:>13.1f}bp{zero:>15.1f}%")

    # ---------- 2. predictive power --------------------------------------
    g = S.grid(S.FULL_START)
    c = g.close.to_numpy(float)
    print("\n2. PREDICTIVE POWER - IC against the next 3 days (6 bars of 12h)")
    fwd = pd.Series(c).pct_change(6).shift(-6).to_numpy()
    print(f"   {'year':>6}" + "".join(f"{s:>16}" for s in SIGS))
    yrs = sorted(set(pd.to_datetime(g.dt).dt.year))
    tab = {s: {} for s in SIGS}
    for y in yrs:
        m = (pd.to_datetime(g.dt).dt.year == y).to_numpy()
        row = ""
        for s in SIGS:
            x = g[f"s_{s}"].to_numpy(float)
            ic, t, n = block_ic(x[m], fwd[m], 6)
            tab[s][y] = ic
            row += (f"{ic:>+9.4f}{t:>+7.1f}" if np.isfinite(ic) else f"{'n/a':>16}")
        print(f"   {y:>6}{row}")
    print(f"\n   {'signal':>10}{'2021-23 mean IC':>18}{'2024-26 mean IC':>18}"
          f"{'change':>10}")
    for s in SIGS:
        early = [v for y, v in tab[s].items() if y <= 2023 and np.isfinite(v)]
        late = [v for y, v in tab[s].items() if y >= 2024 and np.isfinite(v)]
        if not early or not late:
            continue
        e, l = np.mean(early), np.mean(late)
        print(f"   {s:>10}{e:>+18.4f}{l:>+18.4f}"
              f"{(l-e)/max(abs(e),1e-9)*100:>+9.0f}%")

    # ---------- 3. does it still pay -------------------------------------
    print("\n3. SINGLE-SIGNAL BOOKS - what each signal earns on its own")
    a14 = g.atr14.to_numpy(float)
    print(f"   {'signal':>10}" + "".join(f"{y:>10}" for y in yrs) + f"{'Sharpe':>9}")
    for s in SIGS:
        v = S.composite(g, [s])
        u = np.nan_to_num(v)
        arr = dict(entry=u, stop=3.0 * a14, tp=3.0 * 2.0 * a14,
                   exit=(np.abs(u) <= 0.0).astype(float))
        m = backtest(g, arr, "12h", start=S.FULL_START, end=OOS_END, risk=0.048,
                     max_lev=10.0, max_bars_h=21 * 24)
        r = daily(m)
        r.index = pd.to_datetime(r.index)
        if r.index.tz:
            r.index = r.index.tz_localize(None)
        row = ""
        for y in yrs:
            vv = r[r.index.year == y]
            row += (f"{((1+vv).prod()-1)*100:>+9.1f}%" if len(vv) else f"{'—':>10}")
        print(f"   {s:>10}{row}{stats_of(r.to_numpy(float))['sharpe']:>9.2f}")

    print("\n4. CMPX vs THE REST - rolling 1-year IC")
    win = 2 * 365  # 12h bars in a year
    fwds = pd.Series(fwd)
    dts = pd.to_datetime(g.dt)
    for s in ("cmpx", "flow", "btcdom"):
        x = pd.Series(g[f"s_{s}"].to_numpy(float))
        vals = []
        for i in range(win, len(x), 120):
            a = x.iloc[i - win:i]; b = fwds.iloc[i - win:i]
            mm = a.notna() & b.notna()
            if mm.sum() > win // 3:
                vals.append((dts.iloc[i], a[mm].corr(b[mm], method="spearman")))
        if vals:
            first = np.mean([v for _, v in vals[:len(vals)//3]])
            last = np.mean([v for _, v in vals[-len(vals)//3:]])
            print(f"   {s:>8}: first third {first:+.4f}  ->  last third "
                  f"{last:+.4f}   ({(last-first)/max(abs(first),1e-9)*100:+.0f}%)")
    print("\ndone: cmpx health check")
