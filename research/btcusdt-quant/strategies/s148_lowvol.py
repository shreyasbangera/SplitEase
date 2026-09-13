"""
S148 - The low-volatility book across a portfolio of crypto perpetuals.

WHAT S147 ESTABLISHED, AND WHAT IT KILLED
------------------------------------------
Of 17 cross-sectional features, four cleared |t| > 2 at a 5-day horizon with the
same sign in both halves, and only two survived a phase-randomised control:

    vol90   IC +0.1061  t +5.83   p < 0.001 against surrogates   REAL
    vol30   IC +0.0940  t +5.16   p < 0.001                      REAL
    near_hi IC +0.0697  t +4.00   p  = 0.090                     not distinguishable
    illiq   IC -0.0561  t -3.88   p  = 0.560                     not distinguishable

`illiq` is the instructive one: it is a slow, highly autocorrelated series, and
surrogates with the same autocorrelation and NO relationship to returns scored a
median |t| of 4.00 - higher than the real thing. Without that control it would
have looked like the second-best signal in the study.

Cross-sectional MOMENTUM is absent: mom7 through mom180 all sit under |t| = 1.8
and flip sign between halves. That is worth stating plainly because it is the
signal most crypto portfolio papers are built on.

So what is left is one effect, and it has a name and a reason. Low-beta assets
out-earn high-beta ones per unit of risk because investors who want return and
cannot borrow buy high-beta instead of levering low-beta, bidding it up. In
crypto the leverage constraint is weaker but the effect is larger, because the
high-beta end is small speculative coins bought with exactly that motive.

THE HONEST QUALIFICATION, MADE BEFORE THE RESULT RATHER THAN AFTER
------------------------------------------------------------------
The low-vol leg is BTC 95% of days, BNB 79%, TRX 71%, ETH 54%. So this is
substantially ONE BET - long mega-cap, short small alt - held for a 62-day
average, not sixteen independent bets. A portfolio of sixteen names whose
positions are that concentrated and that persistent has far less diversification
than the name suggests, and the Sharpe it can reach is bounded accordingly.
That is tested directly below by removing BTC and ETH from the universe.

WHAT IS CHARGED AGAINST IT
--------------------------
    fees        5bps per side, above Binance's 4bps USDT-M taker
    slippage    swept across 0/5/10/20bps per side; the headline uses 5
    funding     REAL 8-hourly funding for all 16 names, fetched per symbol.
                Longs pay it and shorts receive it, per asset per day, which is
                the carry a long/short perp book actually runs.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, glob, os, zipfile, io

import strategies.s147_panel as P
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"
FEE_BPS = 5.0
GROSS_CAP = 3.0        # max sum|w| before vol targeting
MAX_GROSS = 4.0        # max sum|w| after it


def funding_panel(cols, index):
    """Daily funding paid by a long, per asset. Real 8-hourly rates, summed."""
    out = {}
    for s in cols:
        fs = sorted(glob.glob(f"{D}/fund2/{s}/*.zip"))
        if not fs:
            out[s] = pd.Series(np.nan, index=index); continue
        parts = []
        for f in fs:
            try:
                with zipfile.ZipFile(f) as z:
                    n = z.namelist()[0]
                    d = pd.read_csv(io.BytesIO(z.read(n)))
            except Exception:
                continue
            d.columns = [str(c).strip().lower() for c in d.columns]
            tc = next((c for c in d.columns if "time" in c), d.columns[0])
            # "funding" matches funding_interval_hours, whose value is 8 - a
            # funding rate of 800% per period. Rate must be matched first.
            rc = next((c for c in d.columns if "rate" in c), None) or d.columns[-1]
            d = d[pd.to_numeric(d[tc], errors="coerce").notna()]
            if not len(d):
                continue
            t = pd.to_numeric(d[tc])
            unit = "ms" if t.max() > 1e11 else "s"
            parts.append(pd.Series(pd.to_numeric(d[rc], errors="coerce").to_numpy(),
                                   index=pd.to_datetime(t, unit=unit, utc=True)))
        if not parts:
            out[s] = pd.Series(np.nan, index=index); continue
        r = pd.concat(parts).sort_index()
        r = r[~r.index.duplicated()]
        daily = r.resample("1D").sum()
        daily.index = daily.index.tz_localize(None)
        out[s] = daily.reindex(index)
    return pd.DataFrame(out).reindex(index)


def weights(sig, ok, mode, beta=None, frac=3):
    """Turn a cross-sectional score into portfolio weights.

    `sig` is higher-is-better. Every construction is built from ranks computed
    on the day's tradeable set only, so a name that is not tradeable cannot be
    held and cannot displace one that is.
    """
    s = sig.where(ok)
    n = ok.sum(axis=1)
    k = (n // frac).clip(lower=2)
    rk_lo = s.rank(axis=1, ascending=False)          # 1 = best (lowest vol)
    rk_hi = s.rank(axis=1, ascending=True)           # 1 = worst (highest vol)
    L = rk_lo.le(k, axis=0) & ok
    S = rk_hi.le(k, axis=0) & ok

    if mode == "longonly":
        w = L.astype(float)
        return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    wl = L.astype(float).div(L.sum(axis=1).replace(0, np.nan), axis=0)
    ws = S.astype(float).div(S.sum(axis=1).replace(0, np.nan), axis=0)

    if mode == "dollarneutral":
        return (wl - ws).fillna(0.0)

    if mode == "betaneutral":
        # Frazzini-Pedersen: lever the low-beta leg to beta 1, delever the high-
        # beta leg to beta 1, so the book carries no market exposure at all.
        # The gross is capped: 1/beta is unbounded as beta -> 0, and an uncapped
        # version of this ran at 20x gross and lost more than 100% in a day.
        bl = (wl * beta).sum(axis=1).replace(0, np.nan)
        bs = (ws * beta).sum(axis=1).replace(0, np.nan)
        w = (wl.div(bl, axis=0) - ws.div(bs, axis=0)).fillna(0.0)
        g = w.abs().sum(axis=1)
        return w.div(np.maximum(g / GROSS_CAP, 1.0), axis=0).fillna(0.0)

    if mode == "zscore":
        z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
        z = z.clip(-2, 2).where(ok).fillna(0.0)
        z = z.sub(z.sum(axis=1) / ok.sum(axis=1), axis=0).where(ok).fillna(0.0)
        return z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    raise ValueError(mode)


def book(px, w, fund, slip_bps=5.0, target_vol=None, vol_hl=45, max_lev=3.0,
         band=0.0):
    """Net daily returns of a weight matrix, with costs and funding.

    Weights are lagged one full day: `w` is formed from data through day t-1 and
    earns day t's return. Turnover is charged on the change in weights, funding
    on the weights held.
    """
    r = px.pct_change().fillna(0.0)
    w = w.shift(1).fillna(0.0)

    if target_vol is not None:
        gross = (w * r).sum(axis=1)
        rv = gross.ewm(halflife=vol_hl, min_periods=30).std().shift(1) * np.sqrt(365.25)
        lev = (target_vol / rv.replace(0, np.nan)).clip(upper=max_lev).fillna(0.0)
        if band > 0:
            lv = lev.to_numpy(); out = np.zeros(len(lv)); cur = 0.0
            for i in range(len(lv)):
                if abs(lv[i] - cur) > band * max(cur, 1e-9):
                    cur = lv[i]
                out[i] = cur
            lev = pd.Series(out, index=lev.index)
        w = w.mul(lev, axis=0)
        g = w.abs().sum(axis=1)
        w = w.div(np.maximum(g / MAX_GROSS, 1.0), axis=0).fillna(0.0)

    gross = (w * r).sum(axis=1)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * (FEE_BPS + slip_bps) / 1e4
    carry = (w * fund.reindex_like(w).fillna(0.0)).sum(axis=1)   # long pays
    net = gross - cost - carry
    # A book that loses more than 100% in a day is not a book; it is a margin
    # call. Flag it rather than compounding through negative equity, which is
    # what produced inf/nan figures in the first version of this file.
    if (net <= -1.0).any():
        print(f"      !! {int((net <= -1.0).sum())} day(s) below -100%: "
              f"this book is not survivable as specified")
    return net.clip(lower=-0.95), turn, w


def gate_of(a, tol=0.01):
    """S124's guard: never report a gate the bisection did not actually reach."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return np.nan
    g = at_gate(a, lo=1e-4, hi=60.0, iters=70)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


def summarise(net, turn, tag, extra=""):
    a = net.to_numpy(float)
    st = stats_of(a)
    g = gate_of(a)
    yrs = len(a) / 365.25
    pos_days = int((np.abs(turn) > 0).sum())
    print(f"   {tag:>26}{st['sharpe']:>7.2f}{st['cagr']*100:>9.1f}%"
          f"{st['dd']*100:>8.1f}%{turn.mean():>8.2f}"
          + ("      n/a" if not np.isfinite(g) else f"{g:>8.1f}%") + extra)
    return st, g


if __name__ == "__main__":
    px, dv, ok = P.panel()
    live = ok.sum(axis=1)
    first = live[live >= 8].index.min()
    px, dv, ok = px.loc[first:], dv.loc[first:], ok.loc[first:]

    fund = funding_panel(list(px.columns), px.index)
    print("S148 - the low-volatility book across a crypto portfolio\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}  "
          f"({len(px)} days, {len(px)/365.25:.1f} years), "
          f"median {int(ok.sum(axis=1).median())} tradeable names/day")
    cov = fund.where(ok).notna().sum().sum() / max(ok.sum().sum(), 1) * 100
    print(f"funding coverage over tradeable cells: {cov:.1f}%   "
          f"mean daily funding paid by a long: "
          f"{fund.where(ok).stack().mean()*1e4:.2f}bps "
          f"({fund.where(ok).stack().mean()*365.25*100:+.1f}%/yr)")

    r1 = np.log(px).diff(1)
    sig = (-r1.rolling(90, min_periods=60).std()).where(ok)      # S147's winner
    mkt = r1.mean(axis=1)
    beta = (r1.rolling(180, min_periods=90).cov(mkt)
            .div(mkt.rolling(180, min_periods=90).var(), axis=0)).where(ok).clip(0.4, 2.5)

    print("\nCONSTRUCTIONS, unlevered. gate = CAGR when scaled to a 20% max "
          "drawdown.\n")
    print(f"   {'construction':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    books = {}
    for mode in ("longonly", "dollarneutral", "betaneutral", "zscore"):
        w = weights(sig, ok, mode, beta)
        net, turn, _ = book(px, w, fund)
        st, g = summarise(net, turn, mode)
        books[mode] = net
    # benchmarks
    ew = ok.astype(float).div(ok.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    net, turn, _ = book(px, ew, fund)
    summarise(net, turn, "equal-weight universe")
    btc = pd.DataFrame(0.0, index=px.index, columns=px.columns); btc["BTCUSDT"] = 1.0
    net, turn, _ = book(px, btc, fund)
    summarise(net, turn, "buy and hold BTC")

    print("\nVOL-TARGETED, which is how it would actually be run")
    print(f"   {'construction @ target':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>8}{'at -20%':>9}")
    for mode in ("dollarneutral", "betaneutral", "zscore"):
        for tv in (0.20, 0.40):
            w = weights(sig, ok, mode, beta)
            net, turn, _ = book(px, w, fund, target_vol=tv, band=0.10)
            summarise(net, turn, f"{mode} {tv*100:.0f}%")

    print("\nIS IT JUST 'LONG BTC+ETH, SHORT ALTS'? same book, mega-caps removed")
    ok2 = ok.copy()
    ok2[["BTCUSDT", "ETHUSDT"]] = False
    sig2 = sig.where(ok2)
    print(f"   {'construction':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    for mode in ("dollarneutral", "betaneutral", "zscore"):
        w = weights(sig2, ok2, mode, beta)
        net, turn, _ = book(px, w, fund)
        summarise(net, turn, f"{mode} (no BTC/ETH)")

    print("\nCOST SENSITIVITY on the best neutral construction")
    print(f"   {'slippage per side':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>8}{'at -20%':>9}")
    for sl in (0.0, 5.0, 10.0, 20.0):
        w = weights(sig, ok, "betaneutral", beta)
        net, turn, _ = book(px, w, fund, slip_bps=sl)
        summarise(net, turn, f"{sl:.0f}bps + {FEE_BPS:.0f}bps fee")

    print("\nFUNDING: how much of the result is carry rather than price?")
    print(f"   {'variant':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    w = weights(sig, ok, "betaneutral", beta)
    for tag, f_ in (("real funding", fund),
                    ("funding forced to zero", fund * 0.0),
                    ("funding doubled against us", fund * 2.0)):
        net, turn, _ = book(px, w, f_)
        summarise(net, turn, tag)

    print("\nYEAR BY YEAR, beta-neutral at 40% target")
    w = weights(sig, ok, "betaneutral", beta)
    net, turn, _ = book(px, w, fund, target_vol=0.40, band=0.10)
    for y, v in net.groupby(net.index.year):
        eq = (1 + v).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        print(f"   {y}: {((1+v).prod()-1)*100:>+8.1f}%   in-year DD {dd*100:>6.1f}%"
              f"   ({len(v)} days)")
    print("\ndone: low-vol portfolio")
