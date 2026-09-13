"""
S127 - Macro risk appetite as a signal for a BTCUSDT-only book.

WHERE THIS DATA CAME FROM, HAVING SAID IT WAS UNREACHABLE
----------------------------------------------------------
S126 closed with "no new data is reachable" - every market data host returns
EGRESS_BLOCKED at the network proxy and Binance itself 403s the datacenter IP.
That was true of the container's network and wrong as a conclusion, because this
session also has MCP data servers attached which do not use that path at all.

Checked, with the results: Crypto.com returns at most 50 candles - useless for a
backtest. LunarCrush runs in "limited data mode" and masks every value behind a
subscription, so it yields dates and no numbers. **Twelve Data works**, returns
up to 5,000 daily bars per symbol, and covers equities, ETFs, forex and
commodities. Six series fetched, each verified against an expected price band
rather than trusted by call order: HYG, LQD, TLT, SPY, UUP - 1,830 trading days
from 2019-06 - joined to the VIX history already on disk.

WHY MACRO IS THE RIGHT THING TO SPEND A NEW DATA SOURCE ON
-----------------------------------------------------------
S124 computed the requirement: the brief needs roughly three strategies of V7's
quality that do not correlate with it. Everything tried so far has been
crypto-native - positioning, funding, price, order flow, cross-asset crypto -
and every one of them correlates with the same underlying beta, which is exactly
why the oracle combination gained 4%.

Macro is the first input in this entire study generated **outside the crypto
market**. Credit spreads, the dollar, equity volatility and the long bond are
driven by flows and participants with no mechanical link to perpetual futures
positioning. If a signal exists here it has a real chance of being the
uncorrelated stream the requirement table is asking for, which nothing
crypto-native can be.

ALIGNMENT, WHICH IS WHERE THIS WOULD LEAK
------------------------------------------
Macro trades weekdays and closes at 21:00 UTC; BTC trades continuously and its
daily bar closes at 00:00 UTC. A macro reading stamped date D is therefore known
*before* BTC's D+1 bar, and is forward-filled onto the BTC grid and lagged a
further full day on top. Holidays and weekends carry the last known value
forward, never backward - filling backward across a weekend would hand the book
Monday's panic on Friday, which is the single easiest way to invent a macro edge.

Judged the way S125b was: ICs on both halves, then non-overlapping ICs with a
bootstrap interval, then a phase-randomised control. A feature that cannot beat
surrogates carrying its own autocorrelation is not a feature.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

D = "/home/user/quant/data"
RNG = np.random.default_rng(11)


def panel():
    M = pd.read_parquet(f"{D}/macro/macro_daily.parquet")
    M.index = pd.to_datetime(M.index)
    if M.index.tz is None:
        M.index = M.index.tz_localize("UTC")

    d = pd.read_parquet(f"{D}/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    px = d.set_index("dt")["close"].resample("1D").last().dropna()

    # forward-fill only: a weekend carries FRIDAY's reading, never Monday's
    Mx = M.reindex(px.index.union(M.index)).sort_index().ffill().reindex(px.index)

    F = pd.DataFrame(index=px.index)
    lr = lambda s, k: np.log(s).diff(k)
    z = lambda s, w=252: (s - s.rolling(w, min_periods=60).mean()) / \
                         (s.rolling(w, min_periods=60).std() + 1e-12)

    F["credit"] = lr(Mx.HYG / Mx.LQD, 5)          # high yield vs IG: risk appetite
    F["credit_lvl"] = z(np.log(Mx.HYG / Mx.LQD))
    F["equity"] = lr(Mx.SPY, 20)
    F["vix_lvl"] = z(Mx.VIX)
    F["vix_chg"] = z(Mx.VIX.diff(5), 252)
    F["dollar"] = lr(Mx.UUP, 20)
    F["rates"] = lr(Mx.TLT, 20)
    F["stress"] = -z(Mx.VIX) + z(np.log(Mx.HYG / Mx.LQD)) - z(np.log(Mx.UUP))
    return px, F


COLS = ["credit", "credit_lvl", "equity", "vix_lvl", "vix_chg", "dollar",
        "rates", "stress"]


def nonoverlap_ic(x, y, k, n_boot=2000):
    m = np.isfinite(x) & np.isfinite(y)
    xa, ya = x[m].to_numpy(), y[m].to_numpy()
    xs, ys = xa[::k], ya[::k]
    ic = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    n = len(xs)
    b = np.empty(n_boot)
    for i in range(n_boot):
        j = RNG.integers(0, n, n)
        b[i] = pd.Series(xs[j]).corr(pd.Series(ys[j]), method="spearman")
    return ic, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5), n


def phase_randomise(s):
    a = np.asarray(s, float)
    m = np.isfinite(a)
    v = a[m]
    f = np.fft.rfft(v - v.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(f)); ph[0] = 0.0
    out = np.full(len(a), np.nan)
    out[m] = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(v)) + v.mean()
    return pd.Series(out, index=s.index)


if __name__ == "__main__":
    px, F = panel()
    lr = np.log(px)
    print("S127 - macro risk appetite as a BTCUSDT-only signal\n")
    print(f"daily, {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days")
    print("macro lagged a full day ON TOP of forward-fill; weekends carry "
          "Friday forward\n")

    print(f"{'feature':>12}" + "".join(f"{'IC ' + str(k) + 'd':>10}" for k in (1, 3, 5, 10))
          + f"{'IC5 1st':>10}{'IC5 2nd':>10}{'stable':>8}")
    keep = []
    for c in COLS:
        x = F[c].shift(1)
        ics = []
        for k in (1, 3, 5, 10):
            y = lr.diff(k).shift(-k)
            m = np.isfinite(x) & np.isfinite(y)
            ics.append(pd.Series(x[m].to_numpy()).corr(
                pd.Series(y[m].to_numpy()), method="spearman"))
        y5 = lr.diff(5).shift(-5)
        m = np.isfinite(x) & np.isfinite(y5)
        xa, ya = x[m], y5[m]
        h = len(xa) // 2
        i1 = pd.Series(xa.iloc[:h].to_numpy()).corr(pd.Series(ya.iloc[:h].to_numpy()), method="spearman")
        i2 = pd.Series(xa.iloc[h:].to_numpy()).corr(pd.Series(ya.iloc[h:].to_numpy()), method="spearman")
        st = np.sign(i1) == np.sign(i2) and min(abs(i1), abs(i2)) > 0.03
        if st:
            keep.append(c)
        print(f"{c:>12}" + "".join(f"{v:>10.4f}" for v in ics)
              + f"{i1:>10.4f}{i2:>10.4f}{'YES' if st else '':>8}")

    print(f"\nsurvivors: {keep if keep else 'none'}")
    for c in keep:
        print(f"\n--- {c}: non-overlapping IC with bootstrap 95% interval")
        x = F[c].shift(1)
        for k in (1, 3, 5, 10):
            y = lr.diff(k).shift(-k)
            ic, lo, hi, n = nonoverlap_ic(x, y, k)
            print(f"    {k:>3}d  n={n:>5}  IC {ic:>+7.4f}  [{lo:>+7.4f},{hi:>+7.4f}]"
                  f"  {'EXCLUDES 0' if (lo > 0 or hi < 0) else 'includes 0'}")
        print(f"    control: 200 phase-randomised surrogates")
        y5 = lr.diff(5).shift(-5)
        p = 0
        for _ in range(200):
            s = phase_randomise(F[c]).shift(1)
            m = np.isfinite(s) & np.isfinite(y5)
            xa, ya = s[m], y5[m]
            h = len(xa) // 2
            a1 = pd.Series(xa.iloc[:h].to_numpy()).corr(pd.Series(ya.iloc[:h].to_numpy()), method="spearman")
            a2 = pd.Series(xa.iloc[h:].to_numpy()).corr(pd.Series(ya.iloc[h:].to_numpy()), method="spearman")
            if np.sign(a1) == np.sign(a2) and min(abs(a1), abs(a2)) > 0.03:
                p += 1
        print(f"    surrogates clearing the same bar: {p}/200 ({p/2:.0f}%)")
    print("\ndone: macro diagnostic")
