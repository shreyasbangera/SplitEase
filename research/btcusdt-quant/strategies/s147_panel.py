"""
S147 - The multi-asset panel, and whether a cross-sectional signal exists in it.

WHY THIS IS A DIFFERENT PROBLEM, NOT MORE OF THE SAME
-----------------------------------------------------
Every file from S1 to S146 traded one instrument, and S145 closed that line with
arithmetic rather than with a failure to search: a book at Sharpe s cannot
compound faster than exp(s^2/2) - 1, the best single-instrument book here reached
Sharpe 1.46, and 300% at a 20% drawdown needs Sharpe 6.58.

A portfolio changes the terms of that arithmetic in one specific way, and it is
worth being precise about which. It does NOT find a better signal. It gives
access to a source of return that a single instrument does not have at all:
CROSS-SECTIONAL DISPERSION. The spread between the best and worst crypto in a
given month is far larger than the move in any one of them, and it is harvestable
with a book that is flat the market. A dollar-neutral long/short book has perhaps
a third of the volatility of a directional one, so at a FIXED 20% drawdown it can
be run at three times the size - which is the whole game, because the gate
converts volatility into permitted leverage linearly.

That is the mechanism. Whether it is there in this data is what this file asks.

THE UNIVERSE, AND ITS ONE SERIOUS DEFECT
-----------------------------------------
16 USDT-margined perpetuals, hourly, 2020-01 -> 2026-08. Thinnest name trades
$66m/day, so a book of any size a person is likely to run is inside the top 1% of
daily volume.

**These 16 are coins that still exist.** LUNA, FTT and every other delisted name
is absent, which is survivorship selection in the universe itself, and it is
NOT a small thing in crypto specifically. It is quantified in S148 rather than
mentioned and forgotten; nothing here is claimed to be free of it.

NOTHING IS BUILT IN THIS FILE. S125's mistake - four books built before anyone
asked whether the signal existed - is not repeated. This measures:

  1  CROSS-SECTIONAL IC properly: rank the feature across assets on each day,
     rank forward returns across assets, correlate, average over days. The
     standard error counts DAYS, not day-asset pairs, and is deflated further for
     the overlap in a k-day forward return.
  2  BOTH HALVES, because a feature that changes sign between halves is the thing
     this log has been fooled by twice.
  3  A PHASE-RANDOMISED CONTROL: the same measurement on surrogate features with
     identical autocorrelation and no relationship to returns. If the surrogates
     score what the real feature scores, the feature proved nothing.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

D = "/home/user/quant/data"
RNG = np.random.default_rng(11)

# Universe filter: causal. An asset is tradeable on day t only if, using data
# STRICTLY BEFORE t, it has enough history and enough volume.
MIN_HIST = 120          # days of price history before it may be ranked
MIN_ADV = 25e6          # trailing 30d median dollar volume


def panel():
    """Daily close, dollar volume and tradeable mask for the 16-perp universe."""
    cl = pd.read_parquet(f"{D}/alts_close.parquet")
    qv = pd.read_parquet(f"{D}/alts_qv.parquet")
    b = pd.read_parquet(f"{D}/fut_1h.parquet")
    b["dt"] = pd.to_datetime(b.dt, utc=True)
    b = b.set_index("dt")
    cl["BTCUSDT"] = b["close"].reindex(cl.index)
    qv["BTCUSDT"] = b["quote_volume"].reindex(cl.index)
    cl = cl.sort_index(axis=1)
    qv = qv[cl.columns]

    px = cl.resample("1D").last()
    dv = qv.resample("1D").sum()
    px.index = px.index.tz_localize(None)
    dv.index = dv.index.tz_localize(None)

    hist = px.notna().cumsum()                       # obs seen up to and incl t
    adv = dv.rolling(30, min_periods=20).median()
    ok = (hist.shift(1) >= MIN_HIST) & (adv.shift(1) >= MIN_ADV) & px.notna()
    return px, dv, ok


def features(px, dv, ok):
    """Every feature is a function of data strictly before the day it labels.

    Returned as a dict of DataFrames aligned to px. Sign is NOT asserted - the
    IC is reported raw, because guessing signs is what S118b had to correct.
    """
    lr = np.log(px)
    F = {}
    for k in (7, 14, 30, 60, 90, 180):
        F[f"mom{k}"] = lr.diff(k)
    for k in (1, 3, 5):
        F[f"rev{k}"] = -lr.diff(k)
    r1 = lr.diff(1)
    F["vol30"] = -r1.rolling(30, min_periods=20).std()        # low-vol factor
    F["vol90"] = -r1.rolling(90, min_periods=60).std()
    # idiosyncratic momentum: 30d return with the cross-sectional mean removed,
    # which is the market factor in a universe this correlated
    mkt = r1.mean(axis=1)
    F["imom30"] = (r1.sub(mkt, axis=0)).rolling(30, min_periods=20).sum()
    F["imom90"] = (r1.sub(mkt, axis=0)).rolling(90, min_periods=60).sum()
    # Amihud illiquidity: |return| per dollar traded
    F["illiq"] = (r1.abs() / dv.replace(0, np.nan)).rolling(30, min_periods=20).mean()
    # volume trend: is attention flowing in
    F["dvol30"] = np.log(dv.clip(lower=1.0)).diff(30)
    # distance from the 90d high - a different shape from raw momentum
    F["near_hi"] = px / px.rolling(90, min_periods=60).max() - 1.0
    # realised skew
    F["skew60"] = r1.rolling(60, min_periods=40).skew()
    return {k: v.where(ok) for k, v in F.items()}


def xs_ic(f, fwd, ok, min_n=6):
    """Daily cross-sectional Spearman IC between a feature and a forward return.

    Only days with at least `min_n` tradeable assets count. Returns the per-day
    series, so the caller can split it, average it, and compute a standard error
    that counts days rather than day-asset pairs.
    """
    f = f.where(ok)
    y = fwd.where(ok)
    n = (f.notna() & y.notna()).sum(axis=1)
    rf = f.rank(axis=1)
    ry = y.rank(axis=1)
    rf = rf.sub(rf.mean(axis=1), axis=0)
    ry = ry.sub(ry.mean(axis=1), axis=0)
    num = (rf * ry).sum(axis=1)
    den = np.sqrt((rf ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan)).where(n >= min_n)
    return ic


def ic_stat(ic, k):
    """Mean IC and a t-stat that is honest about overlap.

    A k-day forward return measured daily overlaps k-1 days out of k, so the
    effective sample is about N/k. Using N would inflate every t-stat by sqrt(k),
    which is exactly how a 10-day IC comes to look significant when it is not.
    """
    v = ic.dropna().to_numpy()
    if len(v) < 30:
        return np.nan, np.nan, 0
    neff = max(len(v) / k, 2.0)
    return float(v.mean()), float(v.mean() / (v.std(ddof=1) / np.sqrt(neff))), len(v)


def phase_randomise(df):
    """Surrogate panel: each column keeps its own power spectrum - so the same
    autocorrelation and the same slow wandering - but is decoupled from returns.
    Columns are randomised independently, which also destroys the cross-sectional
    structure, so this is the null for 'this feature ranks assets informatively'.
    """
    out = {}
    for c in df.columns:
        a = df[c].to_numpy(float)
        m = np.isfinite(a)
        if m.sum() < 64:
            out[c] = pd.Series(np.nan, index=df.index); continue
        v = a[m]
        fft = np.fft.rfft(v - v.mean())
        ph = RNG.uniform(0, 2 * np.pi, len(fft)); ph[0] = 0.0
        z = np.full(len(a), np.nan)
        z[m] = np.fft.irfft(np.abs(fft) * np.exp(1j * ph), n=len(v)) + v.mean()
        out[c] = pd.Series(z, index=df.index)
    return pd.DataFrame(out)


if __name__ == "__main__":
    px, dv, ok = panel()
    F = features(px, dv, ok)
    lr = np.log(px)

    live = ok.sum(axis=1)
    first = live[live >= 8].index.min()
    px, dv, ok = px.loc[first:], dv.loc[first:], ok.loc[first:]
    F = {k: v.loc[first:] for k, v in F.items()}
    lr = lr.loc[first:]

    print("S147 - the multi-asset panel and whether a cross-sectional signal is "
          "in it\n")
    print(f"universe   16 USDT perpetuals, tradeable filter = {MIN_HIST}d history "
          f"and ${MIN_ADV/1e6:.0f}m/day trailing volume, both lagged")
    print(f"span       {px.index.min().date()} -> {px.index.max().date()}  "
          f"({len(px)} days, {len(px)/365.25:.1f} years)")
    print(f"breadth    median {int(ok.sum(axis=1).median())} tradeable assets/day, "
          f"min {int(ok.sum(axis=1).min())}, max {int(ok.sum(axis=1).max())}")

    print("\nCROSS-SECTIONAL DISPERSION - the thing a single instrument does not "
          "have")
    r1 = lr.diff(1).where(ok)
    for k in (1, 7, 30):
        rk = lr.diff(k).where(ok)
        sp = (rk.max(axis=1) - rk.min(axis=1)).median()
        mv = rk.mean(axis=1).abs().median()
        print(f"   {k:>3}d   median best-minus-worst spread {sp*100:>6.1f}%   "
              f"median |market move| {mv*100:>5.1f}%   ratio {sp/max(mv,1e-9):>4.1f}x")

    print("\nCROSS-SECTIONAL IC. t-stat counts DAYS and is deflated for the "
          "overlap in a k-day\nforward return. |t| > 2 is the bar; both halves "
          "must agree in sign.\n")
    hdr = f"{'feature':>10}" + "".join(f"{'IC'+str(k)+'d':>9}{'t':>7}" for k in (1, 5, 20))
    print(hdr + f"{'1st h':>9}{'2nd h':>9}{'stable':>8}")
    keep = []
    for name in F:
        row, ics = "", {}
        for k in (1, 5, 20):
            fwd = lr.diff(k).shift(-k)
            ic = xs_ic(F[name].shift(1), fwd, ok)
            m, t, n = ic_stat(ic, k)
            ics[k] = (m, t, ic)
            row += f"{m:>9.4f}{t:>7.2f}"
        _, _, ic5 = ics[5]
        v = ic5.dropna()
        h = len(v) // 2
        i1, i2 = v.iloc[:h].mean(), v.iloc[h:].mean()
        st = (np.sign(i1) == np.sign(i2)) and abs(ics[5][1]) > 2.0
        if st:
            keep.append((name, float(np.sign(i1)), ics[5][0], ics[5][1]))
        print(f"{name:>10}{row}{i1:>9.4f}{i2:>9.4f}{'YES' if st else '':>8}")

    print(f"\nfeatures with |t| > 2 at 5 days AND the same sign in both halves: "
          f"{len(keep)} of {len(F)}")
    for n, s, m, t in sorted(keep, key=lambda x: -abs(x[3])):
        print(f"   {n:>10}  sign {int(s):+d}  IC {m:+.4f}  t {t:+.2f}")

    print("\nCONTROL: the same screen on 100 phase-randomised surrogate panels")
    print("(identical autocorrelation per asset, no relationship to returns)")
    fwd5 = lr.diff(5).shift(-5)
    probe = [n for n, _, _, _ in keep][:4] or ["mom30"]
    for name in probe:
        ts = []
        for _ in range(100):
            s = phase_randomise(F[name])
            ic = xs_ic(s.shift(1), fwd5, ok)
            _, t, _ = ic_stat(ic, 5)
            if np.isfinite(t):
                ts.append(abs(t))
        ts = np.array(ts)
        real = abs([x for x in keep if x[0] == name][0][3]) if keep else np.nan
        pv = float((ts >= real).mean()) if np.isfinite(real) else np.nan
        print(f"   {name:>10}  real |t| {real:>5.2f}   surrogate median "
              f"{np.median(ts):>5.2f}, 95th {np.percentile(ts,95):>5.2f}   "
              f"p = {pv:.3f}  {'REAL' if pv < 0.05 else 'not distinguishable'}")

    print("\ndone: panel diagnostic")
