"""
S174 - Minute-resolution microstructure, traded at daily frequency.

WHY THIS AND NOT ANOTHER PRICE SIGNAL
--------------------------------------
Everything built after V7 that was genuinely independent - the breakout family
(Sharpe 1.08), cross-sectional carry (1.13), low volatility, tail momentum,
cash-and-carry - was computed from DAILY or 12-HOUR bars. They lost to V7 and
they lost to each other for the same reason: a daily bar throws away almost
everything that happened inside it.

Two dense high-frequency datasets have been sitting unused:

    fut_1m    3.5m rows, 6.7 years of 1-minute bars carrying taker_buy_base and
              taker_buy_quote - which is signed order flow, minute by minute
    bvol_1m   1.66m rows, 3.2 years of BTC implied volatility at 1-minute

V7 already reads order flow and implied vol, but only as 12-hour aggregates
(`ofi6_res`, `iv_mom`). Aggregating to 12 hours destroys exactly what makes
microstructure informative: the SHAPE of flow within the window - whether it was
one violent burst or steady accumulation, whether volatility came from jumps or
diffusion, whether the aggressor side persisted or flipped.

THE COST CONSTRAINT, WHICH KILLS THE OBVIOUS VERSION
-----------------------------------------------------
S120 measured the intraday edge on BTC at 0.1-2.2bps against a 16bps taker round
trip. Trading these signals AT minute frequency is arithmetically dead and is not
attempted.

What is attempted is the asymmetry: compute features from minute data, trade them
once a day. That buys the information content of tick data at the cost profile of
a daily book - roughly 8bps a round trip against a signal measured over 1,440
observations per decision rather than one.

WHAT IS MEASURED, WITH THE SAME DISCIPLINE AS EVERYTHING ELSE
--------------------------------------------------------------
Every feature is lagged a full day, screened on a t-statistic deflated for
overlap, and then put against a phase-randomised surrogate control - the test
that killed the illiquidity factor twice, at p = 0.56 and p = 1.00, after it had
looked like the second-best signal in the study.
"""
import sys, os; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"
RNG = np.random.default_rng(23)
FEE_BPS, SLIP_BPS = 5.0, 3.0


def minute_features(cache=f"{D}/micro_daily.parquet", rebuild=False):
    """Daily features built from 1-minute bars. Every one is a within-day
    statistic that a daily bar cannot express."""
    if os.path.exists(cache) and not rebuild:
        return pd.read_parquet(cache)
    m = pd.read_parquet(f"{D}/fut_1m.parquet")
    m["dt"] = pd.to_datetime(m.dt, utc=True)
    m = m.set_index("dt").sort_index()
    m = m[m.volume > 0]
    day = m.index.normalize()

    r = np.log(m["close"]).diff()
    buy = m["taker_buy_quote"].astype(float)
    tot = m["quote_volume"].astype(float).replace(0, np.nan)
    ofi = (2.0 * buy / tot - 1.0)                      # +1 all-buy, -1 all-sell
    n = m["count"].astype(float)

    F = pd.DataFrame(index=pd.DatetimeIndex(sorted(set(day))))
    gb = lambda s: s.groupby(day)

    # --- order flow shape -------------------------------------------------
    F["ofi_mean"] = gb(ofi).mean()
    F["ofi_std"] = gb(ofi).std()
    F["ofi_skew"] = gb(ofi).skew()
    # how persistent is the aggressor side within the day
    F["ofi_ac1"] = gb(ofi).apply(lambda s: s.autocorr(1) if len(s) > 30 else np.nan)
    # dollar-weighted flow, which the mean ignores
    F["ofi_dw"] = gb(ofi * tot).sum() / gb(tot).sum()
    # concentration: did the flow arrive in a burst or evenly
    F["vol_conc"] = gb(tot).apply(
        lambda s: float((s.nlargest(max(len(s)//20, 1)).sum()) / s.sum())
        if s.sum() > 0 else np.nan)

    # --- volatility structure --------------------------------------------
    rv = gb(r ** 2).sum()
    F["rv"] = np.sqrt(rv * 365.25)
    # bipower variation: variance excluding jumps. RV minus BV is the jump part.
    absr = r.abs()
    bv = gb(absr * absr.shift(1)).sum() * (np.pi / 2.0)
    F["jump_frac"] = ((rv - bv) / rv.replace(0, np.nan)).clip(0, 1)
    # signed jump: were the jumps up or down
    F["jump_dir"] = gb(np.sign(r) * (r ** 2)).sum() / rv.replace(0, np.nan)
    # semivariance ratio - downside vs upside minute variance
    up = gb((r.clip(lower=0)) ** 2).sum()
    dn = gb((r.clip(upper=0)) ** 2).sum()
    F["semi_ratio"] = (up - dn) / (up + dn).replace(0, np.nan)
    # signature ratio: 1m RV over 15m RV. >1 means microstructure noise /
    # mean reversion at the minute scale; <1 means trending inside the day.
    r15 = np.log(m["close"]).diff(15)
    rv15 = gb(r15 ** 2).sum() / 15.0
    F["sig_ratio"] = rv / rv15.replace(0, np.nan)
    F["r_skew"] = gb(r).skew()
    F["r_kurt"] = gb(r).apply(lambda s: s.kurt() if len(s) > 30 else np.nan)

    # --- trade intensity --------------------------------------------------
    F["n_trades"] = gb(n).sum()
    F["avg_size"] = gb(tot).sum() / gb(n).sum().replace(0, np.nan)
    F["intensity_ac"] = gb(n).apply(lambda s: s.autocorr(1) if len(s) > 30 else np.nan)
    # Amihud at minute resolution - price impact per dollar
    F["lam"] = gb(absr / tot).mean() * 1e9
    # where the close sat relative to the day's dollar-VWAP
    vwap = gb(m["close"] * tot).sum() / gb(tot).sum()
    F["close_vs_vwap"] = gb(m["close"]).last() / vwap - 1.0

    F["close"] = gb(m["close"]).last()
    F.index = F.index.tz_localize(None)
    F.to_parquet(cache)
    return F


def iv_features(index, cache=f"{D}/iv_daily.parquet", rebuild=False):
    if os.path.exists(cache) and not rebuild:
        return pd.read_parquet(cache).reindex(index)
    b = pd.read_parquet(f"{D}/bvol_1m.parquet")
    b["dt"] = pd.to_datetime(b.dt, utc=True)
    b = b.set_index("dt").sort_index()
    day = b.index.normalize()
    G = pd.DataFrame(index=pd.DatetimeIndex(sorted(set(day))))
    G["iv"] = b["iv"].groupby(day).mean()
    G["iv_close"] = b["iv"].groupby(day).last()
    G["iv_range"] = (b["iv_hi"].groupby(day).max() - b["iv_lo"].groupby(day).min())
    G["iv_vov"] = b["iv"].groupby(day).std()
    G.index = G.index.tz_localize(None)
    G.to_parquet(cache)
    return G.reindex(index)


def block_t(x, y, k=1):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 120:
        return np.nan, np.nan, int(m.sum())
    a = pd.Series(x[m]); b = pd.Series(y[m])
    ic = a.corr(b, method="spearman")
    n_eff = max(m.sum() / k, 3)
    return float(ic), float(ic * np.sqrt(n_eff - 2) / np.sqrt(max(1 - ic ** 2, 1e-9))), int(m.sum())


def phase_rand(s):
    a = np.asarray(s, float); m = np.isfinite(a)
    if m.sum() < 64:
        return pd.Series(np.nan, index=s.index)
    v = a[m]
    f = np.fft.rfft(v - v.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(f)); ph[0] = 0.0
    out = np.full(len(a), np.nan)
    out[m] = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(v)) + v.mean()
    return pd.Series(out, index=s.index)


def book(px, sig, target_vol=0.30, vol_win=60, max_lev=3.0, band=0.10):
    r = px.pct_change().fillna(0.0)
    rv = r.rolling(vol_win, min_periods=40).std() * np.sqrt(365.25)
    want = (sig * (target_vol / rv.replace(0, np.nan))).clip(-max_lev, max_lev).fillna(0.0)
    w = want.copy()
    cur = 0.0
    arr = want.to_numpy(); out = np.zeros(len(arr))
    for i in range(len(arr)):
        if abs(arr[i] - cur) > band or (arr[i] == 0.0 and cur != 0.0):
            cur = arr[i]
        out[i] = cur
    w = pd.Series(out, index=px.index)
    turn = w.diff().abs().fillna(w.abs())
    return w * r - turn * (FEE_BPS + SLIP_BPS) / 1e4, turn


def gate_of(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return np.nan
    g = at_gate(a, lo=1e-4, hi=80.0, iters=80)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


if __name__ == "__main__":
    print("S174 - minute-resolution microstructure, traded daily\n")
    F = minute_features()
    G = iv_features(F.index)
    F = pd.concat([F, G], axis=1)
    px = F["close"].dropna()
    F = F.loc[px.index]
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days "
          f"({len(px)/365.25:.1f} years)")
    cols = [c for c in F.columns if c != "close"]
    print(f"{len(cols)} features built from 1-minute bars\n")

    lr = np.log(px)
    print("INFORMATION CONTENT - IC against forward returns, t deflated for overlap")
    print(f"   {'feature':>15}{'cover':>7}" +
          "".join(f"{'IC'+str(k)+'d':>9}{'t':>7}" for k in (1, 5, 20)))
    surv = []
    for c in cols:
        # trailing z-score so the level is comparable through regimes, lagged
        x = F[c].astype(float)
        z = ((x - x.rolling(365, min_periods=120).mean())
             / (x.rolling(365, min_periods=120).std() + 1e-12)).clip(-3, 3).shift(1)
        row, best = "", (0, 0.0, 0.0)
        for k in (1, 5, 20):
            y = lr.diff(k).shift(-k)
            ic, t, n = block_t(z.to_numpy(), y.to_numpy(), k)
            row += (f"{ic:>9.4f}{t:>7.2f}" if np.isfinite(ic) else f"{'n/a':>16}")
            if np.isfinite(t) and abs(t) > abs(best[2]):
                best = (k, ic, t)
        print(f"   {c:>15}{x.notna().mean()*100:>6.0f}%{row}")
        if abs(best[2]) > 2.5:
            surv.append((c, best[0], best[1], best[2], z))

    print(f"\nfeatures with |t| > 2.5 at their best horizon: {len(surv)}")
    if not surv:
        print("   none. nothing to build."); sys.exit(0)

    print("\nSURROGATE CONTROL - 200 phase-randomised versions of each survivor")
    real = []
    for c, k, ic, t, z in sorted(surv, key=lambda s: -abs(s[3])):
        y = lr.diff(k).shift(-k)
        ts = []
        for _ in range(200):
            s = phase_rand(F[c].astype(float))
            zz = ((s - s.rolling(365, min_periods=120).mean())
                  / (s.rolling(365, min_periods=120).std() + 1e-12)).clip(-3, 3).shift(1)
            _, tt, _ = block_t(zz.to_numpy(), y.to_numpy(), k)
            if np.isfinite(tt):
                ts.append(abs(tt))
        ts = np.array(ts); pv = float((ts >= abs(t)).mean())
        tag = "REAL" if pv < 0.05 else "not distinguishable"
        if pv < 0.05:
            real.append((c, k, ic, t, z))
        print(f"   {c:>15} @{k:>2}d  |t| {abs(t):>5.2f}  surrogate median "
              f"{np.median(ts):>5.2f}, 95th {np.percentile(ts,95):>5.2f}  "
              f"p = {pv:.3f}  {tag}")

    print(f"\n{len(real)} features survive both. BOOKS:")
    print(f"   {'feature':>15}{'sign':>6}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>7}{'at -20%':>10}")
    books = {}
    for c, k, ic, t, z in real:
        sgn = np.sign(t)
        net, turn = book(px, (z * sgn).fillna(0.0))
        a = net.to_numpy(float); s = stats_of(a); g = gate_of(a)
        books[c] = net
        print(f"   {c:>15}{int(sgn):>+6}{s['sharpe']:>7.2f}{s['cagr']*100:>8.1f}%"
              f"{s['dd']*100:>7.1f}%{turn.mean():>7.2f}"
              + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}"))
    if books:
        pd.to_pickle(books, "/home/user/quant/results/s174_books.pkl")
        B = pd.DataFrame(books)
        print(f"\n   equal-weight combination of all {B.shape[1]}:")
        comb = B.mean(axis=1)
        a = comb.to_numpy(float); s = stats_of(a); g = gate_of(a)
        print(f"   {'combined':>15}{'':>6}{s['sharpe']:>7.2f}{s['cagr']*100:>8.1f}%"
              f"{s['dd']*100:>7.1f}%{'':>7}"
              + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}"))
    print("\ndone: microstructure screen")
