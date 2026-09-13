"""
S128 - On-chain data as a signal for a BTCUSDT-only book. The last data class.

HOW THIS DATA GOT HERE
----------------------
S126 concluded no new data was reachable. That was wrong twice over, and the
user was right to push. The container's egress is blocked and Binance 403s the
datacenter IP, but this session also has MCP servers attached which do not use
that path:

    Crypto.com      50 candles maximum. Useless for a backtest.
    LunarCrush      "limited data mode" - every value masked behind a
                    subscription. Dates, no numbers.
    Twelve Data     WORKS. 1,830 trading days of HYG, LQD, TLT, SPY, UUP.
                    Tested in S127 as macro risk appetite: nothing survived.
    Firecrawl       runs server-side, so it REACHES the blocked hosts. Verified
                    against Deribit's API (status 200, live JSON).

That last one is the route that matters, and this is what it was spent on.

WHY ON-CHAIN, GIVEN S124's REQUIREMENT
---------------------------------------
S124 computed what the brief needs: roughly three strategies of V7's quality
that do not correlate with it. Everything crypto-native shares the same beta -
that is why the hindsight oracle gained 4%. S127 tried macro, the first input
generated outside crypto, and found nothing.

On-chain is the last genuinely different generating process available: miners,
holders and network throughput, none of it mechanically linked to perpetual
futures positioning. Eight years of it, 2018-09 onward - longer than any
crypto-native series in this study.

SIX FEATURES, MECHANISM FIRST
-----------------------------
    hashribbon    30d vs 60d hash rate. Below one means miners are switching
                  off, which is the classic capitulation marker.
    miner_margin  revenue per unit of hash. Miners are forced sellers with
                  electricity bills; margin compression means supply.
    nvt           price against on-chain transaction value - the network's
                  price-to-earnings. High means the market is paying more per
                  unit of settled economic activity. NOT purely exogenous,
                  since price is in the numerator, and flagged as such.
    addr          30d growth in unique active addresses - adoption.
    throughput    30d growth in settled USD value.
    mempool       congestion, standardised. Urgency to settle at any fee.

Judged exactly as S125b and S127 were: both halves, then non-overlapping ICs
with a bootstrap interval, then phase-randomised surrogates. Nothing is built on
a feature that cannot beat its own noise.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

D = "/home/user/quant/data"
RNG = np.random.default_rng(23)
MR, HR = "Miners Revenue", "Hash Rate"
AD, TV, MP = ("Number Of Unique Addresses Used", "Estimated USD Transaction Value",
              "Mempool Size")


def panel():
    P = pd.read_parquet(f"{D}/onchain/onchain_daily.parquet")
    P.index = pd.to_datetime(P.index)
    if P.index.tz is None:
        P.index = P.index.tz_localize("UTC")

    d = pd.read_parquet(f"{D}/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    px = d.set_index("dt")["close"].resample("1D").last().dropna()
    ix = px.index.intersection(P.index)
    px, P = px.reindex(ix), P.reindex(ix)

    z = lambda s, w=365: (s - s.rolling(w, min_periods=90).mean()) / \
                         (s.rolling(w, min_periods=90).std() + 1e-12)
    F = pd.DataFrame(index=ix)
    F["hashribbon"] = (P[HR].rolling(30).mean() / P[HR].rolling(60).mean()) - 1.0
    F["miner_margin"] = z(np.log((P[MR] / P[HR].clip(lower=1e-9)).clip(lower=1e-12)))
    F["nvt"] = z(np.log((px / P[TV].clip(lower=1e-9)).clip(lower=1e-12)))
    F["addr"] = np.log(P[AD].clip(lower=1)).diff(30)
    F["throughput"] = np.log(P[TV].clip(lower=1)).diff(30)
    F["mempool"] = z(np.log(P[MP].clip(lower=1)))
    return px, F


COLS = ["hashribbon", "miner_margin", "nvt", "addr", "throughput", "mempool"]


def nonoverlap_ic(x, y, k, n_boot=2000):
    m = np.isfinite(x) & np.isfinite(y)
    xa, ya = x[m].to_numpy(), y[m].to_numpy()
    xs, ys = xa[::k], ya[::k]
    ic = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    n = len(xs); b = np.empty(n_boot)
    for i in range(n_boot):
        j = RNG.integers(0, n, n)
        b[i] = pd.Series(xs[j]).corr(pd.Series(ys[j]), method="spearman")
    return ic, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5), n


def phase_randomise(s):
    a = np.asarray(s, float); m = np.isfinite(a); v = a[m]
    f = np.fft.rfft(v - v.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(f)); ph[0] = 0.0
    out = np.full(len(a), np.nan)
    out[m] = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(v)) + v.mean()
    return pd.Series(out, index=s.index)


def halves(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    xa, ya = x[m], y[m]; h = len(xa) // 2
    a = pd.Series(xa.iloc[:h].to_numpy()).corr(pd.Series(ya.iloc[:h].to_numpy()), method="spearman")
    b = pd.Series(xa.iloc[h:].to_numpy()).corr(pd.Series(ya.iloc[h:].to_numpy()), method="spearman")
    return a, b


if __name__ == "__main__":
    px, F = panel()
    lr = np.log(px)
    print("S128 - on-chain as a BTCUSDT-only signal\n")
    print(f"daily, {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days "
          f"({len(px)/365.25:.1f} years) — the longest sample in this study\n")
    print(f"{'feature':>14}" + "".join(f"{'IC ' + str(k) + 'd':>10}" for k in (1, 3, 5, 10, 20))
          + f"{'IC5 1st':>10}{'IC5 2nd':>10}{'stable':>8}")
    keep = []
    for c in COLS:
        x = F[c].shift(1)
        ics = [pd.Series(x[np.isfinite(x) & np.isfinite(lr.diff(k).shift(-k))].to_numpy()).corr(
               pd.Series(lr.diff(k).shift(-k)[np.isfinite(x) & np.isfinite(lr.diff(k).shift(-k))].to_numpy()),
               method="spearman") for k in (1, 3, 5, 10, 20)]
        i1, i2 = halves(x, lr.diff(5).shift(-5))
        st = np.sign(i1) == np.sign(i2) and min(abs(i1), abs(i2)) > 0.03
        if st:
            keep.append(c)
        print(f"{c:>14}" + "".join(f"{v:>10.4f}" for v in ics)
              + f"{i1:>10.4f}{i2:>10.4f}{'YES' if st else '':>8}")

    print(f"\nsurvivors of the screen: {keep if keep else 'none'}")
    for c in keep:
        print(f"\n--- {c}: non-overlapping IC, bootstrap 95% interval")
        x = F[c].shift(1); any_sig = False
        for k in (1, 3, 5, 10, 20):
            ic, lo, hi, n = nonoverlap_ic(x, lr.diff(k).shift(-k), k)
            ex = (lo > 0) or (hi < 0); any_sig = any_sig or ex
            print(f"    {k:>3}d  n={n:>5}  IC {ic:>+7.4f}  [{lo:>+7.4f},{hi:>+7.4f}]  "
                  f"{'EXCLUDES 0' if ex else 'includes 0'}")
        i1, i2 = halves(x, lr.diff(5).shift(-5)); real = min(abs(i1), abs(i2))
        p = 0
        for _ in range(300):
            s = phase_randomise(F[c]).shift(1)
            a1, a2 = halves(s, lr.diff(5).shift(-5))
            v = min(abs(a1), abs(a2)) if np.sign(a1) == np.sign(a2) else 0.0
            if v >= real:
                p += 1
        print(f"    weaker-half |IC| {real:.4f}; surrogates matching/beating: "
              f"{p}/300 = p {p/300:.3f}")
        print(f"    verdict: {'SURVIVES' if (any_sig and p/300 < 0.05) else 'NOISE'}")
    print("\ndone: on-chain diagnostic")
