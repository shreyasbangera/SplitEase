"""
S125b - Is `rotation` real, or is it the one feature in eight that got lucky?

S125 tested eight cross-asset features and one kept its sign across both halves:
rotation - alt top-trader positioning minus BTC's own - with a 5-day IC of
+0.0425 overall, +0.0317 then +0.0618, and rising with horizon to +0.0696 at 10
days. Unusually for this study it is STRONGER in the second half.

Two reasons not to take that at face value, both of which this log has been
caught by before:

  THE OVERLAP        a 10-day IC measured on daily observations uses windows
                     that overlap nine days out of ten. The effective sample is
                     roughly N/10, not N, so the standard error is about three
                     times what a naive count suggests and the apparent
                     significance at longer horizons is largely manufactured.
                     At 1,728 days the naive standard error on a Spearman IC is
                     0.024, so +0.0425 is t ~ 1.8 - already marginal before the
                     overlap correction, and worse after it.

  THE SEARCH         eight features were tested. The survivor bar - same sign in
                     both halves, |IC| > 0.02 in each - is under one standard
                     error per half, so one survivor out of eight is close to
                     what pure noise produces. Surviving a weak filter is not
                     evidence.

So this file does three things properly rather than trusting the screen:

  1  NON-OVERLAPPING ICs with a bootstrap confidence interval, so the
     significance is honest about its own sample size.
  2  A RANDOM-FEATURE CONTROL: the same screen applied to phase-randomised
     features that preserve the autocorrelation structure of the real ones but
     carry no relationship to returns. If noise clears the bar as often as
     rotation did, the screen proved nothing.
  3  THE BOOK, if and only if it survives both - judged on the brief, standalone,
     and on its correlation to what already exists, because S124 showed the only
     thing worth adding now is a stream that is both strong AND uncorrelated.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s125_cross as X
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

FEE_BPS, SLIP_BPS = 5.0, 3.0
RNG = np.random.default_rng(7)


def nonoverlap_ic(x, y, k, n_boot=2000):
    """IC on non-overlapping k-day windows, with a bootstrap interval."""
    m = np.isfinite(x) & np.isfinite(y)
    xa, ya = x[m].to_numpy(), y[m].to_numpy()
    xs, ys = xa[::k], ya[::k]                       # every k-th, no overlap
    ic = pd.Series(xs).corr(pd.Series(ys), method="spearman")
    b = np.empty(n_boot)
    n = len(xs)
    for i in range(n_boot):
        j = RNG.integers(0, n, n)
        b[i] = pd.Series(xs[j]).corr(pd.Series(ys[j]), method="spearman")
    return ic, np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5), n


def phase_randomise(s):
    """Surrogate with the same power spectrum - so the same autocorrelation and
    the same trending, wandering character - but no link to returns."""
    a = np.asarray(s, float)
    m = np.isfinite(a)
    v = a[m]
    f = np.fft.rfft(v - v.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0.0
    out = np.full(len(a), np.nan)
    out[m] = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(v)) + v.mean()
    return pd.Series(out, index=s.index)


def run(px, sig, target_vol=0.20, thr=0.0, max_lev=3.0, band=0.10, vol_hl=32):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = np.maximum(r.ewm(halflife=vol_hl, adjust=False).std()
                    .shift(1).bfill().to_numpy() * np.sqrt(365.25), 0.05)
    s = np.asarray(sig, float).copy()
    s[np.abs(s) < thr] = 0.0
    want = np.clip(s * target_vol / rv, -max_lev, max_lev)
    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur
    cost = (FEE_BPS + SLIP_BPS) / 1e4
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * px.pct_change().fillna(0.0).to_numpy() - turn * cost
    live = pos != 0
    trades = int(((~live[:-1]) & live[1:]).sum() + (1 if live[0] else 0))
    return pd.Series(net, index=px.index), trades, pos


def gate_of(a, tol=0.01):
    """S124's guard: never report a gate the bisection did not actually reach."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        return np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


if __name__ == "__main__":
    px, F = X.build()
    lr = np.log(px)
    rot = F["rotation"].shift(1)

    print("S125b - is rotation real?\n")
    print("1. NON-OVERLAPPING IC, bootstrap 95% interval")
    print(f"   {'horizon':>9}{'obs':>7}{'IC':>9}{'95% low':>10}{'95% high':>10}"
          f"{'excl 0':>9}")
    sig_ok = False
    for k in (1, 3, 5, 10):
        y = lr.diff(k).shift(-k)
        ic, lo, hi, n = nonoverlap_ic(rot, y, k)
        ex = (lo > 0) or (hi < 0)
        sig_ok = sig_ok or ex
        print(f"   {k:>8}d{n:>7}{ic:>9.4f}{lo:>10.4f}{hi:>10.4f}"
              f"{'YES' if ex else 'no':>9}")

    print("\n2. CONTROL: the same screen on 200 phase-randomised surrogates")
    print("   (same autocorrelation, no relationship to returns)")
    y5 = lr.diff(5).shift(-5)
    passes, ics = 0, []
    for _ in range(200):
        s = phase_randomise(F["rotation"]).shift(1)
        m = np.isfinite(s) & np.isfinite(y5)
        xa, ya = s[m], y5[m]
        hh = len(xa) // 2
        i1 = pd.Series(xa.iloc[:hh].to_numpy()).corr(
            pd.Series(ya.iloc[:hh].to_numpy()), method="spearman")
        i2 = pd.Series(xa.iloc[hh:].to_numpy()).corr(
            pd.Series(ya.iloc[hh:].to_numpy()), method="spearman")
        ics.append(abs((i1 + i2) / 2))
        if np.sign(i1) == np.sign(i2) and abs(i1) > 0.02 and abs(i2) > 0.02:
            passes += 1
    print(f"   surrogates clearing the S125 survivor bar: {passes}/200 "
          f"({passes/2:.0f}%)")
    print(f"   real rotation |mean half-IC| = {abs((0.0317+0.0618)/2):.4f}, "
          f"surrogate median {np.median(ics):.4f}, "
          f"95th pct {np.percentile(ics, 95):.4f}")

    print("\n3. THE BOOK — built regardless, so the verdict rests on money "
          "rather than on an IC")
    z = ((rot - rot.rolling(365, min_periods=120).mean())
         / (rot.rolling(365, min_periods=120).std() + 1e-12)).clip(-2, 2).fillna(0.0)
    print(f"   {'variant':>24}{'Shp':>7}{'realDD':>8}{'trd':>6}{'PF':>7}"
          f"{'at -20%':>9}{'1st h':>8}{'2nd h':>8}")
    best = None
    for thr in (0.0, 0.25, 0.50):
        for tv in (0.20, 0.40):
            net, trd, pos = run(px, z, tv, thr)
            a = net.to_numpy()
            st = stats_of(a)
            g = gate_of(a)
            h = len(a) // 2
            g1, g2 = gate_of(a[:h]), gate_of(a[h:])
            live = pos != 0
            pnl, acc, on = [], 0.0, False
            for i in range(len(a)):
                if live[i]:
                    acc += a[i]; on = True
                elif on:
                    pnl.append(acc); acc, on = 0.0, False
            if on:
                pnl.append(acc)
            p = np.array(pnl)
            pf = (p[p > 0].sum() / -p[p < 0].sum()) if (p < 0).any() else np.inf
            gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
            print(f"   {'thr ' + format(thr, '.2f') + ', tgt ' + format(tv*100, '.0f') + '%':>24}"
                  f"{st['sharpe']:>7.2f}{st['dd']*100:>7.1f}%{trd:>6}{pf:>7.2f}{gs:>9}"
                  f"{g1:>7.1f}%{g2:>7.1f}%")
            if np.isfinite(g) and (best is None or g > best[0]):
                best = (g, net, st["sharpe"])

    if best:
        print("\n4. CORRELATION TO WHAT ALREADY EXISTS — S124 showed only a "
              "strong AND uncorrelated stream is worth anything")
        import strategies.s124_ceiling as C
        S = C.streams()
        nb = best[1]
        nb.index = pd.to_datetime(nb.index)
        nb.index = nb.index.tz_localize(None) if nb.index.tz else nb.index
        for k, v in S.items():
            j = nb.index.intersection(v.index)
            if len(j) > 200:
                print(f"   vs {k:>10}: {nb.reindex(j).corr(v.reindex(j)):+.3f} "
                      f"({len(j)} days)")
        print(f"\n   this stream: Sharpe {best[2]:.2f}, gate {best[0]:.1f}%. "
              f"S124's requirement for the brief is Sharpe ~2.13 at low "
              f"correlation.")
    print("\ndone: rotation")
