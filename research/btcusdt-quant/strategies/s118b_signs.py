"""
S118b - Let the data set the signs, causally, instead of me guessing them.

S118 averaged nine standardised crowding features with signs I assigned from
economic reasoning. An IC check afterwards says I got three of them wrong:

    feature        IC (my sign)   1st half   2nd half
    tt_pos              -0.015     -0.011     -0.016     near zero either way
    taker               -0.035     -0.029     -0.039     I faded it; it is momentum
    basis               -0.021     -0.026     -0.018     I faded it; it is momentum
    btc_dom             +0.012     +0.054     -0.020     FLIPS between halves

So the equal-weight average was carrying three features pointed backwards and
one that reverses. That is a defect in my construction, and it is worth roughly
whatever those features are worth.

**The wrong fix is to flip them now.** Their signs were measured on the whole
sample, and setting them from that is the in-sample fitting this log keeps
catching - S113 turned a +9.9 in-sample edge into -48.8 when it was finally
chosen honestly.

The right fix is to let each feature's sign be decided by a **trailing** IC,
using only data before the day it is applied to, refreshed as it goes. A feature
whose trailing relationship is too weak to call is dropped for that period
rather than voted with. Nine low-dimensional binary choices from a year of data
is a far smaller selection problem than S109's 200-cell grid, but it is still a
selection, so the random control and both halves are reported.

Everything else is unchanged from S118: continuous signal, vol-targeted sizing,
no stops, daily rebalance, funding and costs charged, judged standalone.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s118_crowd as S
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

ICWIN = 365          # days of trailing history used to decide a sign
ICMIN = 0.01         # below this the feature is not voted with at all
REFIT = 30           # days between sign refreshes


def zraw(F, c, zwin=S.ZWIN):
    x = F[c].astype(float)
    return ((x - x.rolling(zwin, min_periods=zwin // 3).mean())
            / (x.rolling(zwin, min_periods=zwin // 3).std() + 1e-12)).shift(1)


def causal_signal(px, F, cols, icwin=ICWIN, icmin=ICMIN, refit=REFIT, fixed=None):
    """Equal-weight mean of standardised features, each signed by TRAILING IC.

    At every refit date the sign of each feature is taken from the rank
    correlation between its own past z-score and the next day's realised return,
    over the preceding `icwin` days only. Nothing after the refit date is used,
    so the sign in force on any given day was decidable on that day.
    """
    Z = pd.DataFrame({c: zraw(F, c) for c in cols})
    fwd = np.log(px / px.shift(1)).shift(-1)      # the label the IC is measured on
    n = len(px)
    # Built as a numpy array and wrapped at the end. `signs[c].iloc[i] = v` on a
    # DataFrame writes to a temporary under pandas 3 copy-on-write and is
    # silently discarded - the first version of this did exactly that, every
    # sign stayed zero, no position was ever taken, and the whole table printed
    # 0.0% without a single error.
    A = np.zeros((n, len(cols)))
    cur = np.array([float(fixed[c]) for c in cols]) if fixed else np.zeros(len(cols))
    for i in range(n):
        if fixed is None and i >= icwin and i % refit == 0:
            lo = i - icwin
            y = fwd.iloc[lo:i]
            for j, c in enumerate(cols):
                z = Z[c].iloc[lo:i]
                m = np.isfinite(z) & np.isfinite(y)
                if m.sum() < icwin // 3:
                    cur[j] = 0.0
                    continue
                ic = pd.Series(z[m].to_numpy()).corr(
                    pd.Series(y[m].to_numpy()), method="spearman")
                cur[j] = 0.0 if not np.isfinite(ic) or abs(ic) < icmin else np.sign(ic)
        A[i] = cur
    signs = pd.DataFrame(A, index=px.index, columns=cols)
    contrib = (np.clip(Z, -S.CLIP, S.CLIP) * signs)
    live = (signs != 0).sum(axis=1).replace(0, np.nan)
    return (contrib.sum(axis=1) / live).fillna(0.0), signs


def report(tag, r, turns):
    a = np.asarray(r, float); a = a[np.isfinite(a)]
    st = stats_of(a); b = bootstrap_dd(a, n=3000, block=90); g = at_gate(a)
    h = len(a) // 2
    g1, g2 = at_gate(a[:h])["cagr"] * 100, at_gate(a[h:])["cagr"] * 100
    print(f"{tag:>34}{st['cagr']*100:8.1f}%{st['dd']*100:8.1f}%{b['dd_median']*100:9.1f}%"
          f"{st['sharpe']:7.2f}{st['calmar']:7.2f}{turns:7d}{g['cagr']*100:9.1f}%"
          f"{g1:9.1f}%{g2:9.1f}%", flush=True)
    return g["cagr"] * 100


if __name__ == "__main__":
    px, fd, F = S.panel()
    cols = [c for c in S.SIGNS if c in F.columns]
    start = F[cols].dropna(thresh=len(cols) - 1).index.min()
    px, fd, F = px[px.index >= start], fd[fd.index >= start], F[F.index >= start]
    print(f"daily, {px.index.min().date()} -> {px.index.max().date()}, "
          f"{len(px)} days. Signs from a trailing {ICWIN}d IC, refreshed every "
          f"{REFIT}d, |IC| < {ICMIN} sits out.\n")

    sig_c, signs = causal_signal(px, F, cols)
    sig_f, _ = causal_signal(px, F, cols, fixed=S.SIGNS)

    print("how often each feature was voted with, and which way:")
    for c in cols:
        s = signs[c]
        print(f"   {c:>14}  long-side {100*(s > 0).mean():3.0f}%   "
              f"short-side {100*(s < 0).mean():3.0f}%   "
              f"sat out {100*(s == 0).mean():3.0f}%   (my guess: {S.SIGNS[c]:+d})")

    print(f"\n{'book':>34}{'CAGR':>9}{'realDD':>8}{'medDD':>9}{'Shp':>7}{'Clm':>7}"
          f"{'turns':>7}{'at -20%':>9}{'1st h':>9}{'2nd h':>9}")
    for tv in (0.20, 0.40, 0.60):
        report(f"FIXED signs (S118), tgt {tv*100:.0f}%", *S.run(px, fd, sig_f, tv))
    print()
    for tv in (0.20, 0.40, 0.60, 0.80):
        report(f"CAUSAL signs, tgt {tv*100:.0f}%", *S.run(px, fd, sig_c, tv))

    print("\ncontrol: random signs, redrawn every refit, 20 draws at tgt 40%")
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(20):
        rs = signs.copy()
        for c in cols:
            flip = rng.choice([-1.0, 1.0])
            rs[c] = rs[c].abs() * flip
        contrib = (np.clip(pd.DataFrame({c: zraw(F, c) for c in cols}),
                           -S.CLIP, S.CLIP) * rs)
        live = (rs != 0).sum(axis=1).replace(0, np.nan)
        sg = (contrib.sum(axis=1) / live).fillna(0.0)
        vals.append(at_gate(np.asarray(S.run(px, fd, sg, 0.40)[0], float))["cagr"] * 100)
    v = np.array(vals)
    print(f"{'random signs':>34}{'':49}{np.median(v):9.1f}%   "
          f"(mean {v.mean():.1f}, sd {v.std(ddof=1):.1f}, max {v.max():.1f})")
    print("\ndone: causal signs")
