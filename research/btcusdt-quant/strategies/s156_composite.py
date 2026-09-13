"""
S156 - Converting a t = 14 signal into Sharpe.

THE DISCONNECT THIS FILE EXISTS TO CLOSE
-----------------------------------------
S155 screened the honest 683-symbol universe and found a clean set of real
signals, every one of them surviving a phase-randomised control:

    vol30  t 14.08   vol90  t 13.65        (low volatility wins)
    mom90  t -4.80   mom30  t -4.58        (past winners lose - contrarian)
    rev10  t  4.52   srev5  t  4.42        (short-term reversal)
    amihud t  8.56 -> p = 1.000            (killed by the control, again)

And yet the low-vol BOOK in S154 ran at Sharpe 0.26. A t-statistic of 14 and a
Sharpe of 0.26 is not a contradiction, it is a construction failure, and there
are three specific reasons for it:

  SKEW      shorting the high-volatility leg means shorting exactly the assets
            with enormous positive skew. The book wins most days and loses
            violently, which is a low-Sharpe payoff no matter how good the
            ranking is.
  SIZING    an equal-dollar leg puts the same money in a 200%-vol coin as in a
            40%-vol one, so a handful of names supply most of the risk and the
            ranking of the rest barely matters.
  HORIZON   the features live at different horizons - vol at 20 days, reversal
            at 1 - and a single rebalance frequency serves one and wastes the
            other.

So this file changes the construction rather than the signal:

    score       equal-weighted mean of the z-scored validated features, which
                is what S106b showed beats informed weighting in this study
    sizing      RISK parity inside the legs: weight proportional to score over
                the asset's own volatility, so each name contributes comparable
                risk rather than comparable dollars
    neutrality  demeaned across names, then beta-neutralised against the
                cross-sectional market
    limits      5% gross per name, causal, and a full haircut on gains in any
                dying contract's last 10 days
    horizon     rebalance frequency swept rather than assumed

Nothing about the signal is re-fitted. The features are exactly the ones S155
validated, with the signs S155 measured.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s151_pit as PIT
import strategies.s148_lowvol as L
import strategies.s153_honest as H
import strategies.s154_stack as ST
from strategies.s96_rank import at_gate, stats_of

# feature -> (sign that makes it higher-is-better-to-be-long), as MEASURED by
# S155, not as guessed.
VALIDATED = {"vol30": +1, "vol90": +1, "mom30": -1, "mom90": -1,
             "rev5": +1, "rev10": +1, "srev3": +1, "srev5": +1}


def raw_features(px, ok, fund):
    lr = np.log(px); r1 = lr.diff(1)
    sd = r1.rolling(30, min_periods=20).std()
    F = {
        "vol30": -r1.rolling(30, min_periods=20).std(),
        "vol90": -r1.rolling(90, min_periods=60).std(),
        "mom30": lr.diff(30),
        "mom90": lr.diff(90),
        "rev5": -lr.diff(5),
        "rev10": -lr.diff(10),
        "srev3": -(lr.diff(3) / (sd * np.sqrt(3) + 1e-12)),
        "srev5": -(lr.diff(5) / (sd * np.sqrt(5) + 1e-12)),
    }
    return {k: v.where(ok) for k, v in F.items()}


def zrank(f, ok):
    """Cross-sectional rank mapped to a standard normal scale, robust to the
    outliers that raw z-scores are not."""
    r = f.where(ok).rank(axis=1, pct=True)
    n = ok.sum(axis=1)
    r = r.sub(0.5).mul(2.0)                 # -1..1, uniform
    return r.where(ok).mul(np.sqrt(3.0))    # unit variance


def composite(F, use, ok):
    parts = [zrank(F[k] * s, ok) for k, s in use.items() if k in F]
    return sum(parts) / len(parts)


def make_weights(score, px, ok, beta=None, risk_parity=True, vol_win=30,
                 neutral="beta"):
    """Score -> weights. Risk parity inside the legs, then neutralised."""
    s = score.where(ok)
    s = s.sub(s.mean(axis=1), axis=0).where(ok)
    if risk_parity:
        rv = px.pct_change().rolling(vol_win, min_periods=20).std()
        rv = rv.where(ok).clip(lower=rv.stack().quantile(0.05))
        s = (s / rv).where(ok)
    s = s.sub(s.mean(axis=1), axis=0).where(ok)          # dollar neutral
    if neutral == "beta" and beta is not None:
        b = beta.where(ok)
        num = (s * b).sum(axis=1)
        den = (b * b).sum(axis=1).replace(0, np.nan)
        s = (s - b.mul(num / den, axis=0)).where(ok)     # beta neutral
    g = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(g, axis=0).fillna(0.0)


def rebalance(w, every):
    """Hold weights for `every` days: only rows on the grid are refreshed."""
    if every <= 1:
        return w
    m = pd.Series(np.arange(len(w)) % every == 0, index=w.index)
    return w.where(m, np.nan).ffill().fillna(0.0)


if __name__ == "__main__":
    pan = PIT.build()
    px = pan["close"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = H.pit_funding(list(px.columns), px.index)
    dead = ST.death_window(px)
    r1 = np.log(px).diff(1); mkt = r1.mean(axis=1)
    beta = (r1.rolling(180, min_periods=90).cov(mkt)
            .div(mkt.rolling(180, min_periods=90).var(), axis=0)).where(ok).clip(0.3, 3.0)
    F = raw_features(px, ok, fund)

    print("S156 - converting the signal into Sharpe\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days, "
          f"median {int(ok.sum(axis=1).median())} tradeable/day")
    print(f"universe {px.shape[1]} symbols ever listed; "
          f"{int(dead.any().sum())} of them die in sample\n")

    print("1. ONE FACTOR AT A TIME, new construction, 5bps+5bps, 5% name limit")
    print(f"   {'factor':>22}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}{'at -20%':>9}")
    singles = {}
    for k, s in VALIDATED.items():
        w = make_weights(zrank(F[k] * s, ok), px, ok, beta)
        net, turn = ST.book(px, w, None, dead=dead, haircut=1.0, max_w=0.05)
        H.show(net, turn, k, w=22)
        singles[k] = net

    print("\n2. THE COMPOSITE - equal weight over all 8 validated features")
    print(f"   {'variant':>22}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}{'at -20%':>9}")
    sc = composite(F, VALIDATED, ok)
    for rp in (False, True):
        for neu in ("none", "beta"):
            w = make_weights(sc, px, ok, beta, risk_parity=rp, neutral=neu)
            net, turn = ST.book(px, w, None, dead=dead, haircut=1.0, max_w=0.05)
            H.show(net, turn, f"{'riskparity' if rp else 'equaldollar'}/{neu}", w=22)

    print("\n3. REBALANCE FREQUENCY - reversal wants speed, low-vol does not")
    print(f"   {'hold (days)':>22}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    w0 = make_weights(sc, px, ok, beta)
    for ev in (1, 2, 3, 5, 10, 20):
        net, turn = ST.book(px, rebalance(w0, ev), None, dead=dead, haircut=1.0,
                            max_w=0.05)
        H.show(net, turn, f"{ev}", w=22)

    print("\n4. TWO FAMILIES SEPARATELY, each at its own speed, then combined")
    slow = {k: v for k, v in VALIDATED.items() if k in ("vol30", "vol90", "mom30", "mom90")}
    fast = {k: v for k, v in VALIDATED.items() if k in ("rev5", "rev10", "srev3", "srev5")}
    print(f"   {'book':>22}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}{'at -20%':>9}")
    ws = make_weights(composite(F, slow, ok), px, ok, beta)
    wf = make_weights(composite(F, fast, ok), px, ok, beta)
    n_s, t_s = ST.book(px, rebalance(ws, 10), None, dead=dead, haircut=1.0, max_w=0.05)
    n_f, t_f = ST.book(px, wf, None, dead=dead, haircut=1.0, max_w=0.05)
    H.show(n_s, t_s, "slow (vol+contrarian)", w=22)
    H.show(n_f, t_f, "fast (reversal)", w=22)
    print(f"   correlation between them: {n_s.corr(n_f):+.3f}")

    def at_vol(s, t=0.30):
        sd = float(np.std(s.to_numpy(float))) * np.sqrt(365.25)
        return s * (t / sd) if sd > 0 else s * 0.0
    Z = pd.DataFrame({"slow": at_vol(n_s), "fast": at_vol(n_f)}).dropna()
    z = pd.Series(0.0, index=Z.index)
    H.show(Z.mean(axis=1), z, "50/50 combination", w=22)

    print("\n5. COST SENSITIVITY on the combination")
    print(f"   {'slippage/side':>22}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    for sl in (2.0, 5.0, 10.0, 20.0):
        a, ta = ST.book(px, rebalance(ws, 10), None, dead=dead, haircut=1.0,
                        max_w=0.05, slip_bps=sl)
        b, tb = ST.book(px, wf, None, dead=dead, haircut=1.0, max_w=0.05,
                        slip_bps=sl)
        c = pd.DataFrame({"s": at_vol(a), "f": at_vol(b)}).dropna().mean(axis=1)
        H.show(c, (ta + tb) / 2, f"{sl:.0f}bps", w=22)
    print("\ndone: composite")
