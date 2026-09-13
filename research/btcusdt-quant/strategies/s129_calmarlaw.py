"""
S129 - Is Calmar really just Sharpe squared? Challenging S124's own bound.

S124 concluded the brief needs Sharpe ~3.27 and that nothing here reaches it.
That conclusion rests entirely on one empirical claim, asserted from three data
points and never tested:

    Calmar_at_the_gate  ~  1.4 x Sharpe^2

If that is the law, the bound is real and the target needs strategies nobody in
this study has built. **But Calmar is a ratio of return to the single worst
excursion, and the worst excursion is a path property, not a moment property.**
Two books with identical Sharpe can have very different drawdowns if one loses in
short sharp bursts and the other bleeds for eight months, or if one has fat right
tails and the other fat left tails.

So the honest question is not "can I find Sharpe 3.27" but:

    what ACTUALLY determines Calmar, and is there a corner of the space with
    high Calmar at attainable Sharpe?

If skew or loss-clustering carries real weight, then a book at Sharpe 2.1 with
the right SHAPE could reach Calmar 15, and this study has a design target rather
than a wall. If Sharpe explains everything, the bound is confirmed properly
instead of asserted, which is worth knowing too.

METHOD
------
Generate a large population of genuinely different return streams from every
signal source this study has - crowding, trend, macro, on-chain, price, and
random controls - across thresholds, horizons, directions and vol targets. For
each, measure:

    sharpe        the moment everyone quotes
    skew, kurt    shape of the daily distribution
    ac1           autocorrelation of returns: does it trend or chop
    dd_dur        mean drawdown DURATION in days - the path property
    downside      fraction of total variance coming from down days
    calmar        at the gate, with S124's guard

then regress log Calmar on log Sharpe and see what the residual is made of. A
random-signal control population is included so the relationship is measured
across real and null books alike rather than only where edges exist.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from scipy import stats as sps

from strategies.s96_rank import at_gate, stats_of

FEE, SLIP = 5.0, 3.0
RNG = np.random.default_rng(5)


def gate(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


def dd_duration(a):
    """Mean length in days of the drawdown episodes a book actually spends in."""
    e = np.cumprod(1.0 + np.asarray(a, float))
    peak = np.maximum.accumulate(e)
    under = e < peak * (1 - 1e-12)
    runs, n = [], 0
    for u in under:
        if u:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return float(np.mean(runs)) if runs else 0.0


def shape(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    st = stats_of(a)
    dn = a[a < 0]
    return dict(sharpe=st["sharpe"], skew=float(sps.skew(a)),
                kurt=float(sps.kurtosis(a)),
                ac1=float(pd.Series(a).autocorr(1)) if a.std() > 0 else 0.0,
                dd_dur=dd_duration(a),
                downside=float((dn ** 2).sum() / max((a ** 2).sum(), 1e-18)),
                winrate=float((a > 0).mean()))


def book(px, fund, sig, tv, thr, vol_hl, max_lev=3.0, band=0.10, longonly=False):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = np.maximum(r.ewm(halflife=vol_hl, adjust=False).std()
                    .shift(1).bfill().to_numpy() * np.sqrt(365.25), 0.05)
    s = np.asarray(sig, float).copy()
    s[np.abs(s) < thr] = 0.0
    if longonly:
        s = np.clip(s, 0, None)
    want = np.clip(s * tv / rv, -max_lev, max_lev)
    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * px.pct_change().fillna(0.0).to_numpy() \
        - pos * fund.to_numpy() - turn * (FEE + SLIP) / 1e4
    return pd.Series(net, index=px.index)


def z(s, w):
    return ((s - s.rolling(w, min_periods=max(20, w // 3)).mean())
            / (s.rolling(w, min_periods=max(20, w // 3)).std() + 1e-12)).clip(-2, 2)


def population():
    """Every signal family in this study, swept. Plus random controls."""
    import strategies.s119_dev as S119
    import strategies.s118_crowd as C
    px, fund, F = S119.build("1D")
    cols = [c for c in C.SIGNS if c in F.columns]

    sigs = {}
    for w in (90, 180, 365):
        sigs[f"crowd{w}"] = S119.signal(F, cols, w)
    lp = np.log(px)
    for fast, slow in ((8, 32), (16, 64), (32, 128), (50, 200)):
        sigs[f"trend{fast}_{slow}"] = np.sign(
            (lp.ewm(span=fast, adjust=False).mean()
             - lp.ewm(span=slow, adjust=False).mean())).shift(1).fillna(0.0)
    for w in (20, 60):
        sigs[f"revert{w}"] = (-z(lp.diff(w), 180)).shift(1).fillna(0.0)
    try:
        import strategies.s128_onchain as OC
        _, FO = OC.panel()
        for c in ("mempool", "hashribbon", "miner_margin"):
            sigs[f"oc_{c}"] = z(FO[c], 365).reindex(px.index).shift(1).fillna(0.0)
    except Exception as e:
        print("  (on-chain unavailable:", e, ")")
    try:
        import strategies.s127_macro as MA
        _, FM = MA.panel()
        for c in ("rates", "vix_lvl", "credit"):
            sigs[f"mc_{c}"] = z(FM[c], 365).reindex(px.index).shift(1).fillna(0.0)
    except Exception as e:
        print("  (macro unavailable:", e, ")")
    for i in range(6):                      # null controls, same autocorrelation
        n = len(px)
        w = pd.Series(RNG.normal(0, 1, n), index=px.index).rolling(30).mean()
        sigs[f"rand{i}"] = z(w, 180).shift(1).fillna(0.0)

    rows = []
    for name, s in sigs.items():
        s = pd.Series(np.asarray(s, float), index=px.index).fillna(0.0)
        for thr in (0.0, 0.3, 0.6):
            for tv in (0.20, 0.40):
                for hl in (16, 64):
                    for lo in (False, True):
                        nb = book(px, fund, s, tv, thr, hl, longonly=lo)
                        g = gate(nb.to_numpy())
                        if not np.isfinite(g) or g <= 0:
                            continue
                        d = shape(nb.to_numpy())
                        d.update(name=name, thr=thr, tv=tv, hl=hl, lo=lo,
                                 calmar=g / 20.0, gate=g,
                                 fam="random" if name.startswith("rand") else "real")
                        rows.append(d)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    P = population()
    P = P[(P.sharpe > 0.05) & np.isfinite(P.calmar)]
    print(f"S129 - what actually determines Calmar?\n")
    print(f"{len(P)} viable books from {P.name.nunique()} signal sources "
          f"({(P.fam=='random').sum()} from random controls)\n")

    x, y = np.log(P.sharpe.to_numpy()), np.log(P.calmar.to_numpy())
    b, a, r, _, _ = sps.linregress(x, y)
    print(f"log Calmar = {a:.3f} + {b:.3f} log Sharpe     R^2 = {r**2:.3f}")
    print(f"  -> Calmar = {np.exp(a):.2f} x Sharpe^{b:.2f}   "
          f"(S124 asserted 1.4 x Sharpe^2)")
    need = np.exp((np.log(15.0) - a) / b)
    print(f"  -> Calmar 15 needs Sharpe {need:.2f}\n")

    resid = y - (a + b * x)
    print("what explains the RESIDUAL — the part Sharpe does not:")
    print(f"   {'driver':>12}{'corr w/ resid':>16}{'partial R^2':>14}")
    for c in ("skew", "kurt", "ac1", "dd_dur", "downside", "winrate"):
        v = P[c].to_numpy()
        cc = float(np.corrcoef(v, resid)[0, 1])
        print(f"   {c:>12}{cc:>16.3f}{cc**2:>14.3f}")

    print(f"\nbest books by Calmar, and their shape:")
    print(f"   {'signal':>16}{'Shp':>7}{'Calmar':>8}{'gate':>9}{'skew':>7}"
          f"{'ddDur':>8}{'ac1':>7}")
    for _, r2 in P.nlargest(8, "calmar").iterrows():
        print(f"   {str(r2['name']):>16}{float(r2['sharpe']):>7.2f}"
              f"{float(r2['calmar']):>8.2f}{float(r2['gate']):>8.1f}%"
              f"{float(r2['skew']):>7.2f}{float(r2['dd_dur']):>8.1f}"
              f"{float(r2['ac1']):>7.3f}")

    hi = P[P.calmar >= 8]
    print(f"\nbooks reaching Calmar 8+: {len(hi)}"
          + (f"   (max Sharpe among them {hi.sharpe.max():.2f})" if len(hi) else ""))
    print(f"books reaching Calmar 15 (the brief): {(P.calmar >= 15).sum()}")
    # ---- where do the REAL reference books sit against this law? ----
    print("\n" + "="*70)
    print("THE RESIDUAL IS THE LEVER. Reference books vs the fitted line:")
    print(f"   {'book':>12}{'Sharpe':>8}{'predicted':>11}{'actual':>9}{'ratio':>8}")
    import strategies.s124_ceiling as CC
    S = CC.streams()
    ref = []
    for k, v in S.items():
        arr = np.asarray(v, float); arr = arr[np.isfinite(arr)]
        g = gate(arr)
        if not np.isfinite(g) or g <= 0:
            continue
        sh = stats_of(arr)["sharpe"]
        if sh <= 0.05:
            continue
        pred = np.exp(a) * sh ** b
        act = g / 20.0
        ref.append((k, sh, pred, act, act / pred))
        print(f"   {k:>12}{sh:>8.2f}{pred:>11.2f}{act:>9.2f}{act/pred:>8.2f}x")

    print("\n   population books, ratio of actual to predicted Calmar:")
    P = P.copy()
    P["ratio"] = P.calmar / (np.exp(a) * P.sharpe ** b)
    for fam, g2 in P.groupby(P["name"].str.replace(r"\d+.*$", "", regex=True)):
        print(f"   {str(fam):>16}  n={len(g2):>4}  median ratio {g2.ratio.median():.2f}x"
              f"  best {g2.ratio.max():.2f}x")
    print("\ndone: calmar law")
