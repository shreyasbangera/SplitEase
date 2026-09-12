"""
S111b - Sizing inversely to predicted excursion. The cheapest version first.

S111 measured what V7's adverse excursion actually looks like, in units of the
stop distance, so it is already ATR-normalised and anything left is information
ATR does not carry:

    median 0.19 R, p90 0.62 R, mean 0.267 R, above 1.0 R on 1.9% of trades

and found residual predictability with no model at all - the **trailing mean of
the last 20 trades' MAE** correlates +0.164 with the next trade's, decaying to
+0.036 by a 200-trade window. That is volatility clustering surviving the ATR
normalisation, and it is a stronger and far more robust signal than the 0.537 AUC
S110 could reach on direction.

Two ways to use it, deliberately in this order.

    trailing    scale the bet by (target / trailing-mean MAE). No fitting at all,
                one window parameter, and nothing to overfit. If the effect is
                real this is where it shows up most honestly.
    modelled    scale by (target / predicted MAE) from a purged walk-forward
                regression on the S110 feature set. More powerful, far more
                overfittable, and disbelieved if it beats the trailing rule by a
                lot.

This is a different intervention from everything that has failed at the
denominator. S34 and S94 scaled DAILY RETURNS on equity-curve state. S105 capped
the ACCOUNT after conviction had set the size. S104 changed how trades END.
This changes the size of an individual bet BEFORE it is placed, using a
cross-sectional risk model rather than a time-series overlay - and the one thing
in this book that has ever worked on the denominator, the ATR stop, is exactly
that kind of object.

The controls are the ones that have killed every previous candidate: a random
multiplier with the same dispersion, and both halves of the sample.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import research.harness as H
from research.harness import backtest
import strategies.s69_calsel as S69
import strategies.s84_gate as S84
import strategies.s87_combined as S87
import strategies.s110_meta as M
from strategies.s96_rank import at_gate, stats_of
from research.robust import bootstrap_dd

RISK, K = 0.08, 3
EMBARGO = pd.Timedelta(days=21)
LO, HI = 0.5, 2.0                  # the multiplier is clipped; no bet may vanish


def trailing_pred(T, win=20):
    """Mean MAE_R of the last `win` trades that had already CLOSED when this one
    opened.

    Sorting by ENTRY time and shifting - the obvious implementation, and the one
    written first here - uses trades that were still open at the moment of the
    decision. Their excursion is not known yet, and holds run to 21 days, so that
    is look-ahead of up to three weeks dressed as a trailing average.
    """
    ex = T.exit_dt.to_numpy()
    order = np.argsort(ex)
    ex_s, mae_s = ex[order], T.mae_R.to_numpy(float)[order]
    out = np.full(len(T), np.nan)
    for k, t0 in enumerate(T.entry_dt.to_numpy()):
        j = np.searchsorted(ex_s, t0, side="left")      # closed strictly before
        if j >= 5:
            out[k] = np.nanmean(mae_s[max(0, j - win):j])
    return out


def modelled_pred(T, X):
    """Purged walk-forward regression on MAE_R."""
    y = T.mae_R.to_numpy(float)
    pred = np.full(len(T), np.nan)
    for q in sorted(T.quarter.unique()):
        te = (T.quarter == q).to_numpy()
        t0 = T.entry_dt[te].min()
        tr = (T.exit_dt < t0 - EMBARGO).to_numpy() & np.isfinite(y)
        if tr.sum() < 200 or te.sum() == 0:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def mult_from(pred, target):
    """Size multiplier: smaller bet where a larger excursion is expected."""
    m = np.full(len(pred), 1.0)
    ok = np.isfinite(pred) & (pred > 1e-6)
    m[ok] = np.clip(target / pred[ok], LO, HI)
    return m


def sim_sized(cfg, start, end, risk, scale):
    p, stp, rr, hold, sp, md = cfg
    c = S69.ctx(); u0 = np.nan_to_num(S69.shape(p)); u = u0; a = c["a"]
    if sp:
        up = S84.trend(sp)
        block = np.zeros(len(u), bool)
        if "s" in md: block |= (u < 0) & up
        if "l" in md: block |= (u > 0) & ~up
        u = np.where(block, 0.0, u)
    arr = dict(entry=u * scale, stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u0) <= 0.0).astype(float))
    return backtest(c["g"], arr, "12h", start=start, end=end, risk=risk,
                    max_lev=10.0, max_bars_h=hold * 24)


def run(T, mult, n):
    """Replay V7 with a per-decision-bar size multiplier."""
    R = S87.rankings()
    idx = {}
    for (q, r, i, mu) in zip(T.quarter, T["rank"], T.sig_i, mult):
        idx.setdefault((q, r), {})[i] = mu
    segs, pnl = [], []
    for s, e, cfgs in R:
        rs = []
        for rank, cfg in enumerate(cfgs[:K]):
            sc = np.ones(n)
            for i, mu in idx.get((s, rank), {}).items():
                sc[i] = mu
            m = sim_sized(cfg, s, e, RISK / K, sc)
            rs.append(S69.daily(m))
            td = m["trades_df"]
            if td is not None and len(td):
                pnl.append(td["pnl"].to_numpy(float))
        segs.append(pd.DataFrame({j: r for j, r in enumerate(rs)}).fillna(0.0).sum(axis=1))
    return pd.concat(segs), (np.concatenate(pnl) if pnl else np.array([]))


def row(tag, r, pl, base=None):
    a = np.asarray(r, float)
    st = stats_of(a); b = bootstrap_dd(a, n=3000)
    h = len(a) // 2
    f_, h1, h2 = (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
                  at_gate(a[h:])["cagr"] * 100)
    pf = pl[pl > 0].sum() / max(-pl[pl < 0].sum(), 1e-9) if len(pl) else np.nan
    d = "" if base is None else f"{f_-base[0]:+9.1f}{min(h1-base[1], h2-base[2]):+12.1f}"
    print(f"{tag:>32}{len(pl):7d}{pf:6.2f}{st['sharpe']:7.2f}{st['calmar']:7.2f}"
          f"{st['dd']*100:8.1f}%{b['p_dd_worse_than_20']*100:6.0f}%"
          f"{f_:9.1f}%{h1:9.1f}%{h2:9.1f}%{d}", flush=True)
    return (f_, h1, h2)


if __name__ == "__main__":
    T = pd.read_parquet("/home/user/quant/results/s111_trades.parquet")
    X = pd.read_parquet("/home/user/quant/results/s111_features.parquet")
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.columns[X.notna().mean() > 0.9]
    X = X[keep].fillna(X[keep].median())
    g, v = M.use_clock()
    n = len(g)
    target = float(np.nanmedian(T.mae_R))
    print(f"target excursion = median MAE_R = {target:.3f} R; multiplier clipped "
          f"to [{LO}, {HI}]\n")

    print(f"{'book':>32}{'N':>7}{'PF':>6}{'Shp':>7}{'Clm':>7}{'realDD':>8}{'P>20%':>6}"
          f"{'full':>9}{'1st h':>9}{'2nd h':>9}{'vs V7':>9}{'worse half':>12}")
    base = row("V7 unsized", *run(T, np.ones(len(T)), n))

    preds = {}
    for w in (10, 20, 50):
        preds[f"trailing {w}"] = trailing_pred(T, w)
    preds["modelled (ridge)"] = modelled_pred(T, X)

    for tag, p in preds.items():
        mu = mult_from(p, target)
        cov = np.isfinite(p).mean() * 100
        row(f"size ~ 1/{tag}", *run(T, mu, n), base)
        print(f"{'':32}   multiplier: median {np.median(mu):.2f} "
              f"p10 {np.quantile(mu, .1):.2f} p90 {np.quantile(mu, .9):.2f}, "
              f"covers {cov:.0f}% of trades", flush=True)

    print("\ncontrol: a RANDOM multiplier with the same dispersion, 3 draws")
    mu_ref = mult_from(preds["trailing 20"], target)
    vals = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        mu = rng.permutation(mu_ref)
        r_, pl_ = run(T, mu, n)
        vals.append(at_gate(np.asarray(r_, float))["cagr"] * 100)
    print(f"{'shuffled multiplier':>32}{'':33}{np.mean(vals):9.1f}%{'':18}"
          f"{np.mean(vals)-base[0]:+9.1f}   (draws: "
          f"{', '.join(f'{x:.0f}' for x in vals)})")
    print("\ndone: excursion sizing")
