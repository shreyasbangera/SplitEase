"""
S110c - Does a 0.537 AUC buy anything? The filtered book, against the controls.

S110b found weak but real separation: out-of-sample AUC **0.537** from penalised
logistic regression, +1.6 sd on the 675 effective events and above the maximum of
twenty shuffled-label runs (0.526). Gradient boosting read 0.517 - worse than the
linear model, which is the correct sign for a flexible model on a small sample
and a reason to trust the linear one rather than the flexible one.

0.537 is a very weak ranking, and on most books it would be worthless. This book
is not most books: V7 wins **46.9%** of its trades at a profit factor of **3.17**,
so the outcome distribution is violently skewed and a filter does not have to
rank well on average - it has to avoid the left tail. A small AUC can pay if the
bottom decile is where the big losses live, and can equally cost everything if
that decile also holds the big winners. Measured, not argued.

The filter is applied where it would really act: the trade's entry is **zeroed on
its decision bar**, so the book is genuinely flat instead and the engine carries
on from there - the next signal re-enters on its own schedule. Removing P&L from
an equity curve after the fact would assume the rest of the path is unchanged,
which it is not.

    predictions are OUT OF SAMPLE only. The first ~300 trades predate the
    classifier's first training window and are never filtered, which is what
    would have happened live.

THREE CONTROLS, AND THE FILTER HAS TO BEAT ALL THREE
    random drop   the same FRACTION of trades removed at random, resampled. A
                  filter that only cuts exposure must not be credited for it.
    split half    the control that killed S105 and the event-bar candidate.
    trade count   the brief requires 100+ trades; a filter that reaches its
                  number by trading 40 times has not met the brief.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import research.harness as H
from research.harness import backtest
import strategies.s69_calsel as S69
import strategies.s84_gate as S84
import strategies.s87_combined as S87
import strategies.s110_meta as M
from strategies.s110b_auc import load, walk, logit
from strategies.s96_rank import at_gate, stats_of
from research.robust import bootstrap_dd

RISK, K = 0.08, 3


def sim_filtered(cfg, start, end, risk, kill):
    """S87.sim with selected decision bars suppressed.

    `kill` is a boolean mask over the decision grid. Zeroing the conviction there
    makes the book FLAT on that bar rather than removing a trade from the
    accounts afterwards, so the rest of the path responds the way it would have.
    """
    p, stp, rr, hold, sp, md = cfg
    c = S69.ctx(); u0 = np.nan_to_num(S69.shape(p)); u = u0; a = c["a"]
    if sp:
        up = S84.trend(sp)
        block = np.zeros(len(u), bool)
        if "s" in md: block |= (u < 0) & up
        if "l" in md: block |= (u > 0) & ~up
        u = np.where(block, 0.0, u)
    u = np.where(kill, 0.0, u)
    arr = dict(entry=u, stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u0) <= 0.0).astype(float))
    return backtest(c["g"], arr, "12h", start=start, end=end, risk=risk,
                    max_lev=10.0, max_bars_h=hold * 24)


def run(T, drop_mask, n):
    """Replay V7 with the flagged (quarter, rank, sig_i) entries suppressed."""
    R = S87.rankings()
    idx = {}
    for (q, r, i) in zip(T.quarter[drop_mask], T["rank"][drop_mask],
                         T.sig_i[drop_mask]):
        idx.setdefault((q, r), []).append(i)
    segs, pnl = [], []
    for s, e, cfgs in R:
        rs = []
        for rank, cfg in enumerate(cfgs[:K]):
            kill = np.zeros(n, bool)
            for i in idx.get((s, rank), []):
                kill[i] = True
            m = sim_filtered(cfg, s, e, RISK / K, kill)
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
    print(f"{tag:>30}{len(pl):7d}{pf:6.2f}{st['sharpe']:7.2f}{st['calmar']:7.2f}"
          f"{st['dd']*100:8.1f}%{b['p_dd_worse_than_20']*100:6.0f}%"
          f"{f_:9.1f}%{h1:9.1f}%{h2:9.1f}%{d}", flush=True)
    return (f_, h1, h2)


if __name__ == "__main__":
    T, X = load()
    y = (T.pnl > 0).astype(int).to_numpy()
    print("refitting the out-of-sample predictions")
    p = walk(T, X, y, logit)
    T["p"] = p
    scored = np.isfinite(p)
    print(f"{scored.sum()} of {len(T)} trades carry an out-of-sample prediction\n")

    g, v = M.use_clock()
    n = len(g)

    print(f"{'book':>30}{'N':>7}{'PF':>6}{'Shp':>7}{'Clm':>7}{'realDD':>8}"
          f"{'P>20%':>6}{'full':>9}{'1st h':>9}{'2nd h':>9}{'vs V7':>9}{'worse half':>12}")
    base = row("V7 unfiltered", *run(T, np.zeros(len(T), bool), n))

    rng = np.random.default_rng(0)
    for q in (0.10, 0.20, 0.30, 0.40):
        thr = np.nanquantile(p[scored], q)
        drop = scored & (p <= thr)
        row(f"drop worst {q*100:.0f}% by model", *run(T, drop, n), base)
        # control: the same number of trades, chosen at random
        vals = []
        for seed in range(3):
            r2 = np.random.default_rng(seed)
            cand = np.flatnonzero(scored)
            pick = r2.choice(cand, drop.sum(), replace=False)
            rd = np.zeros(len(T), bool); rd[pick] = True
            rr_, pl_ = run(T, rd, n)
            vals.append(at_gate(np.asarray(rr_, float))["cagr"] * 100)
        print(f"{f'  random drop {q*100:.0f}% (3 draws)':>30}{'':33}"
              f"{np.mean(vals):9.1f}%{'':18}{np.mean(vals)-base[0]:+9.1f}", flush=True)
    print("\ndone: filtered book")
