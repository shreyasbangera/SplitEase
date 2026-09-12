"""
S110 - Meta-labelling, reopened because the reason it was closed has expired.

S17 is the only experiment in this log recorded as "inconclusive, not
disproven", and its blocker was stated precisely:

    "Base win rate 48.3% on 232 labelled trades - far too few to train a
     classifier. Higher probability thresholds dropped trade counts below the
     40-trade reporting floor. The technique is sound; this signal does not
     generate enough events to use it."

V7 generates **1,697 trades**. The stated obstacle is gone by a factor of 7.3.

This is different in kind from everything since S45. It is not another signal
and not another sampling of the same signals: the primary book decides the SIDE,
and a secondary classifier decides whether to take that trade at all. And it
attacks the right quantity - the brief needs Calmar 8.95 -> 15, and skipping
losers cuts the drawdown faster than it cuts the return, which is exactly what
the gate rewards and exactly what S104's exits and S105's caps could not do.

WHY THIS IS NOT S6 AGAIN
------------------------
S6 put LightGBM on 75 features against a continuous forward-return label and its
IC flipped sign out of sample. That is a much harder problem: predicting the
market. Meta-labelling predicts a BINARY outcome of a trade the primary model
has already chosen, conditioned on the state at entry - the hard part is already
done, the label is far less noisy, and the base rate is near 50% so there is
real information to find rather than a needle in a 0.01-IC haystack.

THE WAYS THIS COULD LIE, AND WHAT IS DONE ABOUT EACH
----------------------------------------------------
    overlapping labels   trades from three sleeves on the same bar are near
                         duplicates, and holds run up to 21 days, so 1,697 rows
                         are nowhere near 1,697 independent events. Training
                         rows are DEDUPLICATED to unique entry bars, and the
                         effective count is reported rather than the raw one.

    look-ahead           every feature is read from the DECISION bar that
                         produced the trade, never the entry bar or later, and
                         walk-forward training is purged with an embargo of the
                         maximum holding period so no training label overlaps a
                         test trade.

    overfitting          logistic regression FIRST, because it is hard to
                         overfit and its coefficients can be read. Gradient
                         boosting second and only as a comparison.

    a filter that is     the control is dropping the SAME FRACTION of trades at
    just fewer trades    random, resampled - without it, any filter that cuts
                         exposure looks like it cut risk.

This file only measures whether the classifier separates winners from losers out
of sample. If the OOS AUC is at the base rate, that is the end of it and no book
gets built.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
import strategies.s45_single as S45
import strategies.s46_net as S46
import strategies.s69_calsel as S69
import strategies.s87_combined as S87

RISK, K = 0.08, 3
EMBARGO_D = 21                       # the longest hold in the config grid


def use_clock():
    g = S45.grid(S45.FULL_START)
    v = S45.composite(g, S46.LONG); nz = np.abs(v) > 0
    S69._C.clear()
    S69._C.update(g=g, v=v, bm=float(np.abs(v[nz]).mean()), nz=nz,
                  a=g.atr14.to_numpy(float))
    H._ctxc.clear()
    return g, v


def features(g, v):
    """State at each DECISION bar, all of it knowable when that bar closes."""
    c = pd.Series(g.close.to_numpy(float))
    a = g.atr14.to_numpy(float)
    F = pd.DataFrame({"dt": g.dt})
    for n in S45.THR:
        if f"s_{n}" in g:
            F[f"z_{n}"] = g[f"s_{n}"].to_numpy(float)
    F["net"] = v
    F["absnet"] = np.abs(v)
    F["nsig"] = np.sum([np.abs(S45.unit(g, n)) > 0 for n in S46.LONG], axis=0)
    F["atrp"] = a / g.close.to_numpy(float)
    F["volrank"] = pd.Series(F.atrp).rolling(360).rank(pct=True).to_numpy()
    for sp in (100, 200, 300):
        F[f"ema{sp}"] = (c.to_numpy() > c.ewm(span=sp, adjust=False).mean().to_numpy()) * 1.0
    for h in (2, 6, 14, 28):
        F[f"mom{h}"] = np.log(c / c.shift(h)).to_numpy()
    F["rv"] = pd.Series(np.log(c / c.shift(1))).rolling(28).std().to_numpy()
    F["volratio"] = F.rv / pd.Series(F.rv).rolling(180).mean().to_numpy()
    return F


def trades():
    """V7's trades, each tagged with the state of the bar that produced it."""
    g, v = use_clock()
    F = features(g, v)
    gdt = pd.to_datetime(g.dt).to_numpy()
    R = S87.rankings()
    rows = []
    for s, e, cfgs in R:
        for rank, cfg in enumerate(cfgs[:K]):
            m = S87.sim(cfg, s, e, RISK / K)
            td = m["trades_df"]
            if td is None or not len(td):
                continue
            td = td.copy()
            td["entry_dt"] = pd.to_datetime(td.entry_dt, utc=True)
            td["exit_dt"] = pd.to_datetime(td.exit_dt, utc=True)
            # the SIGNAL bar is the one that closed at the entry bar's open
            j = np.searchsorted(gdt, td.entry_dt.to_numpy(), side="right") - 1
            td["sig_i"] = np.maximum(j - 1, 0)
            td["quarter"] = s; td["rank"] = rank
            td["exp"], td["stp"] = cfg[0], cfg[1]
            td["rr"], td["hold"], td["gate"] = cfg[2], cfg[3], cfg[4]
            rows.append(td)
    T = pd.concat(rows, ignore_index=True)
    X = F.iloc[T.sig_i.to_numpy()].reset_index(drop=True)
    X = X.drop(columns=["dt"])
    for col in ("exp", "stp", "rr", "hold", "gate", "rank"):
        X[col] = T[col].to_numpy()
    X["side"] = np.sign(T.side.to_numpy())
    T = T.reset_index(drop=True)
    return T, X, g


if __name__ == "__main__":
    T, X, g = trades()
    T["win"] = (T.pnl > 0).astype(int)
    print(f"V7 trades: {len(T)}   base win rate {T.win.mean()*100:.1f}%   "
          f"(S17 had 232 trades at 48.3%)")
    uniq = T.entry_dt.nunique()
    print(f"unique entry bars: {uniq}  -> the three sleeves duplicate each event "
          f"{len(T)/uniq:.2f}x")
    span = (T.exit_dt - T.entry_dt).dt.total_seconds() / 86400
    print(f"holding period days: median {span.median():.1f}  p90 {span.quantile(0.9):.1f} "
          f" max {span.max():.1f}")
    print(f"features: {X.shape[1]}  ({', '.join(X.columns[:8])}, ...)")
    print(f"\nwin rate by exit reason:")
    for k, lab in ((1, "stop"), (2, "target"), (3, "signal"), (4, "hold cap")):
        m = T.reason == k
        if m.sum():
            print(f"   {lab:>9} {int(m.sum()):5d} trades  win {T.win[m].mean()*100:5.1f}%  "
                  f"mean P&L {T.pnl[m].mean():8.1f}")
    T.to_parquet("/home/user/quant/results/s110_trades.parquet")
    X.to_parquet("/home/user/quant/results/s110_features.parquet")
    print("\nsaved trades + features")
    print("done: meta-label dataset")
