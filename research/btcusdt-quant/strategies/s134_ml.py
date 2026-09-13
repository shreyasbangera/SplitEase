"""
S134 - A learned model, walked forward. The one method never used here.

Every strategy in this study specifies BY HAND how its inputs combine: equal
weights over standardised features, a threshold, a sign chosen by reasoning. That
is a strong assumption - it says the relationship is linear, additive, and the
same in every regime. A gradient-boosted model assumes none of those things. It
can find that crowding matters only when volatility is low, or that funding and
open interest interact, which no equal-weight average can express.

This is a different METHOD on the same data, not another data source, and it is
the obvious thing a quant desk would try that this log never has.

THE THREE WAYS THIS GOES WRONG, AND WHAT IS DONE ABOUT EACH
------------------------------------------------------------
  LOOK-AHEAD IN THE LABEL
      A k-day forward return at date t is not known until t+k. Training a model
      at time T on a sample from t > T-k leaks the future into the fit. Every
      retrain here PURGES the last k days of training data for exactly this
      reason. Skipping that purge is the single most common way a backtest of
      this kind produces a beautiful and completely fake equity curve.

  OVERFITTING THE HYPERPARAMETERS
      Nothing is tuned. Depth, learning rate, tree count and the retrain cadence
      are fixed at ordinary defaults before any result is seen, and the whole
      grid of what was tried is printed.

  THE PIPELINE ITSELF MANUFACTURING RETURNS
      This is the one that matters, and it gets a real control: the identical
      pipeline is run with the training LABELS SHUFFLED. A model trained on
      scrambled targets knows nothing, so anything it appears to earn is
      machinery rather than edge. If the real and shuffled versions look alike,
      the result is an artefact regardless of how good the real number looks.

Walk-forward: retrain every 90 days on a trailing 3-year window, predict the next
90 days, never see a bar at or after the prediction date. Sized by prediction
magnitude against trailing volatility, charged 5bps fee, 3bps slippage and
settled funding.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from strategies.s96_rank import at_gate, stats_of
from research.robust import bootstrap_dd

FEE, SLIP = 5.0, 3.0
TRAIN_DAYS, REFIT, HORIZON = 1095, 90, 5


def features():
    """Everything this study has assembled, on one daily frame."""
    import strategies.s119_dev as S119
    import strategies.s118_crowd as C
    px, fd, F = S119.build("1D")
    X = pd.DataFrame(index=px.index)

    # crowding, standardised
    for c in [c for c in C.SIGNS if c in F.columns]:
        s = F[c].astype(float)
        X[f"cr_{c}"] = ((s - s.rolling(180, min_periods=60).mean())
                        / (s.rolling(180, min_periods=60).std() + 1e-12)).clip(-3, 3)
    # price / technical
    lp = np.log(px)
    for k in (1, 5, 10, 20, 60):
        X[f"ret{k}"] = lp.diff(k)
    r = lp.diff()
    X["vol20"] = r.rolling(20).std() * np.sqrt(365.25)
    X["vol60"] = r.rolling(60).std() * np.sqrt(365.25)
    X["volratio"] = X.vol20 / (X.vol60 + 1e-12)
    X["fund"] = fd
    X["fund20"] = fd.rolling(20).mean()

    for mod, pre in (("strategies.s127_macro", "ma"), ("strategies.s128_onchain", "oc")):
        try:
            m = __import__(mod, fromlist=["panel"])
            _, FX = m.panel()
            for c in FX.columns:
                X[f"{pre}_{c}"] = FX[c].reindex(X.index)
        except Exception as e:
            print(f"   ({pre} unavailable: {e})")
    return px, fd, X.shift(1)          # every feature lagged one full day


def walk(px, X, y, shuffle=False, seed=0):
    """Walk-forward prediction. Purges the label horizon at every retrain."""
    rng = np.random.default_rng(seed)
    idx = X.index
    pred = pd.Series(np.nan, index=idx)
    starts = range(TRAIN_DAYS + HORIZON, len(idx), REFIT)
    for s in starts:
        # labels for samples after (s - HORIZON) are not yet observable at s
        tr_end = s - HORIZON
        tr_start = max(0, tr_end - TRAIN_DAYS)
        Xtr = X.iloc[tr_start:tr_end]
        ytr = y.iloc[tr_start:tr_end]
        m = np.isfinite(ytr.to_numpy())
        if m.sum() < 250:
            continue
        yy = ytr.to_numpy()[m].copy()
        if shuffle:
            rng.shuffle(yy)                      # the control
        mdl = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
        mdl.fit(Xtr.to_numpy()[m], yy)
        e = min(s + REFIT, len(idx))
        pred.iloc[s:e] = mdl.predict(X.iloc[s:e].to_numpy())
    return pred


def book(px, fd, pred, tv, thr=0.0, max_lev=3.0, band=0.10):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = np.maximum(r.ewm(halflife=32, adjust=False).std()
                    .shift(1).bfill().to_numpy() * np.sqrt(365.25), 0.05)
    p = pred.copy()
    z = ((p - p.rolling(250, min_periods=60).mean())
         / (p.rolling(250, min_periods=60).std() + 1e-12)).clip(-2, 2).fillna(0.0)
    s = z.to_numpy().copy()
    s[np.abs(s) < thr] = 0.0
    want = np.clip(s * tv / rv, -max_lev, max_lev)
    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * px.pct_change().fillna(0.0).to_numpy() \
        - pos * fd.to_numpy() - turn * (FEE + SLIP) / 1e4
    return pd.Series(net, index=px.index), pos


def gate(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


def score(tag, net, pos):
    a = np.asarray(net, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0):
        print(f"   {tag:>26}  no position"); return np.nan
    st = stats_of(a); g = gate(a)
    live = np.asarray(pos, float) != 0
    trades = int(((~live[:-1]) & live[1:]).sum() + (1 if live[0] else 0))
    pnl, acc, on = [], 0.0, False
    for i in range(len(a)):
        if live[i]:
            acc += a[i]; on = True
        elif on:
            pnl.append(acc); acc, on = 0.0, False
    if on:
        pnl.append(acc)
    p = np.array(pnl) if pnl else np.array([0.0])
    pf = (p[p > 0].sum() / -p[p < 0].sum()) if (p < 0).any() else np.inf
    h = len(a) // 2
    g1, g2 = gate(a[:h]), gate(a[h:])
    gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
    s1 = "n/a" if not np.isfinite(g1) else f"{g1:.0f}%"
    s2 = "n/a" if not np.isfinite(g2) else f"{g2:.0f}%"
    print(f"   {tag:>26}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%{trades:>7}"
          f"{pf:>7.2f}{gs:>9}{s1:>8}{s2:>8}")
    return g


if __name__ == "__main__":
    px, fd, X = features()
    lp = np.log(px)
    y = lp.diff(HORIZON).shift(-HORIZON)          # forward 5-day return
    ok = X.notna().mean() > 0.5
    X = X.loc[:, ok]
    start = X.dropna(thresh=int(X.shape[1] * 0.7)).index.min()
    px, fd, X, y = (px[px.index >= start], fd[fd.index >= start],
                    X[X.index >= start], y[y.index >= start])
    print("S134 - a learned model, walked forward\n")
    print(f"daily, {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days")
    print(f"{X.shape[1]} features; retrain every {REFIT}d on a trailing "
          f"{TRAIN_DAYS}d window, {HORIZON}d label purged at every fit\n")

    print(f"   {'variant':>26}{'Shp':>7}{'realDD':>8}{'trades':>7}{'PF':>7}"
          f"{'at -20%':>9}{'1st h':>8}{'2nd h':>8}")
    pr = walk(px, X, y, shuffle=False)
    print(f"   (model predicted on {pr.notna().sum()} of {len(pr)} days)")
    reals = []
    for tv in (0.20, 0.40):
        for thr in (0.0, 0.5):
            reals.append(score(f"LEARNED tgt{tv*100:.0f}% thr{thr:.1f}",
                               *book(px, fd, pr, tv, thr)))

    print("\n   CONTROL — identical pipeline, training labels shuffled:")
    ctrls = []
    for seed in (0, 1, 2):
        ps = walk(px, X, y, shuffle=True, seed=seed)
        ctrls.append(score(f"SHUFFLED seed {seed}", *book(px, fd, ps, 0.20, 0.0)))

    rv = [v for v in reals if np.isfinite(v)]
    cv = [v for v in ctrls if np.isfinite(v)]
    print(f"\n   best real {max(rv):.1f}%" if rv else "\n   no valid real result")
    if cv:
        print(f"   shuffled control: median {np.median(cv):.1f}%, "
              f"best {max(cv):.1f}%")
    print("\ndone: S134")
