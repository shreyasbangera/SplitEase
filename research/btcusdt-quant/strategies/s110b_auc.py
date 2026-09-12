"""
S110b - Does anything predict a V7 trade's outcome out of sample? One number decides it.

S110 built the dataset and corrected the headline that motivated it. V7 has 1,697
trades but only **675 unique entry bars** - the three sleeves duplicate each
event 2.51x - so this is 2.9x S17's 232 events, not 7.3x. Real, but smaller than
it looked, and 675 events against 28 features is a ratio where a flexible model
will fit noise happily.

It also showed the target is not obvious. The base win rate is **46.9%** with a
profit factor of 3.17: V7 wins less than half its trades and makes money from
skew. A classifier that skips predicted losers also skips the left tail of a
distribution whose right tail pays for everything, so predicting WINS and
predicting PROFIT are different objectives and both are measured.

    win        P(pnl > 0), scored by AUC
    pnl        the actual outcome, scored by rank correlation

No book is built here. If neither reads above chance out of sample, meta-labelling
is finished and S17's verdict becomes a verdict.

THE PROTOCOL
------------
Expanding-window walk-forward by quarter, refit each time, with an **embargo**:
any training trade whose exit falls within 21 days (the longest hold in the grid)
of the test block's start is dropped, so no training label overlaps a test trade.
Only out-of-sample predictions are scored, and they are scored pooled.

Error bars are widened for the duplication: 1,697 test rows are 675 independent
events, so the standard error on AUC is computed on the effective count, not the
row count. A result that needs the row count to look significant is not one.

Two models, deliberately in this order: penalised logistic regression, which is
hard to overfit and whose coefficients can be read, then gradient boosting, which
is not - reported only as a comparison, and disbelieved if it beats the linear
model by a lot on a sample this size.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import lightgbm as lgb

EMBARGO = pd.Timedelta(days=21)
MIN_TRAIN = 200


def load():
    T = pd.read_parquet("/home/user/quant/results/s110_trades.parquet")
    X = pd.read_parquet("/home/user/quant/results/s110_features.parquet")
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.columns[X.notna().mean() > 0.9]
    X = X[keep].fillna(X[keep].median())
    return T.reset_index(drop=True), X.reset_index(drop=True)


def walk(T, X, y, model_fn):
    """Expanding window by quarter, embargoed. Returns OOS predictions."""
    pred = np.full(len(T), np.nan)
    qs = sorted(T.quarter.unique())
    for qi, q in enumerate(qs):
        te = (T.quarter == q).to_numpy()
        t0 = T.entry_dt[te].min()
        tr = (T.exit_dt < t0 - EMBARGO).to_numpy()      # strictly past, embargoed
        if tr.sum() < MIN_TRAIN or te.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = (m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba")
             else m.predict(X[te]))
        pred[te] = p
    return pred


def logit():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000))


def gbm():
    return lgb.LGBMClassifier(n_estimators=120, num_leaves=4, max_depth=3,
                              learning_rate=0.03, min_child_samples=40,
                              subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.6, reg_lambda=5.0,
                              verbose=-1, random_state=0)


def report(tag, y, p, pnl, neff):
    ok = np.isfinite(p)
    if ok.sum() < 50:
        print(f"{tag:>34}  too few OOS predictions ({ok.sum()})")
        return
    auc = roc_auc_score(y[ok], p[ok])
    rho = spearmanr(p[ok], pnl[ok]).statistic
    # SE of AUC under the null, on the EFFECTIVE event count
    n1 = max(int(y[ok].sum() * neff / ok.sum()), 2)
    n0 = max(int((1 - y[ok]).sum() * neff / ok.sum()), 2)
    se = np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))
    print(f"{tag:>34}   n {int(ok.sum()):5d} (eff {int(neff*ok.sum()/len(p)):4d})"
          f"   AUC {auc:.3f}  ({(auc-0.5)/se:+.1f} sd)"
          f"   rho(pred, pnl) {rho:+.3f}")
    return auc


if __name__ == "__main__":
    T, X = load()
    y = (T.pnl > 0).astype(int).to_numpy()
    pnl = T.pnl.to_numpy(float)
    neff = T.entry_dt.nunique()
    print(f"{len(T)} trades, {neff} unique entry bars, {X.shape[1]} features, "
          f"base win rate {y.mean()*100:.1f}%\n")

    print("out-of-sample, expanding window by quarter, 21-day embargo\n")
    for tag, fn in (("logistic regression (C=0.05)", logit),
                    ("gradient boosting (depth 3)", gbm)):
        p = walk(T, X, y, fn)
        report(tag, y, p, pnl, neff)

    print("\ncontrol: the same protocol on a SHUFFLED label, 20 draws")
    rng = np.random.default_rng(0)
    aucs = []
    for i in range(20):
        ys = rng.permutation(y)
        p = walk(T, X, ys, logit)
        ok = np.isfinite(p)
        if ok.sum() > 50:
            aucs.append(roc_auc_score(ys[ok], p[ok]))
    a = np.array(aucs)
    print(f"{'shuffled-label logistic':>34}   AUC median {np.median(a):.3f}  "
          f"mean {a.mean():.3f}  sd {a.std(ddof=1):.3f}  max {a.max():.3f}")

    print("\nif there is signal, where is it? logistic coefficients on the full sample")
    m = logit(); m.fit(X, y)
    co = pd.Series(m[-1].coef_[0], index=X.columns).sort_values()
    for k, v in pd.concat([co.head(6), co.tail(6)]).items():
        print(f"   {k:>12} {v:+.3f}")
    print("\ndone: meta-label AUC")
