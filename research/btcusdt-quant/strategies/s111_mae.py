"""
S111 - Predict the EXCURSION, not the outcome. The target S103 was pointing at.

S110 predicted whether a trade would win. That was the wrong question, and S103
said so before it was asked:

    94% of V7's six deepest drawdowns is mark-to-market on OPEN positions, not
    realised loss. The worst drawdown in the record closed nothing at all - three
    longs, 3.05x aggregate, BTC down 4.1% over three days, exited a fortnight
    later for +3,759.

The gate is a **drawdown** constraint, and a trade's contribution to drawdown is
its adverse excursion, not the sign of its final P&L.

[Corrected after running this file.] The first draft said those two quantities
were "close to unrelated on this book". They are not: rho(MAE_R, P&L) = **-0.655**,
winners average 0.133 R of adverse excursion against losers' 0.386. That
relationship is mostly mechanical - a position that goes far against you and then
exits on signal has by then lost money - so it is not a prediction, but the claim
as written was wrong and is struck rather than quietly dropped.

What S103 actually established is narrower and still holds: the trades that
dominated the DEEPEST EPISODES were ones that eventually paid. Excursion is the
right target because it is what the gate charges for, not because it is unrelated
to P&L.

There is a second reason to expect more here than from S110. Direction is close
to unpredictable - that is why AUC 0.537 was the ceiling. **Excursion is a
volatility quantity, and volatility is the most predictable object in
quantitative finance.** Realised vol has an autocorrelation that direction never
has.

The obvious objection is that V7 already knows this. It sizes every trade by ATR
- the stop is 3xATR and the quantity is risk/stop-distance - so excursion
measured in R units is ALREADY volatility-normalised, and any remaining
predictability is information ATR does not carry. That is exactly the test:

    MAE_R = (worst adverse price move while the position was open) / (stop distance)

    does anything predict MAE_R, GIVEN that ATR is already in the denominator?

If nothing does, ATR is a sufficient risk model and this closes. If something
does, sizing inversely to predicted MAE_R cuts the drawdown denominator without
touching the signal - which is the one thing S94's overlays, S104's exits and
S105's caps all failed to do, each of them because they acted on the equity curve
or the exit rather than on the size of the bet before it was placed.

Note the censoring: a trade stopped out has MAE_R of about 1 by construction, so
the target is right-censored. 32 of 1,697 trades stop, so it is a small effect,
and it is reported rather than corrected for.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
from research.harness import exec_grid
import strategies.s110_meta as M

RISK, K = 0.08, 3


def excursions(T, g):
    """Per-trade maximum adverse and favourable excursion, in R units.

    Walked on the execution grid, between the trade's own entry and exit
    timestamps, so it is the path the engine actually saw.
    """
    ex = exec_grid().copy()
    ex["dt"] = pd.to_datetime(ex.dt, utc=True)
    edt = ex.dt.to_numpy()
    hi, lo = ex.high.to_numpy(float), ex.low.to_numpy(float)
    a = g.atr14.to_numpy(float)

    i0 = np.searchsorted(edt, T.entry_dt.to_numpy(), side="left")
    i1 = np.searchsorted(edt, T.exit_dt.to_numpy(), side="right")
    mae = np.full(len(T), np.nan)
    mfe = np.full(len(T), np.nan)
    for k, (s, e, side, px) in enumerate(zip(i0, i1, T.side.to_numpy(),
                                             T.entry.to_numpy(float))):
        if e <= s:
            continue
        h, l = hi[s:e].max(), lo[s:e].min()
        if side > 0:
            mae[k] = (px - l) / px
            mfe[k] = (h - px) / px
        else:
            mae[k] = (h - px) / px
            mfe[k] = (px - l) / px
    stop = T.stp.to_numpy(float) * a[T.sig_i.to_numpy()]
    R = stop / T.entry.to_numpy(float)              # stop distance as a fraction
    return mae / R, mfe / R, R


if __name__ == "__main__":
    T, X, g = M.trades()
    mae, mfe, R = excursions(T, g)
    T["mae_R"], T["mfe_R"], T["R"] = mae, mfe, R
    ok = np.isfinite(mae)
    print(f"{int(ok.sum())} of {len(T)} trades measured\n")
    print("adverse excursion in R units (R = the stop distance)")
    q = pd.Series(mae[ok]).quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print("   " + "  ".join(f"p{int(k*100)} {v:.2f}" for k, v in q.items()))
    print(f"   mean {np.nanmean(mae):.3f}   >1.0 on {(mae[ok] > 1).mean()*100:.1f}% "
          f"of trades (stop-outs and gaps)")

    print("\ndoes MAE predict the OUTCOME? (if it did, S110 would have found it)")
    win = (T.pnl > 0).to_numpy()
    print(f"   mean MAE_R  winners {np.nanmean(mae[ok & win]):.3f}   "
          f"losers {np.nanmean(mae[ok & ~win]):.3f}")
    from scipy.stats import spearmanr
    print(f"   rho(MAE_R, P&L) = {spearmanr(mae[ok], T.pnl.to_numpy()[ok]).statistic:+.3f}"
          "   <- S103's point, in one number")

    print("\nis MAE_R predictable at all? trailing-mean benchmark, strictly past")
    d = pd.DataFrame({"dt": T.entry_dt, "mae": mae}).sort_values("dt")
    for w in (20, 50, 100, 200):
        prev = d.mae.shift(1).rolling(w, min_periods=max(5, w // 4)).mean()
        m = np.isfinite(prev) & np.isfinite(d.mae)
        print(f"   trailing mean of the last {w:3d} trades: "
              f"rho = {spearmanr(prev[m], d.mae[m]).statistic:+.3f}  (n {int(m.sum())})")

    T.to_parquet("/home/user/quant/results/s111_trades.parquet")
    X.to_parquet("/home/user/quant/results/s111_features.parquet")
    print("\nsaved")
    print("done: excursion dataset")
