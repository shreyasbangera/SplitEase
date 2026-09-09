"""Signal-level edge scanner (excess-return version).

For a signal formed on the CLOSE of bar i, measure the forward return from the
OPEN of bar i+1 to the close of bar i+1+h -- i.e. exactly what a next-bar-open
execution would capture.

Two corrections that matter:
  1. EXCESS return. Gold rose ~10.9%/yr over the sample, so ANY long-biased
     signal shows a large positive mean. We subtract the unconditional mean at
     the same horizon, so what is reported is edge over buy-and-hold drift.
  2. OVERLAP. h-bar forward windows sampled every bar are strongly overlapping,
     which inflates t-stats by roughly sqrt(h). We report t_adj = t / sqrt(h),
     a conservative Newey-West-style haircut.
"""
import numpy as np, pandas as pd


def fwd_matrix(df, horizons):
    o = df["open"]; c = df["close"]
    entry = o.shift(-1)
    return {h: (c.shift(-h) / entry - 1) for h in horizons}


def scan(df, signals: dict, horizons=(1, 4, 12, 24, 72, 168), cost_bps=14.0, min_n=100):
    F = fwd_matrix(df, horizons)
    base = {h: F[h].mean() for h in horizons}
    rows = []
    for name, (mask, side) in signals.items():
        m = mask.fillna(False).to_numpy(bool)
        for h in horizons:
            r = F[h]
            x = r[m].dropna()
            if len(x) < min_n:
                continue
            exc = (x.mean() - base[h]) * side
            sd = x.std()
            t = exc / (sd / np.sqrt(len(x))) if sd > 0 else 0.0
            t_adj = t / np.sqrt(h)
            eff_n = len(x) / h
            rows.append({"label": name, "h": h, "n": len(x), "eff_n": eff_n,
                         "excess_bp": exc * 1e4, "t_adj": t_adj,
                         "net_bp": exc * 1e4 - cost_bps,
                         "hit%": ((side * (x - base[h])) > 0).mean() * 100})
    return pd.DataFrame(rows)
