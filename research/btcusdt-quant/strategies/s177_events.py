"""S177 - Should you flatten V7 around big events?

You cannot know in advance which day is an "event" day, so the kindest possible
version of the question is: if you had PERFECT foresight of the largest-move
days and sat them out, would you be better off?  If even a clairvoyant loses by
sitting out, a human guessing at FOMC dates certainly does.

Proxy for an event day: the largest absolute BTC daily moves in the sample.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from strategies.s87_combined import rankings
from strategies.s69_calsel import ctx
from strategies.s176_lev import blend

RISK = 0.144


def cagr_dd(r):
    e = np.cumprod(1 + r.to_numpy())
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    dd = float((e / np.maximum.accumulate(e) - 1).min())
    return e[-1] ** (1 / yrs) - 1, dd


if __name__ == "__main__":
    R = rankings()
    r, _ = blend(R, 3, RISK, 10.0)
    r = r.groupby(r.index).sum()                       # one row per day

    g = ctx()["g"]
    px = pd.Series(g.close.to_numpy(float), index=pd.to_datetime(g.index if g.index.name else g.dt))
    btc = px.resample("1D").last().pct_change().reindex(r.index).fillna(0.0)

    base_c, base_d = cagr_dd(r)
    print(f"V7 top-3 at {RISK:.1%}, {len(r)} days   CAGR {base_c*100:6.1f}%   DD {base_d*100:6.1f}%\n")

    print("sit out the N biggest BTC move days (with perfect foresight):")
    print(f"{'days out':>9} {'% of days':>10} {'CAGR':>9} {'maxDD':>8} {'vs base':>10}")
    order = btc.abs().sort_values(ascending=False).index
    for n in (0, 5, 10, 20, 40, 80, 160):
        out = set(order[:n])
        rr = r.copy(); rr[rr.index.isin(out)] = 0.0
        c, d = cagr_dd(rr)
        print(f"{n:>9} {n/len(r)*100:>9.1f}% {c*100:>8.1f}% {d*100:>7.1f}% {(c-base_c)*100:>+9.1f}pp")

    print("\nwhere the profit actually is — V7's own best/worst days:")
    s = r.sort_values()
    tot = float(np.cumprod(1 + r.to_numpy())[-1] - 1)
    for lab, idx in (("top 1% of days", s.index[-int(len(s)*0.01):]),
                     ("top 5% of days", s.index[-int(len(s)*0.05):]),
                     ("worst 1% of days", s.index[:int(len(s)*0.01)]),
                     ("worst 5% of days", s.index[:int(len(s)*0.05)])):
        rr = r.copy(); rr[rr.index.isin(set(idx))] = 0.0
        c, _ = cagr_dd(rr)
        print(f"  remove {lab:<18} -> CAGR {c*100:7.1f}%  ({(c-base_c)*100:+.1f}pp)")

    hi = btc.abs() >= btc.abs().quantile(0.95)
    print(f"\non the 5% highest-volatility days: V7 averages {r[hi].mean()*100:+.3f}%/day "
          f"vs {r[~hi].mean()*100:+.3f}% on the rest")
    print(f"   those {int(hi.sum())} days carry {r[hi].sum()/r.sum()*100:.0f}% of the total return")
