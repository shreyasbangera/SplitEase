"""S176 - Does the leverage cap bind?  V7's top-3 at max_lev 10 (as backtested
and as deployed) against 5, at both risk settings.

The cap is `min(q, equity * max_lev / price)` inside the engine.  It only bites
when conviction is high AND the ATR stop is tight, so the question is not what
the cap is but how often the position ever reaches it.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
import strategies.s84_gate as G
from strategies.s87_combined import rankings
from strategies.s69_calsel import daily
from strategies.s77_lookback import stats


def sim_lev(cfg, start, end, risk, span, mode, max_lev):
    """s84_gate.sim_gate with max_lev exposed instead of hardcoded to 10."""
    p, stp, rr, hold = cfg
    c = G.ctx(); u0 = np.nan_to_num(G.shape(p)); u = u0; a = c["a"]
    if span:
        up = G.trend(span)
        block = np.zeros(len(u), bool)
        if "s" in mode: block |= (u < 0) & up
        if "l" in mode: block |= (u > 0) & ~up
        u = np.where(block, 0.0, u)
    arr = dict(entry=u, stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u0) <= 0.0).astype(float))
    return G.backtest(c["g"], arr, "12h", start=start, end=end, risk=risk,
                      max_lev=max_lev, max_bars_h=hold * 24)


def blend(R, k, risk, max_lev):
    segs, pnl = [], []
    for s, e, cfgs in R:
        rs = []
        for p, stp, rr, hold, sp, md in cfgs[:k]:
            m = sim_lev((p, stp, rr, hold), s, e, risk / k, sp, md, max_lev)
            rs.append(daily(m))
            td = m["trades_df"]
            if td is not None and len(td):
                pnl.append(td["pnl"].to_numpy(float))
        segs.append(pd.DataFrame({j: r for j, r in enumerate(rs)}).fillna(0.0).sum(axis=1))
    return pd.concat(segs), (np.concatenate(pnl) if pnl else np.array([]))


if __name__ == "__main__":
    R = rankings()
    print(f"V7 walk-forward, top-3 at risk/3, {len(R)} quarterly segments\n")
    for risk in (0.08, 0.144):
        for ml in (10.0, 5.0):
            r, pl = blend(R, 3, risk, ml)
            stats(r, pl, f"top-3  max_lev {ml:g}", risk)
        print()
