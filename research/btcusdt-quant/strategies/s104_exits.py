"""
S104 - The one lever S94 never pulled: how a trade ENDS.

S94 closed the shape lever, but it closed a specific version of it. Every
overlay there was applied to the book's DAILY RETURNS - vol targets and
high-water-mark throttles, a multiplier on day t from data through t-1. None of
them changed which trades existed or how they finished. The trade-level exit
parameters the engine has always carried - a stop moved to breakeven once a
trade is in profit, and a trailing stop that activates after R multiples - were
used in the pre-S45 studies and never once on this book.

S103 says exactly where to point them, and makes a falsifiable prediction.

    The deepest episodes are NOT realised losses. The single worst drawdown in
    the record, -12.3% over three days, closed nothing at all: it is
    mark-to-market on three long positions that went on to make +3,758. Across
    the six deepest episodes the positions held through them were net positive.

So an exit rule that cuts an adverse excursion should CUT THE RECOVERY WITH IT
and lose money, and if that is what happens the whole family is closed with a
reason rather than by exhaustion.

But the prediction is one-sided, and the other side is why this is worth
running. A breakeven stop does not touch a trade that dips and recovers - it
only arms once the trade is already ahead. It attacks GIVE-BACK, which is a
different pathology from adverse excursion and which S103 did not measure. If
the book's remaining drawdown has any give-back in it, be_r is the rule that
finds it and nothing tested so far would have.

METHOD - S84's, deliberately
---------------------------
The overlay is applied to V7's OWN selection: the cached quarterly rankings are
reused unchanged and only the execution of each selected configuration differs,
so nothing here is fitted and nothing here is adopted. If a rule looks real it
has to go into the quarterly grid and be chosen causally, quarter by quarter, as
S85 did for the trend gate - which is the only reason the gate counts.

Judged at the -20% gate, since that is the only place a shape change can pay.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from research.harness import backtest
from strategies.s69_calsel import ctx, shape, daily
import strategies.s87_combined as S87
from strategies.s84_gate import trend
from strategies.s96_rank import at_gate, stats_of

RISK, K = 0.08, 3


def sim_exit(cfg, start, end, risk, be_r=0.0, trail_after_r=0.0, trail_mult=0.0):
    """S84.sim_gate with the trade's ENDING opened up.

    Copied rather than parameterised upstream because sim_gate is what every
    prior result in this log ran through, and a new keyword on it would put an
    untested default in the path of every one of them.
    """
    p, stp, rr, hold, sp, md = cfg
    c = ctx(); u0 = np.nan_to_num(shape(p)); u = u0; a = c["a"]
    if sp:
        up = trend(sp)
        block = np.zeros(len(u), bool)
        if "s" in md: block |= (u < 0) & up
        if "l" in md: block |= (u > 0) & ~up
        u = np.where(block, 0.0, u)
    arr = dict(entry=u, stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u0) <= 0.0).astype(float))
    if trail_mult:
        arr["trail"] = trail_mult * a
    return backtest(c["g"], arr, "12h", start=start, end=end, risk=risk,
                    max_lev=10.0, max_bars_h=hold * 24,
                    be_r=be_r, trail_after_r=trail_after_r)


def replay(R, **kw):
    segs, pnl = [], []
    for s, e, cfgs in R:
        rs = []
        for cfg in cfgs[:K]:
            m = sim_exit(cfg, s, e, RISK / K, **kw)
            rs.append(daily(m))
            td = m["trades_df"]
            if td is not None and len(td):
                pnl.append(td["pnl"].to_numpy(float))
        segs.append(pd.DataFrame({j: r for j, r in enumerate(rs)}).fillna(0.0).sum(axis=1))
    return pd.concat(segs), (np.concatenate(pnl) if pnl else np.array([]))


VARIANTS = [
    ("baseline - V7 as deployed",        dict()),
    ("breakeven after 0.5R",             dict(be_r=0.5)),
    ("breakeven after 1.0R",             dict(be_r=1.0)),
    ("breakeven after 1.5R",             dict(be_r=1.5)),
    ("trail 1 ATR after 1.0R",           dict(trail_after_r=1.0, trail_mult=1.0)),
    ("trail 2 ATR after 1.0R",           dict(trail_after_r=1.0, trail_mult=2.0)),
    ("trail 3 ATR after 1.0R",           dict(trail_after_r=1.0, trail_mult=3.0)),
    ("trail 3 ATR after 2.0R",           dict(trail_after_r=2.0, trail_mult=3.0)),
    ("trail 2 ATR after 1.0R + BE 1.0R", dict(trail_after_r=1.0, trail_mult=2.0, be_r=1.0)),
]


if __name__ == "__main__":
    R = S87.rankings()
    print("OVERLAY ONLY - V7's own quarterly selection, reused unchanged; only the "
          "exit differs.\nNothing here is adopted. Judged at the -20% gate.\n")
    print(f"{'variant':>34}{'CAGR':>8}{'MaxDD':>8}{'PF':>6}{'N':>7}{'Shp':>7}"
          f"{'Clm':>7}{'at -20%':>10}{'vs base':>9}")
    base = None
    for tag, kw in VARIANTS:
        r, pl = replay(R, **kw)
        a = np.asarray(r, float)
        s, gt = stats_of(a), at_gate(a)
        pf = pl[pl > 0].sum() / max(-pl[pl < 0].sum(), 1e-9) if len(pl) else float("nan")
        if base is None:
            base = gt["cagr"] * 100
        print(f"{tag:>34}{s['cagr']*100:7.1f}%{s['dd']*100:7.1f}%{pf:6.2f}{len(pl):7d}"
              f"{s['sharpe']:7.2f}{s['calmar']:7.2f}{gt['cagr']*100:9.1f}%"
              f"{gt['cagr']*100-base:+9.1f}", flush=True)
    print("\ndone: exit shape")
