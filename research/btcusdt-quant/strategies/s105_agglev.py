"""
S105 - The cap nobody applied to the account that actually exists.

S103 explains V7's maximum drawdown completely, and the explanation is not about
signals. On 2024-07-14 all three sleeves entered the same long at the same
minute, at 1.07x, 0.91x and 1.07x of their own equity. The one real account was
therefore holding **3.05x**. BTC then fell 4.1% over three days and the book lost
12.3% - the number that sets the -20% gate and therefore the entire headline.
The position closed two weeks later for +3,759.

Two measurements in this log missed it, and both missed it the same way.

    S99  "the 10x leverage cap binds on 0.0% of trades, median leverage 0.25x"
         - measured PER SLEEVE. The blend is simulated as three accounts whose
           daily returns are summed, so no quantity in it ever reports what the
           netted account holds. Three sleeves at 0.25x is 0.75x; the cap they
           share is 10x each, so the account's effective cap is 30x.

    S71  "the |net| cap of 3.0 is inert - above 3 it never binds"
         - true of the cap itself, but the conviction scaling it permits means a
           single high-conviction bar can carry THREE TIMES the nominal risk
           budget. At 8% that is 24% of equity risked to the stop, on one bar.

So the book has two ceilings that were each checked in isolation and never
multiplied together. This file caps the product.

    notional   sum over sleeves of |qty| x price / equity      <= cap
    risk       sum over sleeves of risk_i x |u_i|              <= cap

Both are computed from bar t's own conviction, price and ATR and applied to bar
t, so both are causal. Both are implemented by scaling the ENTRY array, which is
the conviction the engine multiplies the risk budget by - no engine change, and
a cap that does nothing is exactly the unmodified book.

WHAT TO EXPECT, SO THE RESULT CANNOT BE READ AFTER THE FACT
-----------------------------------------------------------
S82 found conviction predicts the size of the move. The high-leverage bars are
the high-conviction bars, so a cap cuts precisely the trades the book earns
most from, and the honest prior is that this is another monotone-toward-nothing
family like S94's overlays.

What makes it worth running anyway is that it is not monotone by construction.
S94's vol targets and throttles scale EVERY day; this binds only in a tail and
leaves the other 95% of the record untouched, which is the one shape of
intervention that has never been tested on the denominator.

Overlay only, on V7's own cached quarterly selection. Nothing is adopted here.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from research.harness import backtest
from strategies.s69_calsel import ctx, shape, daily
from strategies.s84_gate import trend
import strategies.s87_combined as S87
from strategies.s96_rank import at_gate, stats_of

RISK, K = 0.08, 3


def shaped(cfg):
    """One sleeve's gated conviction on the full decision grid."""
    p, stp, rr, hold, sp, md = cfg
    u0 = np.nan_to_num(shape(p)); u = u0
    if sp:
        up = trend(sp)
        block = np.zeros(len(u), bool)
        if "s" in md: block |= (u < 0) & up
        if "l" in md: block |= (u > 0) & ~up
        u = np.where(block, 0.0, u)
    return u, u0


def scale_for(cfgs, cap, mode):
    """Per-bar multiplier that brings the ACCOUNT back under `cap`.

    Returns 1.0 everywhere the cap does not bind, so cap=inf reproduces the
    unmodified book exactly rather than approximately.
    """
    if not np.isfinite(cap):
        return np.ones(len(ctx()["g"]))
    c = ctx(); a = c["a"]; px = c["g"].close.to_numpy(float)
    tot = np.zeros(len(a))
    for cfg in cfgs:
        u, _ = shaped(cfg)
        per = RISK / K
        if mode == "risk":
            tot += per * np.abs(u)
        else:                                   # notional / equity
            with np.errstate(divide="ignore", invalid="ignore"):
                tot += per * np.abs(u) * px / (cfg[1] * a)
    s = np.ones(len(a))
    hot = np.isfinite(tot) & (tot > cap)
    s[hot] = cap / tot[hot]
    return s


def sim_capped(cfg, start, end, risk, s):
    u, u0 = shaped(cfg)
    a = ctx()["a"]; stp, rr, hold = cfg[1], cfg[2], cfg[3]
    arr = dict(entry=u * s, stop=stp * a, tp=stp * rr * a,
               exit=(np.abs(u0) <= 0.0).astype(float))
    return backtest(ctx()["g"], arr, "12h", start=start, end=end, risk=risk,
                    max_lev=10.0, max_bars_h=hold * 24)


def replay(R, cap, mode):
    segs, pnl, bind = [], [], []
    for s, e, cfgs in R:
        sc = scale_for(cfgs[:K], cap, mode)
        m0 = (ctx()["g"].dt >= s).to_numpy() & (ctx()["g"].dt < e).to_numpy()
        bind.append(sc[m0])
        rs = []
        for cfg in cfgs[:K]:
            m = sim_capped(cfg, s, e, RISK / K, sc)
            rs.append(daily(m))
            td = m["trades_df"]
            if td is not None and len(td):
                pnl.append(td["pnl"].to_numpy(float))
        segs.append(pd.DataFrame({j: r for j, r in enumerate(rs)}).fillna(0.0).sum(axis=1))
    b = np.concatenate(bind)
    return pd.concat(segs), (np.concatenate(pnl) if pnl else np.array([])), b


if __name__ == "__main__":
    R = S87.rankings()
    print("OVERLAY ONLY - V7's own quarterly selection, reused unchanged; only the "
          "account-level\ncap differs. Nothing adopted. Judged at the -20% gate.\n")
    print(f"{'cap':>34}{'binds':>8}{'CAGR':>8}{'MaxDD':>8}{'PF':>6}{'N':>7}"
          f"{'Shp':>7}{'Clm':>7}{'at -20%':>10}{'vs base':>9}")
    base = None
    for mode, caps in (("notional", (np.inf, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0)),
                       ("risk", (0.16, 0.12, 0.10, 0.08))):
        for cap in caps:
            r, pl, b = replay(R, cap, mode)
            a = np.asarray(r, float)
            st, gt = stats_of(a), at_gate(a)
            pf = pl[pl > 0].sum() / max(-pl[pl < 0].sum(), 1e-9) if len(pl) else np.nan
            lab = ("baseline - no account cap" if not np.isfinite(cap)
                   else (f"{mode} <= {cap:.1f}x equity" if mode == "notional"
                         else f"{mode} <= {cap*100:.0f}% of equity"))
            if base is None:
                base = gt["cagr"] * 100
            print(f"{lab:>34}{(b < 1).mean()*100:7.1f}%{st['cagr']*100:7.1f}%"
                  f"{st['dd']*100:7.1f}%{pf:6.2f}{len(pl):7d}{st['sharpe']:7.2f}"
                  f"{st['calmar']:7.2f}{gt['cagr']*100:9.1f}%"
                  f"{gt['cagr']*100-base:+9.1f}", flush=True)
    print("\ndone: account-level cap")
