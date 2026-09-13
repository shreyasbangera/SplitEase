"""
S160 - Delta-neutral funding capture. The structure the portfolio line missed.

WHY THIS FILE EXISTS, AND WHAT I GOT WRONG BEFORE IT
------------------------------------------------------
S158 closed with "the portfolio does not beat one instrument" and an assertion
that 300% at a 20% drawdown needs Sharpe 6.58. Both need qualifying.

The 6.58 came from an empirical law, Calmar = 0.84 x Sharpe^1.53, fitted across
332 books built in THIS study. That is a regularity of the strategies I happened
to build, not a constraint on strategies. Done from theory instead - log equity
with drift g and volatility v has P(maxDD > d) = exp(-2gd/v^2) - 300% at a 20%
drawdown needs Sharpe ~2.0 for a median outcome and ~3.1 for 90% confidence.
Sharpe 2-3 is reachable. The claim that it was impossible was wrong.

And the carry sleeve was measured wrong. Decomposing it:

    funding leg    +74.3%/yr   vol 10.8%   Sharpe  +6.90
    price leg      -35.6%/yr   vol 58.8%   Sharpe  -0.60
    total          +38.7%/yr   vol 57.7%   Sharpe  +0.67

The edge was never the ranking. It is the funding itself, and it was drowned in
price noise six times its size. The cross-sectional funding spread between top
and bottom deciles runs +148.2% a year and is positive on 79% of days.

THE HONEST TENSION, STATED BEFORE THE RESULT
---------------------------------------------
The price leg is NEGATIVE, which is the market doing its job: a perpetual pays
-98.9%/yr funding precisely when it is being violently sold, so a long collects
that carry while wearing the move that caused it. Funding is compensation for
risk, not a gift. The whole question is what fraction of that compensation
survives a hedge that can actually be put on.

Three hedges are tested, in increasing order of how real they are:

  PERFECT      subtract the price leg entirely. Not attainable; reported only as
               the ceiling, and labelled as such.
  BETA         neutralise against the cross-sectional market using trailing
               betas. Attainable, leaves idiosyncratic risk.
  MATCHED      pair each long against a short of similar beta AND similar
               volatility, so the residual is only the difference between two
               similar assets. Attainable, leaves less.

The only structure that hedges price EXACTLY is holding two contracts on the
same underlying - perp against dated quarterly future. Binance lists dated
futures for BTC and ETH only, so that is tested separately and is a two-
instrument trade, not a portfolio.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s151_pit as PIT
import strategies.s153_honest as H
import strategies.s154_stack as ST
import strategies.s148_lowvol as L
from strategies.s96_rank import at_gate, stats_of

FEE_BPS = 5.0


def neutralise(w, factor):
    """Remove exposure to a factor loading, cross-sectionally, each day."""
    num = (w * factor).sum(axis=1)
    den = (factor * factor).sum(axis=1).replace(0, np.nan)
    return (w - factor.mul(num / den, axis=0)).fillna(0.0)


def carry_weights(fund, px, ok, frac=0.10, beta=None, vol=None, mode="beta",
                  max_w=0.05):
    """Long the most negative funding, short the most positive, then hedge."""
    f = fund.rolling(3, min_periods=1).mean().where(ok)
    r = f.rank(axis=1, pct=True)
    lo = (r <= frac) & ok                      # pays the least (or pays you)
    hi = (r > 1 - frac) & ok                   # costs the most to hold long
    wl = lo.astype(float).div(lo.sum(axis=1).replace(0, np.nan), axis=0)
    ws = hi.astype(float).div(hi.sum(axis=1).replace(0, np.nan), axis=0)
    w = (wl - ws).fillna(0.0)
    if mode in ("beta", "matched") and beta is not None:
        w = neutralise(w, beta.where(ok).fillna(0.0))
    if mode == "matched" and vol is not None:
        w = neutralise(w, vol.where(ok).fillna(0.0))
    w = ST.cap_weights(w, max_w)
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0).fillna(0.0)        # unit gross


def run(px, w, fund, slip_bps=5.0, dead=None):
    r = px.pct_change().fillna(0.0)
    wl = w.shift(1).fillna(0.0)
    price = (wl * r).sum(axis=1)
    carry = -(wl * fund.reindex_like(wl).fillna(0.0)).sum(axis=1)
    if dead is not None:
        d = dead.reindex_like(wl).fillna(False).to_numpy()
        v = (wl * r).to_numpy().copy(); v[d & (v > 0)] = 0.0
        price = pd.DataFrame(v, index=wl.index, columns=wl.columns).sum(axis=1)
    turn = (wl - wl.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * (FEE_BPS + slip_bps) / 1e4
    return price, carry, cost, turn


def show(price, carry, cost, turn, tag, w=30):
    net = (price + carry - cost).clip(lower=-0.95)
    a = net.to_numpy(float); s = stats_of(a); g = L.gate_of(a)
    gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
    print(f"   {tag:>{w}}{s['sharpe']:>7.2f}{s['cagr']*100:>9.1f}%"
          f"{s['dd']*100:>8.1f}%{np.std(net)*np.sqrt(365.25)*100:>7.1f}%"
          f"{turn.mean():>7.2f}{gs:>9}")
    return net, s, g


if __name__ == "__main__":
    pan = PIT.build()
    px = pan["close"]; ok = PIT.mask(pan)
    n = ok.sum(axis=1); first = n[n >= 12].index.min()
    px, ok = px.loc[first:], ok.loc[first:]
    pan = {k: v.loc[first:] for k, v in pan.items()}
    fund = H.pit_funding(list(px.columns), px.index)
    dead = ST.death_window(px)
    r1 = np.log(px).diff(1); mkt = r1.mean(axis=1)
    beta = (r1.rolling(180, min_periods=90).cov(mkt)
            .div(mkt.rolling(180, min_periods=90).var(), axis=0)).where(ok).clip(0.2, 4.0)
    vol = r1.rolling(60, min_periods=40).std().where(ok)

    print("S160 - delta-neutral funding capture\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, {len(px)} days, "
          f"median {int(ok.sum(axis=1).median())} tradeable/day\n")

    print("1. HOW MUCH PRICE RISK DOES EACH HEDGE ACTUALLY REMOVE?")
    print(f"   {'hedge':>30}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'vol':>7}"
          f"{'turn':>7}{'at -20%':>9}")
    books = {}
    for mode, tag in (("none", "unhedged (dollar-neutral)"),
                      ("beta", "beta-neutral"),
                      ("matched", "beta + vol matched")):
        w = carry_weights(fund, px, ok, beta=beta, vol=vol, mode=mode)
        p, c, k, t = run(px, w, fund, dead=dead)
        net, s, g = show(p, c, k, t, tag)
        books[mode] = (net, p, c, k, t)
        print(f"   {'':>30}  price leg {np.mean(p)*365.25*100:>+7.1f}%/yr "
              f"(vol {np.std(p)*np.sqrt(365.25)*100:>5.1f}%), funding leg "
              f"{np.mean(c)*365.25*100:>+7.1f}%/yr, costs "
              f"{np.mean(k)*365.25*100:>5.1f}%/yr")
    w = carry_weights(fund, px, ok, beta=beta, vol=vol, mode="beta")
    p, c, k, t = run(px, w, fund, dead=dead)
    print(f"   {'PERFECT hedge (CEILING, not attainable)':>30}")
    net0 = (c - k).clip(lower=-0.95)
    s0 = stats_of(net0.to_numpy(float)); g0 = L.gate_of(net0.to_numpy(float))
    print(f"   {'funding minus costs only':>30}{s0['sharpe']:>7.2f}"
          f"{s0['cagr']*100:>9.1f}%{s0['dd']*100:>8.1f}%"
          f"{np.std(net0)*np.sqrt(365.25)*100:>7.1f}%{t.mean():>7.2f}"
          + ("      n/a" if not np.isfinite(g0) else f"{g0:>8.1f}%"))

    print("\n2. WIDTH OF THE LEGS - a narrower slice earns more carry per unit")
    print(f"   {'width (each side)':>30}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'vol':>7}"
          f"{'turn':>7}{'at -20%':>9}")
    for fr in (0.05, 0.10, 0.20, 0.33):
        w = carry_weights(fund, px, ok, frac=fr, beta=beta, vol=vol, mode="matched")
        p, c, k, t = run(px, w, fund, dead=dead)
        show(p, c, k, t, f"{fr*100:.0f}%")

    print("\n3. COSTS - funding moves fast, so turnover is the risk to this trade")
    print(f"   {'slippage/side':>30}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'vol':>7}"
          f"{'turn':>7}{'at -20%':>9}")
    for sl in (2.0, 5.0, 10.0, 20.0, 40.0):
        w = carry_weights(fund, px, ok, beta=beta, vol=vol, mode="matched")
        p, c, k, t = run(px, w, fund, slip_bps=sl, dead=dead)
        show(p, c, k, t, f"{sl:.0f}bps")

    print("\n4. HOLD LONGER to cut turnover - does the carry survive staleness?")
    print(f"   {'rebalance every':>30}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'vol':>7}"
          f"{'turn':>7}{'at -20%':>9}")
    import strategies.s156_composite as CP
    for ev in (1, 2, 3, 7, 14):
        w = CP.rebalance(carry_weights(fund, px, ok, beta=beta, vol=vol,
                                       mode="matched"), ev)
        p, c, k, t = run(px, w, fund, dead=dead)
        show(p, c, k, t, f"{ev} day(s)")

    print("\n5. YEAR BY YEAR, beta+vol matched, daily, 5bps slip")
    w = carry_weights(fund, px, ok, beta=beta, vol=vol, mode="matched")
    p, c, k, t = run(px, w, fund, dead=dead)
    net = (p + c - k).clip(lower=-0.95)
    for y, v in net.groupby(net.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"   {y}: {((1+v).prod()-1)*100:>+8.1f}%   in-year DD {dd*100:>6.1f}%"
              f"   ({len(v)} days)")
    print("\ndone: delta-neutral carry")
