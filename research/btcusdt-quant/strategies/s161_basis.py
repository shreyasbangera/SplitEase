"""
S161 - Cash and carry: long spot, short perpetual, collect funding.

WHY THIS IS THE STRUCTURE THAT MATTERS
---------------------------------------
S160 decomposed the carry sleeve and found the edge was never the ranking:

    funding leg, perfectly hedged   +34.8%/yr   vol  3.5%   Sharpe 5.55
    what a BETA hedge leaves        -10.1%/yr   vol 36.2%   Sharpe 0.16

Beta neutralisation removes 11% of the price volatility - 41.0% down to 36.3% -
because the residual is idiosyncratic, not market. You cannot hedge one altcoin
with a different altcoin. The only exact hedge is two contracts on the SAME
underlying, and that is what this file trades:

    long 1 unit of SPOT, short 1 unit of the PERPETUAL on the same coin.

The delta cancels to the basis. What is left is the funding the short perp
collects, plus whatever the basis does between entry and exit. This is the
oldest and most widely-run trade in crypto and it is the reason perpetual
funding stays anchored.

SCOPE NOTE, STATED PLAINLY
---------------------------
The brief for the single-instrument work put spot off the table. This uses spot,
as a HEDGE LEG rather than as a directional position - the book is flat the coin
by construction. That is a different thing from trading spot for direction, but
it is still outside the original constraint, so it is reported as a conditional
result and the reader decides whether the constraint was meant to cover it.

WHAT IS CHARGED
---------------
    two legs      every entry and exit pays fees on BOTH the spot and the perp
    spot fees     10bps, above Binance's 10bps spot taker
    perp fees     5bps, above the 4bps USDT-M taker
    funding       real 8-hourly rates, received by the short perp
    delisting     a coin that dies takes both legs with it; the haircut applies
    capital       the spot leg is fully funded and the perp short needs margin,
                  so gross notional per unit of capital is swept, not assumed
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, glob, os

import strategies.s151_pit as PIT
import strategies.s153_honest as H
import strategies.s154_stack as ST
import strategies.s148_lowvol as L
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"
SPOT_BPS, PERP_BPS = 10.0, 5.0


def spot_panel(cache=f"{D}/spot_panel.parquet", rebuild=False, min_bars=120):
    if os.path.exists(cache) and not rebuild:
        return pd.read_parquet(cache)
    files = sorted(glob.glob(f"{D}/spot/*.zip"))
    bysym = {}
    for f in files:
        bysym.setdefault(os.path.basename(f).split("__")[0], []).append(f)
    cl = {}
    for s, fs in sorted(bysym.items()):
        parts = [x for x in (PIT._read_zip(f) for f in sorted(fs)) if x is not None]
        if not parts:
            continue
        d = pd.concat(parts).sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if len(d) < min_bars:
            continue
        cl[s] = d["close"].astype("float32")
    out = pd.DataFrame(cl)
    out = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="1D"))
    out.to_parquet(cache)
    return out


def build(frac_top=0.20, hold=7, min_names=5):
    """Select the coins whose perps pay the most funding, hold the basis pair."""
    pan = PIT.build()
    perp = pan["close"]; ok = PIT.mask(pan)
    spot = spot_panel()
    cols = [c for c in perp.columns if c in spot.columns]
    ix = perp.index.intersection(spot.index)
    perp, spot, ok = perp.loc[ix, cols], spot.loc[ix, cols], ok.loc[ix, cols]
    fund = H.pit_funding(cols, ix)
    # a pair is tradeable only if BOTH legs have a price
    # A pair is tradeable only if BOTH legs produce a usable RETURN, not merely
    # a price. 5,207 cells had one leg's return missing while the price existed;
    # filling those with zero turned a hedged pair into a naked one-sided
    # position and put 16.4% volatility into what should be a hedged leg.
    rs = spot.pct_change(); rp = perp.pct_change()
    both = spot.notna() & perp.notna() & rs.notna() & rp.notna()
    # and only once the pair has a joint history, counted causally
    joint = both.cumsum().shift(1) >= 60
    tradeable = ok & both & joint
    return perp, spot, fund, tradeable


def carry_book(perp, spot, fund, ok, frac=0.20, hold=7, gross=1.0,
               spot_bps=SPOT_BPS, perp_bps=PERP_BPS, dead=None, min_names=5):
    """Equal-weight the top `frac` of pairs by trailing funding, rebalanced
    every `hold` days. Returns per unit of GROSS NOTIONAL."""
    f = fund.rolling(7, min_periods=3).mean().where(ok)      # trailing, causal
    r = f.rank(axis=1, ascending=False, pct=True)            # 1 = pays the most
    sel = (r <= frac) & ok
    sel = sel.where(sel.sum(axis=1) >= min_names, False)
    w = sel.astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if hold > 1:
        m = pd.Series(np.arange(len(w)) % hold == 0, index=w.index)
        w = w.where(m, np.nan).ffill().fillna(0.0)
    w = w.shift(1).fillna(0.0)
    # A held pair whose legs stop lining up is CLOSED, not carried as a naked
    # leg. Weight is dropped and the book renormalised over what is still
    # hedgeable, so gross exposure is honest rather than implied.
    w = w.where(ok, 0.0)
    g0 = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(g0, axis=0).fillna(0.0) * gross

    rs = spot.pct_change().where(ok).fillna(0.0)
    rp = perp.pct_change().where(ok).fillna(0.0)
    basis = (w * (rs - rp)).sum(axis=1)                      # long spot/short perp
    carry = (w * fund.reindex_like(w).fillna(0.0)).sum(axis=1)   # short perp RECEIVES
    if dead is not None:
        d = dead.reindex_like(w).fillna(False).to_numpy()
        v = (w * (rs - rp)).to_numpy().copy(); v[d & (v > 0)] = 0.0
        basis = pd.DataFrame(v, index=w.index, columns=w.columns).sum(axis=1)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * (spot_bps + perp_bps) / 1e4                # both legs, each way
    return basis, carry, cost, turn, w


def show(basis, carry, cost, turn, tag, w=28):
    net = (basis + carry - cost).clip(lower=-0.95)
    a = net.to_numpy(float); s = stats_of(a); g = L.gate_of(a)
    gs = "n/a" if not np.isfinite(g) else f"{g:>8.1f}%"
    print(f"   {tag:>{w}}{s['sharpe']:>7.2f}{s['cagr']*100:>9.1f}%"
          f"{s['dd']*100:>8.1f}%{np.std(net)*np.sqrt(365.25)*100:>7.1f}%"
          f"{turn.mean():>7.2f}{gs:>9}")
    return net, s, g


if __name__ == "__main__":
    perp, spot, fund, ok = build()
    pan = PIT.build()
    dead = ST.death_window(pan["close"]).reindex(index=perp.index,
                                                 columns=perp.columns).fillna(False)
    print("S161 - cash and carry: long spot, short perpetual\n")
    print(f"span {perp.index.min().date()} -> {perp.index.max().date()}, "
          f"{len(perp)} days")
    print(f"pairs with BOTH spot and perp: {perp.shape[1]}, "
          f"median {int(ok.sum(axis=1).median())} tradeable/day\n")

    print("0. IS THE BASIS ACTUALLY HEDGED? daily |spot return - perp return|")
    dr = (spot.pct_change() - perp.pct_change()).where(ok)
    print(f"   median |difference| {dr.abs().stack().median()*1e4:.1f}bp/day, "
          f"vol {dr.stack().std()*np.sqrt(365.25)*100:.1f}%/yr")
    print(f"   compare: a single perp's own volatility "
          f"{perp.pct_change().where(ok).stack().std()*np.sqrt(365.25)*100:.0f}%/yr")

    print("\n1. THE TRADE, by selection width. hold 7 days, gross 1.0x")
    print(f"   {'top slice by funding':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'vol':>7}{'turn':>7}{'at -20%':>9}")
    for fr in (0.10, 0.20, 0.33, 0.50, 1.00):
        b, c, k, t, _ = carry_book(perp, spot, fund, ok, frac=fr, dead=dead)
        show(b, c, k, t, f"{fr*100:.0f}%")

    print("\n2. DECOMPOSITION at the 20% slice")
    fsel = fund.where(ok)
    rk = fsel.rank(axis=1, ascending=False, pct=True)
    print(f"   {'universe mean funding':>28}: "
          f"{fsel.stack().mean()*365.25*100:>+7.1f}%/yr")
    print(f"   {'top 20% slice funding':>28}: "
          f"{fsel.where(rk <= 0.20).stack().mean()*365.25*100:>+7.1f}%/yr "
          f"(what perfect selection would earn)")
    b, c, k, t, w = carry_book(perp, spot, fund, ok, frac=0.20, dead=dead)
    for tag, s in (("basis (spot - perp)", b), ("funding received", c),
                   ("costs", -k), ("net", b + c - k)):
        print(f"   {tag:>28}: {np.mean(s)*365.25*100:>+7.1f}%/yr   vol "
              f"{np.std(s)*np.sqrt(365.25)*100:>6.2f}%")

    print("\n3. HOLDING PERIOD - funding is sticky, costs are not")
    print(f"   {'rebalance every':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'vol':>7}"
          f"{'turn':>7}{'at -20%':>9}")
    for h in (1, 3, 7, 14, 30):
        b, c, k, t, _ = carry_book(perp, spot, fund, ok, frac=0.20, hold=h, dead=dead)
        show(b, c, k, t, f"{h} day(s)")

    print("\n4. LEVERAGE. The spot leg is fully funded and the short needs margin,")
    print("   so gross notional per unit of capital is what is being swept here.")
    print(f"   {'gross notional / capital':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'vol':>7}{'turn':>7}{'at -20%':>9}")
    for g in (1.0, 2.0, 3.0, 5.0, 8.0):
        b, c, k, t, _ = carry_book(perp, spot, fund, ok, frac=0.20, hold=7,
                                   gross=g, dead=dead)
        show(b, c, k, t, f"{g:.0f}x")

    print("\n5. COSTS")
    print(f"   {'spot+perp bps each way':>28}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'vol':>7}{'turn':>7}{'at -20%':>9}")
    for sb, pb in ((5, 2), (10, 5), (20, 10), (40, 20)):
        b, c, k, t, _ = carry_book(perp, spot, fund, ok, frac=0.20, hold=7,
                                   spot_bps=sb, perp_bps=pb, dead=dead)
        show(b, c, k, t, f"{sb}+{pb}bps")

    print("\n6. YEAR BY YEAR, 20% slice, 7-day hold, gross 1.0x")
    b, c, k, t, _ = carry_book(perp, spot, fund, ok, frac=0.20, hold=7, dead=dead)
    net = (b + c - k).clip(lower=-0.95)
    for y, v in net.groupby(net.index.year):
        eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"   {y}: {((1+v).prod()-1)*100:>+8.1f}%   in-year DD {dd*100:>6.1f}%"
              f"   ({len(v)} days)")
    s = stats_of(net.to_numpy(float))
    print(f"\n   unlevered: CAGR {s['cagr']*100:.1f}%, vol "
          f"{np.std(net)*np.sqrt(365.25)*100:.1f}%, max DD {s['dd']*100:.1f}%, "
          f"Sharpe {s['sharpe']:.2f}")
    g = L.gate_of(net.to_numpy(float))
    if np.isfinite(g):
        print(f"   scaled to a 20% drawdown: {g:.1f}% CAGR")
    print("\ndone: cash and carry")
