"""
S150 - The best single-instrument signal, run across the whole panel.

THE POINT, STATED PRECISELY
---------------------------
S149 measured the two numbers that decide whether a portfolio can help:

    average pairwise correlation of per-asset trend streams   +0.328
    theoretical Sharpe multiplier from combining 16 of them   x1.64
    EMPIRICALLY DELIVERED multiplier                          x1.08

The gap is not a mystery and it is not noise. The formula assumes every stream
has the same Sharpe. They do not - BTC's trend book is far better than ADA's -
and equal-weighting a good stream with fifteen worse ones dilutes it. This log
has now hit that pattern six times.

But S149 also showed where a portfolio DOES pay, and it is not the Sharpe. On
trend30 the Sharpe rose only 8% while the 20%-drawdown gate rose 48%, because
the max drawdown fell from -42% to -32.5%. The gate converts drawdown into
permitted size linearly, so a book that draws down less can be run bigger. That
is the whole portfolio benefit, and it shows up in the constraint the brief
actually imposes.

So the experiment worth running is not "a new factor across sixteen assets". It
is the BEST signal this study has produced, run across sixteen assets instead of
one. That signal is the range-expansion breakout: a bar whose range exceeds the
qth percentile of its own N-day history, taken in the direction of the move,
exited on an ATR trailing stop. On BTC alone, paired with a crowding overlay, it
reached Sharpe 1.46 and a 42.6% gate.

WHAT IS DELIBERATELY NOT DONE HERE
-----------------------------------
No parameter is chosen by looking at the answer. The (N, q) grid is printed in
full, because S142 had to retract a claim built on three sample points of
exactly this surface, which turned out to be spiky rather than smooth. The
headline is the grid AVERAGE across lookbacks, which S146 found to be the one
combination that does not dilute - what is being averaged there is parameter
noise, not signal.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s147_panel as P
import strategies.s148_lowvol as L
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"


def ohlc_panel():
    """Daily OHLC per asset, built from hourly closes.

    High and low are the extremes of the HOURLY closes, not true intrabar
    extremes, so ranges are understated. That is a conservative direction for a
    range-expansion rule - a smaller measured range makes the breakout threshold
    harder to clear, not easier - and it is applied identically to every asset.
    """
    cl = pd.read_parquet(f"{D}/alts_close.parquet")
    b = pd.read_parquet(f"{D}/fut_1h.parquet")
    b["dt"] = pd.to_datetime(b.dt, utc=True); b = b.set_index("dt")
    cl["BTCUSDT"] = b["close"].reindex(cl.index)
    cl = cl.sort_index(axis=1)
    g = cl.resample("1D")
    px, hi, lo = g.last(), g.max(), g.min()
    for d in (px, hi, lo):
        d.index = d.index.tz_localize(None)
    return px, hi, lo


def rex_pos(px, hi, lo, N, q, atr_k=3.0, atr_n=20):
    """Range-expansion breakout with an ATR trailing exit, per asset.

    Everything is lagged: the threshold uses the range quantile through t-1, and
    the resulting position is shifted a further day before it earns anything.
    """
    rng = (hi - lo) / px.replace(0, np.nan)
    thr = rng.rolling(N, min_periods=max(N // 2, 30)).quantile(q).shift(1)
    up = (rng > thr) & (px > px.shift(1))
    dn = (rng > thr) & (px < px.shift(1))
    tr = pd.concat([(hi - lo), (hi - px.shift(1)).abs(),
                    (lo - px.shift(1)).abs()]).groupby(level=0).max()
    atr = tr.rolling(atr_n, min_periods=atr_n // 2).mean()

    out = {}
    for c in px.columns:
        cc = px[c].to_numpy(); aa = atr[c].to_numpy()
        u = np.asarray(up[c].fillna(False)); d = np.asarray(dn[c].fillna(False))
        s = np.zeros(len(cc)); cur = 0.0; peak = 0.0
        for i in range(len(cc)):
            if not np.isfinite(aa[i]) or not np.isfinite(cc[i]):
                s[i] = cur; continue
            if cur == 0.0:
                if u[i]:   cur, peak = 1.0, cc[i]
                elif d[i]: cur, peak = -1.0, cc[i]
            elif cur > 0:
                peak = max(peak, cc[i])
                if cc[i] < peak - atr_k * aa[i]: cur = 0.0
            else:
                peak = min(peak, cc[i])
                if cc[i] > peak + atr_k * aa[i]: cur = 0.0
            s[i] = cur
        out[c] = pd.Series(s, index=px.index)
    return pd.DataFrame(out)


def risk_size(pos, px, ok, target_vol=0.30, vol_win=60, max_pos=2.0):
    """Constant risk contribution per asset, so a volatile coin is held smaller."""
    rv = px.pct_change().rolling(vol_win, min_periods=40).std() * np.sqrt(365.25)
    return (pos * (target_vol / rv.replace(0, np.nan))).clip(-max_pos, max_pos) \
        .where(ok, 0.0).fillna(0.0)


if __name__ == "__main__":
    px_a, hi, lo = ohlc_panel()
    px, dv, ok = P.panel()
    ix = px.index.intersection(px_a.index)
    live = ok.sum(axis=1); first = live[live >= 8].index.min()
    ix = ix[ix >= first]
    px, ok = px.loc[ix], ok.loc[ix]
    px_a, hi, lo = px_a.loc[ix, px.columns], hi.loc[ix, px.columns], lo.loc[ix, px.columns]
    fund = L.funding_panel(list(px.columns), ix)

    print("S150 - the range-expansion breakout across the panel\n")
    print(f"span {ix.min().date()} -> {ix.max().date()}, {len(ix)} days, "
          f"{len(px.columns)} assets, median {int(ok.sum(axis=1).median())} "
          f"tradeable/day\n")

    NS = (55, 89, 144, 233)
    QS = (0.80, 0.90, 0.95)
    print("THE FULL GRID, printed rather than searched. BTC alone vs all 16.")
    print(f"   {'N':>5}{'q':>6} | {'BTC Shp':>9}{'BTC gate':>10} | "
          f"{'16 Shp':>9}{'16 gate':>10}{'16 maxDD':>10}{'mult':>7}")
    store = {}
    for N in NS:
        for q in QS:
            pos = rex_pos(px_a, hi, lo, N, q)
            wb = risk_size(pos[["BTCUSDT"]], px[["BTCUSDT"]], ok[["BTCUSDT"]])
            nb, tb, _ = L.book(px[["BTCUSDT"]], wb, fund[["BTCUSDT"]])
            sb = stats_of(nb.to_numpy(float)); gb = L.gate_of(nb.to_numpy(float))

            w = risk_size(pos, px, ok).div(ok.sum(axis=1).replace(0, np.nan), axis=0)
            n16, t16, _ = L.book(px, w, fund)
            s16 = stats_of(n16.to_numpy(float)); g16 = L.gate_of(n16.to_numpy(float))
            store[(N, q)] = n16
            gbs = "n/a" if not np.isfinite(gb) else f"{gb:>9.1f}%"
            g16s = "n/a" if not np.isfinite(g16) else f"{g16:>9.1f}%"
            print(f"   {N:>5}{q:>6.2f} | {sb['sharpe']:>9.2f}{gbs:>10} | "
                  f"{s16['sharpe']:>9.2f}{g16s:>10}{s16['dd']*100:>9.1f}%"
                  f"{s16['sharpe']/max(sb['sharpe'],1e-9):>7.2f}")

    print("\nTHE AVERAGE ACROSS THE GRID - parameter noise averaged away, which "
          "S146 found\nis the one combination that does not dilute")
    print(f"   {'book':>30}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    avg = sum(store.values()) / len(store)
    L.summarise(avg, pd.Series(0.0, index=avg.index), "average of all 12 configs")
    for q in QS:
        a = sum(store[(N, q)] for N in NS) / len(NS)
        L.summarise(a, pd.Series(0.0, index=a.index), f"average over N at q={q:.2f}")

    print("\nVOL-TARGETED, and combined with the two validated cross-sectional "
          "factors")
    r1 = np.log(px).diff(1)
    lowvol = (-r1.rolling(90, min_periods=60).std()).where(ok)
    carry = (-fund.rolling(3, min_periods=1).mean()).where(ok)
    w_lv = L.weights(lowvol, ok, "betaneutral",
                     (r1.rolling(180, min_periods=90).cov(r1.mean(axis=1))
                      .div(r1.mean(axis=1).rolling(180, min_periods=90).var(), axis=0))
                     .where(ok).clip(0.4, 2.5))
    n_lv, _, _ = L.book(px, w_lv, fund)
    w_cy = L.weights(carry, ok, "zscore")
    n_cy, _, _ = L.book(px, w_cy, fund)

    sleeves = {"rex panel": avg, "low-vol": n_lv, "carry": n_cy}
    print(f"   {'sleeve':>30}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    z = pd.Series(0.0, index=avg.index)
    for k, v in sleeves.items():
        L.summarise(v, z, k)
    C = pd.DataFrame(sleeves).corr()
    print("\n   correlation between sleeves:")
    print("        " + "".join(f"{c:>12}" for c in C.columns))
    for i, row in C.iterrows():
        print(f"   {i:>10}" + "".join(f"{row[c]:>12.3f}" for c in C.columns))

    def at_vol(s, t=0.30):
        sd = float(np.std(s.to_numpy(float))) * np.sqrt(365.25)
        return s * (t / sd) if sd > 0 else s * 0.0
    Z = pd.DataFrame({k: at_vol(v) for k, v in sleeves.items()}).dropna()
    print(f"\n   {'combination':>30}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    for tag, w in (("equal weight, 3 sleeves", np.ones(3)),
                   ("rex + low-vol", np.array([1.0, 1.0, 0.0])),
                   ("rex + carry", np.array([1.0, 0.0, 1.0]))):
        w = w / np.abs(w).sum()
        L.summarise((Z * w).sum(axis=1), z.reindex(Z.index), tag)
    print("\ndone: rex across the panel")
