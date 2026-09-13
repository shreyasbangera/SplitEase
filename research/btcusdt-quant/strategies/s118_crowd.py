"""
S118 - Crowding inputs through a managed-futures architecture. A third strategy.

Two things are now established independently:

    S117   price alone on BTCUSDT is worth **Sharpe 1.0** under a
           vol-targeted, continuously-sized, stopless architecture
    V7     crowding data is worth **Sharpe 2.15** under a thresholded,
           ATR-stop-sized, quarterly-selected architecture

Nobody has crossed them. This is the crowding inputs under S117's architecture,
and it is neither of the two books above:

    no thresholds          V7 turns each z-score into -1/0/+1 outside a band.
                           Here the signal is continuous, so a signal at 0.9
                           sigma is not discarded and one at 1.1 is not treated
                           as identical to one at 3.
    no conviction exponent no |net|^p reshaping.
    no ATR stops           V7 sizes to a stop and is stopped out. This sizes to
                           VOLATILITY and is never stopped out at all.
    no quarterly selection no 200-cell grid, no top-3 blend, no trend gate, and
                           therefore none of S96c's 24-point selection noise and
                           neither of S96b's or S106b's oracle bounds, which are
                           statements about a selector this book does not have.
    daily, not 12h         and no clock phase to be fragile about (S101).

The combination rule is a plain equal-weight average of standardised features
with **nothing fitted** - no weights, no regression, no selection. S106b showed
equal weights beat every informed weighting of V7's signals by 8 sd, so this is
the choice that result argues for, applied to a different architecture.

FEATURES, AND WHY THESE
-----------------------
Built from the raw published series rather than V7's five engineered signals, so
the two do not share construction:

    tt_pos, tt_acct     top traders' position and account ratios - the informed
    retail_acct         all-account ratio - the crowd
    tt_vs_retail        the log gap between them
    taker_ratio         aggressive buy vs sell volume
    oi_chg              open-interest change: leverage building or flushing
    funding             what longs pay to be long
    basis               annualised front-quarterly basis - the term structure
    btc_dom             BTC dominance

Each is z-scored on a trailing window, lagged a full day, clipped, and averaged.
Positive means "the crowd is short and the informed are long", which is the
direction the whole study says pays.

Judged standalone against the brief: 300% at a 20% drawdown, 100+ trades,
profit factor above 1.10. Funding and costs charged as in S117.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from engine.data import load
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

FEE_BPS, SLIP_BPS = 5.0, 3.0
D = "/home/user/quant/data"
ZWIN = 180          # days, trailing standardisation window
CLIP = 2.0
VOL_HL = 32
MAX_LEV = 3.0
BAND = 0.10


def daily(s, how="last"):
    return getattr(s.resample("1D"), how)()


def panel():
    """One daily frame: price, funding, and the crowding features."""
    d = load("fut_1h").copy(); d["dt"] = pd.to_datetime(d.dt, utc=True)
    px = daily(d.set_index("dt")["close"]).dropna()

    f = load("funding").copy(); f["dt"] = pd.to_datetime(f.dt, utc=True)
    fund_day = daily(f.set_index("dt")["rate"], "sum").reindex(px.index).fillna(0.0)

    m = pd.read_parquet(f"{D}/metrics_1h.parquet")
    m["dt"] = pd.to_datetime(m.dt, utc=True)
    m = m.set_index("dt")
    F = pd.DataFrame(index=px.index)
    F["tt_pos"] = daily(m["tt_pos"]).reindex(px.index)
    F["tt_acct"] = daily(m["tt_acct"]).reindex(px.index)
    F["retail"] = daily(m["retail_acct"]).reindex(px.index)
    F["tt_vs_retail"] = np.log(F.tt_pos / F.retail.clip(lower=1e-6))
    F["taker"] = daily(m["taker_ratio"]).reindex(px.index)
    oi = daily(m["oi"]).reindex(px.index)
    F["oi_chg"] = oi.pct_change(7)
    F["funding"] = fund_day

    q = pd.read_parquet(f"{D}/quarterly_front.parquet")
    q["dt"] = pd.to_datetime(q.dt, utc=True)
    q = q[(q.dte > 2) & (q.basis.abs() < 0.25)]
    F["basis"] = daily(q.set_index("dt")["ann"]).reindex(px.index)

    b = pd.read_parquet(f"{D}/breadth.parquet")
    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.to_datetime(b.index, utc=True)
    b.index = b.index.tz_convert("UTC") if b.index.tz else b.index.tz_localize("UTC")
    F["btc_dom"] = daily(b["btc_dom"]).reindex(px.index)
    return px, fund_day, F


# Sign convention: +1 where a HIGH reading should mean GO LONG.
# Crowd positioning and funding are faded; informed positioning is followed.
SIGNS = {"tt_pos": +1, "tt_acct": -1, "retail": -1, "tt_vs_retail": +1,
         "taker": -1, "oi_chg": -1, "funding": -1, "basis": -1, "btc_dom": +1}


def zsig(F, cols, zwin=ZWIN, clip=CLIP):
    """Equal-weight average of trailing-standardised, lagged, clipped features."""
    parts = []
    for c in cols:
        x = F[c].astype(float)
        z = (x - x.rolling(zwin, min_periods=zwin // 3).mean()) / \
            (x.rolling(zwin, min_periods=zwin // 3).std() + 1e-12)
        parts.append(np.clip(z * SIGNS[c], -clip, clip))
    S = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return S.shift(1).fillna(0.0)               # strictly past


def run(px, fund_day, sig, target_vol, band=BAND, max_lev=MAX_LEV, long_only=False):
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = (r.ewm(halflife=VOL_HL, adjust=False).std().shift(1).bfill()
          .to_numpy() * np.sqrt(365.25))
    rv = np.maximum(rv, 0.05)
    s = sig.to_numpy()
    if long_only:
        s = np.clip(s, 0.0, None)
    want = np.clip(s * target_vol / rv, -max_lev, max_lev)

    pos, cur = np.zeros(len(px)), 0.0
    for i in range(len(px)):
        if abs(want[i] - cur) > band:
            cur = want[i]
        pos[i] = cur

    cost = (FEE_BPS + SLIP_BPS) / 1e4
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    simple = px.pct_change().fillna(0.0).to_numpy()
    net = pos * simple - pos * fund_day.to_numpy() - turn * cost
    return pd.Series(net, index=px.index), int((turn > 1e-9).sum())


def report(tag, r, turns):
    a = np.asarray(r, float)
    a = a[np.isfinite(a)]
    st = stats_of(a); b = bootstrap_dd(a, n=3000, block=90); g = at_gate(a)
    h = len(a) // 2
    g1, g2 = at_gate(a[:h])["cagr"] * 100, at_gate(a[h:])["cagr"] * 100
    print(f"{tag:>32}{st['cagr']*100:8.1f}%{st['dd']*100:8.1f}%{b['dd_median']*100:9.1f}%"
          f"{st['sharpe']:7.2f}{st['calmar']:7.2f}{turns:7d}{g['cagr']*100:9.1f}%"
          f"{g1:9.1f}%{g2:9.1f}%", flush=True)
    return g["cagr"] * 100


if __name__ == "__main__":
    px, fund_day, F = panel()
    cols = [c for c in SIGNS if c in F.columns]
    ok = F[cols].notna().mean()
    start = F[cols].dropna(thresh=len(cols) - 1).index.min()
    px, fund_day, F = px[px.index >= start], fund_day[fund_day.index >= start], \
        F[F.index >= start]
    print(f"daily, {px.index.min().date()} -> {px.index.max().date()} "
          f"({len(px)} days, {len(px)/365.25:.1f} years)")
    print("feature coverage: " + "  ".join(f"{c} {ok[c]*100:.0f}%" for c in cols))
    print(f"equal-weight mean of {len(cols)} standardised features, "
          f"{ZWIN}d window, clipped at {CLIP}, lagged 1 day\n")

    sig = zsig(F, cols)
    print(f"signal: mean {sig.mean():+.3f}  sd {sig.std():.3f}  "
          f"long {100*(sig>0).mean():.0f}% of days\n")

    print(f"{'book':>32}{'CAGR':>9}{'realDD':>8}{'medDD':>9}{'Shp':>7}{'Clm':>7}"
          f"{'turns':>7}{'at -20%':>9}{'1st h':>9}{'2nd h':>9}")
    for tv in (0.20, 0.40, 0.60, 0.80):
        report(f"crowding L/S, tgt vol {tv*100:.0f}%", *run(px, fund_day, sig, tv))
    print()
    for tv in (0.20, 0.40, 0.60):
        report(f"crowding LONG-ONLY, tgt {tv*100:.0f}%",
               *run(px, fund_day, sig, tv, long_only=True))
    print("\ndone: crowding under a managed-futures architecture")
