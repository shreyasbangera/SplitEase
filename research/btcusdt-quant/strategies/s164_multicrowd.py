"""
S164 - V7's signal family, ported to eight other instruments.

THE OPENING S163 LEFT
----------------------
S163 reached 217.0% at a realised 20% drawdown by weighting V7 at 70% against
four weak sleeves. The binding constraint is now the tail penalty: the idealised
frontier at Sharpe 2.30 is 525% and the book delivers 217%, a 2.4x gap. Volatility
targeting and a drawdown rail were both tested against it and both made the book
worse, so the penalty is not coming from clustering that a scaling rule can undo.

What DOES flatten a tail is averaging independent things. Five sleeves at near-zero
correlation already helped; the reason it did not help more is that four of the
five are Sharpe ~0.5 and dilute at any weight high enough to matter.

What is needed is more streams of COMPARABLE strength, and there is an obvious
source that has been sitting unopened: **V7's signal family is not BTC-specific.**
Of its nine crowding features, seven are properties of a single contract's
positioning data -

    tt_pos +1, tt_acct -1, retail -1, tt_vs_retail +1, taker -1, oi_chg -1,
    funding -1

- and `altmetrics_1h.parquet` carries exactly those fields for ADA, AVAX, BNB,
DOGE, ETH, LINK, SOL and XRP from 2021-12. The eighth, btc_dom, is market-wide
and applies to all of them. Only `basis` is genuinely BTC-only, because Binance
lists dated futures for BTC and ETH alone.

So this builds the same book on eight more instruments, with the same signs, the
same trailing standardisation and the same execution, and asks whether they are
strong enough and different enough to be worth adding.

WHAT WOULD MAKE THIS FAIL, STATED FIRST
----------------------------------------
S119 established that BTC's crowding edge DECAYS - 21.9% in 2021 down to 0.8% by
2025 - so an alt version measured mostly over 2022-2026 is being measured over
exactly the decayed period. And alt positioning series are shorter and noisier.
If the alt books come in at Sharpe 0.3 they are more of what S163 already has too
much of. The test is whether any of them clears Sharpe 1.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd, glob, zipfile, io

import strategies.s118_crowd as C
import strategies.s119_dev as S119
import strategies.s162_maxprofit as MP
import strategies.s158_final as F8
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"
ALTS = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
        "DOGEUSDT", "LINKUSDT"]


def daily(s, how="last"):
    return getattr(s.resample("1D"), how)()


def alt_funding(sym, index):
    fs = sorted(glob.glob(f"{D}/pitfund/{sym}__*.zip")) or \
         sorted(glob.glob(f"{D}/fund2/{sym}/*.zip"))
    parts = []
    for f in fs:
        try:
            with zipfile.ZipFile(f) as z:
                d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        except Exception:
            continue
        d.columns = [str(c).strip().lower() for c in d.columns]
        tc = next((c for c in d.columns if "time" in c), None)
        rc = next((c for c in d.columns if "rate" in c), None)
        if tc is None or rc is None:
            continue
        d = d[pd.to_numeric(d[tc], errors="coerce").notna()]
        if not len(d):
            continue
        t = pd.to_numeric(d[tc])
        unit = "us" if t.max() > 1e14 else ("ms" if t.max() > 1e11 else "s")
        parts.append(pd.Series(pd.to_numeric(d[rc], errors="coerce").to_numpy(),
                               index=pd.DatetimeIndex(pd.to_datetime(
                                   t.to_numpy(), unit=unit, utc=True))))
    if not parts:
        return pd.Series(0.0, index=index)
    r = pd.concat(parts).sort_index()
    r = r[~r.index.duplicated()]
    return r.resample("1D").sum().reindex(index).fillna(0.0)


def alt_panel(sym):
    """The same nine-feature crowding panel S118 builds for BTC, for one alt."""
    a = pd.read_parquet(f"{D}/altmetrics_1h.parquet")
    a["dt"] = pd.to_datetime(a.dt, utc=True)
    a = a[a.symbol == sym].set_index("dt").sort_index()
    if not len(a):
        return None, None, None
    cl = pd.read_parquet(f"{D}/alts_close.parquet")[sym].dropna()
    px = daily(cl).dropna()
    ix = px.index.intersection(daily(a["sum_open_interest_value"]).dropna().index)
    px = px.reindex(ix)
    fd = alt_funding(sym, ix)

    F = pd.DataFrame(index=ix)
    F["tt_pos"] = daily(a["sum_toptrader_long_short_ratio"]).reindex(ix)
    F["tt_acct"] = daily(a["count_toptrader_long_short_ratio"]).reindex(ix)
    F["retail"] = daily(a["count_long_short_ratio"]).reindex(ix)
    F["tt_vs_retail"] = np.log(F.tt_pos / F.retail.clip(lower=1e-6))
    F["taker"] = daily(a["sum_taker_long_short_vol_ratio"]).reindex(ix)
    F["oi_chg"] = daily(a["sum_open_interest_value"]).reindex(ix).pct_change(7)
    F["funding"] = fd
    b = pd.read_parquet(f"{D}/breadth.parquet")
    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.to_datetime(b.index, utc=True)
    b.index = b.index.tz_convert("UTC") if b.index.tz else b.index.tz_localize("UTC")
    F["btc_dom"] = daily(b["btc_dom"]).reindex(ix)
    return px, fd, F


if __name__ == "__main__":
    print("S164 - V7's signal family on eight other instruments\n")
    print("same nine features, same signs, same trailing standardisation, same")
    print("execution as the BTC book. Only `basis` is dropped - no dated futures"
          "\nexist for alts.\n")

    books = {}
    print(f"{'instrument':>12}{'from':>12}{'to':>12}{'days':>7}{'feat':>6}"
          f"{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'gate':>9}")
    for sym in ALTS:
        try:
            px, fd, F = alt_panel(sym)
        except Exception as e:
            print(f"{sym:>12}  ERR {e}"); continue
        if px is None or len(px) < 400:
            print(f"{sym:>12}  insufficient data"); continue
        cols = [c for c in C.SIGNS if c in F.columns and F[c].notna().sum() > 300]
        start = F[cols].dropna(thresh=max(len(cols) - 1, 1)).index.min()
        px, fd, F = px[px.index >= start], fd[fd.index >= start], F[F.index >= start]
        sig = C.zsig(F, cols, 365)
        res = C.run(px, fd, sig, 0.30)
        net = res[0] if isinstance(res, tuple) else res
        n = pd.Series(np.asarray(net, float), index=pd.to_datetime(px.index))
        n.index = n.index.tz_localize(None) if n.index.tz else n.index
        n = n.groupby(n.index.normalize()).sum()
        a = n.to_numpy(float); st = stats_of(a); g, _ = MP.realised_gate(a)
        print(f"{sym:>12}{str(n.index.min().date()):>12}{str(n.index.max().date()):>12}"
              f"{len(n):>7}{len(cols):>6}{st['sharpe']:>7.2f}{st['cagr']*100:>8.1f}%"
              f"{st['dd']*100:>7.1f}%"
              + (f"{g:>8.1f}%" if np.isfinite(g) else f"{'n/a':>9}"))
        books[sym.replace("USDT", "")] = n

    if not books:
        print("\nno alt book built. nothing to add."); sys.exit(0)

    S = MP.sleeves()
    S.update(books)
    R = pd.DataFrame(S).dropna()
    print(f"\nintersection with the existing sleeves: {R.index.min().date()} -> "
          f"{R.index.max().date()}, {len(R)} days ({len(R)/365.25:.1f} years)")
    C_ = R.corr()
    print("\ncorrelation to V7 and to btc_rex_crowd:")
    for c in R.columns:
        if c in ("V7", "btc_rex_crowd"):
            continue
        print(f"   {c:>14}  vs V7 {C_.loc[c,'V7']:>+6.3f}   "
              f"vs btc_rex {C_.loc[c,'btc_rex_crowd']:>+6.3f}")
    rho = C_.to_numpy()[np.triu_indices(len(C_), 1)]
    print(f"\n   average pairwise correlation across all {len(C_)} sleeves "
          f"{np.nanmean(rho):+.3f}, worst {np.nanmax(rho):+.3f}")

    Z = pd.DataFrame({c: F8.at_vol(R[c], MP.TARGET_VOL) for c in R.columns})
    others = [c for c in Z.columns if c != "V7"]
    print(f"\nV7 WEIGHT SWEEP with the alt books added "
          f"({len(others)} non-V7 sleeves)")
    print(f"   {'V7 weight':>22}{'Shp':>7}{'realised':>11}{'honest':>10}{'maxDD':>8}")
    best = None
    for wv in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
        rest = (1 - wv) / max(len(others), 1)
        comb = Z["V7"] * wv + Z[others].sum(axis=1) * rest
        a = comb.to_numpy(float); st = stats_of(a)
        rg, sc = MP.realised_gate(a); hg, _ = F8.honest_gate(a)
        print(f"   {f'{wv*100:.0f}% V7':>22}{st['sharpe']:>7.2f}"
              + (f"{rg:>10.1f}%" if np.isfinite(rg) else f"{'n/a':>11}")
              + (f"{hg:>9.1f}%" if np.isfinite(hg) else f"{'n/a':>10}")
              + f"{st['dd']*100:>8.1f}%")
        if np.isfinite(rg) and (best is None or rg > best[0]):
            best = (rg, wv, comb, sc, hg)

    if best:
        rg, wv, comb, sc, hg = best
        scaled = comb * sc
        s = stats_of(scaled.to_numpy(float))
        print(f"\n{'='*74}\nBEST: {wv*100:.0f}% V7 / {(1-wv)*100:.0f}% across "
              f"{len(others)} sleeves")
        print(f"{'='*74}")
        print(f"   Sharpe {s['sharpe']:.2f}   realised gate {rg:.1f}%   "
              f"honest gate {hg:.1f}%")
        for y, v in scaled.groupby(scaled.index.year):
            eq = (1 + v).cumprod(); dd = (eq / eq.cummax() - 1).min()
            print(f"      {y}: {((1+v).prod()-1)*100:>+9.1f}%   in-year DD "
                  f"{dd*100:>6.1f}%   ({len(v)} days)")
        idl = 2 * 0.20 * s["sharpe"] / (np.log(2) + 0.20)
        g = s["sharpe"] * idl - idl * idl / 2
        print(f"   idealised {(np.exp(g)-1)*100:.0f}%  ->  tail penalty "
              f"{(np.exp(g)-1)/max(s['cagr'],1e-9):.1f}x")
        print(f"   target 300%: "
              + ("REACHED" if s["cagr"] * 100 >= 300
                 else f"{300/max(s['cagr']*100,1e-9):.2f}x short"))
    print("\ndone: multi-instrument crowding")
