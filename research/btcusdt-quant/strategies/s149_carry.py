"""
S149 - Cross-sectional carry, and whether sixteen instruments beat one.

TWO QUESTIONS, BOTH OF WHICH S148 LEFT OPEN
--------------------------------------------
S148 built the one factor S147 validated - low volatility - and it reached a
20%-drawdown gate of 11.7%. That is real, it beats buy-and-hold BTC at 6.0%, and
it is nowhere near the brief. But S148 answered a narrower question than the one
the portfolio was opened for, and left two things untested.

  1  CARRY. Funding is the price of leverage in a perpetual, it is quoted
     separately for every contract, and it varies enormously across them. A book
     that is long the contracts paying the least to hold and short the ones
     paying the most collects that spread while carrying no net market position.
     This is the factor a portfolio uniquely enables: there is no
     cross-sectional funding spread in a universe of one. S147 could not test it
     because funding was on disk for 8 of 16 names; it is now fetched for all 16.

  2  DOES BREADTH HELP AT ALL. The single-instrument result that closed the
     previous line of work was Sharpe 1.46. The portfolio question underneath
     the brief is not "is there a new signal" but "does running a signal I
     already trust on sixteen instruments instead of one raise its Sharpe, and
     by how much". For N streams of Sharpe s at average pairwise correlation
     rho the answer is s*sqrt(N/(1+(N-1)rho)), and every term is measurable
     here rather than assumed. This measures it directly by running the SAME
     time-series trend rule on one asset and on all sixteen.

Question 2 is the one that decides whether the portfolio route can reach the
brief at all, so it is answered with the empirical correlation rather than a
hopeful one.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s147_panel as P
import strategies.s148_lowvol as L
from strategies.s96_rank import at_gate, stats_of


def carry_features(fund, px, ok):
    """Funding-based cross-sectional scores. Sign convention: higher = better to
    be LONG, so a low funding rate scores high (a long pays funding)."""
    F = {}
    for k in (1, 3, 7, 30):
        F[f"carry{k}"] = -fund.rolling(k, min_periods=1).mean().where(ok)
    f30 = fund.rolling(30, min_periods=20)
    F["carry_z"] = -((fund - f30.mean()) / (f30.std() + 1e-12)).where(ok)
    F["carry_chg"] = -(fund.rolling(7, min_periods=4).mean()
                       - fund.rolling(30, min_periods=20).mean()).where(ok)
    # funding relative to the cross-section rather than to its own history
    F["carry_rel"] = -(fund.rolling(7, min_periods=4).mean()
                       .sub(fund.rolling(7, min_periods=4).mean().mean(axis=1),
                            axis=0)).where(ok)
    return F


def tsmom_panel(px, fund, cols, lookback=90, target_vol=0.30, slip_bps=5.0,
                vol_win=60, max_pos=2.0):
    """The same time-series trend rule run independently on each asset.

    Position in asset i = sign of its own `lookback`-day return, sized to a
    constant risk contribution by its own realised volatility. This is what a
    managed-futures book does, and it is the construction in which adding
    instruments is supposed to pay.
    """
    lr = np.log(px[cols])
    sig = np.sign(lr.diff(lookback))
    rv = px[cols].pct_change().rolling(vol_win, min_periods=40).std() * np.sqrt(365.25)
    w = (sig * (target_vol / rv.replace(0, np.nan))).clip(-max_pos, max_pos)
    return w.fillna(0.0)


def per_asset_streams(px, fund, ok, lookback=90):
    """Each asset's trend book as its own return stream, so their correlation
    can be measured rather than guessed."""
    out = {}
    for c in px.columns:
        m = ok[c]
        if m.sum() < 400:
            continue
        w = tsmom_panel(px, fund, [c], lookback).where(m, 0.0)
        net, _, _ = L.book(px[[c]], w, fund[[c]])
        out[c] = net
    return pd.DataFrame(out)


if __name__ == "__main__":
    px, dv, ok = P.panel()
    live = ok.sum(axis=1)
    first = live[live >= 8].index.min()
    px, dv, ok = px.loc[first:], dv.loc[first:], ok.loc[first:]
    fund = L.funding_panel(list(px.columns), px.index)
    lr = np.log(px)

    print("S149 - cross-sectional carry, and whether breadth pays\n")
    print(f"span {px.index.min().date()} -> {px.index.max().date()}, "
          f"{len(px)} days, median {int(ok.sum(axis=1).median())} names/day\n")

    print("FUNDING DISPERSION - the raw material for a carry trade")
    fw = fund.where(ok)
    print(f"   cross-sectional mean funding   {fw.stack().mean()*365.25*100:>+7.1f}%/yr")
    sp = (fw.max(axis=1) - fw.min(axis=1))
    print(f"   median cross-sectional spread  {sp.median()*365.25*100:>+7.1f}%/yr "
          f"(high-minus-low funding, annualised)")
    print(f"   days funding is negative somewhere: "
          f"{(fw.min(axis=1) < 0).mean()*100:.0f}%")

    print("\n1. CARRY AS A CROSS-SECTIONAL SIGNAL - IC with the S147 machinery")
    Fc = carry_features(fund, px, ok)
    print(f"   {'feature':>11}" + "".join(f"{'IC'+str(k)+'d':>9}{'t':>7}"
                                          for k in (1, 5, 20)) +
          f"{'1st h':>9}{'2nd h':>9}{'stable':>8}")
    keep = []
    for n, f in Fc.items():
        row, ic5 = "", None
        for k in (1, 5, 20):
            ic = P.xs_ic(f.shift(1), lr.diff(k).shift(-k), ok)
            m, t, _ = P.ic_stat(ic, k)
            row += f"{m:>9.4f}{t:>7.2f}"
            if k == 5:
                ic5 = (ic, m, t)
        v = ic5[0].dropna(); h = len(v) // 2
        i1, i2 = v.iloc[:h].mean(), v.iloc[h:].mean()
        st = (np.sign(i1) == np.sign(i2)) and abs(ic5[2]) > 2.0
        if st:
            keep.append((n, ic5[1], ic5[2]))
        print(f"   {n:>11}{row}{i1:>9.4f}{i2:>9.4f}{'YES' if st else '':>8}")

    if keep:
        print("\n   surrogate control on the survivors (100 phase-randomised "
              "panels)")
        for n, m, t in keep:
            ts = []
            for _ in range(100):
                s = P.phase_randomise(Fc[n])
                ic = P.xs_ic(s.shift(1), lr.diff(5).shift(-5), ok)
                _, tt, _ = P.ic_stat(ic, 5)
                if np.isfinite(tt):
                    ts.append(abs(tt))
            ts = np.array(ts); pv = float((ts >= abs(t)).mean())
            print(f"      {n:>11} real |t| {abs(t):>5.2f}  surrogate median "
                  f"{np.median(ts):>5.2f}, 95th {np.percentile(ts,95):>5.2f}  "
                  f"p = {pv:.3f}  {'REAL' if pv < 0.05 else 'not distinguishable'}")

        print("\n   the carry book itself")
        print(f"   {'construction':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
              f"{'turn':>8}{'at -20%':>9}")
        best = max(keep, key=lambda x: abs(x[2]))[0]
        for mode in ("dollarneutral", "zscore"):
            for tv in (None, 0.20, 0.40):
                w = L.weights(Fc[best], ok, mode)
                net, turn, _ = L.book(px, w, fund, target_vol=tv, band=0.10)
                L.summarise(net, turn, f"{best} {mode} "
                            + ("raw" if tv is None else f"{tv*100:.0f}%"))
    else:
        print("\n   no carry feature clears the bar. nothing is built on it.")

    print("\n2. DOES BREADTH PAY? the SAME trend rule on 1 asset vs on 16")
    print(f"   {'universe':>26}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}{'turn':>8}"
          f"{'at -20%':>9}")
    for lb in (30, 90, 180):
        w1 = tsmom_panel(px, fund, ["BTCUSDT"], lb)
        n1, t1, _ = L.book(px[["BTCUSDT"]], w1, fund[["BTCUSDT"]], target_vol=0.30,
                           band=0.10)
        s1, g1 = L.summarise(n1, t1, f"trend{lb} BTC only")
        cols = list(px.columns)
        wn = tsmom_panel(px, fund, cols, lb).where(ok, 0.0)
        wn = wn.div(len(cols))
        nn, tn, _ = L.book(px, wn, fund, target_vol=0.30, band=0.10)
        mult = stats_of(nn.to_numpy(float))["sharpe"] / max(s1["sharpe"], 1e-9)
        sn, gn = L.summarise(nn, tn, f"trend{lb} all 16",
                             f"   x{mult:.2f} Sharpe")

    print("\n3. WHY THAT MULTIPLIER AND NOT A BIGGER ONE - measured correlation")
    S = per_asset_streams(px, fund, ok, 90)
    C = S.corr()
    rho = C.to_numpy()[np.triu_indices(len(C), 1)]
    N = len(C.columns)
    print(f"   {N} per-asset trend streams, average pairwise correlation "
          f"{np.nanmean(rho):+.3f} (min {np.nanmin(rho):+.2f}, max "
          f"{np.nanmax(rho):+.2f})")
    pred = np.sqrt(N / (1 + (N - 1) * np.nanmean(rho)))
    print(f"   theoretical Sharpe multiplier from combining them: x{pred:.2f}")
    print(f"   so a single-instrument book of Sharpe 1.46 becomes "
          f"{1.46*pred:.2f} at best, against the {np.sqrt(2*np.log(4.0)):.2f} "
          f"the brief needs at a 20% drawdown.")
    print("\ndone: carry and breadth")
