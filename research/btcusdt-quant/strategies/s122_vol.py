"""
S122 - The volatility leg, which this study wrote off on the wrong file.

S114 named the three sources of return available to any book on one instrument:
**direction, carry, volatility.** It then measured carry at 0.2% a year and
recorded volatility as unreachable, on the grounds that "volatility needs options
and this study has five months of them."

That was `opt_1h.parquet` - 158 days, correctly recorded as noise. But
`bvol_1m.parquet` holds **1.66 million minutes of implied volatility from
2023-06 to 2026-09**: 1,178 days, 26 missing, IV ranging 34.9% to 81.0%. Three
and a quarter years of the one input this study declared it did not have. The
volatility leg was never opened; it was written off on the wrong file.

WHY THIS IS THE MOST VALUABLE THING LEFT
----------------------------------------
S116 established that no negatively correlated complement to a directional book
exists *inside direction* - every sleeve shares the same beta, correlations +0.20
to +0.41. That is a statement about direction, and it is exactly why the study
stalled: with one return stream you are stuck with that stream's Calmar.

Volatility is not direction. The variance risk premium is among the most
persistent and highest-Sharpe effects documented anywhere, and it is close to
orthogonal to the direction of the underlying. If there is anything here, it is
the diversifying stream S116 proved could not be found by looking harder at
direction.

FOUR QUESTIONS, IN THE ORDER THAT MATTERS
-----------------------------------------
No options venue is reachable from here, so the premium cannot be sold directly.
That makes the *forecasting* questions more important than the premium itself:

  1  IS THERE A PREMIUM AT ALL
     IV at day t against realised vol over the following 30 days. If implied
     runs above subsequent realised, the premium exists and is worth sizing.

  2  DOES IV FORECAST VOL BETTER THAN TRAILING VOL      <- the quiet one
     Every book in this study sizes off a 32-day EWMA of past returns. If
     implied vol is a better forecast of forward realised vol, then every book
     is mis-sized: too large going into turbulence, too small coming out. Better
     vol forecasting tightens drawdown control, and tighter drawdown control is
     worth more size at the gate, which is worth more return. This improves
     books MECHANICALLY, without needing a single new directional insight.

  3  DOES THE PREMIUM PREDICT RETURNS
     In equities a wide variance premium predicts positive forward returns - it
     is a risk-appetite signal. If it holds here it is a directional signal from
     a non-directional data source.

  4  DO VOL SPIKES MARK CAPITULATION
     A jump in implied vol is fear being priced. Whether that is a level to fade
     is an empirical question and this asks it rather than assuming.

Everything is lagged a full day. Nothing is fitted. Both halves are reported on
every number, because a 3.2-year sample is short and this log has watched one
effect after another live entirely in its first half.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

D = "/home/user/quant/data"


def panel():
    """Daily: price, implied vol, realised vol, funding. IV lagged one day."""
    b = pd.read_parquet(f"{D}/bvol_1m.parquet")
    b["dt"] = pd.to_datetime(b.dt, utc=True)
    iv = b.set_index("dt")["iv"].resample("1D").last() / 100.0   # to a fraction

    d = pd.read_parquet(f"{D}/fut_1h.parquet")
    d["dt"] = pd.to_datetime(d.dt, utc=True)
    px = d.set_index("dt")["close"].resample("1D").last()

    f = pd.read_parquet(f"{D}/funding.parquet")
    f["dt"] = pd.to_datetime(f.dt, utc=True)
    fd = f.set_index("dt")["rate"].resample("1D").sum()

    ix = iv.dropna().index.intersection(px.dropna().index)
    P = pd.DataFrame({"px": px.reindex(ix), "iv": iv.reindex(ix),
                      "fund": fd.reindex(ix).fillna(0.0)}).dropna(subset=["px", "iv"])
    P["r"] = np.log(P.px).diff()
    # trailing realised vol, the estimator every book in this study sizes off
    P["rv32"] = P.r.ewm(halflife=32, adjust=False).std() * np.sqrt(365.25)
    P["rv30"] = P.r.rolling(30).std() * np.sqrt(365.25)
    # FORWARD realised vol over the next 30 days - the thing being forecast
    P["fwd30"] = P.r.shift(-1).rolling(30).std().shift(-29) * np.sqrt(365.25)
    return P.dropna(subset=["r"])


def q1_premium(P):
    print("\n1. IS THERE A VARIANCE RISK PREMIUM")
    m = P.dropna(subset=["iv", "fwd30"])
    prem = m.iv - m.fwd30
    print(f"   implied (mean {m.iv.mean()*100:.1f}%) vs forward realised "
          f"(mean {m.fwd30.mean()*100:.1f}%)")
    print(f"   premium: mean {prem.mean()*100:+.2f} vol points, "
          f"median {prem.median()*100:+.2f}, positive on "
          f"{(prem > 0).mean()*100:.0f}% of days")
    print(f"   {'year':>8}{'implied':>10}{'realised':>10}{'premium':>10}{'% pos':>8}")
    for y, g in m.groupby(m.index.year):
        p = g.iv - g.fwd30
        print(f"   {y:>8}{g.iv.mean()*100:>9.1f}%{g.fwd30.mean()*100:>9.1f}%"
              f"{p.mean()*100:>+9.2f}{(p > 0).mean()*100:>7.0f}%")


def q2_forecast(P):
    print("\n2. DOES IMPLIED VOL FORECAST BETTER THAN TRAILING VOL")
    m = P.dropna(subset=["iv", "rv32", "rv30", "fwd30"])
    y = m.fwd30
    print(f"   forecasting realised vol over the NEXT 30 days, {len(m)} days")
    print(f"   {'forecaster':>22}{'corr':>9}{'MAE':>9}{'bias':>9}"
          f"{'corr 1st':>10}{'corr 2nd':>10}")
    h = len(m) // 2
    for name, x in (("trailing EWMA 32d", m.rv32), ("trailing 30d window", m.rv30),
                    ("IMPLIED vol", m.iv),
                    ("0.5 x implied + 0.5 x EWMA", 0.5 * m.iv + 0.5 * m.rv32)):
        c = float(np.corrcoef(x, y)[0, 1])
        c1 = float(np.corrcoef(x.iloc[:h], y.iloc[:h])[0, 1])
        c2 = float(np.corrcoef(x.iloc[h:], y.iloc[h:])[0, 1])
        mae = float(np.abs(x - y).mean()) * 100
        bias = float((x - y).mean()) * 100
        print(f"   {name:>22}{c:>9.3f}{mae:>8.2f}%{bias:>+8.2f}%{c1:>10.3f}{c2:>10.3f}")


def q3_premium_returns(P):
    print("\n3. DOES THE PREMIUM PREDICT RETURNS")
    m = P.dropna(subset=["iv", "rv32"]).copy()
    # the premium as it is KNOWN on the day: implied against TRAILING realised.
    # forward realised vol is not knowable, so it cannot be in a tradeable signal.
    m["vrp"] = m.iv - m.rv32
    print(f"   signal = implied minus TRAILING realised (knowable on the day), "
          f"lagged 1d")
    print(f"   {'horizon':>10}{'IC':>9}{'IC 1st':>10}{'IC 2nd':>10}"
          f"{'top tercile':>14}{'bot tercile':>14}")
    s = m.vrp.shift(1)
    for hz in (1, 5, 10, 20):
        fwd = np.log(m.px).diff(hz).shift(-hz)
        k = np.isfinite(s) & np.isfinite(fwd)
        x, y = s[k], fwd[k]
        ic = pd.Series(x.to_numpy()).corr(pd.Series(y.to_numpy()), method="spearman")
        hh = len(x) // 2
        i1 = pd.Series(x.iloc[:hh].to_numpy()).corr(
            pd.Series(y.iloc[:hh].to_numpy()), method="spearman")
        i2 = pd.Series(x.iloc[hh:].to_numpy()).corr(
            pd.Series(y.iloc[hh:].to_numpy()), method="spearman")
        t = x.quantile([1 / 3, 2 / 3])
        hi = y[x >= t.iloc[1]].mean() * 100 * (365.25 / hz)
        lo = y[x <= t.iloc[0]].mean() * 100 * (365.25 / hz)
        print(f"   {hz:>9}d{ic:>9.4f}{i1:>10.4f}{i2:>10.4f}"
              f"{hi:>13.1f}%{lo:>13.1f}%")
    print("   (terciles annualised, so they are comparable across horizons)")


def q4_spikes(P):
    print("\n4. DO IMPLIED-VOL SPIKES MARK CAPITULATION")
    m = P.dropna(subset=["iv"]).copy()
    m["dz"] = (m.iv.diff(5) /
               m.iv.diff().rolling(60).std().replace(0, np.nan)).shift(1)
    print(f"   signal = 5-day change in implied vol, standardised, lagged 1d")
    print(f"   {'horizon':>10}{'IC':>9}{'IC 1st':>10}{'IC 2nd':>10}"
          f"{'after spike':>14}{'after crush':>14}")
    for hz in (1, 5, 10, 20):
        fwd = np.log(m.px).diff(hz).shift(-hz)
        k = np.isfinite(m.dz) & np.isfinite(fwd)
        x, y = m.dz[k], fwd[k]
        ic = pd.Series(x.to_numpy()).corr(pd.Series(y.to_numpy()), method="spearman")
        hh = len(x) // 2
        i1 = pd.Series(x.iloc[:hh].to_numpy()).corr(
            pd.Series(y.iloc[:hh].to_numpy()), method="spearman")
        i2 = pd.Series(x.iloc[hh:].to_numpy()).corr(
            pd.Series(y.iloc[hh:].to_numpy()), method="spearman")
        t = x.quantile([0.15, 0.85])
        hi = y[x >= t.iloc[1]].mean() * 100 * (365.25 / hz)
        lo = y[x <= t.iloc[0]].mean() * 100 * (365.25 / hz)
        print(f"   {hz:>9}d{ic:>9.4f}{i1:>10.4f}{i2:>10.4f}"
              f"{hi:>13.1f}%{lo:>13.1f}%")


if __name__ == "__main__":
    P = panel()
    print(f"S122 - the volatility leg\n")
    print(f"daily, {P.index.min().date()} -> {P.index.max().date()}, "
          f"{len(P)} days ({len(P)/365.25:.1f} years)")
    print(f"implied vol: mean {P.iv.mean()*100:.1f}%, "
          f"range {P.iv.min()*100:.1f}% to {P.iv.max()*100:.1f}%")
    q1_premium(P)
    q2_forecast(P)
    q3_premium_returns(P)
    q4_spikes(P)
    print("\ndone: volatility diagnostic")
