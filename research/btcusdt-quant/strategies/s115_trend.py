"""
S115 - V7's opposite. A trend sleeve, judged as a partner rather than a rival.

Spot is off the table, so the instrument is one futures contract, and a single
futures contract offers exactly three sources of return: **direction, carry, and
volatility**. S114 measured the carry between perp and quarterly at 0.2% a year -
absent. Volatility needs options, and this study has five months of them, which
S36's successor called noise. So whatever is left has to be directional.

That sounds like the end, because 110 experiments have been directional. It is
not, because they have all been the SAME direction:

    V7 = flow, cmpx, btcdom, fundz, posn

Four of those five are crowding measures. V7 is a **fade-the-crowd** book. It
sells what the crowd is buying and buys what it is dumping, and its single worst
episode (S84) was *29 shorts into the strongest trend of the sample* - which is
precisely what a crowding book does wrong, and precisely what a trend book does
right. The trend gate was bolted on afterwards to stop the bleeding. A trend
SLEEVE was never built.

WHY A WEAK SLEEVE MIGHT STILL WIN HERE
--------------------------------------
S4 built a Donchian trend book and rejected it: best Calmar 0.97, CAGR 8.0%,
DD -9.5%. That verdict was correct **as a standalone book**, and it was reached
years before S102c gave this study a way to judge a partner rather than a rival:

        a second sleeve improves the pair when   S2 > S1 * (sqrt(2 + 2*rho) - 1)

At rho = 0 that asks Sharpe 0.89 of a partner to V7. **At rho = -0.3 it asks
0.39. At rho = -0.5 it asks zero, and below that any positive Sharpe helps.**
Every sleeve tested in this log so far has correlated POSITIVELY with V7 -
0.27 to 0.43 for event bars, 0.56 to 0.65 for other horizons, 0.70 to 0.82 for
other phases - because all of them were V7's signals rearranged. A trend book is
the first candidate with a structural reason to correlate negatively.

DISCIPLINE
----------
The rules are textbook and fixed in advance, and **every span is reported**.
Picking the best span after seeing the table is the in-sample selection that
turned S113's +9.9 into -48.8, so the family either helps as a family or it does
not count. One configuration per rule, chosen from standard practice and not
from results: 3 ATR stop, no take-profit (a target caps the winners, which is
where trend earns), exit when the signal flips.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import research.harness as H
from research.harness import backtest, OOS_END
import strategies.s45_single as S45
import strategies.s69_calsel as S69
import strategies.s87_combined as S87
import strategies.s110_meta as M
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

RISK = 0.08
STOP_ATR, HOLD_D = 3.0, 60
EMAS = (50, 100, 200)
DONCH = (20, 40, 60)


def signals(g):
    """Textbook trend entries on V7's own decision grid. No tuning, no gating."""
    c = pd.Series(g.close.to_numpy(float))
    out = {}
    for n in EMAS:
        e = c.ewm(span=n, adjust=False).mean()
        out[f"EMA{n}"] = np.where(c > e, 1.0, -1.0)
    for n in DONCH:
        hi = c.rolling(n).max().shift(1)
        lo = c.rolling(n).min().shift(1)
        s = np.where(c >= hi, 1.0, np.where(c <= lo, -1.0, np.nan))
        out[f"DC{n}"] = pd.Series(s).ffill().fillna(0.0).to_numpy()
    return out


def run(g, u, risk):
    a = g.atr14.to_numpy(float)
    u = np.nan_to_num(np.asarray(u, float))
    arr = dict(entry=u, stop=STOP_ATR * a,
               tp=1e6 * a,                      # no target: never let it cap a winner
               exit=(np.abs(u) <= 0.0).astype(float))
    H._ctxc.clear()
    return backtest(g, arr, "12h", start=S45.FULL_START, end=OOS_END, risk=risk,
                    max_lev=10.0, max_bars_h=HOLD_D * 24)


def line(tag, r, pl=None, base=None, rho=None, need=None):
    a = np.asarray(r, float)
    st = stats_of(a)
    h = len(a) // 2
    f_, h1, h2 = (at_gate(a)["cagr"] * 100, at_gate(a[:h])["cagr"] * 100,
                  at_gate(a[h:])["cagr"] * 100)
    extra = ""
    if rho is not None:
        extra = f"{rho:+8.3f}{need:8.2f}{'YES' if st['sharpe'] > need else 'no':>6}"
    d = "" if base is None else f"{f_-base[0]:+9.1f}{min(h1-base[1], h2-base[2]):+12.1f}"
    print(f"{tag:>24}{st['cagr']*100:8.1f}%{st['dd']*100:8.1f}%{st['sharpe']:7.2f}"
          f"{st['calmar']:7.2f}{f_:9.1f}%{h1:9.1f}%{h2:9.1f}%{extra}{d}", flush=True)
    return (f_, h1, h2)


if __name__ == "__main__":
    M.use_clock()
    g = S69.ctx()["g"]
    v7, _ = S87.blend(S87.rankings(), 3, RISK)
    v7 = pd.Series(np.asarray(v7, float), index=v7.index)
    s7 = stats_of(v7.to_numpy())["sharpe"]
    print(f"V7 Sharpe {s7:.2f}. A partner needs S2 > {s7:.2f} x (sqrt(2+2rho) - 1):")
    print("   rho  +0.4  +0.2   0.0  -0.2  -0.3  -0.4  -0.5")
    print("  needs " + " ".join(f"{s7*(np.sqrt(2+2*r)-1):5.2f}"
                                for r in (0.4, 0.2, 0.0, -0.2, -0.3, -0.4, -0.5)))

    S = signals(g)
    print(f"\n{'trend sleeve':>24}{'CAGR':>9}{'MaxDD':>8}{'Shp':>7}{'Clm':>7}"
          f"{'at -20%':>9}{'1st h':>9}{'2nd h':>9}{'rho v7':>8}{'needs':>8}{'ok':>6}")
    sleeves = {}
    for tag, u in S.items():
        m = run(g, u, RISK)
        d = S69.daily(m)
        both = pd.DataFrame({"a": v7, "b": d}).fillna(0.0)
        rho = float(both.corr().iloc[0, 1])
        need = s7 * (np.sqrt(2 + 2 * rho) - 1)
        sleeves[tag] = d
        line(f"{tag}  ({m['trades']} trades)", d, rho=rho, need=need)

    print(f"\n{'blended 50/50 with V7':>24}{'CAGR':>9}{'MaxDD':>8}{'Shp':>7}{'Clm':>7}"
          f"{'at -20%':>9}{'1st h':>9}{'2nd h':>9}{'vs V7':>9}{'worse half':>12}")
    base = line("V7 alone", v7)
    v7h, _ = S87.blend(S87.rankings(), 3, RISK / 2)
    v7h = pd.Series(np.asarray(v7h, float), index=v7h.index)
    for tag in S:
        half = S69.daily(run(g, S[tag], RISK / 2))
        comb = pd.DataFrame({"a": v7h, "b": half}).fillna(0.0).sum(axis=1)
        line(f"V7 + {tag}", comb, base=base)
    print("\ndone: trend sleeve")
