"""
S95e - Closing the macro class properly: the dollar, not just equity vol.

S95d killed the VIX candidate with a control S95c did not have. Once both books
are put on the same realised drawdown, the six-signal book is 3.1 points WORSE,
and 25.0 points worse over the first half. The apparent Calmar gain was the
equal-weight average cutting every existing signal from 1/5 to 1/6 - the
dilution control reproduces the raw control to three significant figures.

But closing an entire source class on one proxy would be sloppy, and VIX is
equity volatility, which is not the factor most often argued to drive BTC. The
dollar is. So the same pipeline, the same controls, run on a real
DXY-weighted dollar index built from the Fed's H.10 daily rates - all six DXY
constituents, quoted consistently, geometric weights, 1999 to date.

The sign prior, stated before looking, is the strong one in this whole study:
**a stronger dollar should mean a weaker BTC.** It is the most widely asserted
macro relationship in crypto and the one with the clearest mechanism - BTC is a
long-duration dollar-denominated asset and global dollar liquidity is its
funding condition. If macro has anything for this book, this is where it is.

The VIX failure also produced a testable explanation worth carrying over. The
year rows showed s_vixchg helping exactly where the prior said it should - 2022
went +4.9% to +11.2% - and destroying 2023, when the VIX spiked on the regional
banking crisis while BTC rallied *because* of it. If BTC's macro sign is
regime-dependent rather than absent, the dollar should show the same shape: a
useful 2022, a bad 2023. That is checked explicitly below rather than left as a
story.

Judged against the dilution control at the gate, exactly as S95d.
"""
import sys
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd
from scipy import stats

import strategies.s45_single as S
import strategies.s46_net as S46
from research.harness import OOS_END, IS_END
from research.ic import newey_west_t
from research import macro
from strategies.s95d_gate import run, at_gate, line

START = S.FULL_START


def zs(x, n):
    s = pd.Series(np.asarray(x, float))
    return ((s - s.rolling(n).mean()) / s.rolling(n).std()).to_numpy(float)


def fwd(close, h):
    c = pd.Series(np.asarray(close, float))
    return (np.log(c.shift(-h)) - np.log(c)).to_numpy()


def add_dollar(g):
    """Two macro signals on the book's 12h grid, strict lag, house idiom.

    Sign convention: both are stated so that POSITIVE means buy BTC, which is
    the book's convention for every other signal.
    """
    usd = macro.dollar()
    v = macro.onto_bars(g.dt, usd, mode="strict")
    g = g.copy()
    g["usd"] = v
    # dollar strengthening -> short BTC
    g["s_usdchg"] = -zs(pd.Series(np.log(v)).diff(20).to_numpy(float), 240)
    # dollar high against its own range -> short BTC
    g["s_usdlvl"] = -zs(v, 240)
    return g


if __name__ == "__main__":
    g = add_dollar(S.grid(START))
    v = g.usd.to_numpy(float)
    print(f"dollar index on the 12h grid: {np.isfinite(v).mean()*100:.0f}% coverage, "
          f"{g.dt.iloc[0].date()} -> {g.dt.iloc[-1].date()}")
    print(f"  index moved {v[np.isfinite(v)][0]:.4f} -> {v[np.isfinite(v)][-1]:.4f} "
          f"({(v[np.isfinite(v)][-1]/v[np.isfinite(v)][0]-1)*100:+.1f}%)\n")

    # --- 1. is there any IC at all, overlap-corrected ---------------------
    print("1. IC, with the overlap correction S95b established as necessary.")
    print("   Sign prior: POSITIVE IC, because both signals are already written")
    print("   so that positive means buy.\n")
    print(f"{'signal':>10}{'h':>4}{'days':>6}{'IC':>9}{'NW t':>8}"
          f"{'1st':>9}{'2nd':>9}   stable")
    dt = pd.Series(pd.to_datetime(g.dt))
    mid_t = dt.iloc[len(dt) // 2]
    h1 = (dt < mid_t).to_numpy(); h2 = (dt >= mid_t).to_numpy()
    close = g.close.to_numpy(float)
    for name in ("s_usdchg", "s_usdlvl"):
        x = g[name].to_numpy(float)
        for h in (1, 2, 4, 8, 24):
            y = fwd(close, h)
            ok = np.isfinite(x) & np.isfinite(y)
            rho = stats.spearmanr(x[ok], y[ok])[0]
            t = newey_west_t(x[ok], y[ok], lags=max(1, h * 2))
            r1 = stats.spearmanr(x[ok & h1], y[ok & h1])[0]
            r2 = stats.spearmanr(x[ok & h2], y[ok & h2])[0]
            flag = "BOTH" if np.sign(r1) == np.sign(r2) else ""
            print(f"{name:>10}{h:>4}{h/2:6.1f}{rho:+9.4f}{t:+8.2f}"
                  f"{r1:+9.4f}{r2:+9.4f}   {flag}")

    # --- 2. in the book, at the gate, against dilution --------------------
    base5 = S.composite(g, S46.LONG)
    S.THR["usdchg"] = 1.0
    S.THR["usdlvl"] = 1.0
    BOOKS = [("5 signals (control)", base5),
             ("5 signals x 5/6 (dilution)", base5 * (5.0 / 6.0)),
             ("6 signals with usdchg", S.composite(g, S46.LONG + ["usdchg"])),
             ("6 signals with usdlvl", S.composite(g, S46.LONG + ["usdlvl"]))]

    print("\n2. AT THE -20% GATE, risk bisected for every row.\n")
    b0 = None
    res = {}
    for tag, net in BOOKS:
        m = at_gate(g, net)
        res[tag] = m
        b0 = m["cagr"] if b0 is None else b0
        line(tag, m, None if tag.startswith("5 signals (") else b0)

    # --- 3. correlation, which is the entire premise ----------------------
    print("\n3. CORRELATION to the five-signal composite\n")
    for e in ("usdchg", "usdlvl"):
        u = S.unit(g, e)
        ok = np.isfinite(u) & np.isfinite(base5)
        print(f"   s_{e:<8} corr {np.corrcoef(u[ok], base5[ok])[0,1]:+.3f}"
              f"   non-zero on {(np.abs(u)>0).mean()*100:4.1f}% of bars")

    # --- 4. the regime story, checked rather than told --------------------
    print("\n4. YEAR BY YEAR at a common 8% risk. The VIX run helped 2022 and")
    print("   wrecked 2023; if BTC's macro sign is regime-dependent rather than")
    print("   absent, the dollar should show the same shape.\n")
    eq = {}
    for tag, net in BOOKS:
        m = run(g, net, 0.08)
        s = pd.Series(m["equity"], index=pd.to_datetime(m["dt"]))
        eq[tag] = s.resample("1D").last().dropna()
    yrs = sorted({d.year for d in eq[BOOKS[0][0]].index})
    print(f"{'year':>6}" + "".join(f"{t.split(' (')[0][:16]:>18}" for t, _ in BOOKS))
    for y in yrs:
        row = f"{y:>6}"
        for tag, _ in BOOKS:
            s = eq[tag][eq[tag].index.year == y]
            row += f"{(s.iloc[-1]/s.iloc[0]-1)*100:17.1f}%" if len(s) > 1 else f"{'-':>18}"
        print(row)

    print("\ndone: the dollar")
