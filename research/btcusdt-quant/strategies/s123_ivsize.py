"""
S123 - Sizing off implied vol instead of trailing vol. The mechanical lever.

S122 found the thing this study has been getting wrong in every single book.
Forecasting realised volatility over the next 30 days:

    trailing EWMA 32d    corr 0.082     <- what V7, S117, S118, S119 all size off
    trailing 30d window  corr 0.255
    IMPLIED vol          corr 0.409

The estimator underneath every strategy here is close to useless, and negatively
correlated with the thing it forecasts across the first half of the sample. Every
book in this log is therefore mis-sized in a specific direction: it carries full
size into turbulence that implied vol had already priced, and shrinks only after
the move has happened. That is precisely the mechanism S103 identified when it
found **94% of the deepest drawdown is mark-to-market on positions still open**.

WHY THIS IS WORTH MORE THAN A BETTER SIGNAL
-------------------------------------------
The binding constraint is drawdown, not return. At the gate the book is scaled
until drawdown sits on 20%, so anything that cuts drawdown without cutting return
by as much is worth size, and size is worth return linearly. A forecast that is
five times better at anticipating vol should let the book be smaller exactly when
being smaller matters and larger the rest of the time.

**A constant bias in the forecast cannot matter here.** Implied vol runs 6.8
points above realised - that is the premium - but sizing is `target / forecast`,
so a uniform bias rescales the whole book and the gate rescales it straight back.
Only the SHAPE of the forecast can change anything, which is what makes this a
clean test of forecast quality rather than of a fitted constant.

WHAT IS TESTED
--------------
The same three books, sized three ways, nothing else touched:

    books       trend (S117's signal), crowding (S119's signal), always-long
    sizing      EWMA 32d           the incumbent
                implied vol        S122's winner
                half and half      best mean absolute error in S122

Implied vol starts 2023-06, so every variant is run on 2023-06 onward and the
incumbent is re-measured on that window too. This is a **3.1 year** comparison
and it excludes 2021-2022 entirely; that is short, and it is stated rather than
buried. Both halves are printed on every row for the same reason.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s122_vol as V
import strategies.s118_crowd as C
import strategies.s119_dev as S119
from research.robust import bootstrap_dd
from strategies.s96_rank import at_gate, stats_of

FEE_BPS, SLIP_BPS = 5.0, 3.0
MAX_LEV, BAND = 3.0, 0.10
SPEEDS = ((8, 32), (16, 64), (32, 128))


def trend_signal(px):
    lp = np.log(px)
    s = np.zeros(len(px))
    for f, sl in SPEEDS:
        s += np.sign((lp.ewm(span=f, adjust=False).mean()
                      - lp.ewm(span=sl, adjust=False).mean()).to_numpy())
    return pd.Series(s / len(SPEEDS), index=px.index).shift(1).fillna(0.0)


def crowd_signal(idx):
    """S119's crowding signal, reindexed onto the implied-vol window."""
    px, fd, F = S119.build("1D")
    cols = [c for c in C.SIGNS if c in F.columns]
    s = S119.signal(F, cols, 365)
    s.index = pd.to_datetime(s.index)
    return s.reindex(idx).fillna(0.0)


def run(P, sig, vol, target_vol, thr=0.0):
    """One book. `vol` is the ANNUALISED vol forecast used for sizing."""
    v = np.maximum(np.asarray(vol, float), 0.05)
    s = np.asarray(sig, float).copy()
    if thr > 0:
        s[np.abs(s) < thr] = 0.0
    want = np.clip(s * target_vol / v, -MAX_LEV, MAX_LEV)

    pos, cur = np.zeros(len(P)), 0.0
    for i in range(len(P)):
        if abs(want[i] - cur) > BAND or (want[i] == 0.0 and cur != 0.0):
            cur = want[i]
        pos[i] = cur

    cost = (FEE_BPS + SLIP_BPS) / 1e4
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    simple = P.px.pct_change().fillna(0.0).to_numpy()
    net = pos * simple - pos * P.fund.to_numpy() - turn * cost
    return pd.Series(net, index=P.index), pos


def pf_of(net, pos):
    live, out, acc, on = pos != 0, [], 0.0, False
    for i in range(len(net)):
        if live[i]:
            acc += net.iloc[i]; on = True
        elif on:
            out.append(acc); acc, on = 0.0, False
    if on:
        out.append(acc)
    p = np.array(out)
    if not len(p):
        return np.nan, 0
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    return (g / l if l > 0 else np.inf), len(p)


def show(tag, net, pos, base=None):
    a = np.asarray(net, float); a = a[np.isfinite(a)]
    st, g = stats_of(a), at_gate(a)
    b = bootstrap_dd(a, n=1500, block=90)
    pf, n = pf_of(net, pos)
    h = len(a) // 2
    g1, g2 = at_gate(a[:h])["cagr"] * 100, at_gate(a[h:])["cagr"] * 100
    d = "" if base is None else f"{(g['cagr']*100/base - 1)*100:>+8.0f}%"
    print(f"  {tag:>26}{st['sharpe']:>7.2f}{st['dd']*100:>8.1f}%"
          f"{b['dd_median']*100:>8.1f}%{n:>7}{pf:>7.2f}{g['cagr']*100:>9.1f}%"
          f"{g1:>8.1f}%{g2:>8.1f}%{d}", flush=True)
    return g["cagr"] * 100


HDR = (f"  {'sizing':>26}{'Shp':>7}{'realDD':>8}{'medDD':>8}{'trd':>7}{'PF':>7}"
       f"{'at -20%':>9}{'1st h':>8}{'2nd h':>8}{'vs base':>9}")

if __name__ == "__main__":
    P = V.panel().dropna(subset=["iv", "rv32"]).copy()
    print("S123 - does sizing off implied vol beat sizing off trailing vol?\n")
    print(f"daily, {P.index.min().date()} -> {P.index.max().date()}, "
          f"{len(P)} days ({len(P)/365.25:.1f} years). SHORT SAMPLE - "
          f"2021-2022 is not in it.\n")

    P["blend"] = 0.5 * P.iv + 0.5 * P.rv32
    SIZERS = (("EWMA 32d (incumbent)", P.rv32), ("implied vol", P.iv),
              ("half implied, half EWMA", P.blend))

    books = (("ALWAYS LONG", pd.Series(1.0, index=P.index), 0.40, 0.0),
             ("TREND", trend_signal(P.px), 0.40, 0.0),
             ("CROWDING", crowd_signal(P.index), 0.20, 0.30))

    for name, sig, tv, thr in books:
        print(f"\n=== {name}, target vol {tv*100:.0f}%")
        print(HDR)
        base = None
        for lab, vol in SIZERS:
            net, pos = run(P, sig, vol, tv, thr)
            g = show(lab, net, pos, base)
            if base is None:
                base = g
    print("\ndone: implied-vol sizing")
