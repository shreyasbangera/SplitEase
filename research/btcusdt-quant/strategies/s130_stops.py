"""
S130 - Do per-trade stops buy Calmar? Chasing V7's 3.3x residual.

S129 fitted the law across 332 books from 17 signal sources:

    Calmar = 0.84 x Sharpe^1.53        R^2 = 0.969

Sharpe explains 97% of Calmar and no shape variable explains the rest - skew
correlates with the residual at r = 0.005. **But V7 sits 3.33x above that line**
(2.40x even on the stricter bootstrap gate), while every other book in the study
sits between 0.70x and 1.55x. That residual is worth more than any signal found
in this entire log: 3.3x on Calmar is 3.3x on the gate figure.

THE HYPOTHESIS
--------------
Every one of the 332 population books is **stopless** - vol-targeted, continuously
sized, riding whatever the position does. V7 is not: it sizes to an ATR stop and
gets stopped out. A stop truncates the loss of any single trade, and max drawdown
is a path property built out of exactly those episodes. Sharpe, a moment of the
daily return distribution, barely notices; Calmar, a function of the single worst
excursion, should notice enormously.

If that is the mechanism it is a **general architectural lever**, not a fact about
V7, and it should transfer to any book.

THE CONTROL THAT DECIDES IT
---------------------------
Random-signal books are swept with the same stops. This is the whole experiment:

    if stops raise Calmar for RANDOM signals too, the effect is mechanical - the
    gate rewards a reshaped path regardless of edge - and it explains V7's outlier
    status as architecture rather than skill, but buys nothing real.

    if stops raise Calmar only where there is genuine edge, it is a real lever and
    every stopless book in this study has been leaving Calmar on the table.

Either answer is worth having. Nothing is selected on its own result; the whole
grid prints.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from scipy import stats as sps

from strategies.s96_rank import at_gate, stats_of

FEE, SLIP = 5.0, 3.0
A, B = 0.84, 1.53          # the S129 law, for measuring the residual


def gate(a, tol=0.01):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a) or np.allclose(a, 0) or a.sum() <= 0:
        return np.nan
    g = at_gate(a, lo=1e-3, hi=40.0, iters=60)
    if not np.isfinite(g["dd"]) or abs(g["dd"] + 0.20) > tol:
        return np.nan
    return g["cagr"] * 100


def book(px, fund, sig, tv, thr, vol_hl, stop=None, max_lev=3.0, band=0.10):
    """Vol-targeted book, optionally with a per-trade stop.

    `stop` is the loss on the POSITION, as a fraction of equity, at which the
    book goes flat and stays flat until the signal changes sign. It is evaluated
    on the running P&L since entry, using only realised bars - no intrabar fill
    assumption, which would be the optimistic way to do this.
    """
    r = np.log(px / px.shift(1)).fillna(0.0)
    rv = np.maximum(r.ewm(halflife=vol_hl, adjust=False).std()
                    .shift(1).bfill().to_numpy() * np.sqrt(365.25), 0.05)
    s = np.asarray(sig, float).copy()
    s[np.abs(s) < thr] = 0.0
    want = np.clip(s * tv / rv, -max_lev, max_lev)
    simple = px.pct_change().fillna(0.0).to_numpy()
    fu = fund.to_numpy()

    n = len(px)
    pos = np.zeros(n)
    cur, since, blocked_sign = 0.0, 0.0, 0.0
    for i in range(n):
        w = want[i]
        if stop is not None and blocked_sign != 0.0:
            # stay flat until the signal changes sign or goes flat
            if np.sign(w) == blocked_sign and w != 0.0:
                w = 0.0
            else:
                blocked_sign = 0.0
        if np.sign(w) != np.sign(cur) or cur == 0.0:
            since = 0.0                       # new trade
        if abs(w - cur) > band or (w == 0.0 and cur != 0.0):
            cur = w
        pos[i] = cur
        if cur != 0.0:
            since += cur * simple[i] - cur * fu[i]
            if stop is not None and since < -stop:
                blocked_sign = np.sign(cur)
                cur, since = 0.0, 0.0
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = pos * simple - pos * fu - turn * (FEE + SLIP) / 1e4
    return pd.Series(net, index=px.index)


def z(s, w):
    return ((s - s.rolling(w, min_periods=max(20, w // 3)).mean())
            / (s.rolling(w, min_periods=max(20, w // 3)).std() + 1e-12)).clip(-2, 2)


if __name__ == "__main__":
    import strategies.s119_dev as S119
    import strategies.s118_crowd as C
    px, fund, F = S119.build("1D")
    cols = [c for c in C.SIGNS if c in F.columns]
    lp = np.log(px)
    rng = np.random.default_rng(5)

    sigs = {"crowd180": S119.signal(F, cols, 180),
            "crowd365": S119.signal(F, cols, 365),
            "trend16_64": np.sign((lp.ewm(span=16, adjust=False).mean()
                                   - lp.ewm(span=64, adjust=False).mean())).shift(1).fillna(0.0),
            "trend32_128": np.sign((lp.ewm(span=32, adjust=False).mean()
                                    - lp.ewm(span=128, adjust=False).mean())).shift(1).fillna(0.0)}
    for i in range(4):
        w = pd.Series(rng.normal(0, 1, len(px)), index=px.index).rolling(30).mean()
        sigs[f"RANDOM{i}"] = z(w, 180).shift(1).fillna(0.0)

    print("S130 - do per-trade stops buy Calmar?\n")
    print(f"law from S129: Calmar = {A} x Sharpe^{B}. 'ratio' is actual / law.")
    print("V7 sits at 3.33x. Every stopless book in this study sits 0.70-1.55x.\n")
    print(f"{'signal':>12}{'stop':>8}{'Sharpe':>8}{'Calmar':>8}{'gate':>9}"
          f"{'ratio':>8}{'vs no stop':>12}")

    summary = {}
    for name, s in sigs.items():
        s = pd.Series(np.asarray(s, float), index=px.index).fillna(0.0)
        base_ratio = None
        for stop in (None, 0.20, 0.12, 0.08, 0.05, 0.03):
            nb = book(px, fund, s, 0.30, 0.30, 32, stop=stop)
            g = gate(nb.to_numpy())
            if not np.isfinite(g) or g <= 0:
                print(f"{name:>12}{str(stop):>8}{'':>8}{'':>8}{'n/a':>9}")
                continue
            sh = stats_of(nb.to_numpy())["sharpe"]
            cal = g / 20.0
            ratio = cal / (A * max(sh, 1e-9) ** B)
            if stop is None:
                base_ratio = ratio
                d = ""
            else:
                d = f"{(ratio/base_ratio - 1)*100:>+11.0f}%" if base_ratio else ""
            summary.setdefault(name, []).append((stop, ratio))
            print(f"{name:>12}{str(stop):>8}{sh:>8.2f}{cal:>8.2f}{g:>8.1f}%"
                  f"{ratio:>8.2f}{d}")
        print()

    print("=" * 68)
    real = [k for k in summary if not k.startswith("RANDOM")]
    rand = [k for k in summary if k.startswith("RANDOM")]

    def lift(keys):
        out = []
        for k in keys:
            v = dict(summary[k])
            base = v.get(None)
            best = max((r for s, r in v.items() if s is not None), default=None)
            if base and best:
                out.append(best / base)
        return np.median(out) if out else np.nan

    print(f"median best-stop lift in Calmar ratio:")
    print(f"   REAL signals   {lift(real):.2f}x   ({len(real)} books)")
    print(f"   RANDOM signals {lift(rand):.2f}x   ({len(rand)} books)")
    print("\nif the two are similar, stops reshape the path mechanically and buy")
    print("nothing; if real >> random, per-trade loss truncation is a genuine lever.")
    print("\ndone: stops")
