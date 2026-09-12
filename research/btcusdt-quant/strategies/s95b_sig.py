"""
S95b - Does the VIX read survive its own overlap, and does it clear cost?

S95's screen produced a coherent-looking picture under the punitive `strict`
lag: VIX rising predicts BTC weakness over ~12 days (IC about -0.05 to -0.06,
same sign in both halves and in both IS and OOS), while an elevated VIX LEVEL
predicts strength over the same span (+0.038, also stable). Level is
already-priced fear, change is new fear - economically sensible, and the
magnitudes sit right among the book's own signals, three of which have IC below
0.05.

Two things have to be checked before any of that counts.

OVERLAP. The stable horizon is h=24 bars of 12h = 12 days. Consecutive
observations share 23 of their 24 bars, so 4,020 rows are nowhere near 4,020
independent facts - roughly 167 of them are. A Spearman IC of 0.06 on 167
effective observations is about 0.8 standard errors from zero. This computes
the Newey-West t-statistic, which corrects for exactly that, rather than the
naive p-value the screen printed.

COST. The book pays ~16 bps a round turn. A decile spread below that is not a
signal, whatever its t-statistic. S50 is the cautionary case in the other
direction - the options chain showed a +69 bps spread that was worthless
because the sign was unstable - so both tests have to pass together, and
they are reported side by side.

The honest prior going in: the horizons with statistical power (12h to 2 days,
no overlap) are the ones whose sign FLIPS between halves, and the horizon whose
sign is stable (12 days) is the one with almost no power. That is the shape of
a result that is about to evaporate, and this is the measurement that says so
one way or the other.
"""
import sys
sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd
from scipy import stats

from research.harness import panel, IS_END
from research.ic import newey_west_t
from research import macro

START = "2021-03-01"
COST_BPS = 16.0


def fwd(close, h):
    c = pd.Series(np.asarray(close, float))
    return (np.log(c.shift(-h)) - np.log(c)).to_numpy()


def deciles(x, y, q=10):
    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[ok], y[ok]
    b = pd.qcut(pd.Series(xs), q, labels=False, duplicates="drop")
    g = pd.DataFrame({"b": b, "y": ys}).groupby("b").y.mean()
    return float(g.iloc[-1] - g.iloc[0]) * 1e4, float(g.iloc[0]) * 1e4, float(g.iloc[-1]) * 1e4


if __name__ == "__main__":
    fut, f12 = panel("12h")
    f = f12[f12.dt >= START].reset_index(drop=True)
    close = f.close.to_numpy(float)
    vix = macro.load_csv("vix")
    v = macro.onto_bars(f.dt, vix, mode="strict")
    feat = macro.features(v)

    n = len(f)
    print(f"{n} bars of 12h, {f.dt.iloc[0].date()} -> {f.dt.iloc[-1].date()}\n")

    print("The candidates S95 left standing - stable sign in BOTH splits under")
    print("the strict lag - re-tested for overlap and against cost.\n")
    print(f"{'feature':>10}{'h':>4}{'days':>6}{'IC':>9}{'NW t':>8}{'eff N':>7}"
          f"{'spread':>9}{'D1':>9}{'D10':>9}   verdict")

    cand = [("lvl_z120", 24), ("lvl_z120", 8), ("chg3d", 24), ("chg5d", 24),
            ("chg10d", 24), ("chg20d", 24), ("chg20d", 8), ("chg10d", 1),
            ("chg20d", 1), ("lvl_z120", 1)]

    for c, h in cand:
        x = feat[c].to_numpy(float)
        y = fwd(close, h)
        ok = np.isfinite(x) & np.isfinite(y)
        rho = stats.spearmanr(x[ok], y[ok])[0]
        t = newey_west_t(x[ok], y[ok], lags=max(1, h * 2))
        eff = int(ok.sum() / max(1, h))
        sp, d1, d10 = deciles(x, y)
        passes_t = abs(t) >= 2.0
        passes_c = abs(sp) >= COST_BPS
        verdict = ("real" if (passes_t and passes_c) else
                   "under cost" if passes_t else
                   "not significant" if passes_c else "neither")
        print(f"{c:>10}{h:>4}{h/2:6.1f}{rho:+9.4f}{t:+8.2f}{eff:7d}"
              f"{sp:+9.1f}{d1:+9.1f}{d10:+9.1f}   {verdict}")

    print(f"\nspread, D1, D10 in bps over the horizon; cost is ~{COST_BPS:.0f} bps a round turn")
    print("NW t uses 2h lags, i.e. twice the overlap it has to absorb")

    # --- the same question asked of the book's own signals, for calibration --
    print("\nCALIBRATION - the same test on signals already IN the book, so the")
    print("numbers above have something to be judged against.\n")
    print(f"{'signal':>10}{'h':>4}{'days':>6}{'IC':>9}{'NW t':>8}{'eff N':>7}"
          f"{'spread':>9}")
    own = {}
    for name, col in (("fundz", "fund_z"), ("basis", "basis_z"),
                      ("ofi96res", "ofi96_resz"), ("ofi6res", "ofi6_resz"),
                      ("ttpos", "tt_pos_z"), ("ttvsret", "tt_vs_retail"),
                      ("doi24", "doi24_z")):
        if col not in f.columns:
            continue
        own[name] = f[col].to_numpy(float)
    for name, x in own.items():
        for h in (4, 24):
            y = fwd(close, h)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 500:
                continue
            rho = stats.spearmanr(x[ok], y[ok])[0]
            t = newey_west_t(x[ok], y[ok], lags=max(1, h * 2))
            sp = deciles(x, y)[0]
            print(f"{name:>10}{h:>4}{h/2:6.1f}{rho:+9.4f}{t:+8.2f}"
                  f"{int(ok.sum()/h):7d}{sp:+9.1f}")

    print("\ndone: significance and cost")
