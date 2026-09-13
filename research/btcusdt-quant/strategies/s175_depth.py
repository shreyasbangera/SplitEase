"""
S175 - Standing liquidity: does the order book's SHAPE predict anything?

WHY THIS IS NOT S174 AGAIN
---------------------------
S174 built 22 features from minute bars and none reached |t| = 1.4. The reason is
economic rather than statistical: order FLOW is a transient. Information arrives,
it is traded on within seconds to minutes, and it is in the price. Aggregate a
day of it and there is nothing left to predict tomorrow with. S120 found the same
thing from the other side - the intraday edge is real at 0.1-2.2bps and dies
against a 16bps round trip.

DEPTH is a different object. It is not an event, it is a STATE: how much size is
resting on each side of the book and how quickly it thickens away from mid. A
state variable persists, which is exactly the property flow lacks and exactly
what a daily-horizon signal needs. Nothing in this study has tested it on more
than 250 days, because that is all `book_1m.parquet` covers and it ends 2024-02.

Binance's archive carries `bookDepth` daily from 2023-01 to 2026-09 - 1,348 days,
2,880 snapshots each, notional resting size at +/-1%, 2%, 3%, 4% and 5% from mid.
That is 3.7 years of the standing book, and it has never been looked at here.

WHAT THE BOOK'S SHAPE MIGHT CARRY
----------------------------------
    imbalance      more notional bid than ask, at a given distance from mid, is
                   standing demand. Whether it PRECEDES returns or merely
                   accompanies them is the question.
    slope          how fast depth grows from 1% to 5% out. A flat book is thin
                   and fragile; a steep one absorbs size. This is resilience,
                   and it should matter most before large moves.
    near vs far    imbalance at 1% against imbalance at 5%. Disagreement between
                   them separates real standing interest from the layer that
                   evaporates when touched.
    level          total notional depth, and its change - the market's willingness
                   to provide liquidity at all.

The same screen as everything else: lagged a full day, t deflated for overlap,
and a phase-randomised surrogate control on anything that clears - the test that
killed the illiquidity factor at p = 0.56 and again at p = 1.00 after it looked
like the second-best signal in the study.
"""
import sys, os, glob, zipfile, io; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s174_micro as MI
from strategies.s96_rank import at_gate, stats_of

D = "/home/user/quant/data"
RNG = np.random.default_rng(29)


def depth_features(cache=f"{D}/depth_daily.parquet", rebuild=False):
    """Daily features from 30-second order book snapshots."""
    if os.path.exists(cache) and not rebuild:
        return pd.read_parquet(cache)
    rows = []
    files = sorted(glob.glob(f"{D}/bookdepth/*.zip"))
    for i, f in enumerate(files):
        try:
            with zipfile.ZipFile(f) as z:
                d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        except Exception:
            continue
        if not {"timestamp", "percentage", "notional"} <= set(d.columns):
            continue
        d["percentage"] = pd.to_numeric(d["percentage"], errors="coerce")
        d["notional"] = pd.to_numeric(d["notional"], errors="coerce")
        d = d.dropna(subset=["percentage", "notional"])
        w = d.pivot_table(index="timestamp", columns="percentage",
                          values="notional", aggfunc="last")
        need = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
        if not set(need) <= set(w.columns):
            continue
        bid1, ask1 = w[-1], w[1]
        bid5, ask5 = w[-5], w[5]
        bidT = w[[-1, -2, -3, -4, -5]].sum(axis=1)
        askT = w[[1, 2, 3, 4, 5]].sum(axis=1)
        imb1 = (bid1 - ask1) / (bid1 + ask1).replace(0, np.nan)
        imb5 = (bid5 - ask5) / (bid5 + ask5).replace(0, np.nan)
        imbT = (bidT - askT) / (bidT + askT).replace(0, np.nan)
        rows.append(dict(
            day=os.path.basename(f)[:10],
            imb1=imb1.mean(), imb5=imb5.mean(), imbT=imbT.mean(),
            imb_near_far=(imb1 - imb5).mean(),
            imb_vol=imbT.std(),
            imb_ac=imbT.autocorr(1) if len(imbT) > 30 else np.nan,
            depth_tot=np.log(max((bidT + askT).mean(), 1.0)),
            # how fast the book thickens away from mid: 5% depth over 1% depth
            slope_bid=float((bid5 / bid1.replace(0, np.nan)).mean()),
            slope_ask=float((ask5 / ask1.replace(0, np.nan)).mean()),
            slope_asym=float(((bid5 / bid1.replace(0, np.nan))
                              - (ask5 / ask1.replace(0, np.nan))).mean()),
            depth_intraday_vol=float((bidT + askT).std()
                                     / max((bidT + askT).mean(), 1.0)),
            n_snap=len(w)))
        if (i + 1) % 200 == 0:
            print(f"   parsed {i+1}/{len(files)}", flush=True)
    F = pd.DataFrame(rows)
    F["day"] = pd.to_datetime(F["day"])
    F = F.set_index("day").sort_index()
    F.to_parquet(cache)
    return F


if __name__ == "__main__":
    print("S175 - does the order book's standing shape predict returns?\n")
    F = depth_features()
    print(f"parsed {len(F)} days, {F.index.min().date()} -> {F.index.max().date()} "
          f"({len(F)/365.25:.1f} years), median {int(F.n_snap.median())} snapshots/day\n")

    M = MI.minute_features()
    px = M["close"].reindex(F.index).dropna()
    F = F.loc[px.index]
    lr = np.log(px)
    cols = [c for c in F.columns if c != "n_snap"]
    print(f"{len(cols)} features from the standing book\n")
    print("INFORMATION CONTENT - lagged a full day, t deflated for overlap")
    print(f"   {'feature':>19}" +
          "".join(f"{'IC'+str(k)+'d':>9}{'t':>7}" for k in (1, 5, 20)))
    surv = []
    for c in cols:
        x = F[c].astype(float)
        z = ((x - x.rolling(180, min_periods=90).mean())
             / (x.rolling(180, min_periods=90).std() + 1e-12)).clip(-3, 3).shift(1)
        row, best = "", (0, 0.0, 0.0)
        for k in (1, 5, 20):
            y = lr.diff(k).shift(-k)
            ic, t, n = MI.block_t(z.to_numpy(), y.to_numpy(), k)
            row += (f"{ic:>9.4f}{t:>7.2f}" if np.isfinite(ic) else f"{'n/a':>16}")
            if np.isfinite(t) and abs(t) > abs(best[2]):
                best = (k, ic, t)
        print(f"   {c:>19}{row}")
        if abs(best[2]) > 2.5:
            surv.append((c, best[0], best[1], best[2], z))

    print(f"\nfeatures with |t| > 2.5 at their best horizon: {len(surv)}")
    if not surv:
        print("   none. the standing book carries no daily-horizon signal either.")
        sys.exit(0)

    print("\nSURROGATE CONTROL - 200 phase-randomised versions of each survivor")
    real = []
    for c, k, ic, t, z in sorted(surv, key=lambda s: -abs(s[3])):
        y = lr.diff(k).shift(-k)
        ts = []
        for _ in range(200):
            s = MI.phase_rand(F[c].astype(float))
            zz = ((s - s.rolling(180, min_periods=90).mean())
                  / (s.rolling(180, min_periods=90).std() + 1e-12)).clip(-3, 3).shift(1)
            _, tt, _ = MI.block_t(zz.to_numpy(), y.to_numpy(), k)
            if np.isfinite(tt):
                ts.append(abs(tt))
        ts = np.array(ts); pv = float((ts >= abs(t)).mean())
        tag = "REAL" if pv < 0.05 else "not distinguishable"
        if pv < 0.05:
            real.append((c, k, ic, t, z))
        print(f"   {c:>19} @{k:>2}d  |t| {abs(t):>5.2f}  surrogate median "
              f"{np.median(ts):>5.2f}, 95th {np.percentile(ts,95):>5.2f}  "
              f"p = {pv:.3f}  {tag}")

    if not real:
        print("\n   nothing survived the control."); sys.exit(0)
    print(f"\n{len(real)} survive both. BOOKS (5bps fee + 3bps slip, vol-targeted):")
    print(f"   {'feature':>19}{'sign':>6}{'Shp':>7}{'CAGR':>9}{'maxDD':>8}"
          f"{'turn':>7}{'at -20%':>10}")
    books = {}
    for c, k, ic, t, z in real:
        sgn = np.sign(t)
        net, turn = MI.book(px, (z * sgn).fillna(0.0))
        a = net.to_numpy(float); s = stats_of(a); g = MI.gate_of(a)
        books[c] = net
        print(f"   {c:>19}{int(sgn):>+6}{s['sharpe']:>7.2f}{s['cagr']*100:>8.1f}%"
              f"{s['dd']*100:>7.1f}%{turn.mean():>7.2f}"
              + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}"))
    if len(books) > 1:
        B = pd.DataFrame(books); comb = B.mean(axis=1)
        a = comb.to_numpy(float); s = stats_of(a); g = MI.gate_of(a)
        print(f"   {'combined':>19}{'':>6}{s['sharpe']:>7.2f}{s['cagr']*100:>8.1f}%"
              f"{s['dd']*100:>7.1f}%{'':>7}"
              + (f"{g:>9.1f}%" if np.isfinite(g) else f"{'n/a':>10}"))
    pd.to_pickle(books, "/home/user/quant/results/s175_books.pkl")
    print("\ndone: standing book screen")
