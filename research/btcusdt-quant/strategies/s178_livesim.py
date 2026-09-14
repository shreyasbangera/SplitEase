"""S178 - Does the backtest describe the deployed bot?

THE TWO MACHINES

engine/core.py is a TRADE simulator.  It opens a position when the signal fires,
freezes the quantity and the stop for the life of that trade, and closes it on the
stop, the target, a flat signal, a reversal, or the holding cap.  V7's headline
runs three of these as INDEPENDENT books at risk/3 each and sums their returns.

webapp/ is a TARGET simulator.  Every 12h it recomputes what it wants to hold from
the CURRENT conviction, sends the difference as one order on one netted position,
and cancels and re-places the whole stop ladder against the CURRENT price.  It
never applies the holding cap - `hold_bars` is recorded on the sleeve and nothing
reads it.

This file runs both on the same bars, the same costs and the same quarterly config
selection, so the gap is the answer to "does the 179% describe my bot".

VALIDATION
`python strategies/s178_livesim.py check` runs ONE config in backtest mode against
engine/core.py on that same config.  The two are different implementations of the
same rules, so they should land on the same CAGR.  Nothing below is worth reading
until that passes.
"""
import sys; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd
from research.harness import exec_grid, slice_period
from engine.core import funding_array
from strategies.s69_calsel import ctx, shape, daily
from strategies.s84_gate import trend, sim_gate
from strategies.s87_combined import rankings
from strategies.s77_lookback import stats

FEE, SLIP = 5e-4, 3e-4          # per side, on notional - as the backtest
MIN_NOTIONAL = 100.0            # webapp/engine.py
MAX_LEV = 10.0                  # webapp/strategies/v7.py, and the backtest
EQ0 = 10_000.0
BAR_H = 0.25                    # fut_15m execution grid


class Sleeve:
    """One configuration.  In LIVE mode it is pure bookkeeping against a shared
    netted position; in BACKTEST mode it is an independent trade with a frozen
    size, a fixed stop and a holding cap."""

    __slots__ = ("cfg", "qty", "stop", "tp", "ej")

    def __init__(self, cfg):
        self.cfg = cfg; self.qty = 0.0; self.stop = self.tp = 0.0; self.ej = -1


def wanted(cfg, u_raw, up, eq, per, px, a, max_lev):
    """webapp/strategies/v7.py decide(): what this sleeve wants to hold now."""
    p, stp, rr, hold, sp, md = cfg
    u = u_raw
    if sp and up is not None:
        if ("s" in md and u < 0 and up) or ("l" in md and u > 0 and not up):
            u = 0.0
    if u == 0.0 or not (a > 0):
        return 0.0, 0.0, 0.0
    side = 1.0 if u > 0 else -1.0
    sd = stp * a
    q = min((eq * per * abs(u)) / sd, eq * max_lev / px)
    return side * q, px - side * sd, px + side * rr * sd


def run(cfgs_at, U, U0, UP, sig, close, atr, risk, k, live=None,
        start=None, end=None, eq0=EQ0, max_lev=MAX_LEV,
        resize=None, roll=None, cap=None):
    """Three independent switches, so each difference can be priced on its own:
       resize -> recompute the target every bar (bot) vs freeze it at entry
       roll   -> re-anchor the stop every bar (bot) vs fix it at entry
       cap    -> honour the 14/21-day holding cap (backtest) vs ignore it (bot)
       `live` sets all three at once."""
    if resize is None: resize = live
    if roll is None:   roll = live
    if cap is None:    cap = not live
    ex, _ = slice_period(exec_grid(), start, end)
    eo, eh, el, ec = (ex.open.to_numpy(float), ex.high.to_numpy(float),
                      ex.low.to_numpy(float), ex.close.to_numpy(float))
    fund = funding_array(ex)
    dec = np.searchsorted(sig.asi8, pd.DatetimeIndex(ex.dt).asi8, side="right") - 1
    n = len(ex)

    eq = eq0                      # cash + realised
    pos = 0.0; entry = 0.0        # netted position and its average entry
    S = {}                        # cfg -> Sleeve
    pnl = []
    eqc = np.empty(n)
    last = -1
    per = risk / k

    def fill(dq, px):
        nonlocal eq, pos, entry
        if dq == 0.0:
            return
        f = px * (1.0 + (1.0 if dq > 0 else -1.0) * SLIP)
        eq -= FEE * abs(dq) * f
        if pos == 0.0 or (pos > 0) == (dq > 0):
            entry = (entry * abs(pos) + f * abs(dq)) / (abs(pos) + abs(dq))
            pos += dq; return
        c = min(abs(dq), abs(pos)); s = 1.0 if pos > 0 else -1.0
        p = s * (f - entry) * c
        eq += p; pnl.append(p); pos += dq
        if abs(pos) < 1e-12: pos = 0.0; entry = 0.0
        elif (pos > 0) != (s > 0): entry = f

    for j in range(n):
        if pos and fund[j]:
            eq -= fund[j] * pos * ec[j]

        # ---- reduce-only orders resting on the exchange.  Stop checked before
        #      target, so a bar spanning both is scored the unfavourable way.
        for sl in S.values():
            if sl.qty == 0.0:
                continue
            sd = 1.0 if sl.qty > 0 else -1.0
            for lvl, is_stop in ((sl.stop, True), (sl.tp, False)):
                if lvl <= 0 or sl.qty == 0.0:
                    continue
                hit = (el[j] <= lvl) if (sd > 0) == is_stop else (eh[j] >= lvl)
                if hit:
                    fill(-sl.qty, lvl)
                    sl.qty = 0.0; sl.stop = sl.tp = 0.0; sl.ej = -1
                    break

        d = dec[j] - 1
        if dec[j] != last and d >= 0 and cfgs_at[d] is not None and eq > 0:
            last = dec[j]
            nxt = eo[j + 1] if j + 1 < n else ec[j]   # engine fills at o[i+1]
            a, cpx = atr[d], close[d]
            live_cfgs = cfgs_at[d]
            for cf in live_cfgs:
                S.setdefault(cf, Sleeve(cf))
            for cf, sl in list(S.items()):
                if cf not in live_cfgs and sl.qty:        # quarter rolled over
                    fill(-sl.qty, nxt); sl.qty = 0.0; sl.ej = -1

            for cf in live_cfgs:
                sl = S[cf]
                p, stp, rr, hold, sp, md = cf
                u0 = float(U0[p][d])
                q, st, tp = wanted(cf, float(U[p][d]),
                                   bool(UP[sp][d]) if sp else None,
                                   eq, per, cpx, a, max_lev)
                if resize:
                    tgt = q                                # resize, every bar
                else:
                    capped = cap and sl.ej >= 0 and (j - sl.ej) * BAR_H >= hold * 24
                    flat = abs(u0) <= 0.0
                    flip = sl.qty and q and (q > 0) != (sl.qty > 0)
                    if sl.qty and (capped or flat or flip):
                        tgt = 0.0                          # close, then re-enter
                    elif sl.qty:
                        tgt = sl.qty                       # frozen
                    else:
                        tgt = q
                dq = tgt - sl.qty
                if dq and abs(dq) * nxt >= (MIN_NOTIONAL if resize else 0.0):
                    fill(dq, nxt)
                    if sl.qty == 0.0 and tgt != 0.0:
                        sl.ej = j
                    sl.qty = tgt
                    if tgt == 0.0:
                        sl.ej = -1
                if sl.qty:
                    if roll or sl.stop == 0.0:             # roll, or set at entry
                        sl.stop, sl.tp = st, tp
                else:
                    sl.stop = sl.tp = 0.0

            # a flip in backtest mode needs the re-entry in the same decision
            if not resize:
                for cf in live_cfgs:
                    sl = S[cf]
                    if sl.qty == 0.0:
                        q, st, tp = wanted(cf, float(U[cf[0]][d]),
                                           bool(UP[cf[4]][d]) if cf[4] else None,
                                           eq, per, cpx, a, max_lev)
                        if q and abs(q) * nxt >= MIN_NOTIONAL:
                            fill(q, nxt); sl.qty = q; sl.stop, sl.tp = st, tp
                            sl.ej = j

        eqc[j] = eq + (pos * (ec[j] - entry) if pos else 0.0)
        if eqc[j] <= 0:
            eqc[j:] = 0.0; break

    s = pd.Series(eqc, index=pd.to_datetime(ex.dt)).resample("1D").last().dropna()
    return s.pct_change().dropna(), np.array(pnl)


def prep(R, k):
    c = ctx(); g = c["g"]
    sig = pd.DatetimeIndex(pd.to_datetime(g.dt))
    cfgs_at = [None] * len(g)
    for s0, e0, cs in R:
        m = (sig >= pd.Timestamp(s0, tz="UTC")) & (sig < pd.Timestamp(e0, tz="UTC"))
        for i in np.flatnonzero(m):
            cfgs_at[i] = [tuple(x) for x in cs[:k]]
    used = [tuple(cf) for _, _, cs in R for cf in cs[:k]]
    U0 = {p: np.nan_to_num(shape(p)) for p in {cf[0] for cf in used}}
    UP = {sp: trend(sp) for sp in {cf[4] for cf in used} if sp}
    U = dict(U0)                       # gating is applied per sleeve in wanted()
    return cfgs_at, U, U0, UP, sig, g.close.to_numpy(float), c["a"]


def cagr_of(r):
    e = np.cumprod(1 + r.to_numpy())
    return e[-1] ** (365.25 / (r.index[-1] - r.index[0]).days) - 1


if __name__ == "__main__":
    R = rankings()
    S0, S1 = R[0][0], R[-1][1]

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print("VALIDATION - one config, backtest rules, my engine vs engine/core.py\n")
        for cfg in (R[0][2][0], R[4][2][0], R[9][2][0]):
            cfg = tuple(cfg)
            cfgs_at, U, U0, UP, sig, close, atr = prep(
                [(S0, S1, [list(cfg)])], 1)
            r, _ = run(cfgs_at, U, U0, UP, sig, close, atr, 0.08, 1,
                       live=False, start=S0, end=S1)
            m = sim_gate(cfg[:4], S0, S1, 0.08, span=cfg[4], mode=cfg[5])
            ref = daily(m)
            print(f"  {str(cfg):<34} mine {cagr_of(r)*100:8.1f}%   "
                  f"engine {cagr_of(ref)*100:8.1f}%   "
                  f"diff {(cagr_of(r)-cagr_of(ref))*100:+6.1f}pp")
        sys.exit(0)

    cfgs_at, U, U0, UP, sig, close, atr = prep(R, 3)
    for risk in (0.08, 0.144):
        print(f"\n{'='*112}\nrisk {risk:.1%}\n{'='*112}")
        for rs, rl, cp, tag in (
                (False, False, True,  "BACKTEST  freeze size, fixed stop, hold cap"),
                (False, True,  True,  "  + rolling stop only"),
                (True,  False, False, "  + resizing only"),
                (True,  True,  False, "THE BOT   resize + rolling stop, no cap")):
            r, pl = run(cfgs_at, U, U0, UP, sig, close, atr, risk, 3,
                        start=S0, end=S1, resize=rs, roll=rl, cap=cp)
            stats(r, pl, tag, risk)
