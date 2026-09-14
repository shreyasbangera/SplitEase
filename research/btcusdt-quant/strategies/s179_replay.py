"""S179 - Backtest the ACTUAL bot code, before and after the stop-anchor change.

s178 was my reimplementation of what webapp/ appeared to do.  This is not: it
imports webapp.engine and webapp.strategies.v7 and calls plan_orders() and
execute() exactly as deploy/run.ps1 does, once per 12h bar, against a broker that
replays history.  The only thing replaced is load_panels(), which is data
plumbing - every line of decision, sizing, gating and ladder logic is the code
that is running on the laptop.

  OLD = anchors.apply() neutered, so the stop is re-derived from the close each
        decision, which is what shipped
  NEW = anchors.apply() live, so the stop stays where the trade opened

Same code path, same bars, same costs, one function toggled.
"""
import os, sys, json, tempfile
os.environ.setdefault("BOOK_STORE", "/tmp/replaystore")
os.environ.setdefault("BOT_MODE", "paper")
sys.path.insert(0, "/home/user/quant")

import numpy as np, pandas as pd
from webapp.broker.base import Position
from webapp import engine as wengine, anchors
from webapp.strategies.v7 import V7
from engine.core import funding_array
from research.harness import exec_grid, slice_period
from strategies.s87_combined import rankings
from strategies.s77_lookback import stats

FEE, SLIP = 5e-4, 3e-4                      # per side, as the backtest
STORE = os.environ["BOOK_STORE"]
PLAN = os.path.join(STORE, "v7_plan.json")
P12 = pd.read_parquet("/home/user/quant/data/live/panel_12h.parquet")
P4 = pd.read_parquet("/home/user/quant/data/live/panel_4h.parquet")


class Replay:
    """A venue the real engine can talk to.  Implements webapp/broker/base.py."""
    mode = "test"

    def __init__(self, eq0):
        self.cash = eq0; self.qty = 0.0; self.entry = 0.0
        self.orders = []; self.px = 0.0; self.fillpx = 0.0
        self.pnl = []; self.sent = 0; self.stopped = 0; self.tp = 0

    def equity(self):
        return self.cash + (self.qty * (self.px - self.entry) if self.qty else 0.0)

    # -- the interface the engine uses -----------------------------------
    def position(self):
        return Position(self.qty, self.entry, self.equity())

    def price(self):
        return self.px

    def market(self, side, qty, note=""):
        self._fill(qty if side == "BUY" else -qty, self.fillpx)
        self.sent += 1
        return dict(ok=True, side=side, qty=qty)

    def cancel_all(self):
        self.orders = []
        return dict(ok=True)

    def place_stop(self, side, qty, stop, kind):
        self.orders.append((side, float(qty), float(stop), kind))
        return dict(ok=True)

    # -- replay mechanics -------------------------------------------------
    def _fill(self, dq, px):
        if dq == 0.0:
            return
        f = px * (1.0 + (1.0 if dq > 0 else -1.0) * SLIP)
        self.cash -= FEE * abs(dq) * f
        if self.qty == 0.0 or (self.qty > 0) == (dq > 0):
            self.entry = ((self.entry * abs(self.qty) + f * abs(dq))
                          / (abs(self.qty) + abs(dq)))
            self.qty += dq
            return
        c = min(abs(dq), abs(self.qty)); s = 1.0 if self.qty > 0 else -1.0
        p = s * (f - self.entry) * c
        self.cash += p; self.pnl.append(p); self.qty += dq
        if abs(self.qty) < 1e-12:
            self.qty = 0.0; self.entry = 0.0
        elif (self.qty > 0) != (s > 0):
            self.entry = f

    def touch(self, hi, lo):
        """Fire resting reduce-only orders.  Stops before targets, so a bar that
        spans both is scored the unfavourable way."""
        if not self.qty or not self.orders:
            return
        keep = []
        for o in sorted(self.orders, key=lambda x: x[3] != "STOP_MARKET"):
            side, q, lvl, kind = o
            if not self.qty:
                keep.append(o); continue
            stop = kind == "STOP_MARKET"
            hit = ((lo <= lvl) if stop else (hi >= lvl)) if side == "SELL" \
                  else ((hi >= lvl) if stop else (lo <= lvl))
            if hit:
                self._fill(-np.sign(self.qty) * min(q, abs(self.qty)), lvl)
                self.stopped += stop; self.tp += (not stop)
            else:
                keep.append(o)
        self.orders = keep


def replay(R, risk, use_anchors, eq0=10_000.0):
    real_apply = anchors.apply
    anchors.apply = real_apply if use_anchors else (lambda s, a, px, flat: 0)
    for f in os.listdir(STORE):
        os.remove(os.path.join(STORE, f))

    ex, _ = slice_period(exec_grid(), R[0][0], R[-1][1])
    eo, eh, el, ec = (ex.open.to_numpy(float), ex.high.to_numpy(float),
                      ex.low.to_numpy(float), ex.close.to_numpy(float))
    fund = funding_array(ex)
    ens = pd.DatetimeIndex(ex.dt).asi8
    d12 = pd.DatetimeIndex(P12.dt).asi8

    strat = V7()
    strat.params = dict(strat.params); strat.params["plan"] = PLAN
    b = Replay(eq0)
    eqc = np.full(len(ex), np.nan)
    reused = 0

    for s0, e0, cfgs in R:
        json.dump({"asof": s0, "configs": [list(c[:4]) + (["ema", c[4]] if c[4] else [None, 0])
                                           for c in cfgs[:3]]}, open(PLAN, "w"))
        lo_i = int(np.searchsorted(d12, pd.Timestamp(s0, tz="UTC").value, "left"))
        hi_i = int(np.searchsorted(d12, pd.Timestamp(e0, tz="UTC").value, "left"))
        for i in range(lo_i, hi_i):
            j = int(np.searchsorted(ens, d12[i], "right"))      # first exec bar after the close
            if j >= len(ex) or b.equity() <= 0:
                continue
            a12, a4 = P12.iloc[:i + 1], P4[P4.dt <= P12.dt.iloc[i]]
            wengine.load_panels = lambda names, x=a12, y=a4: {"panel_12h": x, "panel_4h": y}
            b.px = float(a12.close.iloc[-1]); b.fillpx = eo[j]
            plan = wengine.plan_orders(strat, b, b.equity(), risk)
            reused += plan.get("anchors_reused", 0)
            wengine.execute(plan, b, armed=True)

            nxt = d12[i + 1] if i + 1 < len(d12) else ens[-1] + 1
            k = int(np.searchsorted(ens, nxt, "right"))
            for t in range(j, min(k, len(ex))):
                b.px = ec[t]
                if b.qty and fund[t]:
                    b.cash -= fund[t] * b.qty * ec[t]
                b.touch(eh[t], el[t])
                eqc[t] = b.equity()

    anchors.apply = real_apply
    s = pd.Series(eqc, index=pd.to_datetime(ex.dt)).ffill().dropna()
    s = s.resample("1D").last().dropna()
    return s.pct_change().dropna(), np.array(b.pnl), reused, b


if __name__ == "__main__":
    R = rankings()
    risks = [float(x) for x in (sys.argv[1:] or ["0.08", "0.144"])]
    for risk in risks:
        print(f"\n{'='*112}\nREAL webapp/ CODE REPLAYED   risk {risk:.1%}\n{'='*112}")
        for use, tag in ((False, "OLD  stop re-derived from the close"),
                         (True,  "NEW  stop anchored to the trade")):
            r, pl, reu, b = replay(R, risk, use)
            stats(r, pl, tag, risk)
            print(f"        orders sent {b.sent}   stops fired {b.stopped}   "
                  f"targets hit {b.tp}   anchors reused {reu}")
