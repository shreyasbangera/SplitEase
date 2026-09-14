#!/usr/bin/env python3
"""End to end: the engine hands the trade's own stop to the exchange, and only an
armed send is allowed to change what is remembered.

Run:  python tests/anchors_engine.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp import engine, anchors
from webapp.broker.base import Position
from webapp.strategies.base import Sleeve, Decision

FAIL = []
LABEL = "#1 exp 3.0 · 3.0ATR ×2.0R · 14d"


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


class Strat:
    """One short sleeve whose stop is re-derived from the close every call -
    exactly what webapp/strategies/v7.py does."""
    name = "stub"

    def __init__(self, close):
        self.close = close

    def needs(self):
        return []

    def decide(self, panels, equity, risk):
        sd = 4_064.8
        return Decision([Sleeve(LABEL, -0.011, self.close + sd, self.close - 2 * sd)])


class Broker:
    mode = "test"

    def __init__(self, qty, px):
        self.qty, self.px = qty, px
        self.cancelled = 0
        self.placed = []

    def position(self):
        return Position(self.qty, 76_911.0, 5_000.0)

    def price(self):
        return self.px

    def market(self, side, qty, note=""):
        return dict(side=side, qty=qty)

    def cancel_all(self):
        self.cancelled += 1

    def place_stop(self, side, qty, stop, kind):
        self.placed.append((kind, stop))
        return dict(kind=kind, stop=stop)


with tempfile.TemporaryDirectory() as store:
    engine.STORE = store
    engine.load_panels = lambda names: {}

    print("first decision - nothing remembered yet")
    b = Broker(qty=0.0, px=76_805.0)
    p1 = engine.plan_orders(Strat(76_805.0), b, 5_000.0, 0.08)
    stop1 = next(l["stop"] for l in p1["ladder"] if l["kind"] == "STOP_MARKET")
    check("stop comes from the close", abs(stop1 - 80_869.8) < 0.1, stop1)
    check("nothing reused", p1["anchors_reused"] == 0)
    check("a plan alone writes nothing", anchors.load(store) == {})

    engine.execute(p1, b, armed=True)
    check("an armed send remembers the ladder", LABEL in anchors.load(store))

    print("\nsecond decision, price has run 1,500 against the short")
    b2 = Broker(qty=-0.011, px=78_300.0)
    p2 = engine.plan_orders(Strat(78_300.0), b2, 5_000.0, 0.08)
    stop2 = next(l["stop"] for l in p2["ladder"] if l["kind"] == "STOP_MARKET")
    check("the stop does NOT chase price", abs(stop2 - 80_869.8) < 0.1, stop2)
    check("it is reported as reused", p2["anchors_reused"] == 1)
    check("without the fix it would have moved here", abs(78_300.0 + 4_064.8 - 82_364.8) < 0.1)

    print("\na dry run must not disturb what is remembered")
    before = anchors.load(store)
    engine.execute(p2, Broker(-0.011, 78_300.0), armed=False)
    check("not armed -> anchors untouched", anchors.load(store) == before)

    print("\nthe position went flat")
    b3 = Broker(qty=0.0, px=78_300.0)
    p3 = engine.plan_orders(Strat(78_300.0), b3, 5_000.0, 0.08)
    stop3 = next(l["stop"] for l in p3["ladder"] if l["kind"] == "STOP_MARKET")
    check("a new trade gets a new stop", abs(stop3 - 82_364.8) < 0.1, stop3)

print("\n" + ("FAILED: " + ", ".join(FAIL) if FAIL else "all good"))
sys.exit(1 if FAIL else 0)
