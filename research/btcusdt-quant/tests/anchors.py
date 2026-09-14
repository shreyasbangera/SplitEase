#!/usr/bin/env python3
"""The stop must belong to the trade, not to the latest bar.

Run:  python tests/anchors.py     (exit 0 = all good)
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp import anchors
from webapp.strategies.base import Sleeve

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def short(stop=79_000.0, tp=70_000.0, qty=-0.011):
    return [Sleeve(label="#1 exp 3.0 · 3.0ATR ×2.0R · 14d", qty=qty, stop=stop, take_profit=tp)]


SAVED = {"#1 exp 3.0 · 3.0ATR ×2.0R · 14d": dict(side=-1, stop=80_869.8, tp=68_675.4)}

print("anchor reuse")
s = short()
n = anchors.apply(s, SAVED, px=77_000.0, flat=False)
check("an open short keeps the stop it opened with", s[0].stop == 80_869.8 and n == 1, s[0].stop)
check("...and its target too", s[0].take_profit == 68_675.4)

print("\nan anchor is void when there is no trade behind it")
s = short()
anchors.apply(s, SAVED, px=77_000.0, flat=True)
check("flat account -> fresh levels", s[0].stop == 79_000.0)

s = short(qty=+0.011)
anchors.apply(s, SAVED, px=77_000.0, flat=False)
check("side flipped -> fresh levels", s[0].stop == 79_000.0)

s = short()
anchors.apply(s, SAVED, px=80_900.0, flat=False)
check("price reached the stop -> fresh levels", s[0].stop == 79_000.0)

s = short()
anchors.apply(s, SAVED, px=68_000.0, flat=False)
check("price reached the target -> fresh levels", s[0].stop == 79_000.0)

s = short()
anchors.apply(s, {}, px=77_000.0, flat=False)
check("no anchor stored -> fresh levels", s[0].stop == 79_000.0)

s = [Sleeve(label="#1 exp 2.0 · 2.5ATR ×3.0R · 21d", qty=-0.011, stop=79_000.0, take_profit=70_000.0)]
anchors.apply(s, SAVED, px=77_000.0, flat=False)
check("quarterly reselection changed the label -> fresh levels", s[0].stop == 79_000.0)

print("\nthe same rules for a long")
L = {"#1": dict(side=1, stop=70_000.0, tp=90_000.0)}
s = [Sleeve(label="#1", qty=0.011, stop=74_000.0, take_profit=85_000.0)]
anchors.apply(s, L, px=77_000.0, flat=False)
check("open long keeps its stop", s[0].stop == 70_000.0)
s = [Sleeve(label="#1", qty=0.011, stop=74_000.0, take_profit=85_000.0)]
anchors.apply(s, L, px=69_000.0, flat=False)
check("price reached the long's stop -> fresh levels", s[0].stop == 74_000.0)

print("\nwhat gets remembered")
snap = anchors.snapshot([Sleeve("#1", -0.004, 80_869.8, 68_675.4),
                         Sleeve("#2", 0.0, None, None),
                         Sleeve("#3", -0.003, 80_869.8, 68_675.4)])
check("only sleeves actually holding something", set(snap) == {"#1", "#3"}, sorted(snap))
check("side is recorded", snap["#1"]["side"] == -1)

with tempfile.TemporaryDirectory() as d:
    check("a missing file reads as empty, not an error", anchors.load(d) == {})
    anchors.save(d, snap)
    check("round trips through disk", anchors.load(d) == snap)
    open(os.path.join(d, "v7_anchors.json"), "w").write("{ not json")
    check("a corrupt file reads as empty rather than killing the run",
          anchors.load(d) == {})

print("\n" + ("FAILED: " + ", ".join(FAIL) if FAIL else "all good"))
sys.exit(1 if FAIL else 0)
