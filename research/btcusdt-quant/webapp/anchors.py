"""Where a trade's stop lives between decisions.

THE BUG THIS FIXES
------------------
V7 recomputes its target every 12h from the current conviction, and until now it
also recomputed the stop from the current close.  So the stop was a property of
the latest bar, not of the trade: while price ground against a position the stop
was re-placed further away every 12 hours, and the loss was never capped where
the risk budget said it would be.

s178_livesim.py prices that at Sharpe 0.99 against 2.77 for the same strategy
with the stop held still - the difference between a live edge and no edge.

Anchoring to the position's average entry does NOT work: as the bot resizes into
a trade the average entry drifts toward the current price, so the stop chases
anyway (measured: Sharpe 1.01).  The level has to be remembered.

WHAT IS STORED
--------------
One record per sleeve label, mirroring the ladder actually resting on the
exchange.  Written only after an armed send, so a dry run can never corrupt it.

An anchor is void when:
  * the account is flat            - there is no trade for it to belong to
  * the side flipped               - it is a different trade
  * price has already reached the  - that trade is over; the exchange filled the
    stop or the target               order even if we have not seen it yet
  * the label changed              - the quarterly reselection picked new configs
"""
import json, os


def path(store):
    return os.path.join(str(store), "v7_anchors.json")


def load(store):
    try:
        with open(path(store)) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save(store, obj):
    p = path(store)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)                      # never leave a half-written ladder


def spent(rec, px):
    """True if price has already reached this trade's stop or target."""
    side, stop, tp = rec["side"], rec["stop"], rec["tp"]
    if side > 0:
        return px <= stop or (tp and px >= tp)
    return px >= stop or (tp and px <= tp)


def apply(sleeves, saved, px, flat):
    """Replace each sleeve's freshly-derived stop/target with the level the trade
    opened on, where one is still valid.  Returns the number reused."""
    if flat:
        return 0
    n = 0
    for s in sleeves:
        if not s.qty:
            continue
        rec = saved.get(s.label)
        if not rec:
            continue
        side = 1 if s.qty > 0 else -1
        if rec["side"] != side or spent(rec, px):
            continue
        s.stop, s.take_profit = rec["stop"], rec["tp"]
        n += 1
    return n


def snapshot(sleeves):
    """The ladder as it should be remembered, mirroring what was just placed."""
    return {s.label: dict(side=(1 if s.qty > 0 else -1),
                          stop=s.stop, tp=s.take_profit)
            for s in sleeves if s.qty and s.stop}
