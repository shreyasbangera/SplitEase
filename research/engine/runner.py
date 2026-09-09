"""Evaluation harness: in-sample / out-of-sample splits, cost sensitivity,
and a persistent ledger of every attempt."""
import json, os, sys, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from backtest import run, Costs, RiskCfg, summary

CLEAN = "/tmp/claude-0/-home-user-SplitEase/9d81b630-73e8-5554-a848-3dc9a73c8211/scratchpad/clean"
LEDGER = "/home/user/SplitEase/research/results/ledger.jsonl"

BAR_HOURS = {"m15": 0.25, "h1": 1.0, "h4": 4.0, "d1": 24.0, "w1": 168.0, "m1": 730.0}

_cache = {}
def data(name):
    if name not in _cache:
        _cache[name] = pd.read_parquet(f"{CLEAN}/{name}.parquet")
    return _cache[name]


def bar_hours(name):
    return BAR_HOURS[name.split("_")[-1]]


def qualifies(r):
    return (r["cagr_pct"] > 500 and r["trades"] >= 100 and
            r["profit_factor"] > 1.10 and r["max_dd_pct"] < 20 and not r["ruined"])


def evaluate(name, strat, dataset, params=None, costs=None, risk=None,
             is_end="2018-12-31", verbose=True, log=True, notes=""):
    """Run full-period, in-sample and out-of-sample."""
    params = params or {}
    costs = costs or Costs()
    risk = risk or RiskCfg()
    df = data(dataset)
    bh = bar_hours(dataset)
    sig = strat(df, **params)
    out = {}
    for label, sl in [("full", slice(None)),
                      ("IS", slice(None, is_end)),
                      ("OOS", slice(is_end, None))]:
        d = df.loc[sl]
        if len(d) < 500:
            continue
        r = run(d, sig.loc[d.index], costs=costs, risk=risk, bar_hours=bh)
        out[label] = r
        if verbose:
            print(summary(f"{name} [{label}]", r))
    if log:
        rec = {"ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
               "name": name, "dataset": dataset, "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()},
               "costs": costs.__dict__, "risk": risk.__dict__, "notes": notes}
        for label, r in out.items():
            rec[label] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                          for k, v in r.items() if not k.startswith("_")}
        rec["qualifies_full"] = bool(qualifies(out["full"])) if "full" in out else False
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return out


def sweep(strat, dataset, grid, costs=None, risk=None, is_end="2018-12-31", top=12, metric="cagr_pct"):
    """Parameter sweep scored on IN-SAMPLE only; OOS reported for the survivors."""
    import itertools
    keys = list(grid)
    rows = []
    df = data(dataset); bh = bar_hours(dataset)
    costs = costs or Costs(); risk = risk or RiskCfg()
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, combo))
        try:
            sig = strat(df, **p)
        except Exception as e:
            continue
        d_is = df.loc[:is_end]
        r_is = run(d_is, sig.loc[d_is.index], costs=costs, risk=risk, bar_hours=bh)
        d_oos = df.loc[is_end:]
        r_oos = run(d_oos, sig.loc[d_oos.index], costs=costs, risk=risk, bar_hours=bh)
        rows.append({**p,
                     "is_n": r_is["trades"], "is_cagr": r_is["cagr_pct"], "is_dd": r_is["max_dd_pct"],
                     "is_pf": r_is["profit_factor"], "is_sharpe": r_is["sharpe"],
                     "oos_n": r_oos["trades"], "oos_cagr": r_oos["cagr_pct"], "oos_dd": r_oos["max_dd_pct"],
                     "oos_pf": r_oos["profit_factor"], "oos_sharpe": r_oos["sharpe"]})
    t = pd.DataFrame(rows)
    if len(t):
        t = t.sort_values("is_" + metric.replace("cagr_pct", "sharpe"), ascending=False)
    return t


def scale_to_dd(strat, dataset, params=None, target_dd=18.0, costs=None,
                max_lev=20.0, lo=0.002, hi=0.60, iters=18, sl=slice(None)):
    """Find the fixed-fractional risk per trade that puts max drawdown at the
    drawdown budget, and report what the strategy earns at that risk level.
    This is the honest way to ask 'how much can this edge make inside a 20% DD
    constraint' - it scales the bet, it does not change the edge."""
    params = params or {}
    costs = costs or Costs()
    df = data(dataset).loc[sl]
    bh = bar_hours(dataset)
    sig = strat(data(dataset), **params).loc[df.index]
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = run(df, sig, costs=costs, risk=RiskCfg(risk_frac=mid, max_leverage=max_lev), bar_hours=bh)
        if r["ruined"] or r["max_dd_pct"] > target_dd:
            hi = mid
        else:
            lo = mid
            best = (mid, r)
    if best is None:
        r = run(df, sig, costs=costs, risk=RiskCfg(risk_frac=lo, max_leverage=max_lev), bar_hours=bh)
        best = (lo, r)
    return best


def walk_forward(strat, dataset, grid, train_years=3, test_years=1, costs=None,
                 risk=None, metric="sharpe", min_trades=15, start=None):
    """Anchored-window walk-forward. Parameters are chosen ONLY on the training
    window and then applied, unchanged, to the immediately following test window.
    Concatenating the test windows gives an out-of-sample equity path that never
    saw its own parameters."""
    import itertools
    costs = costs or Costs(); risk = risk or RiskCfg()
    df = data(dataset); bh = bar_hours(dataset)
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*[grid[k] for k in keys])]
    sigs = {}
    for p in combos:
        try:
            sigs[tuple(sorted(p.items()))] = strat(df, **p)
        except Exception:
            pass
    t0 = pd.Timestamp(start) if start else df.index[0]
    segs = []
    cur = t0 + pd.DateOffset(years=train_years)
    while cur < df.index[-1]:
        tr = df.loc[cur - pd.DateOffset(years=train_years):cur]
        te = df.loc[cur:cur + pd.DateOffset(years=test_years)]
        if len(te) < 50:
            break
        best, bestv = None, -1e9
        for p in combos:
            key = tuple(sorted(p.items()))
            if key not in sigs:
                continue
            r = run(tr, sigs[key].loc[tr.index], costs=costs, risk=risk, bar_hours=bh)
            if r["trades"] < min_trades:
                continue
            v = r[metric]
            if v > bestv:
                best, bestv = p, v
        if best is None:
            cur += pd.DateOffset(years=test_years); continue
        rt = run(te, sigs[tuple(sorted(best.items()))].loc[te.index], costs=costs, risk=risk, bar_hours=bh)
        segs.append({"train_end": cur, "params": best, "test_ret_pct": rt["net_profit_pct"],
                     "test_n": rt["trades"], "test_pf": rt["profit_factor"],
                     "test_dd": rt["max_dd_pct"], "test_sharpe": rt["sharpe"],
                     "_eq": rt["_equity"]})
        cur += pd.DateOffset(years=test_years)
    return segs


def stitch(segs, init=10000.0):
    """Chain the walk-forward test windows into one continuous equity curve."""
    eq = []
    level = init
    for s in segs:
        e = s["_eq"] / s["_eq"].iloc[0] * level
        eq.append(e)
        level = e.iloc[-1]
    if not eq:
        return None
    return pd.concat(eq)


def portfolio(modules, weights=None, init=10000.0, bars_per_year=None, freq="4h"):
    """Combine independent strategy modules as a continuously-rebalanced
    portfolio of their per-bar returns. Each module is an equity Series; they are
    aligned on a common bar grid (modules on slower bars are forward-filled, so
    their return lands on the bar where it was actually earned)."""
    idx = None
    for m in modules:
        idx = m.index if idx is None else idx.union(m.index)
    rets = []
    for m in modules:
        r = m.reindex(idx).ffill().pct_change().fillna(0.0)
        rets.append(r)
    W = np.array(weights if weights is not None else [1 / len(modules)] * len(modules), float)
    R = sum(w * r for w, r in zip(W, rets))
    eq = init * (1 + R).cumprod()
    dd = eq / eq.cummax() - 1
    years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 24 * 3600)
    bpy = bars_per_year or (len(idx) / years)
    return {"equity": eq, "cagr_pct": ((eq.iloc[-1] / init) ** (1 / years) - 1) * 100,
            "max_dd_pct": -dd.min() * 100,
            "sharpe": R.mean() / R.std() * np.sqrt(bpy) if R.std() > 0 else 0,
            "net_profit_pct": (eq.iloc[-1] / init - 1) * 100, "years": years,
            "corr": pd.concat(rets, axis=1).corr()}
