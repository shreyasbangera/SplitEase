"""
S106 - Is the ARCHITECTURE bounded below 300%, or only the search of it?

Every negative in this log is a negative about one channel. S96b is different in
kind: it bounded a whole channel by giving the selector perfect foresight and
showing that even then it tops out at 268.9%, below the brief. That closed
config selection permanently rather than by exhaustion.

But S96b's oracle ranged over CONFIGURATIONS - exponent, stop, target, hold,
gate. V7 has a second free parameter that no selection rule has ever touched:
**which signals are in the composite, and in what proportion**. The five are
equal-weighted because S46 found the five-signal set best on the full sample and
nobody has revisited it since. If an oracle over weights also lands below 300%,
then the architecture itself is bounded and no amount of further searching
inside it can reach the brief - which is a terminal result rather than another
closed door.

THREE THINGS ARE MEASURED, AND ONLY THE FIRST IS TRADABLE
---------------------------------------------------------
    control   equal weights over all five - what V7 deploys
    causal    the weighting chosen each quarter on TRAILING 12-month Calmar,
              the same rule and the same lookback the config selection uses
    oracle    the weighting chosen each quarter on that quarter's REALISED
              Calmar. Pure look-ahead, deliberately, as a ceiling.

The oracle is the point. The causal row is nearly free once the machinery
exists, and it is the one that could actually be adopted - so it is reported
first and judged against the same bar as everything else.

The bet SIZE is held constant across weightings by construction: `shape()`
renormalises so mean |conviction| matches the equal-weight book whatever the
weights are. Without that, a narrower composite would read as a different
strategy simply for betting bigger, which is the dilution trap S95d fell into
from the other direction.
"""
import sys, itertools, json, os; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s45_single as S45
import strategies.s69_calsel as S69
import strategies.s87_combined as S87
from strategies.s96_rank import at_gate, stats_of, quarters

RISK, K = 0.08, 3
NAMES = ["flow", "cmpx", "btcdom", "fundz", "posn"]
CACHE = "/home/user/quant/results/s106_scores.json"


def install_w(w):
    """Rebuild the composite under weight vector `w` and hand it to the stack."""
    c = S69.ctx()
    g = c["g"]
    v = S45.composite(g, NAMES, w=np.asarray(w, float))
    nz = np.abs(v) > 0
    if not nz.any():
        return False
    c["v"] = v; c["nz"] = nz; c["bm"] = float(np.abs(v[nz]).mean())
    return True


def blend_over(cfgs, s, e, w):
    """V7's top-3 blend for one window, with the composite built from `w`."""
    if not install_w(w):
        return None
    rs, ok = [], False
    for cfg in cfgs[:K]:
        m = S87.sim(cfg, s, e, RISK / K)
        rs.append(S69.daily(m))
        ok |= m["trades"] > 0
    if not ok:
        return None
    return pd.DataFrame({j: r for j, r in enumerate(rs)}).fillna(0.0).sum(axis=1)


def calmar(r):
    if r is None or len(r) < 20:
        return -9e9
    s = stats_of(np.asarray(r, float))
    return s["calmar"] if s["dd"] < -1e-6 else -9e9


# the weight menu: every non-empty subset, plus a double weight on each single
# signal inside the full set. 31 + 5 = 36 vectors, coarse on purpose - a finer
# grid would be a larger search, and a larger search is exactly what an oracle
# bound is supposed to make unnecessary.
def menu():
    out = []
    for k in range(1, 6):
        for c in itertools.combinations(range(5), k):
            w = [0.0] * 5
            for i in c:
                w[i] = 1.0
            out.append(tuple(w))
    for i in range(5):
        w = [1.0] * 5
        w[i] = 2.0
        out.append(tuple(w))
    return out


def scores():
    """Per quarter: the realised Calmar of every weighting, on the TEST window
    and on the preceding 12 months. Cached - this is the whole cost of the file."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    R = S87.rankings()
    M, W = {}, menu()
    for s, e, cfgs in R:
        tr0 = str((pd.Timestamp(s, tz="UTC") - pd.DateOffset(months=12)).date())
        row = {}
        for w in W:
            row["|".join(f"{x:g}" for x in w)] = [
                calmar(blend_over(cfgs, tr0, s, w)),      # trailing, causal
                calmar(blend_over(cfgs, s, e, w))]        # realised, hindsight
        M[s] = row
        print(f"    {s[:7]} done ({len(M)}/{len(R)})", flush=True)
        json.dump(M, open(CACHE, "w"))
    return M


def assemble(M, pick):
    """Concatenate each quarter's returns under the weighting `pick` selects."""
    R = S87.rankings()
    segs = []
    for s, e, cfgs in R:
        row = M[s]
        key = pick(row)
        if row[key][1] <= -9e8:          # every weighting unusable in this quarter
            key = "1|1|1|1|1"            # fall back to the control rather than drop it
        w = tuple(float(x) for x in key.split("|"))
        segs.append(blend_over(cfgs, s, e, w))
    return pd.concat(segs)


def line(tag, r, base=None):
    a = np.asarray(r, float)
    st, gt = stats_of(a), at_gate(a)
    d = "" if base is None else f"{gt['cagr']*100 - base:+9.1f}"
    print(f"{tag:>32}{st['cagr']*100:8.1f}%{st['dd']*100:7.1f}%{st['sharpe']:7.2f}"
          f"{st['calmar']:7.2f}{gt['cagr']*100:10.1f}%{d}", flush=True)
    return gt["cagr"] * 100


if __name__ == "__main__":
    print(f"{len(menu())} weightings x {len(quarters())} quarters, "
          f"trailing and realised Calmar for each\n")
    M = scores()
    EQ = "1|1|1|1|1"

    print(f"\n{'book':>32}{'CAGR':>9}{'MaxDD':>8}{'Shp':>7}{'Clm':>7}"
          f"{'at -20%':>10}{'vs V7':>10}")
    base = line("control - equal weights (V7)", assemble(M, lambda r: EQ))
    line("causal - trailing 12m Calmar",
         assemble(M, lambda r: max(r, key=lambda k: r[k][0])), base)
    line("ORACLE - realised Calmar",
         assemble(M, lambda r: max(r, key=lambda k: r[k][1])), base)

    print("\nwhat the oracle picks, and what the causal rule picks")
    from collections import Counter
    co = Counter(max(r, key=lambda k: r[k][1]) for r in M.values())
    cc = Counter(max(r, key=lambda k: r[k][0]) for r in M.values())
    print(f"  oracle : {dict(co.most_common(6))}")
    print(f"  causal : {dict(cc.most_common(6))}")
    hit = sum(max(r, key=lambda k: r[k][0]) == max(r, key=lambda k: r[k][1])
              for r in M.values())
    print(f"  the causal rule picks the oracle's weighting in {hit}/{len(M)} quarters")

    rho = np.corrcoef(
        [r[k][0] for r in M.values() for k in r],
        [r[k][1] for r in M.values() for k in r])[0, 1]
    print(f"  rho(trailing Calmar, next-quarter Calmar) over all weightings = {rho:+.3f}")
    print("\ndone: weight bound")
