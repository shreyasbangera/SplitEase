"""
S106b - The weight oracle, with the controls it needs before anyone believes it.

S106 found the result this study has been missing for a hundred experiments, and
it points the OPPOSITE way to S96b:

    oracle over CONFIGURATIONS   (S96b, 200 cells)    268.9%   - below the brief
    oracle over SIGNAL WEIGHTS   (S106, 36 cells)     836.7%   - five times V7

If that survives, the architecture is not bounded. The headroom is not in a
missing signal or a missing structure; it is in **which of the five signals the
book uses each quarter**, and the brief is reachable in principle without a
single new data source.

Three things could make 836.7% an artefact, and each gets a control here.

    1. THE CRITERION. S106's oracle picked on realised Calmar, whose denominator
       is a single worst day. A quarter that happens to have a shallow drawdown
       scores enormously whatever its edge. S96b found foresight of CAGR (93.1%)
       far WORSE than foresight of Sharpe (268.9%) for exactly this reason, so
       the criterion is not a detail. Every criterion is reported.

    2. SELECTION NOISE. A max over 36 cells per quarter is a strong selection,
       and some of the gain is the luckiest PATH rather than the best signal set.
       The control is random-of-36, resampled, which is the same selection
       pressure with none of the foresight.

    3. THE PATH ITSELF. The sharpest test: pick the weighting on the FIRST HALF
       of each quarter's realised returns and collect the SECOND half. That is
       still look-ahead - no tradable rule knows the first half of a quarter
       before it starts - but it can only exploit structure that PERSISTS inside
       the quarter, not the single lucky day Calmar rewards. If most of the gain
       survives it, the oracle is finding something real.

Every quarter's returns are cached per weighting, so all of the above, and any
later question, costs nothing after the first run.
"""
import sys, itertools, json, os; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

import strategies.s45_single as S45
import strategies.s69_calsel as S69
import strategies.s87_combined as S87
from strategies.s96_rank import at_gate, stats_of

RISK, K = 0.08, 3
NAMES = ["flow", "cmpx", "btcdom", "fundz", "posn"]
NPZ = "/home/user/quant/results/s106b_returns.npz"
TRAIL = "/home/user/quant/results/s106_scores.json"
EQ = "1|1|1|1|1"


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


def key_of(w):
    return "|".join(f"{x:g}" for x in w)


def install_w(w):
    c = S69.ctx()
    v = S45.composite(c["g"], NAMES, w=np.asarray(w, float))
    nz = np.abs(v) > 0
    if not nz.any():
        return False
    c["v"] = v; c["nz"] = nz; c["bm"] = float(np.abs(v[nz]).mean())
    return True


def blend_over(cfgs, s, e, w):
    if not install_w(w):
        return None
    rs = []
    for cfg in cfgs[:K]:
        rs.append(S69.daily(S87.sim(cfg, s, e, RISK / K)))
    return pd.DataFrame({j: r for j, r in enumerate(rs)}).fillna(0.0).sum(axis=1)


def build():
    """Daily returns for every (quarter, weighting). The whole cost of the file."""
    if os.path.exists(NPZ):
        z = np.load(NPZ, allow_pickle=True)
        return z["R"].item(), list(z["Q"]), list(z["W"])
    R = S87.rankings()
    W = [key_of(w) for w in menu()]
    out, Q = {}, []
    for s, e, cfgs in R:
        Q.append(s)
        for w in menu():
            r = blend_over(cfgs, s, e, w)
            out[(s, key_of(w))] = (np.asarray(r, float) if r is not None
                                   else np.zeros(1))
        print(f"    {s[:7]} ({len(Q)}/{len(R)})", flush=True)
        np.savez(NPZ, R=np.array(out, dtype=object), Q=np.array(Q),
                 W=np.array(W))
    return out, Q, W


def series(R, Q, pick):
    return np.concatenate([R[(q, pick(q))] for q in Q])


def crit(a, how):
    s = stats_of(a) if len(a) > 20 else None
    if s is None or not np.isfinite(s["sharpe"]):
        return -9e9
    return dict(calmar=s["calmar"] if s["dd"] < -1e-6 else -9e9,
                sharpe=s["sharpe"], ret=float(np.prod(1 + a) - 1))[how]


def line(tag, a, base=None):
    st, gt = stats_of(a), at_gate(a)
    d = "" if base is None else f"{gt['cagr']*100 - base:+10.1f}"
    print(f"{tag:>38}{st['cagr']*100:9.1f}%{st['dd']*100:8.1f}%{st['sharpe']:7.2f}"
          f"{st['calmar']:8.2f}{gt['cagr']*100:11.1f}%{d}", flush=True)
    return gt["cagr"] * 100


if __name__ == "__main__":
    R, Q, W = build()
    TR = json.load(open(TRAIL)) if os.path.exists(TRAIL) else {}

    print(f"\n{len(W)} weightings x {len(Q)} quarters\n")
    print(f"{'book':>38}{'CAGR':>10}{'MaxDD':>9}{'Shp':>7}{'Clm':>8}"
          f"{'at -20%':>11}{'vs V7':>11}")
    base = line("control - equal weights (V7)", series(R, Q, lambda q: EQ))

    if TR:
        line("causal - trailing 12m Calmar",
             series(R, Q, lambda q: max(W, key=lambda k: TR[q][k][0])), base)

    print()
    for how in ("calmar", "sharpe", "ret"):
        line(f"ORACLE on realised {how}",
             series(R, Q, lambda q, h=how: max(W, key=lambda k: crit(R[(q, k)], h))),
             base)

    print("\ncontrol 1 - random of 36 each quarter: the same selection pressure,\n"
          "            none of the foresight (200 draws)")
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(200):
        pick = {q: W[rng.integers(len(W))] for q in Q}
        vals.append(at_gate(series(R, Q, lambda q: pick[q]))["cagr"] * 100)
    v = np.array(vals)
    print(f"{'random-of-36':>38}{'':35}{np.median(v):11.1f}%{np.median(v)-base:+10.1f}")
    print(f"{'':38}   mean {v.mean():.1f}%  sd {v.std(ddof=1):.1f}  "
          f"p95 {np.percentile(v, 95):.1f}%  max {v.max():.1f}%")

    print("\ncontrol 2 - pick on the FIRST half of each quarter, collect the SECOND.\n"
          "            Still look-ahead, but only rewards structure that persists.")

    def split_pick(q, how):
        best, bk = -9e9, EQ
        for k in W:
            a = R[(q, k)]
            h = len(a) // 2
            sc = crit(a[:h], how)
            if sc > best:
                best, bk = sc, k
        return bk

    for how in ("calmar", "sharpe"):
        half = {q: split_pick(q, how) for q in Q}
        a = np.concatenate([R[(q, half[q])][len(R[(q, half[q])]) // 2:] for q in Q])
        ctl = np.concatenate([R[(q, EQ)][len(R[(q, EQ)]) // 2:] for q in Q])
        g1, g0 = at_gate(a), at_gate(ctl)
        print(f"{'half-sample oracle on ' + how:>38}{'':35}{g1['cagr']*100:11.1f}%"
              f"{g1['cagr']*100 - g0['cagr']*100:+10.1f}")
    print(f"{'  (2nd-half control, equal weights)':>38}{'':35}"
          f"{at_gate(ctl)['cagr']*100:11.1f}%")

    print("\nwhat the oracle actually asks for")
    from collections import Counter
    for how in ("calmar", "sharpe"):
        c = Counter(max(W, key=lambda k: crit(R[(q, k)], how)) for q in Q)
        n = Counter(k.count("0") for k in
                    (max(W, key=lambda k: crit(R[(q, k)], how)) for q in Q))
        print(f"  {how:>7}: {dict(c.most_common(5))}")
        print(f"{'':11}signals dropped per quarter: "
              + " ".join(f"{5-z}sig:{m}" for z, m in sorted(n.items())))
    print("\ndone: weight bound, controlled")
