"""
S107 - Where the RETURN comes from, which is the question S103 asked of the losses.

S103 opened the drawdowns and found they are mark-to-market excursions rather
than losing trades. The mirror question was never asked: is the 168.8% spread
across the record, or does it live in a handful of quarters?

It matters for one specific reason. The brief needs Calmar 8.95 -> 15, and every
route to that through selection is now bounded - S96b over configurations,
S106b over signal weights. What is left to say honestly about V7 is not "how do
we improve it" but "what is it", and a book whose return is concentrated in
three quarters is a different object from one that earns steadily, even when
both print the same CAGR.

Free: the per-quarter daily returns of the equal-weight book are already cached
by S106b, so this adds no simulation.
"""
import sys, os; sys.path.insert(0, "/home/user/quant")
import numpy as np, pandas as pd

from strategies.s96_rank import at_gate, stats_of

NPZ = "/home/user/quant/results/s106b_returns.npz"
EQ = "1|1|1|1|1"


if __name__ == "__main__":
    z = np.load(NPZ, allow_pickle=True)
    R, Q = z["R"].item(), list(z["Q"])
    per = {q: R[(q, EQ)] for q in Q}
    a = np.concatenate([per[q] for q in Q])
    full = at_gate(a)
    print(f"V7, equal weights: {len(a)} days, {len(Q)} quarters, "
          f"{full['cagr']*100:.1f}% at the -20% gate (scale {full['scale']:.2f})\n")

    print(f"{'quarter':>10}{'days':>6}{'return':>10}{'maxDD':>9}{'Sharpe':>8}"
          f"{'share of total log-growth':>28}")
    tot = float(np.sum([np.log1p(per[q]).sum() for q in Q]))
    rows = []
    for q in Q:
        r = per[q]
        g = float(np.log1p(r).sum())
        s = stats_of(r)
        rows.append((q, g))
        print(f"{q:>10}{len(r):6d}{(np.exp(g)-1)*100:9.1f}%{s['dd']*100:8.1f}%"
              f"{s['sharpe']:8.2f}{g/tot*100:26.1f}%")

    rows.sort(key=lambda x: -x[1])
    cum = np.cumsum([g for _, g in rows]) / tot
    print(f"\nconcentration of log-growth")
    for k in (1, 2, 3, 4, 6, 9):
        print(f"  best {k:2d} of {len(Q)} quarters ({k/len(Q)*100:4.1f}% of the record) "
              f"= {cum[k-1]*100:5.1f}% of all growth   [{', '.join(q for q, _ in rows[:k])}]"
              if k <= 3 else
              f"  best {k:2d} of {len(Q)} quarters ({k/len(Q)*100:4.1f}% of the record) "
              f"= {cum[k-1]*100:5.1f}% of all growth")
    neg = sum(1 for _, g in rows if g < 0)
    print(f"  losing quarters: {neg} of {len(Q)}")

    print("\ndropping the best quarters, one at a time, and re-gating")
    for k in (0, 1, 2, 3):
        drop = {q for q, _ in rows[:k]}
        b = np.concatenate([per[q] for q in Q if q not in drop])
        g = at_gate(b)
        print(f"  drop best {k}: {g['cagr']*100:7.1f}% at -20%"
              + (f"   ({(g['cagr']-full['cagr'])*100:+.1f} vs the full record)" if k else ""))

    print("\nby calendar year")
    idx = pd.date_range("2022-03-01", periods=len(a), freq="D")[:len(a)]
    s = pd.Series(a, index=idx)
    for y, gg in s.groupby(s.index.year):
        st = stats_of(gg.to_numpy())
        print(f"  {y}  {len(gg):4d}d  return {(np.exp(np.log1p(gg).sum())-1)*100:8.1f}%"
              f"   maxDD {st['dd']*100:6.1f}%   Sharpe {st['sharpe']:5.2f}")
    print("\ndone: concentration")
