"""
S101 - Is V7's number partly an artefact of WHICH 12-hour bars it uses?

The book decides at 00:00 and 12:00 UTC. Nobody chose that: it is what
pandas resamples to by default. S42 tested BLENDING phases and found
cross-phase correlation 0.70-0.82, so there was nothing to diversify. It never
asked the prior question - whether a different single phase gives a different
ANSWER.

This matters more after S96c, which showed the quarterly selection is
noise-dominated in level (sd about 24 points under a small perturbation). If a
six-hour shift moves the result by a comparable amount, then 179% is a draw
from a distribution rather than a property of the strategy, and the honest
headline is the centre of that distribution, not this particular draw.

Note what a result here can and cannot justify. If a shifted phase reads
HIGHER, that is not a better book to adopt - picking the phase on full-sample
results is exactly the in-sample selection this log keeps refusing. It would be
evidence of fragility either way. The only outcome that supports the headline
is the phases agreeing.

METHOD
------
ONLY GRID-ALIGNED OFFSETS ARE COMPARABLE. The positioning panel is 4-hourly
(00,04,08,12,16,20 UTC) and funding settles 8-hourly (00,08,16). A first run
used 3, 6 and 9-hour offsets and every one of them collapsed - but every one of
them is also OFF both native grids, so it measured the pipeline being sampled
off-grid as much as it measured phase sensitivity. 4 and 8 hours are the only
shifts that keep the 12h bars aligned to the data underneath, so they are the
only honest comparison.

Every 12-hour resample in the pipeline is shifted by the same offset - the
futures and spot aggregation in the harness, the cross-panel in S38, the
implied-vol panel in S36, and the 4h-to-12h positioning roll in S45 - by
patching the resample origin globally for the "12h" rule only. Everything
downstream (execution alignment, the 15-minute grid, the backtest) follows the
signal timestamps, so it shifts with them.

The quarterly rankings are RECOMPUTED on each shifted panel rather than reused,
because a selection fitted on one phase applied to another would be measuring
neither.
"""
import sys, os, json; sys.path.insert(0, "/home/user/quant")
import numpy as np
import pandas as pd

from strategies.s96_rank import quarters, at_gate, stats_of

OFFSETS = (0, 4, 8)
_df_res, _s_res = pd.DataFrame.resample, pd.Series.resample
_STATE = {"origin": None}


def _patch(orig):
    def wrapper(self, rule, *a, **kw):
        if (_STATE["origin"] is not None and isinstance(rule, str)
                and rule.lower() == "12h" and "origin" not in kw):
            kw["origin"] = _STATE["origin"]
        return orig(self, rule, *a, **kw)
    return wrapper


pd.DataFrame.resample = _patch(_df_res)
pd.Series.resample = _patch(_s_res)


def set_phase(hours):
    _STATE["origin"] = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=hours)
    # every cache downstream of the panel has to go
    import research.harness as H
    import strategies.s38_orth as s38, strategies.s36_ivmom as s36
    import strategies.s69_calsel as C, strategies.s84_gate as G
    H._cache.clear(); H._ctxc.clear()
    s38._P.clear(); s36._C.clear(); C._C.clear(); G._T.clear()


def rankings_resumable(off):
    """S87.rankings(), but checkpointed after every quarter.

    The upstream version writes its JSON only once the whole 18-quarter loop
    has finished, so a 13-minute run that dies at minute 10 loses all of it -
    which is exactly what happened when this container was idle-reclaimed
    mid-run. Each quarter is ~40 seconds of work and there is no reason to risk
    more than that.
    """
    import json, os
    import strategies.s87_combined as S87
    from strategies.s69_calsel import calmar_of

    path = f"/home/user/quant/results/ranks_phase{off:02d}.json"
    done = []
    if os.path.exists(path):
        done = [(s, e, [tuple(c) for c in cs]) for s, e, cs in json.load(open(path))]
    Q = quarters()
    if len(done) >= len(Q):
        return done
    for s, e in Q[len(done):]:
        tr0 = str((pd.Timestamp(s, tz="UTC") - pd.DateOffset(months=12)).date())
        sc = sorted(((calmar_of(S87.sim(cfg, tr0, s, 0.10)), cfg) for cfg in S87.GRID),
                    key=lambda x: -x[0])
        done.append((s, e, [tuple(c) for c in (c for _, c in sc)]))
        json.dump([[a, b, [list(c) for c in cs]] for a, b, cs in done], open(path, "w"))
        print(f"    {s[:7]}  best Calmar {sc[0][0]:.2f}   "
              f"[{len(done)}/{len(Q)} checkpointed]", flush=True)
    return done


if __name__ == "__main__":
    import strategies.s87_combined as S87

    print("V7 rebuilt from scratch on each 12h bar phase, rankings recomputed.\n")
    print(f"{'phase (UTC)':>16}{'bars':>7}{'CAGR':>9}{'MaxDD':>8}{'Shp':>7}"
          f"{'Clm':>7}{'trades':>8}{'at -20%':>10}")
    res = {}
    for off in OFFSETS:
        set_phase(off)
        S87.RANKS = f"/home/user/quant/results/ranks_phase{off:02d}.json"
        import strategies.s45_single as S
        g = S.grid(S.FULL_START)
        first = pd.Timestamp(g.dt.iloc[0]).strftime("%H:%M")
        R = rankings_resumable(off)
        r, pnl = S87.blend(R, 3, 0.08)
        a = np.asarray(r, float)
        s, gt = stats_of(a), at_gate(a)
        res[off] = gt["cagr"]
        lbl = f"{off:02d}:00 / {(off+12)%24:02d}:00"
        print(f"{lbl:>16}{len(g):7d}{s['cagr']*100:8.1f}%{s['dd']*100:7.1f}%"
              f"{s['sharpe']:7.2f}{s['calmar']:7.2f}{len(pnl):8d}"
              f"{gt['cagr']*100:9.1f}%", flush=True)

    v = np.array(list(res.values())) * 100
    print(f"\nacross {len(v)} phases at the gate: mean {v.mean():.1f}%  "
          f"sd {v.std(ddof=1):.1f}  range {v.min():.1f}-{v.max():.1f}%")
    print(f"deployed phase (00:00/12:00) = {res[0]*100:.1f}%, which is "
          f"{(res[0]*100-v.mean())/v.std(ddof=1):+.2f} sd from the mean of the {len(v)}")
    print("\ndone: bar phase")
