"""
XAUUSD backtest engine.
 - Signals computed on COMPLETED signal-timeframe bars.
 - Execution no earlier than the NEXT bar's open.
 - Stop/target resolution walked at M1 resolution (removes intrabar ambiguity).
 - Conservative: if SL and TP are both touchable inside the same M1 bar, SL wins.
 - Costs: real per-minute broker spread + commission (bps of notional) + slippage.
 - Risk-based position sizing with a leverage cap; fully compounding.
"""
import numpy as np, pandas as pd
from dataclasses import dataclass, field

W = "/tmp/claude-0/-home-user-SplitEase/dab1516d-4d3c-5d20-84aa-3aeaebb0401b/scratchpad/work"

# ----------------------------------------------------------------- data
_M1 = None
def m1():
    global _M1
    if _M1 is None:
        d = pd.read_parquet(f"{W}/m1_clean.parquet")
        d.index = pd.to_datetime(d.index, utc=True)
        _M1 = d
    return _M1

def bars(tf):
    """Resample M1 -> signal timeframe. Right-labelled=False (bar stamped at its open)."""
    d = m1()
    o = d.resample(tf).agg({'open':'first','high':'max','low':'min','close':'last',
                            'tick_volume':'sum','spread':'mean'}).dropna()
    return o

# ----------------------------------------------------------------- costs
@dataclass
class Costs:
    comm_bps_side: float = 2.0     # commission per side, bps of notional
    slip_usd_side: float = 0.05    # slippage per side, USD/oz
    stop_slip_usd:  float = 0.15   # EXTRA slippage when exiting on a stop
    spread_mult:    float = 1.0    # scale the broker spread from the data

@dataclass
class RiskCfg:
    risk_pct:      float = 0.01    # fraction of equity risked per trade
    max_leverage:  float = 10.0    # notional / equity cap
    start_equity:  float = 10000.0

# ----------------------------------------------------------------- core
def run(sig_df, entries, sl_dist, tp_dist, tf,
        costs=Costs(), risk=RiskCfg(),
        max_hold_bars=200, trail_dist=None, be_trigger=None,
        allow_long=True, allow_short=True, name="strategy"):
    """
    sig_df   : signal-timeframe OHLC (index = bar OPEN time)
    entries  : int array on sig_df, +1 long / -1 short / 0 flat. Decision uses
               data through that bar's CLOSE; fill happens at the NEXT bar open.
    sl_dist  : USD/oz stop distance, aligned to sig_df
    tp_dist  : USD/oz target distance, aligned to sig_df
    trail_dist / be_trigger : optional USD/oz trailing stop / breakeven trigger
    """
    d1 = m1()
    mt = d1.index.values.astype('datetime64[ns]')
    mo, mh, ml, mc = (d1[c].values.astype(np.float64) for c in ('open','high','low','close'))
    msp = d1['spread'].values.astype(np.float64) * 0.001 * costs.spread_mult  # points -> USD

    st = sig_df.index.values.astype('datetime64[ns]')
    tf_delta = pd.Timedelta(tf)
    # next-bar open time for each signal bar
    nxt = st + np.timedelta64(int(tf_delta.total_seconds()), 's')
    # map each signal bar's next-open to the first M1 bar at/after it
    m1_entry_idx = np.searchsorted(mt, nxt, side='left')

    ent = np.asarray(entries, dtype=np.int64)
    sld = np.asarray(sl_dist,  dtype=np.float64)
    tpd = np.asarray(tp_dist,  dtype=np.float64)

    H = int(max_hold_bars * (tf_delta.total_seconds() / 60.0))   # hold horizon in minutes
    equity = risk.start_equity
    trades = []
    eq_t, eq_v = [], []
    block_until = np.datetime64('1970-01-01')   # one position at a time
    n_m1 = len(mt)

    for i in range(len(st) - 1):
        s = ent[i]
        if s == 0: continue
        if s > 0 and not allow_long: continue
        if s < 0 and not allow_short: continue
        k = m1_entry_idx[i]
        if k >= n_m1 - 2: break
        if mt[k] < block_until: continue
        if not (np.isfinite(sld[i]) and sld[i] > 0): continue

        spr = msp[k] if np.isfinite(msp[k]) and msp[k] > 0 else 0.20
        # fill: bars are BID. long pays ask (+spread), short sells bid. slippage always adverse.
        px_ref = mo[k]
        entry_px = px_ref + s*(spr + costs.slip_usd_side)

        sl = entry_px - s*sld[i]
        tp = entry_px + s*tpd[i]

        # ---- position size: risk_pct of equity over the stop distance, leverage-capped
        risk_usd = equity * risk.risk_pct
        units = risk_usd / sld[i]
        max_units = (equity * risk.max_leverage) / entry_px
        units = min(units, max_units)
        if units <= 0: continue

        # ---- walk M1 forward to find the exit
        hi = min(k + H, n_m1)
        h = mh[k:hi]; l = ml[k:hi]
        nb = len(h)
        if nb < 2: break

        if s > 0:
            hit_sl = l <= sl
            hit_tp = h >= tp
        else:
            hit_sl = h >= sl
            hit_tp = l <= tp

        # optional trailing stop / breakeven, based on info through the PREVIOUS bar
        if trail_dist is not None or be_trigger is not None:
            if s > 0:
                run_ext = np.maximum.accumulate(h)
                prev_ext = np.concatenate(([entry_px], run_ext[:-1]))
                dyn = np.full(nb, -np.inf)
                if trail_dist is not None:
                    dyn = np.maximum(dyn, prev_ext - trail_dist)
                if be_trigger is not None:
                    dyn = np.where(prev_ext >= entry_px + be_trigger,
                                   np.maximum(dyn, entry_px), dyn)
                hit_sl = hit_sl | (l <= dyn)
            else:
                run_ext = np.minimum.accumulate(l)
                prev_ext = np.concatenate(([entry_px], run_ext[:-1]))
                dyn = np.full(nb, np.inf)
                if trail_dist is not None:
                    dyn = np.minimum(dyn, prev_ext + trail_dist)
                if be_trigger is not None:
                    dyn = np.where(prev_ext <= entry_px - be_trigger,
                                   np.minimum(dyn, entry_px), dyn)
                hit_sl = hit_sl | (h >= dyn)

        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else 10**9
        i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else 10**9

        if i_sl == 10**9 and i_tp == 10**9:
            j = nb - 1; reason = 'time'; exit_ref = mc[k + j]
        elif i_sl <= i_tp:                       # conservative tie-break -> stop
            j = i_sl; reason = 'sl'
            gap = (ml[k+j] > sl) if s > 0 else (mh[k+j] < sl)   # gapped through
            exit_ref = mo[k+j] if ((s>0 and mo[k+j] < sl) or (s<0 and mo[k+j] > sl)) else sl
        else:
            j = i_tp; reason = 'tp'
            exit_ref = mo[k+j] if ((s>0 and mo[k+j] > tp) or (s<0 and mo[k+j] < tp)) else tp

        ex_slip = costs.slip_usd_side + (costs.stop_slip_usd if reason == 'sl' else 0.0)
        exit_px = exit_ref - s*ex_slip

        gross = s * (exit_px - entry_px) * units
        notional = units * entry_px
        comm = 2.0 * notional * costs.comm_bps_side / 10000.0
        pnl = gross - comm

        # intra-trade worst mark-to-market (true drawdown, not just bar closes)
        worst = (np.min(l[:j+1]) - entry_px) if s > 0 else (entry_px - np.max(h[:j+1]))
        mae_usd = worst * units

        eq_t.append(mt[k]);       eq_v.append(equity + min(0.0, mae_usd))  # trough
        equity += pnl
        eq_t.append(mt[k+j]);     eq_v.append(equity)

        trades.append(dict(entry_time=mt[k], exit_time=mt[k+j], side=int(s),
                           entry=entry_px, exit=exit_px, units=units,
                           notional=notional, pnl=pnl, reason=reason,
                           equity=equity, hold_min=int(j), mae=mae_usd,
                           R=pnl/risk_usd if risk_usd else 0.0))
        block_until = mt[k+j]
        if equity <= risk.start_equity * 0.02:
            break

    return Result(name, tf, pd.DataFrame(trades), pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)), risk)

# ----------------------------------------------------------------- results
class Result:
    def __init__(self, name, tf, tr, eq, risk):
        self.name, self.tf, self.trades, self.eq, self.risk = name, tf, tr, eq, risk
        self.m = self._metrics()

    def _metrics(self):
        t = self.trades
        if len(t) == 0:
            return dict(trades=0, net_pct=0, cagr=0, pf=0, mdd=1, win=0, qualifies=False)
        e0 = self.risk.start_equity
        ef = t['equity'].iloc[-1]
        eqc = self.eq
        peak = eqc.cummax()
        mdd = float(((peak - eqc) / peak).max())
        yrs = (t['exit_time'].iloc[-1] - t['entry_time'].iloc[0]) / np.timedelta64(1, 'D') / 365.25
        gp = t.loc[t.pnl > 0, 'pnl'].sum(); gl = -t.loc[t.pnl < 0, 'pnl'].sum()
        pf = gp / gl if gl > 0 else np.inf
        cagr = ((ef / e0) ** (1 / yrs) - 1) if yrs > 0 and ef > 0 else -1
        # per-trade R Sharpe-ish
        return dict(trades=len(t), net_pct=(ef/e0-1)*100, final_eq=ef, years=yrs,
                    cagr=cagr*100, pf=pf, mdd=mdd, win=(t.pnl>0).mean(),
                    avg_R=t.R.mean(), expectancy=t.pnl.mean(),
                    qualifies=bool(cagr*100 > 500 and len(t) >= 100 and pf > 1.10 and mdd < 0.20))

    def line(self):
        m = self.m
        if m['trades'] == 0: return f"{self.name:<34} {self.tf:>4}  NO TRADES"
        return (f"{self.name:<34} {self.tf:>4} | trades {m['trades']:>5} | CAGR {m['cagr']:>9.1f}% | "
                f"PF {m['pf']:>5.2f} | mDD {m['mdd']*100:>5.1f}% | win {m['win']*100:>4.1f}% | "
                f"net {m['net_pct']:>12,.0f}% | {'QUALIFIES' if m['qualifies'] else ''}")
