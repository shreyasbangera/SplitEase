"""Bar-by-bar backtest engine with realistic frictions.

Execution model (no look-ahead by construction):
  * A strategy sees bars 0..i (all completed) and emits a signal on bar i.
  * The engine opens/closes that position at the OPEN of bar i+1, plus slippage.
  * Stops/targets are checked against the high/low of every bar the position is open.
  * If a single bar touches both the stop and the target, the STOP is assumed to
    fill first (conservative; verified separately against 15m intrabar data).
  * Equity is marked to market on every bar close, so drawdown includes open risk.

Costs:
  * taker fee, in bps of notional, charged on entry and on exit
  * slippage, in bps, applied against us on entry and exit (and on stop fills)
  * perp funding, in bps of notional per 8h, charged while a position is held
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

BPS = 1e-4


@dataclass
class Costs:
    fee_bps: float = 5.0        # 0.05% taker per side
    slip_bps: float = 2.0       # 0.02% per side
    funding_bps_8h: float = 1.0  # 0.01% / 8h on notional (perp)


@dataclass
class RiskCfg:
    risk_frac: float = 0.01     # fraction of equity risked per trade (stop distance)
    max_leverage: float = 5.0   # cap on notional / equity
    compound: bool = True


@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    qty: float
    pnl: float
    ret_on_equity: float
    r_multiple: float
    bars: int
    reason: str


def run(df: pd.DataFrame, sig: pd.DataFrame, costs: Costs = Costs(), risk: RiskCfg = RiskCfg(),
        init_equity: float = 10_000.0, bar_hours: float = 1.0, allow_pyramid: bool = False):
    """df: OHLCV. sig: columns [sig, sl_dist, tp_dist] and optionally
    [exit, trail_atr, max_bars, tp2_dist, tp2_frac, be_at_r]. Index must match df."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)

    s = sig.reindex(idx)
    S = s["sig"].fillna(0).to_numpy(float)
    SL = s["sl_dist"].to_numpy(float)
    TP = s["tp_dist"].to_numpy(float) if "tp_dist" in s else np.full(n, np.nan)
    EX = s["exit"].fillna(0).to_numpy(float) if "exit" in s else np.zeros(n)
    TRAIL = s["trail"].to_numpy(float) if "trail" in s else np.full(n, np.nan)
    MAXB = s["max_bars"].to_numpy(float) if "max_bars" in s else np.full(n, np.nan)
    BE_R = s["be_at_r"].to_numpy(float) if "be_at_r" in s else np.full(n, np.nan)

    equity = init_equity
    peak = equity
    eq_curve = np.empty(n); eq_curve[:] = np.nan
    trades = []
    pos = 0           # 0 flat, +1 long, -1 short
    qty = 0.0; entry_px = 0.0; stop = np.nan; targ = np.nan
    entry_i = 0; entry_stop_dist = np.nan; be_done = False
    trail_dist = np.nan; max_bars = np.nan; extreme = np.nan
    fees_paid = 0.0; funding_paid = 0.0
    fee = costs.fee_bps * BPS
    slip = costs.slip_bps * BPS
    fund_per_bar = costs.funding_bps_8h * BPS * (bar_hours / 8.0)

    def close_pos(i, px, reason, ts):
        nonlocal equity, pos, qty, fees_paid
        px_eff = px * (1 - slip) if pos > 0 else px * (1 + slip)
        gross = pos * (px_eff - entry_px) * qty
        f = fee * px_eff * qty
        fees_paid += f
        pnl = gross - f
        equity += pnl
        r = pnl / (entry_stop_dist * qty) if entry_stop_dist and qty else 0.0
        trades.append(Trade(pos, ts_entry, ts, entry_px, px_eff, qty, pnl,
                            pnl / eq_at_entry, r, i - entry_i, reason))
        pos = 0; qty = 0.0

    ts_entry = idx[0]; eq_at_entry = equity

    for i in range(n - 1):
        # ---------- manage an open position on bar i ----------
        if pos != 0:
            funding = fund_per_bar * qty * c[i]
            equity -= funding; funding_paid += funding
            hit_stop = (l[i] <= stop) if pos > 0 else (h[i] >= stop)
            hit_targ = (not np.isnan(targ)) and ((h[i] >= targ) if pos > 0 else (l[i] <= targ))
            if hit_stop:
                close_pos(i, stop, "stop", idx[i])
            elif hit_targ:
                close_pos(i, targ, "target", idx[i])
            else:
                # move stop to breakeven once price has run be_at_r * initial risk
                if not be_done and not np.isnan(be_r) and be_r > 0:
                    run_px = (h[i] - entry_px) if pos > 0 else (entry_px - l[i])
                    if run_px >= be_r * entry_stop_dist:
                        stop = entry_px if pos > 0 else entry_px
                        be_done = True
                # trailing stop on closed-bar extremes
                if not np.isnan(trail_dist):
                    extreme = max(extreme, h[i]) if pos > 0 else min(extreme, l[i])
                    ns = extreme - trail_dist if pos > 0 else extreme + trail_dist
                    stop = max(stop, ns) if pos > 0 else min(stop, ns)
                # signal-based / time-based exit -> next bar open
                want_exit = (EX[i] == 1) or (S[i] == -pos and S[i] != 0)
                if not np.isnan(max_bars) and (i - entry_i) >= max_bars:
                    want_exit = True
                if want_exit:
                    close_pos(i + 1, o[i + 1], "signal", idx[i + 1])

        # ---------- open a new position at bar i+1 open ----------
        if pos == 0 and S[i] != 0 and not np.isnan(SL[i]) and SL[i] > 0:
            side = int(np.sign(S[i]))
            px = o[i + 1] * (1 + slip) if side > 0 else o[i + 1] * (1 - slip)
            stop_dist = SL[i]
            if stop_dist <= 0 or not np.isfinite(stop_dist):
                pass
            else:
                base = equity if risk.compound else init_equity
                q = (base * risk.risk_frac) / stop_dist
                q = min(q, base * risk.max_leverage / px)
                if q > 0:
                    f = fee * px * q
                    equity -= f; fees_paid += f
                    pos = side; qty = q; entry_px = px
                    entry_stop_dist = stop_dist
                    stop = px - side * stop_dist
                    targ = px + side * TP[i] if np.isfinite(TP[i]) else np.nan
                    trail_dist = TRAIL[i] if np.isfinite(TRAIL[i]) else np.nan
                    max_bars = MAXB[i] if np.isfinite(MAXB[i]) else np.nan
                    be_r = BE_R[i] if np.isfinite(BE_R[i]) else np.nan
                    be_done = False
                    extreme = px
                    entry_i = i + 1; ts_entry = idx[i + 1]; eq_at_entry = equity

        # ---------- mark to market ----------
        mtm = equity + (pos * (c[i] - entry_px) * qty if pos != 0 else 0.0)
        eq_curve[i] = mtm
        peak = max(peak, mtm)
        if mtm <= 0:
            eq_curve[i:] = 0.0
            return _stats(df, eq_curve, trades, init_equity, bar_hours, fees_paid, funding_paid, ruined=True)

    if pos != 0:
        close_pos(n - 1, c[-1], "eod", idx[-1])
    eq_curve[-1] = equity
    return _stats(df, eq_curve, trades, init_equity, bar_hours, fees_paid, funding_paid)


def _stats(df, eq, trades, init_equity, bar_hours, fees_paid, funding_paid, ruined=False):
    eq = pd.Series(eq, index=df.index).ffill().fillna(init_equity)
    years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 24 * 3600)
    total_ret = eq.iloc[-1] / init_equity - 1
    cagr = (eq.iloc[-1] / init_equity) ** (1 / years) - 1 if eq.iloc[-1] > 0 and years > 0 else -1.0
    dd = eq / eq.cummax() - 1
    maxdd = -dd.min()
    pnl = np.array([t.pnl for t in trades])
    wins = pnl[pnl > 0].sum(); losses = -pnl[pnl < 0].sum()
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else 0.0)
    rets = eq.pct_change().fillna(0)
    bars_per_year = 365.25 * 24 / bar_hours
    sharpe = rets.mean() / rets.std() * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(bars_per_year) if downside and downside > 0 else 0.0
    exposure = np.mean([t.bars for t in trades]) * len(trades) / len(df) if trades else 0.0
    return {
        "trades": len(trades),
        "net_profit_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "max_dd_pct": maxdd * 100,
        "profit_factor": pf,
        "win_rate_pct": (pnl > 0).mean() * 100 if len(pnl) else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": (cagr / maxdd) if maxdd > 0 else np.nan,
        "avg_r": float(np.mean([t.r_multiple for t in trades])) if trades else 0.0,
        "expectancy_pct_eq": float(np.mean([t.ret_on_equity for t in trades]) * 100) if trades else 0.0,
        "avg_bars": float(np.mean([t.bars for t in trades])) if trades else 0.0,
        "exposure": exposure,
        "years": years,
        "fees_pct_of_init": fees_paid / init_equity * 100,
        "funding_pct_of_init": funding_paid / init_equity * 100,
        "ruined": ruined,
        "final_equity": float(eq.iloc[-1]),
        "_equity": eq,
        "_trades": trades,
    }


def summary(name, r, extra=""):
    return (f"{name:36s} n={r['trades']:>5d} net={r['net_profit_pct']:>10.1f}% "
            f"CAGR={r['cagr_pct']:>8.1f}% DD={r['max_dd_pct']:>6.1f}% PF={r['profit_factor']:>5.2f} "
            f"WR={r['win_rate_pct']:>5.1f}% Sharpe={r['sharpe']:>5.2f} Calmar={r['calmar'] if r['calmar']==r['calmar'] else 0:>6.2f} {extra}")
