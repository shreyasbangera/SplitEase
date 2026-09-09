"""Final validation battery for the surviving candidates."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/user/SplitEase/research/strategies")
import pandas as pd, numpy as np
from runner import data, bar_hours, portfolio
from backtest import run, Costs, RiskCfg, summary
import s11_momo_pulse as s11, s07_trend_ensemble as s7, s12_bear_short as s12
import s10_regime_long as s10

TAKER = Costs(5, 2, 1)
MODULES = {
    "M1 BTC momentum-pulse H4": (s11.signal, "btc_h4", dict(ma_n=200, hold_bars=48, sl_atr=3.0, atr_n=14)),
    "M2 BTC trend-ensemble D1": (s7.signal, "btc_d1", dict(looks=(20, 60, 120), k=3, sl_atr=3.0, trail_atr=4.0, atr_n=14)),
    "M3 BTC bear-short H4":     (s12.signal, "btc_h4", dict(fast=120, slow=1500, trig_n=60, hold_bars=240, sl_atr=4.0, atr_n=14)),
    "M4 BTC regime-long D1":    (s10.signal, "btc_d1", dict(ma_n=200, vol_cap_pct=0.85, sl_atr=8.0, atr_n=20)),
    "G1 GOLD trend-ensemble D1": (s7.signal, "gold_d1", dict(looks=(20, 60, 120), k=3, sl_atr=3.0, trail_atr=4.0, long_only=True, atr_n=14)),
}


def one(f, ds, p, costs=TAKER, risk_frac=0.02, sl=slice(None), max_lev=10):
    df = data(ds).loc[sl]
    sig = f(data(ds), **p).loc[df.index]
    return run(df, sig, costs=costs, risk=RiskCfg(risk_frac=risk_frac, max_leverage=max_lev), bar_hours=bar_hours(ds))


def pf_from_returns(r):
    g = r[r > 0].sum(); l = -r[r < 0].sum()
    return g / l if l > 0 else np.inf


def main():
    out = {}
    print("=" * 118)
    print("MODULE DETAIL - full period / in-sample / out-of-sample (BTC IS<=2021, GOLD IS<=2016), risk 2% per trade")
    print("=" * 118)
    for name, (f, ds, p) in MODULES.items():
        cut = "2021-12-31" if ds.startswith("btc") else "2016-12-31"
        for lbl, s in [("full", slice(None)), ("IS", slice(None, cut)), ("OOS", slice(cut, None))]:
            r = one(f, ds, p, sl=s)
            print(summary(f"{name} [{lbl}]", r))
            out[f"{name}|{lbl}"] = {k: v for k, v in r.items() if not k.startswith("_")}
        print()

    print("=" * 118)
    print("COST SENSITIVITY (fee+slippage per side, round trip in brackets) - full period, risk 2%")
    print("=" * 118)
    for name, (f, ds, p) in MODULES.items():
        line = f"{name:28s}"
        for c in [Costs(2, 0.5, 1), Costs(5, 2, 1), Costs(10, 5, 2), Costs(15, 10, 3)]:
            r = one(f, ds, p, costs=c)
            rt = (c.fee_bps + c.slip_bps) * 2
            line += f"  RT{rt:>4.0f}bp: CAGR {r['cagr_pct']:>6.1f}% PF {r['profit_factor']:>4.2f} |"
        print(line)

    print()
    print("=" * 118)
    print("PORTFOLIO of the four BTC modules, equal weight, continuously rebalanced")
    print("=" * 118)
    eqs, trades = [], 0
    for name, (f, ds, p) in MODULES.items():
        if not name.startswith("M"):
            continue
        r = one(f, ds, p)
        eqs.append(r["_equity"]); trades += r["trades"]
    P = portfolio(eqs)
    eq = P["equity"]
    daily = eq.resample("1D").last().dropna().pct_change().dropna()
    print(f"  base (each module at 2% risk): CAGR {P['cagr_pct']:.1f}%  maxDD {P['max_dd_pct']:.1f}%  "
          f"Sharpe {P['sharpe']:.2f}  trades {trades}  PF(daily) {pf_from_returns(daily):.2f}")
    for cut, lbl in [("2021-12-31", "IS 2014-2021"), ("2021-12-31", "OOS 2022-2026")]:
        e = eq.loc[:cut] if lbl.startswith("IS") else eq.loc[cut:]
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        d = -(e / e.cummax() - 1).min() * 100
        rr = e.pct_change().fillna(0)
        print(f"    {lbl}: CAGR {100*((e.iloc[-1]/e.iloc[0])**(1/yrs)-1):>6.1f}%  maxDD {d:>5.1f}%  "
              f"Sharpe {rr.mean()/rr.std()*np.sqrt(365.25*6):.2f}")

    print()
    print("  Leverage scaling of the portfolio (what it takes to chase a 500%/yr target):")
    base_r = eq.pct_change().fillna(0)
    for k in [1, 2, 3, 4, 6, 8, 12, 20, 40]:
        r = base_r * k
        e = (1 + r).cumprod()
        if (1 + r).min() <= 0:
            print(f"    x{k:<3d} ACCOUNT WIPED OUT (a single bar loses 100% of equity)")
            continue
        yrs = P["years"]
        d = -(e / e.cummax() - 1).min() * 100
        print(f"    x{k:<3d} CAGR {100*(e.iloc[-1]**(1/yrs)-1):>10.1f}%   maxDD {d:>6.1f}%   "
              f"{'QUALIFIES' if (100*(e.iloc[-1]**(1/yrs)-1) > 500 and d < 20) else ''}")
    with open("/home/user/SplitEase/research/results/final_modules.json", "w") as fh:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()}, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
