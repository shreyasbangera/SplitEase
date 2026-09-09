# XAUUSD / XAUUSDT Strategy Research — Final Report (two research rounds)

**Instrument:** XAUUSD spot gold (Exness MT5 feed) — proxy for XAUUSDT
**Data:** 3,276,127 1-minute bars, 2017-04-28 → 2026-08-18 (9.3 years), incl. per-minute broker spread
**In-sample:** 2017-04-28 → 2022-12-31   ·   **Out-of-sample:** 2023-01-01 → 2026-08-18 (held out)

---

## Bottom line

**No strategy met the qualifying criteria**, across two rounds covering 12 strategy families, a
systematic feature scan, a multi-factor model, and machine learning — roughly 300 configurations.

The best strategy that survives out-of-sample delivers **10.3% CAGR at 34.4% max drawdown**
(full sample). Buy-and-hold gold returned 14.3% CAGR at 27.7% DD over the same period.

Round 2 established something stronger than "I didn't find one": it measured the **ceiling**, and the
ceiling is far below the target. Details in §4.

---

## 1. Data validation

- 0 invalid OHLC rows, 0 NaN, 0 non-positive prices; ~352k bars/year (consistent with 24/5 gold)
- **Cross-check vs independent same-broker M5 download:** mean abs diff **$0.015** (333,038 bars)
- **Cross-check vs third-party feed:** price corr **0.9991**, 1-min return corr **0.985**
- Broker spread: median $0.20, p90 $0.30, p99 $0.65

*Note:* every crypto-exchange API (Binance, Bybit incl. `bytick` mirror, MEXC, OKX, Gate, KuCoin)
is blocked by this session's egress policy (403 on CONNECT), so true XAUUSDT perpetual data was
unreachable. XAUUSD spot is what XAUUSDT tracks; perpetual fee regimes are modelled explicitly.
Perpetual **funding is not modelled**, making the perp figures optimistic rather than conservative.

## 2. Anti-look-ahead controls

Signals on completed bars only; execution no earlier than next bar's open; stop/target resolved at
**1-minute resolution**; stop wins ties; gaps fill at the gapped open; trailing stops use the
previous bar's extreme; ML training set purged of the target horizon before the test split.

**Null test:** random entries lose ≈ the transaction cost (avg −0.41 to −0.52 R); at **zero** cost
they are **slightly negative** (−0.06 R), never positive. No look-ahead profit leaks into the engine.

## 3. What was tested

**Round 1 — hypothesis-driven (9 families).** Session drift; Asian-range ORB (Zarattini);
ORB fade; volatility-conditioned mean reversion; prior-day liquidity sweeps; multi-horizon TSMOM;
1h Donchian breakout; calendar/event effects (NFP, month-end, seasonality); 1-minute mean reversion.

**Round 2 — systematic.** 21-feature conditional-edge scan with an IS→OOS stability filter;
multi-factor composite (discrete and continuous, with turnover analysis); zero-cost ceiling
analysis; 108-configuration filtered HF search; gradient-boosted trees.

## 4. Why the target is unreachable — three independent measurements

### (a) The edge/cost floor
Systematic scan of per-trade edge in USD/oz against the round-trip cost floor:

| Signal | Gross edge | t-stat | n | Cost floor | Verdict |
|---|---|---|---|---|---|
| 1m fade, 5m z>3, hold 15m | $0.113/oz | **+5.5** | 18,197 | $0.26 | below cost |
| 1m fade, 5m z>2, hold 5m | $0.065/oz | **+10.4** | 53,066 | $0.26 | below cost |
| 1D vol-scaled 50d momentum | $5.37/oz | +3.6 | 1,000 | $0.26 | clears — but ~15 trades/yr |

Gold's intraday inefficiency is **real and highly significant** (t = +10.4 over 53,066 observations)
but worth only $0.065–$0.113/oz against a $0.26 minimum round trip. A 108-configuration search over
z-thresholds, holding periods, spread filters (down to the tightest 25%) and liquid-hours filters
found **0 configurations net-positive in both IS and OOS**. Best case: gross $0.135 vs cost $0.267.

Because cost is fixed per trade, clearing it forces daily holding periods — where trade count
collapses to ~15/year and compounding capacity collapses with it.

### (b) The zero-cost ceiling
Running the highest-Sharpe effect with **execution assumed free**:

| | Sharpe | CAGR at DD<20% |
|---|---|---|
| In-sample | 5.33 | ~340% |
| **Out-of-sample** | **1.96** | **~50%** |

Even with perfect, costless execution the out-of-sample ceiling is ~50% CAGR at the drawdown limit.
The target is unreachable *regardless of execution quality*. With real costs this strategy returns
−100%: continuous rebalancing generates **714,566× turnover**.

### (c) The leverage wall
The best surviving strategy, scaled up (full sample):

| Target vol | CAGR | Max DD |
|---|---|---|
| 30% | 10.3% | 34.4% |
| 60% | 18.9% | 61.0% |
| 120% | **25.6%** | 88.8% |
| 240% | 0.6% | 99.6% |
| 480% | −84.2% | 100.0% |

Return peaks near 26% CAGR (at 89% DD) and then **collapses** — volatility drag. Leverage cannot
bridge the gap; past a point it destroys the account while *reducing* return.

### The arithmetic
500% CAGR at <20% DD implies Calmar > 25, requiring **Sharpe ≈ 9–22**. Measured Sharpe, after costs:

| | Sharpe |
|---|---|
| Best tradeable (TREND, daily rebalance) | **0.75 IS / 0.59 OOS** |
| Daily TSMOM ensemble | 0.40 IS / 1.26 OOS |
| Buy & hold | 0.80 |
| HF mean reversion, **zero cost** | 5.33 IS / 1.96 OOS |

Nothing tradeable exceeds ~0.8. Even the free-execution ceiling (1.96 OOS) is 5–10× short.

### What overfitting looks like (control)
Gradient-boosted trees: in-sample IC **+0.58**, decile spread **$16.50/oz** — and out-of-sample
IC **−0.013**, trading at **−$1.01/oz**. Pure noise fitting, caught by the held-out split. This is
the shape a "qualifying" result would take if the OOS discipline were dropped.

## 5. Best strategy found (does NOT qualify)

**TREND factor, daily-rebalanced, vol-targeted**

| Period | CAGR | Max DD | Sharpe | PF |
|---|---|---|---|---|
| IS | 12.2% | 25.7% | 0.78 | 1.02 |
| OOS | 7.5% | 25.5% | 0.51 | 1.02 |
| **Full** | **10.3%** | **34.4%** | 0.67 | 1.02 |

**Rules (reproducible).** 15-minute bars, UTC. Six trend features — momentum over 16 and 48 bars,
deviation from EMA20 and EMA50 (both in ATR units), RSI(14), and position within the 96-bar range —
each converted to a past-only rolling percentile rank over 480 bars, mapped to [−1,+1], then averaged.
Position = score × (0.30 / realized vol), where realized vol = 5-day stdev of 15m returns annualized;
capped at 10× leverage. Rebalance every 96 bars (daily) with a 0.5 deadband. Position decided at bar
close, **held from the next bar's open**. No stop loss — vol targeting is the risk control.
Costs: actual per-minute broker spread (half-spread per side) + 0.04 bps/side commission + $0.03/side slippage.

## 6. Scorecard

| Criterion | Required | Best achieved | Pass |
|---|---|---|---|
| Net yearly profit | > 500% | 10.3% (26% at 89% DD) | ✗ |
| Completed trades | ≥ 100 | 1,700 | ✓ |
| Profit factor | > 1.10 | 1.20 | ✓ (not jointly) |
| Max drawdown | < 20% | 25.5% (OOS best) | ✗ |
| Realistic risk management | yes | vol targeting, leverage caps | ✓ |
| No look-ahead bias | yes | verified by null test | ✓ |

**No configuration satisfies all criteria simultaneously**, and none comes within an order of
magnitude of the profit target while respecting the drawdown limit.

## 7. Limitations

XAUUSD is a proxy for XAUUSDT (exchange data unreachable); perpetual funding unmodelled; both primary
datasets come from the same broker; the wide scan carries multiple-comparison risk, which is why only
IS→OOS-stable effects are reported; one asset, one 9.3-year window. A multi-asset portfolio is the one
untested lever that could raise Sharpe materially — diversification across uncorrelated markets is the
standard route to higher Calmar, and it is not available within a single instrument.
