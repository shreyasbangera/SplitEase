# XAUUSDT Strategy Research — Findings Report

**Date:** 2026-09-09  ·  **Analyst:** automated quantitative research loop

---

## Bottom line

**No strategy qualified.** Across 9 strategy families and ~120 parameter configurations, nothing
came close to the required bar (>500% net yearly profit, ≥100 trades, PF >1.10, max DD <20%).

The best strategy discovered achieved **22.4% CAGR with 52.4% max drawdown** (PF 1.20) over the
full 9.3-year sample — versus **14.3% CAGR / 27.7% DD** for simply buying and holding gold.

This is not a reporting of failure to find the right parameters. The study identified a specific,
quantified reason the target is unreachable, set out in [Why the target is unreachable](#why-the-target-is-unreachable).

---

## 1. Market, data and provenance

| Item | Value |
|---|---|
| **Instrument** | XAUUSD spot gold (`XAUUSDm`), used as proxy for XAUUSDT — see note below |
| **Source feed** | Exness MT5 broker feed, via public dataset `github.com/sherwynjoel/xauusd-historical-data` |
| **Resolution** | 1-minute OHLCV + per-minute broker spread |
| **Period** | 2017-04-28 → 2026-08-18 (**3,276,127 M1 bars**, ~9.3 years) |
| **In-sample (IS)** | 2017-04-28 → 2022-12-31 (~5.7 yrs) |
| **Out-of-sample (OOS)** | 2023-01-01 → 2026-08-18 (~3.6 yrs) — held out, untouched during design |

**Note on XAUUSDT vs XAUUSD.** Direct exchange APIs (Binance, Bybit, MEXC, OKX, Gate, KuCoin,
Crypto.com, Yahoo, Stooq, Dukascopy) are **all blocked by this session's egress policy** (HTTP 403
on CONNECT). Bulk XAUUSDT perpetual data could not be obtained. XAUUSD spot gold is the underlying
that XAUUSDT perpetuals track (typically within a few bps plus funding), so it is the correct
research proxy — but results are stated for spot gold, and the two crypto-perp fee regimes are
modelled explicitly in §3. This substitution is a genuine limitation, not a silent one.

### Data validation (all checks passed)

- **0** invalid OHLC rows (high<low, high<open/close, low>open/close), **0** NaN, **0** non-positive prices
- Coverage stable at ~352,000 bars/year, consistent with gold's 24/5 schedule
- Largest 1-minute move 2.32% (a genuine event, not a glitch); only 1 move >2% in 3.28M bars
- Broker spread: median **$0.20**, p90 $0.30, p99 $0.65 — realistic for retail gold
- **Cross-check vs an independent same-broker M5 download** (`Anilk2978/xauusd-5yr-data`, 333,038
  overlapping bars): mean absolute difference **$0.015** on a ~$2,000 price. Confirms both the data
  and my M1→M5 resampling.
- **Cross-check vs a third-party feed** (`getdata-finance`, 40,556 overlapping M1 bars): price
  correlation **0.9991**, 1-minute **return** correlation **0.985**. (A constant level offset exists
  between feeds — different venues — but the dynamics are genuine.)

---

## 2. Backtest methodology

Engine: `code/engine.py`. Event-driven, one position at a time, fully compounding.

**Look-ahead controls**
- Every signal is computed on **completed** signal-timeframe bars only (all indicators use `.shift()`
  where they reference the current bar).
- Execution occurs **no earlier than the next bar's open**.
- Stop/target resolution is walked at **1-minute resolution** inside the signal bar, which removes the
  usual "did the stop or target hit first?" ambiguity rather than assuming an answer.
- Where both stop and target are touchable within the same M1 bar, **the stop wins** (conservative).
- Price gaps through a stop are filled at the **gapped open**, not at the stop price.
- Trailing stops / breakeven use the running extreme **through the previous bar only**.

**Null test (the critical validation).** Random entries with the same stop/target geometry:

| Regime | avg R per trade | Profit factor |
|---|---|---|
| With realistic costs | **−0.41 to −0.52** | 0.37 – 0.48 |
| At **zero** cost | **−0.06 to −0.11** | 0.83 – 0.91 |

Random entries lose approximately the transaction cost, and at zero cost they are slightly
**negative** (the conservative stop-wins tie-break), never positive. No look-ahead profit is
leaking into the engine. This −0.06R zero-cost baseline is used as the reference when judging
whether a signal has genuine edge.

**Position sizing / risk management**
- Risk-based sizing: units = (equity × risk%) / stop distance, so every trade risks a fixed fraction of equity
- Hard leverage cap (notional/equity), default 10–20×
- Compounding; run terminated at 98% capital loss
- Max drawdown measured on a mark-to-market equity curve that includes each trade's **intra-trade worst excursion** (measured at M1 resolution), not just closed-trade equity — this is stricter than the common practice

---

## 3. Cost assumptions

Costs are charged per trade as: **actual per-minute broker spread** (from the data) + commission
(bps of notional, both sides) + slippage, with **additional** slippage on stop-loss exits.

| Regime | Commission/side | Slippage/side | Stop extra | Round-trip @ $2,000 gold | Represents |
|---|---|---|---|---|---|
| **CFD** | 0.04 bps ($3.5/lot) | $0.03 | $0.10 | **≈ $0.26** | Exness-style gold CFD (matches the data source) |
| **PERP-2** | 2.0 bps | $0.05 | $0.15 | **≈ $1.10** | MEXC-style XAUUSDT perp |
| **PERP-5.5** | 5.5 bps | $0.05 | $0.15 | **≈ $2.50** | Bybit-style XAUUSDT perp |

Funding rates on perpetuals are **not** modelled (data unavailable); this makes the perp figures
*optimistic*, not conservative.

---

## 4. Strategies tested

All results are drift-adjusted where relevant (a long-only signal is measured against gold's
unconditional drift, so a bull market cannot masquerade as edge).

| # | Family | Concept / research basis | Result |
|---|---|---|---|
| A | Intraday session drift | Gold Asian-session drift; my IS finding of hours 22–23 UTC positive on *every* weekday (t = 2.0–3.1) | **Fail.** Edge ≈ 2–3 bps vs ≈ 4 bps round-trip cost. IS CAGR −10% to −38% |
| B | Asian-range opening breakout | Zarattini/Aziz ORB, adapted to the 00–07 UTC range | **Fail.** IS PF 0.24–0.46, *worse than random* |
| C | Fade the Asian-range break | Inverse of B, motivated by negative autocorrelation | **Fail.** PF 0.12–0.43 |
| D | Volatility-conditioned mean reversion | Fade z-score extremes; ac1 most negative in top ATR quintile | **Fail.** PF 0.26–0.59 |
| E | Prior-day liquidity sweep | Sweep PDH/PDL then close back inside → fade | **Fail.** PF 0.29–0.67 |
| F | **Multi-horizon TSMOM ensemble** | Moskowitz/Ooi/Pedersen time-series momentum (1/3/6/12-month), vol-targeted | **Best found.** Full-sample CAGR 22.4%, DD 52.4%, PF 1.20 — still fails |
| G | **1h Donchian-20 breakout** | Best cost-surviving intraday signal from the scan | Full-sample CAGR 2.1%, DD 22.4%, PF 1.03, 1,700 trades — fails |
| H | Calendar / event effects | NFP (1st Friday), month-end, month-start, seasonality | **Fail.** Only month-end survives (+23.8 bps over 5 days, t=4.83) — ~12 trades/yr, far too small |
| I | 1-minute mean reversion | High-frequency overreaction | **Fail on cost** — see below. The clearest result in the study |

### The central finding: a real edge that sits below the cost floor

A broad predictive scan (`code/scan.py`, `scan_tf.py`) measured drift-adjusted edge in **USD per ounce**
for ~50 signal variants across 15m / 1h / 4h / 1D. The result is consistent and decisive:

| Signal | Gross edge | t-stat | n | Min. cost | Tradeable? |
|---|---|---|---|---|---|
| 1m fade, 5m z>3, hold 15m | **$0.113/oz** | **+5.50** | 18,197 | $0.26 | ✗ |
| 1m fade, 5m z>2, hold 5m | $0.065/oz | **+10.37** | 53,066 | $0.26 | ✗ |
| 15m Donchian-32 fade, 4 bars | $0.11/oz | +3.41 | 10,797 | $0.26 | ✗ |
| 1h Donchian-20 breakout, 72 bars | $0.78/oz | +2.40 | 3,378 | $0.26 | marginal |
| 1D vol-scaled 50d momentum, 20d hold | **$5.37/oz** | +3.60 | 1,000 | $0.26 | ✓ but ~15 trades/yr |

Gold's intraday inefficiency is **real and highly significant** — the 1-minute reversion effect has a
t-statistic of **+10.4** over 53,066 observations, which is not noise. But it is worth only
**$0.065–$0.113 per ounce**, while the cheapest realistic round trip is **$0.26**. The edge is
2.5–4× too small to survive execution, and on a crypto perpetual (round trip $1.10–$2.50) it is
10–20× too small.

Because cost is essentially **fixed per trade**, the only way to clear it is to hold for larger
moves — which drives you to the daily timeframe, where trade count collapses to ~15/year and
compounding capacity collapses with it. This is the core bind, and it is structural, not a
parameter-tuning problem.

---

## 5. Why the target is unreachable

The qualifying bar (>500% CAGR **and** <20% max drawdown) implies a Calmar ratio > 25. Working
backwards to the Sharpe ratio it requires:

- 500% CAGR ⇒ continuous drift μ ≈ ln(6) ≈ **1.79**
- Under the *optimistic* approximation max DD ≈ 1.0σ: σ < 0.20 ⇒ required Sharpe **> 9.0**
- Under the realistic approximation max DD ≈ 2.5σ: σ < 0.08 ⇒ required Sharpe **> 22**

Measured Sharpe ratios in this study, after costs:

| Strategy | Sharpe |
|---|---|
| Daily TSMOM (best single) | 0.34 |
| TSMOM ensemble, IS | 0.40 |
| TSMOM ensemble, OOS | 1.26 *(but buy-and-hold OOS Sharpe was 1.18 — this is mostly bull-market beta, not alpha)* |
| Buy & hold, full sample | 0.80 |

A Sharpe of 9–22 is roughly an order of magnitude beyond what the best-documented systematic
strategies achieve in liquid markets. **Leverage cannot bridge the gap**: the frontier sweep in
`code/tsmom.py` shows drawdown growing faster than return past optimal leverage, exactly as
volatility drag predicts —

| Target vol | CAGR | Max DD |
|---|---|---|
| 15% | 5.1% | 34.5% |
| 30% | 7.3% | 59.0% |
| 60% | 2.1% | 86.0% |
| 120% | −36.5% | 99.1% |

Raising leverage to chase 500% destroys the account long before it reaches the return target.

---

## 6. Ranked results (best strategies found — **none qualify**)

### #1 — Family F: Multi-horizon TSMOM ensemble *(best risk-adjusted)*

| Period | CAGR | Max DD | Sharpe | Calmar | PF (daily) |
|---|---|---|---|---|---|
| IS | 8.3% | 52.4% | 0.40 | 0.16 | 1.10 |
| **OOS** | **48.2%** | **24.9%** | **1.26** | 1.93 | 1.31 |
| **Full** | **22.4%** | **52.4%** | 0.78 | 0.43 | 1.20 |

Fails: net profit (22% vs 500%), max DD (52% vs 20%). The strong OOS is largely gold's 2023–2026
bull market — buy-and-hold returned 27.5% CAGR at Sharpe 1.18 over the same window, so the honest
attribution is beta, not alpha.

**Full rules (reproducible).** Daily bars (UTC). Signal = mean of sign(close − close[−k]) for
k ∈ {21, 63, 126, 252}, computed at day *t* close. Realized vol = 60-day stdev of daily returns × √252.
Position = signal × (0.30 / realized vol), capped at 10× leverage, **held from day t+1**. Rebalanced
daily; costs charged on turnover at the CFD regime (0.04 bps/side + $0.06/oz). No stop loss (vol
targeting is the risk control). 482 direction changes over the full sample.

### #2 — Family G: 1h Donchian-20 breakout *(most trades, closest to PF bar)*

| Period | Trades | CAGR | Max DD | PF | Win % |
|---|---|---|---|---|---|
| IS | 1,035 | −1.1% | 21.5% | 0.98 | 28.2% |
| OOS | 665 | 8.0% | 11.8% | 1.12 | 30.1% |
| **Full** | **1,700** | **2.1%** | **22.4%** | **1.03** | 28.9% |

Fails: net profit, PF (1.03 vs 1.10), max DD. Meets only the trade-count criterion. **Under
XAUUSDT perpetual fees (2 bps/side) it turns outright negative: CAGR −6.7%, DD 56.9%, PF 0.89** —
which is the single clearest illustration of the cost problem.

**Full rules (reproducible).** 1-hour bars (UTC). Long when close > max(high, 20 bars, excluding
current); short when close < min(low, 20 bars, excluding current). Enter at next bar's open.
Stop = 2 × ATR(32); target = 6 × ATR(32); time exit after 72 bars. Risk 0.5% of equity per trade,
leverage cap 20×. One position at a time.

### Benchmark — buy & hold gold

| Period | CAGR | Max DD | Sharpe |
|---|---|---|---|
| IS | 6.6% | 21.4% | 0.48 |
| OOS | 27.5% | 27.7% | 1.18 |
| Full | 14.3% | 27.7% | 0.80 |

Neither strategy beats buy-and-hold on a risk-adjusted basis over the full sample.

---

## 7. Criteria scorecard

| Criterion | Required | Best achieved | Pass? |
|---|---|---|---|
| Net yearly profit after costs | > 500% | 22.4% (Family F, full sample) | ✗ |
| Completed trades | ≥ 100 | 1,700 (Family G) | ✓ |
| Profit factor | > 1.10 | 1.20 (Family F, daily) / 1.03 (Family G) | partial |
| Max drawdown | < 20% | 22.4% (Family G, full) | ✗ |
| Realistic risk management | yes | risk-based sizing, leverage caps, vol targeting | ✓ |
| No look-ahead bias | yes | verified by null test | ✓ |

**No single strategy satisfies all criteria simultaneously.** No configuration came within an order
of magnitude of the profit target while respecting the drawdown limit.

---

## 8. Honest limitations

- XAUUSD spot is a **proxy** for XAUUSDT; exchange data was unreachable (egress policy). Basis and
  funding are unmodelled.
- Perpetual **funding rates are excluded**, making the perp results optimistic.
- Both primary datasets derive from the same broker (Exness); the third-party cross-check covers
  only ~6 weeks.
- Signals were explored on IS and confirmed on OOS, but the wide scan (~50 variants × 4 timeframes)
  carries multiple-comparison risk — the surviving daily-momentum effect is the one with independent
  literature support, which is why it is ranked first.
- Results reflect one asset over one 9.3-year window.

## 9. Reproducing this study

```bash
pip install pandas numpy pyarrow
git clone https://github.com/sherwynjoel/xauusd-historical-data   # 90MB M1 parquet
python code/prep.py         # validate + clean  -> m1_clean.parquet
python code/xcheck.py       # cross-source validation
python code/test_null.py    # look-ahead / null test
python code/scan.py         # predictive scan (15m)
python code/scan_tf.py      # predictive scan (1h/4h/1D)
python code/tsmom.py        # leverage frontier
python code/consolidate.py  # final ranked results
```
Adjust the hard-coded scratchpad paths at the top of `engine.py` / `lab.py` to your data location.
