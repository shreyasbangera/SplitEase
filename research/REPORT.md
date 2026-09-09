# XAUUSDT / BTCUSDT systematic strategy research

**Author:** automated quant research loop
**Date:** 2026-09-09
**Primary instrument:** XAUUSDT (spot gold quoted in USD-stablecoin terms)
**Secondary / confirmation instrument:** BTCUSDT

---

## 1. Headline result

**No strategy qualified.** Across 13 distinct strategy families, ~40 logged full
backtests, several hundred parameter combinations and ~200 signal-level edge
tests, nothing came close to the required combination of

> net yearly profit > 500% **and** max drawdown < 20% **and** ≥ 100 trades **and** profit factor > 1.10.

The binding constraint is not the profit factor, the trade count, or the
drawdown taken alone — it is the *ratio* between them. The target asks for a
Calmar ratio (CAGR ÷ max drawdown) of **25 or better**. The best fully-validated
system found here has a Calmar of **1.9**, and that is already a good number by
the standards of published systematic strategies.

This is a property of the target, not of the search. Return and drawdown scale
together with position size, so the ratio cannot be levered into existence. The
table below is the same portfolio at nine different leverage settings:

| leverage | CAGR | max drawdown | qualifies? |
|---|---|---|---|
| x1 | 11.6% | 7.0% | no (return) |
| x2 | 24.2% | 13.6% | no (return) |
| **x2.8** | **34.9%** | **18.7%** | no (return) |
| x3 | 37.7% | 20.0% | no (return) |
| x4 | 52.2% | 26.0% | no (both) |
| x6 | 84.3% | 37.2% | no (both) |
| x8 | 120.3% | 47.1% | no (both) |
| x12 | 203.2% | 63.4% | no (both) |
| x20 | 394.8% | 84.3% | no (both) |
| x40 | 598.2% | **99.1%** | no (drawdown) |

The only leverage that reaches 500%/yr destroys 99% of the account on the way.
Nothing was altered, omitted or invented to make any of this look better.

What *was* found, and is reported in full below, is a genuinely validated BTC
momentum system earning **32.2%/yr at 19.1% max drawdown with 332 trades and a
1.79 profit factor**, whose edge survives a permutation test at p < 0.025 — and
the clean, statistically-supported finding that **gold has no exploitable
technical edge at these cost levels**, which is itself an actionable result.

---

## 2. Market, data and trading assumptions

### 2.1 Instruments and data sources

| dataset | instrument | source | period | bars |
|---|---|---|---|---|
| `gold_h1` / `h4` / `d1` / `w1` | XAU/USD spot gold | MetaTrader broker feed via [FeziweMelvin/XAUUSD-Gold-Price](https://github.com/FeziweMelvin/XAUUSD-Gold-Price) | 2004-06-11 → 2025-06-06 | 122,028 hourly |
| `gold2_m15` / `h1` / `h4` | XAU/USD spot gold | MT5 feed via [ejtraderLabs/historical-data](https://github.com/ejtraderLabs/historical-data) | 2012-05-15 → 2022-03-04 | 230,400 15-min |
| `btc_m15` / `h1` / `h4` / `d1` / `w1` | BTC/USD | Bitstamp 1-minute via [ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data) | 2014-01-01 → 2026-09-09 | 111,099 hourly |

**Instrument caveat, stated plainly.** Direct exchange APIs (Binance, Bybit,
OKX, Kraken) are blocked by this environment's egress policy, so the actual
`XAUUSDT` and `BTCUSDT` perpetual order books could not be downloaded. The gold
series used is spot XAU/USD from a retail broker feed — the underlying that
XAUUSDT/XAUT-USDT perpetuals track, and economically the same price series to
within the perp basis. The BTC series is Bitstamp BTC/USD spot rather than the
Binance BTCUSDT perp. Both substitutions are conservative for the *signal* work
(the same price process) but they do differ from a perp in two ways that were
modelled explicitly: **perp funding is charged** in the engine, and gold's spot
feed has weekend gaps that a 24/7 XAUUSDT perp would not have. The two
independent gold feeds turned out to be the same underlying data (return
correlation 1.000 on 57,600 overlapping bars), so no cross-vendor validation of
the gold result was possible.

### 2.2 Cost and execution assumptions (baseline)

| item | assumption | rationale |
|---|---|---|
| taker fee | **0.05% per side** (10 bps round trip) | Binance/Bybit perp taker tier |
| slippage | **0.02% per side** (4 bps round trip) | one tick plus queue for a liquid perp |
| **total round trip** | **0.14% of notional** | |
| funding | **0.01% per 8h on notional** while a position is held | typical perp funding |
| execution | signal on the **close of bar `i`**, fill at the **open of bar `i+1`** | no same-bar execution |
| intrabar sequencing | if a bar touches both stop and target, the **stop** is assumed to fill first | conservative |
| stop fills | at the stop price plus slippage (no gap modelling beyond that) | mildly optimistic; noted |
| position sizing | fixed-fractional: risk a constant % of current equity per trade, sized from the stop distance; leverage capped | standard, and it makes sizing inverse to volatility |
| starting equity | 10,000 units, fully compounded | |

Cost sensitivity was run at 5, 14, 30 and 50 bps round trip for every surviving
candidate (§6.3).

### 2.3 Anti-look-ahead controls

1. Every indicator is computed on completed bars only; the engine physically
   cannot fill before `open[i+1]`.
2. **Buy-and-hold reproduction:** the engine reproduces gold's 10.9%/yr and
   BTC's 44.5%/yr buy-and-hold CAGR exactly.
3. **Zero-edge control:** random entries with symmetric stops and zero costs
   returned profit factor 0.91–1.03 and win rate 48–51% over five seeds — no
   hidden edge leaks from the harness.
4. **Execution-delay test:** delaying the best strategy's entry by 1, 2 and 3
   *extra* bars degrades it gracefully (32.2% → 31.3% → 28.1% → 27.1% CAGR).
   A look-ahead artefact collapses under this test; this does not.
5. **Permutation test:** entry timestamps shuffled 40× per strategy, sizing and
   trade count preserved, to separate genuine timing skill from mere long
   exposure to a rising asset (§6.4). This test disqualified the best *gold*
   strategy.

---

## 3. The arithmetic that governs everything: cost per unit of risk

The single most useful quantity in this study is **cost ÷ R**, where R is the
stop distance:

```
cost/R  =  round-trip cost (% of price)  ÷  stop distance (% of price)
```

A strategy is only viable when its average edge per trade, expressed in R,
exceeds cost/R. With a 14 bps round trip:

| timeframe | typical ATR | stop = 3 ATR | cost/R | measured edge |
|---|---|---|---|---|
| gold H1 | 0.25% | 0.75% | **0.19 R** | ~0.037 R |
| gold M15 | 0.10% | 0.30% | **0.47 R** | ~0.005 R |
| gold D1 | 1.0% | 3.0% | 0.047 R | ~0.02 R |
| BTC H4 | 1.6% | 4.8% | **0.029 R** | ~0.10 R |

This is why every gold intraday strategy tested lost money *even where its gross
edge was positive*, and why BTC on 4-hour bars is the only combination in the
study with comfortable headroom. It is also why the tight-stop patterns (S9
liquidity sweep, stop ≈ 0.3% of price) were the worst performers in the entire
study: they run at roughly 0.5 R of cost per trade.

---

## 4. Research log — all attempts

Each family is a genuinely different economic idea, not a re-parameterisation of
the previous one. Full-period figures, 0.05% fee + 0.02% slippage per side, 2%
risk per trade, unless noted.

| # | Strategy family | Research basis | Market/TF | Trades | CAGR | MaxDD | PF | Verdict & lesson |
|---|---|---|---|---|---|---|---|---|
| S1 | **Asian-range / session opening-range breakout** | intraday session literature: 60–70% of gold's daily range forms in the London–NY overlap | gold H1 | 4,834 | −34.1% | 100% | 0.86 | **Fail.** Gross edge is real but tiny (+0.037 R); costs are 0.19 R. Killed by cost, not by the idea. |
| S2 | **Donchian channel breakout** (Turtle) | Moskowitz/Ooi/Pedersen time-series momentum; classic CTA | gold H4/D1, BTC D1 | 65–373 | −4.5% to −1.1% (gold) | 39–67% | 1.04–1.29 | **Fail on gold.** Works weakly on BTC. Gold trend-following has decayed. |
| S3 | **Vol-conditioned mean reversion** (z-score + ADX + vol-regime filter) | reversion dominates in calm regimes | gold H4 | 352 | −3.9% | 57% | 0.74 | **Fail.** Negative even gross (PF 0.87 at zero cost). Gold does not mean-revert at 4h. |
| S4 | **Market intraday momentum** | Gao, Han, Li & Zhou (JFE 2018): first half-hour predicts last half-hour | gold H1 | 5,232 | −35.2% | 100% | 0.51 | **Fail.** Gross PF only 1.06; the published effect does not replicate on gold at this cost level. |
| S5 | **Session drift harvesting** | own measurement: +2.1 bp at 01:00 UTC (t=6.2), −1.1 bp at 16:00 UTC (t=−2.3) | gold H1 | 5,466 | −32.0% | 100% | 0.63 | **Fail.** The drift is statistically real but ~5 bp per trade against 14 bp of cost. Even at maker fees (5 bps RT) it loses. |
| S6 | **Time-series momentum** (12-1 style) | Moskowitz/Ooi/Pedersen | gold D1/M1, BTC M1 | 9–289 | −0.4% (gold) / +2.1% (BTC) | 9–27% | 1.4–25 | **Fail.** Directionally right on BTC, far too few trades and too little return. |
| S7 | **Multi-lookback trend ensemble** | consensus of 20/60/120-bar trend signs; standard CTA robustness fix | gold + BTC, H4/D1 | 107–612 | −3.9% (gold) / **+14.7%** (BTC H4) | 11–70% | 1.12–2.40 | **Partial.** Best gold variant is indistinguishable from random (§6.4). Kept as BTC module M2. |
| S8 | **Volatility-squeeze expansion breakout** | volatility clustering; compressed range → expansion | gold H4/D1, BTC H4 | 95–421 | −4.4% (gold) / +8.6% (BTC) | 33–70% | 1.06–1.25 | **Fail.** BTC version is strong in-sample (Sharpe 1.44) and negative out-of-sample (−9.0%) — a textbook overfit. |
| S9 | **Liquidity-sweep / stop-run reversal** | stops cluster beyond prior-session extremes; failed breakout microstructure | gold H1/H4, BTC H4 | 1,631–8,149 | −98.9% to −42.8% | 100% | 0.41–0.83 | **Fail, worst in study.** The tight invalidation that makes the pattern attractive puts cost at ~0.5 R. Win rate 20–25%. |
| S10 | **Regime-filtered leveraged exposure** | Faber tactical asset allocation: hold above the long MA, cash below | gold + BTC D1 | 41–111 | +0.8% (gold, DD-scaled) / **+3.7%** (BTC) | 3.7–18% | 2.05–5.29 | **Partial.** Excellent risk control on BTC (PF 5.29, DD 3.7%). Kept as module M4. Adds nothing on gold. |
| S11 | **Trend-gated momentum pulse** | built from the measured edge table, not from a chart pattern | **BTC H4** | **332** | **+32.2%** | **19.1%** | **1.79** | **Best single strategy.** See §5.1. On gold the same construction returns −7.6%. |
| S12 | **Bear-regime short module** | measured: 20d and 250d returns both negative → +296 to +513 bp excess to the short side over 10–20 days | BTC H4/D1 | 34 | +3.6% | 11.4% | 2.27 | **Partial.** Small but genuine (p<0.025) and uncorrelated with the long modules. Kept as module M3. |
| S13 | **Buy-the-dip in an uptrend** (pullback) | the natural opposite of S11: buy short-term oversold within a long uptrend | gold + BTC, H1–D1 | 102–1,087 | −8.2% to +1.3% | 9–81% | 0.92–1.20 | **Fail.** Confirms the scan: crypto pays for continuation, not reversion. |

### 4.1 Signal-level edge scans (~200 tests)

Before and between the backtests, candidate signals were screened by measuring
the **excess** forward return over the unconditional drift, with a `t/√h`
haircut for overlapping windows. Two corrections mattered enormously:

* **Drift contamination.** In the first scan, "hold gold on Fridays for 168h"
  showed +33 bp with t = 18. Gold's *unconditional* 168h return over the sample
  is also +33 bp. The entire apparent effect was buy-and-hold drift. Every
  reported number in this study is excess-over-drift.
* **Overlap.** Sampling h-bar forward windows every bar inflates t-statistics by
  roughly √h.

Results after both corrections:

| market / timeframe | best |t_adj| found | best net-of-cost excess |
|---|---|---|---|
| gold H1 (2004–2025, 22 signals × 6 horizons) | 3.8 (Friday, +2.4 bp / 4h) | **all negative after 14 bps** |
| gold M15 (2012–2022) | 2.3 (4-bar reversal, +0.21 bp) | all negative even at 0 bps |
| gold D1 (35 signals × 5 horizons) | 2.9 (Monday weakness, −8.9 bp) | all negative after cost |
| gold W1 (20 signals × 5 horizons) | **0.9 — nothing at all** | — |
| BTC H4 | 4.1; several at 2.5–3.1 | **+20 to +60 bp net** (Donchian-50 break, RSI(2) thrust above MA200, dual-momentum) |
| BTC D1 | 2.7 | +45 bp (Donchian-20 break), +500 bp (bear-regime short, 20d) |

With ~200 tests, roughly 10 results with |t| > 2 are expected by chance alone, so
the isolated gold hits are consistent with noise. The BTC H4 momentum cluster is
not: it is many related signals all pointing the same way, in the direction
predicted by the momentum literature, and it survives out-of-sample.

---

## 5. Ranked best strategies

Ranked by robustness first (out-of-sample survival and permutation
significance), then profit factor, drawdown, trade count and net profit.
**None of these qualifies under the stated bar.**

### 🥇 #1 — BTC 4-module portfolio (best overall)

Four modules on one instrument, equal weight, continuously rebalanced. Their
bar-return correlations are 0.00–0.09 (except M2/M4 at 0.60), so the portfolio's
Sharpe (1.96) is well above any single module's, and that is what buys return
per unit of drawdown.

| metric | full 2014–2026 | IS 2014–2021 | **OOS 2022–2026** |
|---|---|---|---|
| CAGR (at x2.8 sizing) | **34.9%** | 47.0% | **16.5%** |
| max drawdown | **18.7%** | 14.7% | 18.7% |
| Sharpe | 1.96 | 2.43 | 1.10 |
| profit factor (daily P&L) | 1.38 | — | — |
| trades | **537** | — | — |
| Calmar | **1.87** | 3.20 | 0.88 |

Pass/fail against the bar: trades ✅ (537 ≥ 100) · PF ✅ (1.38 > 1.10) ·
drawdown ✅ (18.7% < 20%) · **return ❌ (34.9% vs 500%)**.

### 🥈 #2 — M1, BTC trend-gated momentum pulse (best single strategy)

| metric | full 2014–2026 | IS 2014–2021 | **OOS 2022–2026** |
|---|---|---|---|
| CAGR | **32.2%** | 42.7% | **16.0%** |
| max drawdown | **19.1%** | 14.9% | 18.7% |
| profit factor | **1.79** | 2.28 | 1.60 |
| trades | **332** | 207 | 125 |
| win rate | 46.1% | 51.2% | 37.6% |
| Sharpe / Calmar | 1.64 / 1.68 | 2.02 / 2.86 | 0.95 / 0.85 |

Permutation test: real Sharpe 1.64 vs shuffled mean 0.92 (p < 0.025 on 40
shuffles) — the timing is doing real work, not just holding a rising asset.
Walk-forward (3y train → 1y test, re-optimised each year, 2017–2026): +19.9%/yr,
38.4% drawdown, Sharpe 1.01, 363 out-of-sample trades. Note the walk-forward
drawdown is *worse* than the fixed-parameter version — annual re-optimisation
chases regimes and hurts. The fixed parameters below are the recommendation.

Pass/fail: trades ✅ · PF ✅ · drawdown ✅ (19.1%) · **return ❌**.

### 🥉 #3 — M4, BTC regime-filtered long

CAGR 3.7%, max drawdown **3.7%**, PF **5.29**, 64 trades, Sharpe 1.14, Calmar
0.99; OOS PF 4.83. The best *risk* profile in the study and the cleanest
component, but only 64 trades — fails the trade-count criterion on its own.

### #4 — M3, BTC bear-regime short

CAGR 3.6%, drawdown 11.4%, PF 2.27, 34 trades. Fails trade count. Included
because it is the only module that makes money in bear markets, and its
permutation p-value is < 0.025.

### #5 — G1, gold trend ensemble — **the best gold strategy found, and it is not real**

CAGR 0.9%, drawdown 19.3%, PF 1.97, 81 trades, Sharpe 0.23.
The permutation test settles it: shuffling the entry dates gives a mean Sharpe
of **0.23** (identical), mean PF **2.02** (higher than the real 1.97), and
p-values of 0.475–0.700. **This strategy is statistically indistinguishable from
placing the same trades at random.** It is reported for completeness and should
not be traded.

---

## 6. Complete reproduction specifications

### 6.1 M1 — BTC trend-gated momentum pulse (the recommended single strategy)

```
Market        BTCUSDT (tested on Bitstamp BTC/USD spot)
Timeframe     4-hour bars, UTC
Period        2014-01-01 → 2026-09-09  (IS to 2021-12-31, OOS from 2022-01-01)
Direction     long / flat only  (short down-breaks are negative-edge on BTC)

INDICATORS (all on completed bars)
  MA      = 200-period simple moving average of close
  ATR     = 14-period Wilder ATR
  DCH     = highest high of the previous 20 bars (current bar excluded)
  RSI2    = 2-period Wilder RSI of close
  BBW     = 20-period Bollinger width (2σ) ÷ close
  SQ      = percentile rank of BBW within its own trailing 120 bars

ENTRY  (evaluated on the close of bar i; filled at the open of bar i+1)
  Regime : close > MA
  AND at least one trigger:
      A) close > DCH                       (20-bar breakout)
      B) RSI2 > 95                         (short-term thrust)
      C) SQ[i-1] < 0.20 AND close > DCH    (squeeze expansion)
  One position at a time; no pyramiding; no new entry while in a trade.

EXIT (whichever comes first)
  - stop loss at entry ± 3.0 × ATR (checked against every bar's high/low;
    if a bar touches both stop and target, the stop is assumed to fill first)
  - time stop after 48 bars held (8 days), filled at the next bar's open
  (no take-profit; no trailing stop — both tested and both reduced returns)

POSITION SIZING
  qty = (equity × 0.02) ÷ (3.0 × ATR)          i.e. risk 2% of equity per trade
  capped at 10× equity notional (cap essentially never binds: 3×ATR on 4h BTC
  is ~5% of price, so typical leverage is ~0.4×)
  Fully compounded on current equity.

COSTS APPLIED
  0.05% fee + 0.02% slippage on entry and on exit; 0.01%/8h funding on notional.
```

**Parameter stability** (this is what makes it credible, not the peak number):
`ma_n` ∈ {100, 200, 400} and `hold_bars` ∈ {24, 48} all give IS Sharpe 1.6–2.0
and positive OOS. The surface is a plateau, not a spike.

### 6.2 The #1 portfolio

Run all four modules simultaneously on BTC with **equal capital weight**, each
at 2% risk per trade, and apply a **2.8× overall leverage multiplier** to the
combined return stream (equivalently 5.6% risk per trade in each module).

| module | strategy | timeframe | rules |
|---|---|---|---|
| M1 | momentum pulse | 4h | as §6.1 |
| M2 | trend ensemble | 1d | long/short on the sign-consensus of the 20/60/120-day returns (all three must agree); stop 3×ATR(14); 4×ATR(14) trailing stop from the closed-bar extreme; exit when consensus breaks |
| M3 | bear-regime short | 4h | short when the 120-bar AND 1500-bar returns are both negative AND close breaks the 60-bar Donchian low; stop 4×ATR(14); time stop 240 bars |
| M4 | regime long | 1d | long while close > SMA(200) and 20-day realised vol is below its 85th percentile (trailing 500 days); stop 8×ATR(20); exit when the condition fails |

### 6.3 Cost sensitivity (full period, 2% risk)

| strategy | 5 bps RT | 14 bps RT (base) | 30 bps RT | 50 bps RT |
|---|---|---|---|---|
| M1 momentum pulse | 33.6% / PF 1.82 | **32.2% / 1.79** | 25.5% / 1.75 | 19.2% / 1.69 |
| M2 trend ensemble | 7.7% / 2.42 | 7.5% / 2.40 | 6.0% / 2.39 | 4.4% / 2.37 |
| M3 bear short | 3.7% / 2.29 | 3.6% / 2.27 | 2.7% / 2.10 | 2.1% / 2.06 |
| M4 regime long | 3.7% / 5.36 | 3.7% / 5.29 | 3.3% / 5.19 | 2.8% / 5.05 |
| G1 gold ensemble | 1.1% / 2.03 | 0.9% / 1.97 | **−1.3%** / 1.82 | **−3.5%** / 1.67 |

The BTC modules stay profitable even at 3.5× the assumed cost. The gold strategy
turns negative at 30 bps — another sign it has no margin of safety.

### 6.4 Permutation significance (40 shuffles each)

| strategy | real Sharpe | shuffled mean | shuffled 95th pct | p-value | verdict |
|---|---|---|---|---|---|
| M1 BTC momentum pulse | **1.64** | 0.92 | 1.15 | **< 0.025** | genuine |
| M3 BTC bear short | **0.51** | −0.31 | 0.12 | **< 0.025** | genuine |
| M4 BTC regime long | **1.14** | 0.83 | 0.94 | **< 0.025** | genuine |
| G1 gold trend ensemble | 0.23 | **0.23** | 0.31 | **0.475** | **no skill** |

---

## 7. What was learned

1. **The 500%/20% target is not reachable by any legitimate strategy.** It
   requires a Calmar of 25; real systematic strategies live at 0.5–3. Leverage
   moves along the frontier, it does not move the frontier.
2. **Cost per unit of risk is the master constraint.** Any strategy whose stop
   is tighter than ~1% of price is uneconomic at perp taker fees, regardless of
   how good the pattern looks. This one ratio predicted the outcome of every
   intraday test in the study before it was run.
3. **Gold (XAUUSDT) is efficient at every horizon tested.** 15-minute, hourly,
   4-hour, daily and weekly; ~90 conditional signals; momentum, reversion,
   breakout, seasonality, session and volatility-regime families. The strongest
   gold effects (Asian-session drift, Monday weakness) are statistically real but
   worth 2–9 bp against a 14 bp round trip, and the best complete gold strategy
   is indistinguishable from random entries. **The correct conclusion is not to
   trade gold systematically on price-derived signals alone.** A genuine gold
   edge, if one exists, most likely needs data this study could not reach: real
   yields, DXY, ETF flows, COT positioning, or options-implied skew.
4. **BTC still pays for momentum continuation, not reversion** — but far less
   than it used to: every BTC system earns roughly half as much out-of-sample
   (2022-2026) as in-sample (2014-2021).
5. **Diversification is the only free lunch on offer.** Four near-uncorrelated
   modules on the *same* instrument lifted the Sharpe from 1.64 to 1.96 and cut
   drawdown from 19.1% to 7.0% before leverage — worth more than any single
   parameter choice in the study.
6. **Always shuffle your entries.** The permutation control killed the best gold
   strategy, which looked respectable on every conventional metric (PF 1.97,
   positive out-of-sample). Standard IS/OOS splitting would have passed it.

## 8. Known limitations

* Exchange APIs were unreachable, so spot proxies were used instead of the
  actual XAUUSDT/BTCUSDT perpetual books (§2.1); the perp basis, real funding
  history and true perp liquidity are not modelled from live data.
* The gold data ends 2025-06-06, so the most recent 15 months of gold are not
  covered; BTC runs to 2026-09-09.
* Both gold feeds proved to be the same underlying source, so the gold result
  has no independent-vendor confirmation.
* Stop fills are modelled at the stop price plus slippage; a real gap through a
  stop fills worse. Strategies here use wide stops, which limits the error, but
  it remains a mild optimism in the tail.
* Single-instrument portfolios only. Trading gold *and* BTC together (plus other
  assets) is the obvious next step and would raise the Sharpe further — but it
  changes the mandate from "an XAUUSDT strategy" to "a multi-asset program".
