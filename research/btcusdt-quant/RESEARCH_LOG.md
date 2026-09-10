# BTCUSDT Systematic Strategy Research Log

## Market & data
| item | value |
|---|---|
| Exchange / market | **Binance USDⓈ-M Perpetual `BTCUSDT`** (spot `BTCUSDT` used for basis) |
| Source | Official Binance public archive (`data.binance.vision`, S3 origin) |
| Decision data | perp 1h / 4h klines (aggregated from 5m) |
| Execution data | perp **15m** klines (intrabar stop/target resolution) |
| Period | 2020-01-01 → 2026-08-31 (perp), spot back to 2017-08-17 |
| Bars | 58,440 × 1h · 233,760 × 15m · 701,280 × 5m — **zero gaps, zero OHLC violations** |
| Funding | real Binance historical funding (7,305 settlements, mean 0.0108%/8h ≈ 12.5%/yr, positive 85.9% of the time) |

## Trading assumptions (applied to every result below)
* Taker fee **5 bps per side**, slippage **3 bps per side** → **16 bps round-turn**.
* Real funding cash-flows at 00/08/16 UTC on open notional (longs pay a positive rate, shorts receive it).
* Signals from **closed** bars only; entry fills at the **open of the next execution bar**.
* Stops/targets resolved on 15m bars; if one bar spans both levels the **stop** fills first.
* Position size = `risk_fraction × equity / stop_distance`, capped by a max-leverage limit.
* Equity compounds; account is marked to market every 15m; equity ≤ 0 = liquidated.

## Validation of the engine
| test | result |
|---|---|
| Frictionless 1x long vs BTC price return | 994.3% vs 992.1% (diff = 1 bar of execution lag) ✔ |
| Random signals, zero cost | PF 0.991 (≈1.0) ✔ |
| Random signals, costs on | PF 0.77 → 0.58 as costs rise ✔ |
| Signal that peeks 1h ahead | CAGR 3034%, PF 4.28 (detector fires) ✔ |
| Same signal, past-only | CAGR −56%, PF 0.53 ✔ |

## Benchmarks (2020-01-01 → 2026-08-31)
| strategy | CAGR | MaxDD | Sharpe |
|---|---|---|---|
| BTC spot buy & hold | 43.2% | −77.2% | 0.90 |

## In-sample signal screen (2020-01 → 2024-06, 1h bars)
Rank-IC and decile economics vs a 16 bps round-turn.

**Rejected — edge smaller than costs**
* Short-horizon price reversion (`rsi4`, `mom2-6`, `clv`): IC ≈ −0.07 at h=1 but decile spread < 0 net of costs.
* 15m sign-reversal (arXiv 2608.21888): reversal spread only **+1.54 bps** over 30 min (t=5.8) vs 16 bps cost. Statistically real, economically dead.
* Raw order-flow imbalance: contaminated by short-horizon reversal (IC −0.04 at h=1).

**Accepted — survives costs in the tails**
| signal | horizon | top-decile fwd ret | net of cost | note |
|---|---|---|---|---|
| `-funding_z` (contrarian funding) | 96h | +109.5 bps (t 10.8) | +93.5 | long side strong |
| `-basis_z` (perp discount to spot) | 96h | +139.8 bps (t 13.6) | +123.8 | **two-sided**: D1 −42.9 bps |
| `ofi96_res` (order flow ⟂ past returns) | 96h | +252.6 bps (t 23.2) | +236.6 | long-biased |
| `ofi24_res` | 96h | +246.1 bps (t 23.6) | +230.1 | long-biased |

Unconditional mean 96h return in-sample ≈ +47 bps, so these are genuine discriminators, not just beta.

---
# Attempts

## S1 — Positioning-Reversal with Flow Confirmation (PRFC)
**Idea.** Perp basis measures how crowded *levered* positioning is; spot taker order flow
orthogonalised to recent returns measures what *unlevered* flow is doing. When they
disagree, the derivatives crowd gets squeezed.
**Rules.** 4h bars. Long if `basis_z < −1.2` and `ofi24_resz > 0`; short if `basis_z > +1.8`
and `ofi24_resz < 0`. Stop 2.0×ATR(14), target 3.5×ATR, time-stop 7 days, risk 1%/trade,
max leverage 5.

| period | CAGR | MaxDD | PF | N | WR | Sharpe |
|---|---|---|---|---|---|---|
| IS 2020-01→2024-06 | 5.8% | −14.6% | 1.28 | 190 | 45.3% | 0.72 |
| OOS 2024-07→2026-08 | 10.6% | −7.8% | 1.32 | 151 | 47.0% | 1.02 |
| ALL | 7.3% | −14.6% | 1.30 | 341 | 46.0% | 0.83 |

**Verdict: FAILS** (CAGR 7.3% ≪ 300%). Edge is real and *holds up out-of-sample*, and the
flow filter is what makes it work (PF 1.04 → 1.30 when added). But the per-trade
expectancy (~0.14% of equity on 51 trades/yr) is two orders of magnitude too small.
On 1h bars the same idea loses money — costs dominate.

## Seasonality screen — REJECTED
Hour-of-day: best hour (20:00 UTC) +4.5 bps, t=2.39; day-of-week best (Wed) +2.2 bps, t=2.14.
Neither survives a multiple-testing correction across 24 hours / 7 days. Funding-settlement
hours are slightly *negative* (−0.83 bps vs +0.75 bps). No tradable calendar effect.

## S2 — Crowding-Reversal Ensemble (CRE)
Equal-weight z-score blend of −funding, −basis, +ofi24_res, +ofi96_res (pairwise
correlations 0.05–0.49, so genuinely additive). Trade the tails; ATR stop/target; time-stop.
Swept 3 timeframes × 3 thresholds × 3 holding periods (27 variants).
**Best:** 4h, thr 0.5, 4-day hold — ALL: CAGR 9.0%, DD −16.8%, PF 1.19, N 684; OOS CAGR 1.6%.
**Verdict: FAILS.** Blending *reduced* per-trade edge vs S1 (PF 1.19 vs 1.30) because the
weakest component (funding) dilutes the strongest. Max Calmar across all 27 variants: **0.59**.

## S3 — Orthogonal Order-Flow Swing (OFS)
Long when spot taker-buy pressure orthogonal to recent returns is in its top decile,
optionally gated by a 100-period EMA trend filter. 32 variants.
**Best:** 4h, `ofi24_resz > 0.75`, 10-day hold, trend filter on — ALL: CAGR 4.9%, DD −9.3%,
**PF 1.75**, N 113; OOS CAGR 3.7%, PF 1.43.
**Verdict: FAILS.** Highest profit factor of any directional idea and it holds up OOS, but
only ~17 trades/yr at 1% risk. Max Calmar **0.58**.

## S4 — Adaptive Volatility-Regime Trend (AVT)
Donchian breakout gated by ADX + efficiency ratio, ATR trailing stop, optional
volatility-regime scaling of the trail. 36 variants.
**Best:** 4h, dc40, ADX>20, ER>0.3, fixed trail — ALL: CAGR 8.0%, DD −9.5%, PF 1.53, N 251;
OOS CAGR 9.6%, PF 1.57 (good IS/OOS agreement).
**Negative result worth recording:** scaling the trailing stop by the volatility regime
*hurt* in 11 of 12 pairings (e.g. Calmar 0.97 → 0.75). Wider trails in high vol give back
too much; the ATR already carries the regime information.
**Verdict: FAILS.** Max Calmar **0.97**.

## S5 — Dynamic Funding Carry (delta-neutral) — best risk-adjusted result so far
Long spot BTCUSDT / short perp BTCUSDT in equal size; collect funding while the trailing
9-settlement funding average exceeds a threshold; flat otherwise. Costs: 4 legs × 8 bps per
round trip, **USDT borrow at 8% APR on the levered portion**, spot borrow 3% APR for reverse carry.

| variant | CAGR | MaxDD | Sharpe | N | Calmar |
|---|---|---|---|---|---|
| gross 1x, thr 0.8 bps/8h, no reverse | 15.4% | −3.9% | **6.72** | 29 | 3.97 |
| gross 2x, thr 0.8 bps/8h, no reverse | 22.8% | −6.7% | 5.55 | 29 | 3.41 |
| gross 3x, thr 0.8 bps/8h, no reverse | 29.0% | −8.8% | 5.09 | 29 | 3.29 |
| gross 3x, always on | 12.4% | −37.1% | 1.77 | 137 | 0.34 |

**Verdict: FAILS** — on trade count (29 < 100) and on CAGR. But Sharpe 5–6.7 and Calmar ~4 are
by far the best risk-adjusted numbers in this study. Two corrections mattered a lot: realising
basis P&L at unwind, and charging USDT borrow on the levered spot leg (that alone cut the
3x variant from 41.5% to 29.0% CAGR).

## S6 — Walk-Forward Gradient Boosting (WFGB)
LightGBM on the full ~75-feature panel, expanding window, refit every 180 bars, with an
h-bar **purge gap** so no training label overlaps the test block. Label = h-bar forward
return / ATR%.
| horizon | early-WF IC | late-WF IC |
|---|---|---|
| 24h | +0.013 | +0.009 |
| 48h | +0.037 | **−0.033** |

**Verdict: FAILS.** The model does not beat the hand-built signals and its IC flips sign out
of sample — the classic overfitting failure mode for ML on a single noisy series.

## New data source added mid-study — Binance futures *metrics*
`data/futures/um/daily/metrics/BTCUSDT/` publishes, every 5 minutes since 2021-01-01:
open interest, the long/short **account** ratio of top traders by margin balance, their
long/short **position** ratio, the long/short account ratio of all accounts (retail), and
the taker long/short volume ratio. 595,268 rows, 604 missing bars (0.1%).

IC screen (IS 2021-01 → 2024-06, 1h bars):
| feature | h=24 | h=48 | h=96 |
|---|---|---|---|
| `tt_vs_retail` = z(log(top-trader position ratio ÷ retail account ratio)) | +0.067 | +0.101 | **+0.108** |
| `tt_acct_z` (crowded count of top accounts) | −0.054 | −0.081 | **−0.102** |
| `retail_acct_z` (fade the crowd) | −0.041 | −0.064 | −0.054 |
| `oi_rank` | −0.023 | −0.030 | −0.041 |

`tt_vs_retail` at IC +0.108 is the **strongest single predictor found in this study** — 50%
above the best price/flow feature. Top decile earned +374 bps over 96h vs +26 bps
unconditional.

## S7 — Smart-Money vs Retail Positioning Divergence (SMRD) — best single directional book
4h bars. Composite = z(log(tt_pos/retail_acct)) − z(tt_acct) − z(retail_acct), each clipped
to ±3. Long if composite > 0.7, short if < −0.7. Stop 3.5×ATR(14), target 2.5R,
time-stop 10 days, risk 1%/trade. 36 variants swept.

| period | CAGR | MaxDD | PF | N | WR | Sharpe | Calmar |
|---|---|---|---|---|---|---|---|
| ALL 2021-01→2026-08 | 9.1% | −7.8% | 1.54 | 229 | 46.3% | 1.13 | **1.17** |
| OOS 2024-07→2026-08 | 9.7% | — | 1.49 | 108 | — | — | — |

**Verdict: FAILS on CAGR**, but the IS and OOS numbers are nearly identical (9.1% vs 9.7%
CAGR, PF 1.54 vs 1.49) — the most convincing single-strategy edge found.

## S10 — Volatility-Managed Trend (VMT)
Exposure = target_vol / realised_vol, gated by a 200-EMA trend filter, on the continuous-
weight simulator (turnover-costed, real funding). 54 variants.
**Best:** 4h, long/flat, target vol 0.8, 240-bar vol window — CAGR 69.5%, DD −61.3%,
PF 1.10, Sharpe 1.13, Calmar 1.13.
**Verdict: FAILS.** Vol-targeting raises *return* substantially but does nothing for Calmar,
because scaling up in quiet markets simply scales the drawdown too. Always-on (no trend
filter) is far worse (DD −66% to −91%).

## S8 — Multi-Strategy Portfolio, first pass
Daily-return correlations of the four sleeves (2021-2026):
|  | SMRD | OFS | AVT | CARRY |
|---|---|---|---|---|
| SMRD | 1.00 | 0.41 | 0.44 | **0.02** |
| OFS | | 1.00 | 0.38 | **−0.01** |
| AVT | | | 1.00 | **−0.03** |

The carry sleeve is essentially uncorrelated with all three directional books — exactly the
diversification the Calmar constraint needs. Risk-parity weights at 8× gave CAGR 140.8%,
DD −14.6%, Sharpe 3.74, Calmar 9.66.
**But this run is not trustworthy**: it scaled each sleeve's *return series* by the leverage
knob, which silently gives the carry sleeve 8× leverage for free. Superseded by S9.

## S9 / S11 — Multi-Strategy Portfolio with honest leverage (MSP-R)
Four sub-accounts (SMRD, OFS, AVT, CARRY), rebalanced monthly, each **re-simulated at its
true size** at every leverage setting so the carry sleeve pays its real USDT borrow cost.
Risk-parity weights are estimated on the in-sample window only and applied unchanged OOS.

Equal weight:
| knob | IS CAGR / DD | OOS CAGR / DD | ALL CAGR / DD | PF | Sharpe | Calmar |
|---|---|---|---|---|---|---|
| 1 | 10.1% / −3.6% | 6.4% / −3.9% | 8.9% / −3.9% | 1.41 | 1.98 | 2.31 |
| 4 | 30.6% / −14.2% | 22.1% / −16.4% | 27.7% / −16.4% | 1.26 | 1.51 | 1.69 |
| 6 | 44.2% / −20.8% | 32.2% / −24.0% | 40.1% / −24.0% | 1.23 | 1.44 | 1.67 |
| 12 | — | — | 72.8% / −42.7% | 1.18 | 1.34 | 1.70 |

Risk parity (IS-derived weights: CARRY 47.8%, OFS 18.4%, SMRD 18.2%, AVT 15.6%):
| knob | IS CAGR / DD / Sharpe | OOS CAGR / DD / Sharpe | ALL Calmar |
|---|---|---|---|
| 1 | 13.6% / −2.3% / 4.36 | 5.6% / −1.8% / 1.51 | 4.73 |

**The correction mattered enormously.** The first-pass S8, which scaled return series instead
of re-simulating, reported Calmar 9.66; done honestly the same construction gives 1.7–4.7.

## Structural finding — the funding premium is decaying
Annualised funding run-rate, half-year buckets:
| period | 2021 H1 | 2021 H2 | 2022 H2 | 2023 | 2024 H1 | 2025 H1 | 2025 H2 | 2026 |
|---|---|---|---|---|---|---|---|---|
| run-rate | **42.0%** | 18.3% | 4.2% | 5.5% | 17.2% | 9.0% | 5.3% | **3.0–3.4%** |

This is the single most important caveat in the whole study. The carry sleeve — the highest
Sharpe component and the one risk parity leans on for ~48% of capital — earns a premium that
has fallen by an order of magnitude as the basis trade became crowded. At a 3% funding
run-rate against an 8% APR USDT borrow, **levered carry is unprofitable today**. Its
historical contribution is not repeatable, and the portfolio's forward-looking Sharpe is
much closer to the OOS figure (1.5) than the full-sample one (3.3).

## S12 — ETH→BTC Lead-Lag (ELL)
ETH/BTC relative momentum plus ETH taker-flow imbalance as a risk-appetite proxy for BTC.
First variants: CAGR −6.6%, PF 0.91, DD −42.6%. **Verdict: FAILS** — no exploitable lead-lag
at 4h once costs are applied; ETH does not lead BTC at this frequency.

## Portfolio frontier — risk parity, honest leverage (the best result in the study)
Weights fixed from the in-sample window (CARRY 47.8%, OFS 18.4%, SMRD 18.2%, AVT 15.6%) and
applied unchanged out-of-sample. The knob raises every sleeve's true size simultaneously.

| knob | IS CAGR / DD / Shp | OOS CAGR / DD / Shp | ALL CAGR | ALL DD | PF | Sharpe | Calmar | N |
|---|---|---|---|---|---|---|---|---|
| 1 | 13.6% / −2.3% / 4.36 | 5.6% / −1.8% / 1.51 | 10.9% | −2.3% | 1.70 | 3.27 | **4.73** | 622 |
| 2 | 20.6% / −5.0% / 3.38 | 9.1% / −4.8% / 1.24 | 16.8% | −5.0% | 1.48 | 2.53 | 3.34 | 622 |
| 3 | 27.6% / −7.7% / 2.99 | 12.6% / −7.8% / 1.15 | 22.5% | −7.8% | 1.39 | 2.26 | 2.89 | 622 |
| 4 | 34.5% / −10.2% / 2.77 | 16.1% / −10.7% / 1.10 | 28.2% | −10.7% | 1.35 | 2.10 | 2.65 | 622 |
| **6** | 48.4% / −15.8% / 2.52 | 22.8% / −16.2% / 1.06 | **39.5%** | **−16.2%** | **1.29** | 1.93 | 2.45 | **622** |
| 8 | 62.2% / −21.4% / 2.37 | 29.3% / −21.4% / 1.03 | 50.7% | −21.4% | 1.25 | 1.82 | 2.37 | 622 |

Knob 6 is the **largest size that still respects the 20% drawdown limit**. It satisfies four of
the five numeric gates (trades 622 ≥ 100 ✔, PF 1.29 > 1.10 ✔, MaxDD −16.2% < 20% ✔, realistic
risk management ✔) and misses only on net yearly profit: **39.5% vs the required 300%**.

Block-bootstrap of the knob-6 daily returns (1,500 resamples, 5-day blocks):
median max drawdown −17.4%, 5th percentile −27.1%, **P(drawdown worse than −20%) = 29%**.
So even the drawdown gate is only marginally met — the realised −16.2% is a favourable draw.

## S13 — Liquidation-Cascade Reversal (LCR)
Long after a ≥3–7% 24h fall accompanied by a ≥2–6% collapse in open interest (forced closing,
not informed selling); mirror image for short. 1h and 4h, 54 variants.
First variants: CAGR −14.8% to −6.9%, PF 0.85–0.95, DD −56% to −68%.
**Verdict: FAILS.** The `flush` condition does mark capitulation (IC +0.023 at h=4) but the
cascade keeps going far enough to take out a 2×ATR stop before the snap-back arrives. The
signal is real; it is not survivable with sane risk control.

## S14 — Volatility-Squeeze Expansion (VSE)
Bollinger bandwidth in the bottom 15–30% of its trailing distribution, then the first close
outside the band. 1h and 4h, 24 variants.
Range: CAGR −36.8% to −9.2%, PF 0.85–0.98, DD −70% to −96%, win rate 27–33%.
**Verdict: FAILS badly.** BTC squeeze breaks are predominantly false; the compression is real
but the direction of the expansion is a coin flip and the stops pay for every one.

---

# Final ranking

**No strategy qualified.** The five numeric gates were: net yearly profit > 300%, ≥100 trades,
profit factor > 1.10, max drawdown < 20%, realistic risk management, no look-ahead bias.
The best configuration clears every gate except net yearly profit, which it misses by 7.6×.

| # | Strategy | Period | CAGR | MaxDD | PF | N | Sharpe | Calmar | OOS CAGR | PF @2× cost | P(DD>20%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Portfolio, risk parity, size 6 | 2020-01→2026-08 | **39.5%** | −16.2% | 1.29 | 622 | 1.93 | 2.45 | 22.8% | — | 29% |
| 2 | S7 SMRD, 1.0% risk | 2021-01→2026-08 | 9.1% | −7.8% | 1.54 | 229 | 1.13 | 1.17 | 9.7% | 1.48 | **2%** |
| 3 | S7 SMRD, 2.5% risk | 2021-01→2026-08 | 22.9% | −18.7% | 1.49 | 229 | 1.14 | 1.23 | 24.1% | 1.43 | 76% |
| 4 | S4 AVT, 1.0% risk | 2020-01→2026-08 | 8.0% | −9.5% | 1.53 | 251 | 0.77 | 0.85 | 9.6% | 1.45 | 24% |
| 5 | S3 OFS, 2.0% risk | 2020-01→2026-08 | 9.7% | −18.1% | **1.71** | 113 | 0.88 | 0.53 | 7.2% | 1.65 | 28% |
| 6 | S5 Carry, gross 1× | 2020-01→2026-08 | 15.4% | **−3.9%** | — | 29 ✗ | **6.72** | 3.97 | — | — | — |
| — | BTC spot buy & hold | 2020-01→2026-08 | 43.2% | −77.2% | — | 1 | 0.90 | 0.56 | — | — | 100% |

Note that **buy & hold beat every strategy in absolute return** over this window. What the
portfolio buys is a −16.2% worst drawdown instead of −77.2%, and Sharpe 1.93 instead of 0.90.

## Why 300% at <20% drawdown is not reachable here
Two independent calculations agree.

1. **The ratio is fixed by the constraint.** 300% ÷ 20% = Calmar ≥ 15. Measured across every
   portfolio size, `Calmar ≈ 1.3 × Sharpe` (3.27→4.73, 1.93→2.45, 1.82→2.37). Leverage moves
   return and drawdown together and leaves the ratio alone — which is why raising the knob
   from 1 to 14 never helped. Calmar 15 needs **Sharpe ≈ 11** net of costs. Best measured:
   3.27 full-sample, 1.51 out-of-sample.

2. **The signal-strength ceiling.** `Sharpe ≈ IC × √(independent bets/yr)`. The best IC found
   anywhere was 0.108 (top-trader vs retail, 96h horizon). A 96h horizon on one instrument
   gives ~91 non-overlapping bets/yr, so `0.108 × √91 ≈ 1.03` — essentially the 1.13 that
   strategy actually delivered. Sharpe 11 at that IC needs ~10,400 independent bets a year.
   Shortening the horizon to buy breadth fails because IC collapses (0.013 at 1h) and the
   16 bps round-turn eats everything under roughly a 24h hold.

Breadth of that order requires either several hundred weakly-correlated instruments (outside
a BTCUSDT-only brief) or sub-second market making (not evaluable from OHLCV — it needs
order-book reconstruction and a queue model, and its return is a function of latency and fee
tier, not of a signal). **Within the brief, the ceiling observed here is about 40% net annual
at a 16% drawdown.**
