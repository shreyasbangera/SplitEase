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
