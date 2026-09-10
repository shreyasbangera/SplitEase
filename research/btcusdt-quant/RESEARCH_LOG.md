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

---
# Round 2 — futures-only, and attacking the drawdown constraint directly

**Scope correction.** This is futures-only research. S1–S4, S7 and S15–S20 were already pure
Binance USDⓈ-M perpetual BTCUSDT (perp klines, perp funding, futures positioning metrics,
perp execution). **S5, the long-spot / short-perp carry, is out of scope** and is removed —
which matters, because it was 47.8% of the portfolio behind the 39.5% headline.

## Futures-native carry: tested and rejected
Downloaded 24 BTCUSDT **quarterly delivery** contracts (2021-02 → 2026-08, 75,823 hourly bars)
to replace the spot leg with a perp-vs-quarterly calendar trade.

| quantity | mean | sd | positive |
|---|---|---|---|
| quarterly annualised carry | +8.6% | 8.5% | 98% |
| perp funding annualised | +9.9% | 16.8% | — |
| **spread (quarterly − perp)** | **−1.3%** | 10.9% | **49%** |

**Verdict: no edge.** The spot-perp carry earned its 15% by harvesting funding against a leg
that pays none. Once both legs are futures, funding sits on both sides and the spread prices
to zero — exactly as arbitrage should. There is no futures-native replacement for that sleeve.

## New engine capability
Added two mechanisms aimed squarely at the drawdown constraint rather than at returns:
* **High-water-mark throttle** — nominal risk is full while drawdown is shallower than
  `dd_soft`, tapering linearly to `dd_floor` × nominal at `dd_hard`.
* **Pyramiding** — up to N extra units added as a trade advances, each after a further step of
  R, with the stop pulled to the new average entry so the enlarged package never risks more
  than the original unit.

## S15 — Convex Positioning Trend
Deliberately breaks the symmetric return distribution: no take-profit (winners ride an ATR
trail), pyramiding, and the HWM throttle.
* Pyramiding raised profit factor from 1.46 to **2.21** and cut win rate to ~20% — the intended
  convex shape.
* Best unthrottled: risk 5%, pyramid 3 → CAGR 59.6%, DD −70.4%, PF 1.71, **OOS CAGR 110.6%**.
* **The throttle works but locks out.** Tapering to zero caps drawdown at exactly −20.0% but
  collapses trade count from 266 to 15 and CAGR to −0.4%. With a floor of 0.15 it keeps 263
  trades: CAGR 30.6%, DD −40.9%, PF 1.89. Bounding drawdown by shrinking size costs more
  return than it saves.

## S18 — Growth-optimal leverage curve  ← the decisive result
Compound growth is **not monotone in size**. Traced for each book:

| book | peak CAGR | at risk/trade | drawdown at peak | at 22–30% risk |
|---|---|---|---|---|
| S15 convex, pyramid 3 | **59.6%** | 5.0% | −70.4% | **−71.6%** CAGR |
| S15 convex, no pyramid | **63.1%** | 12.0% | −83.6% | −43.2% CAGR |
| S4 adaptive trend | 35.9% | 9.0% | −64.2% | +9.0% CAGR |

Past the growth-optimal point, more size buys **less** compound return and strictly more
drawdown. This is a property of the return distribution, not a tuning failure: no leverage
setting reaches 300%. Under a 20% drawdown budget the same curves sit at roughly 1% risk and
~20% CAGR.

## S17 — Meta-labelling: inconclusive, not disproven
Lopez de Prado meta-labelling (secondary classifier predicting whether each primary signal
wins, expanding window, purge gap = max holding period). Base win rate 48.3% on **232 labelled
trades** — far too few to train a classifier. Higher probability thresholds dropped trade
counts below the 40-trade reporting floor. The technique is sound; this signal does not
generate enough events to use it.

## S20 — Continuous Positioning Signal
Target exposure proportional to signal strength, volatility-targeted, rebalanced every 4h,
costs charged only on the change in position. Higher breadth at lower turnover per bet.
* Best: CAGR **100.6%**, DD −73.9%, Sharpe 1.20, Calmar 1.36, OOS CAGR 50.5%.
* At a 20% drawdown budget: ~25% CAGR.
Sharpe improves slightly over the discrete implementation (1.20 vs 1.13) — the breadth
argument is real but small.

## S19 — Futures-only portfolio (corrected headline)
Four directional perp books, monthly rebalance, IS-derived risk-parity weights.
Sleeve correlations are high because they are all directional on one asset:
SMRD↔CONVEX **0.70**, CONVEX↔AVT 0.60, SMRD↔AVT 0.44, OFS↔others 0.25–0.41.

| knob | IS CAGR / DD | OOS CAGR / DD | ALL CAGR / DD | PF | Sharpe | Calmar | N |
|---|---|---|---|---|---|---|---|
| 1 | 9.0% / −6.7% | 11.6% / −8.9% | 9.9% / −8.9% | 1.24 | 1.12 | 1.11 | 811 |
| 2 | 17.9% / −12.9% | 23.3% / −17.1% | **19.8% / −17.1%** | 1.23 | 1.12 | 1.16 | 811 |
| 3 | 25.3% / −18.8% | 34.6% / −24.5% | 28.6% / −24.5% | 1.22 | 1.10 | 1.17 | 813 |

**Removing the spot leg costs most of the diversification**: Sharpe 1.93 → 1.12, Calmar
2.45 → 1.17. The futures-only answer under a 20% drawdown budget is **≈20% net annual**, not
39.5%.

## S16 — Drawdown-constrained grid search (480 configs, IS-selected then OOS-validated)
Grid over base risk × pyramid depth × throttle parameters × trail width; selection maximises
in-sample CAGR subject to a drawdown budget; the winner is then reported out-of-sample untouched.

| DD budget | IS CAGR / DD | OOS CAGR / DD | full CAGR / DD | winning config |
|---|---|---|---|---|
| 15% | 19.3% / −14.6% | 9.9% / −20.0% | 14.1% / −20.0% | risk 2%, pyr 2, throttle 5/16 floor 0.40 |
| 20% | 31.1% / −17.9% | **11.0% / −23.0%** | 22.1% / −23.0% | risk 2%, pyr 2, throttle 10/22 floor 0.45 |
| 25% | 37.8% / −22.1% | 14.7% / −31.7% | 28.3% / −31.7% | risk 2%, pyr 2, **throttle off** |
| 40% | **73.3%** / −38.5% | **20.2%** / −54.2% | 50.3% / −54.2% | risk 4%, pyr 2, **throttle off** |

**Verdict: FAILS, and instructively.** In-sample CAGR rises to 73.3% as the budget loosens
while out-of-sample sits at 20% regardless — the textbook signature of selection bias once the
real edge is exhausted. The 20% budget is also *breached out of sample* (−23.0%). Note that at
the two loosest budgets the winning configuration has the **throttle switched off**: the grid
independently rediscovered that capping drawdown by shrinking size is not worth its cost.

## S21 — Multi-timeframe ensemble
Same positioning signal on 2h/4h/8h/12h/1D sleeves, equal weight, monthly rebalance.
Per-sleeve IS Sharpe: 2h 1.46 · 4h 1.17 · 8h 0.71 · 12h 0.54 · 1D 0.49.
Cross-sleeve correlations 0.18–0.79.
Best: risk 3.5% → CAGR 19.1%, DD −16.7%, PF 1.17, Sharpe 1.10, N 970.
**Verdict: FAILS.** Equal weighting drags the blend *below* its best single sleeve. Time
diversification is real but the weak long-horizon sleeves cost more than the decorrelation
gains.

## S22 — Short-timeframe positioning book
1h / 2h / 3h / 6h versions of the S7 signal, chasing the higher breadth implied by the 2h
in-sample Sharpe of 1.46.

| tf | IS CAGR / Sharpe | OOS CAGR / Sharpe | full CAGR / DD / PF / N |
|---|---|---|---|
| 1h thr0.5 7d | 23.8% / **1.48** | 11.6% / **0.72** | 19.2% / −14.3% / 1.26 / 722 |
| 2h thr0.5 7d | 15.4% / 1.35 | 8.8% / 0.76 | 12.8% / −11.4% / 1.32 / 468 |
| 4h (baseline) | 22.3% / 1.17 | 24.1% / 1.10 | 22.9% / −18.7% / 1.49 / 229 |

**Verdict: FAILS — and the apparent short-timeframe advantage was in-sample only.** Sharpe
halves out of sample at 1h and 2h, while the 4h book is stable (1.17 → 1.10). Buying breadth
by shortening the horizon does not work for this signal.

---
# Final answer — futures-only

**No strategy qualified.** Under the 20% drawdown limit the best books reach ≈23% net annual.
Removing the limit does not help: the strongest single book peaks at 59.6% CAGR at its
growth-optimal size, and beyond that point more leverage returns less.

| # | Strategy | CAGR | MaxDD | PF | N | Sharpe | IS | OOS | gate missed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S7 Smart-money vs retail, 2.5% risk | **22.9%** | −18.7% | 1.49 | 229 | 1.14 | 22.3% | 24.1% | profit only |
| 2 | S19 futures-only portfolio, size 2 | 22.6% | −19.3% | 1.24 | 811 | 1.10 | 17.9% | 23.3% | profit only |
| 3 | S15 convex trend, 2% risk | 37.2% | −36.0% | **1.91** | 261 | 0.95 | — | 56.4% | profit + drawdown |
| 4 | S20 continuous vol-targeted | 31.0% | −25.9% | 1.18 | — | 1.21 | — | 22.2% | profit + drawdown |
| 5 | S21 multi-timeframe ensemble | 19.1% | −16.7% | 1.17 | 970 | 1.10 | 21.6% | 15.2% | profit only |
| 6 | S4 adaptive trend, 2.5% risk | 17.1% | −22.2% | 1.46 | 211 | 0.77 | 16.4% | 22.4% | profit + drawdown |
| — | BTCUSDT perp buy & hold | 16.8% | −75.2% | — | 1 | 0.55 | — | — | — |

Every book beats the underlying on this window (perp: 10k → 24.2k; convex book: 10k → 60.0k
at half the drawdown). None comes within an order of magnitude of 300%.

---
# Round 3

## S23 — USDT-perp vs COIN-margined-perp funding differential — REJECTED
Second attempt to rebuild a market-neutral sleeve without a spot leg. Downloaded Binance
COIN-margined `BTCUSD_PERP` (52,985 hourly bars, 2020-08 → 2026-08) and its independent
funding series; matched 2,115 settlements against the USDⓈ-M perp.

| leg | mean funding |
|---|---|
| USDⓈ-M perp `BTCUSDT` | +8.31%/yr |
| COIN-M perp `BTCUSD_PERP` | +8.38%/yr |
| **differential** | **−0.07%/yr**, sd 7.8%, positive **32%** of the time |

**Verdict: no edge.** The two funding rates track each other to within 7 bps a year. Combined
with the quarterly calendar spread (−1.3%/yr, positive 49%), this settles the question:
**Binance's futures complex is internally well-arbitraged and contains no futures-only carry.**
The market-neutral sleeve that carried the earlier portfolio genuinely required the spot leg.

## S24 — Session Range Breakout — REJECTED
Asia (00–08 UTC) range traded in the London session, London range traded in the US session,
US range traded into Asia. 1h bars, ATR or range-width stops, 24 variants.
Range across all variants: CAGR −29% to −54%, **PF 0.83–0.97**, win rate 34–42%, DD −90% to −99%.

## S25 — Session Range FADE — REJECTED, and the pair is informative
Same structure with the signal inverted, on the hypothesis that if breaking out loses then
fading should win. It does not: **PF 0.60–0.76**, CAGR −50% to −63%.

**The pair is the finding.** Costs subtract from both directions, so inverting a post-cost
PF 0.83 strategy does not hand you PF 1.20 — but if range breaks were pure noise the two sides
would lose roughly symmetrically around the cost drag. The fade losing *materially more* than
the breakout (0.60–0.76 vs 0.83–0.97) means breakouts do carry **weak genuine continuation**;
it is simply far too small to clear a 16 bps round turn. Together with S14 (volatility-squeeze
breakout, PF 0.85–0.98) this closes the whole intraday-breakout family: real but sub-cost
momentum at range edges, in both the volatility-compression and session-structure versions.

## Tick-level microstructure — tested and REJECTED as an information source
Downloaded 30 days of `aggTrades` (638 MB, ~50M prints) spanning three regimes (2022-06,
2024-03, 2025-10) and built per-minute features that klines **cannot** express: signed volume
restricted to large prints (≥$50k) vs small prints (≤$1k), aggressor run lengths (the
signature of one participant working an order), true VPIN, trade-size concentration, and
Kyle's lambda (price impact per unit of signed flow). 43,200 minute observations.

| feature | h=5m | h=15m | h=60m | h=240m |
|---|---|---|---|---|
| `kline_imb` (baseline, from klines) | −0.0155 | −0.0187 | −0.0078 | −0.0090 |
| `lg_imb` large-print imbalance | −0.0176 | −0.0201 | −0.0080 | −0.0090 |
| `lg_sm_div` institutions vs retail | −0.0196 | −0.0137 | −0.0006 | −0.0041 |
| `run_len` order-splitting signature | +0.0101 | +0.0101 | +0.0098 | +0.0115 |
| `vpin` | +0.0077 | +0.0131 | +0.0084 | +0.0045 |
| `lam` Kyle's lambda | −0.0102 | −0.0161 | +0.0013 | −0.0054 |

**Residual IC after orthogonalising to the kline baseline: 0.003–0.015** — noise at this sample
size (SE ≈ 0.005). And `corr(tick imbalance, kline imbalance) = 0.9999`: the aggregate
`taker_buy_base ÷ volume` already in the klines is a near-perfect proxy for the tick-level
computation.

**Verdict: no incremental information.** The 51 GB full-history download is not justified. This
closes the last untested data source for a directional futures-only study — OHLCV, taker
volume, trade count, funding, open interest, trader positioning, quarterly futures,
coin-margined perp and now tick prints have all been mined. Only the order book remains, and
that is a market-making dataset rather than a directional-signal one.

## Execution-granularity validation — results hold at 1-minute resolution
Every result in this study resolved stops and targets on **15-minute** bars, with the stop
assumed to fill first whenever one bar straddled both levels. That worst-case rule could cut
either way at finer resolution: fewer bars straddle both levels, but stops also trigger on
wicks a coarse bar smooths over. Downloaded the full **1-minute** perp history (3,506,400 bars,
zero gaps, zero OHLC violations) and re-ran the finalists on it.

| book | exec grid | CAGR | MaxDD | PF | N | Sharpe |
|---|---|---|---|---|---|---|
| S7 SMRD 2.5% | 15m | 22.9% | −18.7% | 1.49 | 229 | 1.14 |
| | **1m** | **21.8%** | **−19.2%** | 1.46 | 229 | 1.10 |
| S15 convex 2% pyr3 | 15m | 37.2% | −36.0% | 1.91 | 261 | 0.95 |
| | **1m** | **36.4%** | **−30.5%** | 1.66 | 283 | 0.93 |

The 15-minute grid was **mildly optimistic on return** (−0.8 to −1.1 pp) and, for the convex
book, **pessimistic on drawdown** (−36.0% at 15m vs −30.5% at 1m — the stop-first rule fires
less often when the path is resolved finely). Net: the headline numbers are robust to
execution granularity, and no conclusion in this study changes.

## S26 — Adaptive Sleeve Allocation — REJECTED
Monthly re-weighting of the four perp sleeves in proportion to trailing risk-adjusted
performance (3/6/12-month lookbacks, Sharpe- and mean-weighted), using only data available
before each month begins.

| scheme (knob 2) | IS CAGR / DD | OOS CAGR / DD | ALL CAGR / DD | Sharpe | Calmar |
|---|---|---|---|---|---|
| **fixed equal weight** | 19.4% / −13.9% | 28.2% / −19.3% | **22.6% / −19.3%** | **1.10** | **1.17** |
| adaptive Sharpe 3m | 20.8% / −15.4% | 21.3% / −19.1% | 21.5% / −19.1% | 1.01 | 1.12 |
| adaptive mean 6m | 24.3% / −16.2% | 24.3% / −24.7% | 24.8% / −24.5% | 0.99 | 1.01 |
| adaptive Sharpe 12m | 21.8% / −14.1% | 24.4% / −18.2% | 22.0% / −18.9% | 1.10 | 1.16 |

**Verdict: FAILS.** No adaptive scheme beats fixed equal weight on Sharpe or Calmar; the
higher-CAGR variants buy it entirely with deeper drawdowns. Relative sleeve performance is not
persistent enough month-to-month to chase.

## S27 — Parameter Ensemble — the one technique that helped
72 configurations of the S7 book (4 thresholds × 3 stops × 3 reward:risk × 2 holding caps) run
simultaneously at 1/N size instead of selecting one.

| | ensemble | median single config | gain |
|---|---|---|---|
| IS Sharpe | **1.54** | 1.34 | **+0.20** |
| OOS Sharpe | **0.80** | 0.73 | +0.07 |
| IS CAGR / DD | 28.1% / −12.2% | 26.5% / −14.1% | — |
| OOS CAGR / DD | 15.0% / −22.2% | 15.6% / −21.8% | — |

**Verdict: helps, modestly.** The ensemble beats the median single configuration in both
windows, and unlike a selected configuration it cannot be the one that happened to fit the
sample. Single-config CAGR ranged from −0.6% to +36.0% out of sample — that spread is exactly
the selection risk S16 exposed. This is the correct default; it does not change the ceiling.

## Market-wide breadth screen — one genuinely new signal
Downloaded 15 liquid USDⓈ-M alt perps (ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, LINK, DOT, LTC,
TRX, BCH, ATOM, NEAR, FIL — 58,440 hours, 99.9% coverage) to use as **sensors only**; BTCUSDT
remains the sole traded instrument.

| feature | h=24 | h=48 | h=96 | note |
|---|---|---|---|---|
| `breadth24` (fraction of alts up over 24h) | **−0.074** | −0.045 | −0.026 | crypto-wide overbought |
| `flow_breadth96` (median 96h taker imbalance across 15 perps) | +0.024 | +0.044 | **+0.062** | vs BTC's own ofi24_z at +0.039 |
| `altrel24` (alt median return − BTC) | −0.063 | −0.053 | −0.034 | |
| `btc_dom` (BTC share of complex volume) | +0.037 | +0.039 | +0.050 | risk-off rotation into BTC |

**Averaging the same measure across 15 instruments beats BTC's own version** (+0.062 vs
+0.039) — more breadth in the *estimator*, not in the bets. `flow_breadth96`'s top decile earns
+131.9 bps over 96h (t=10.9), +115.9 bps net of cost, and correlates only **0.22** with the S7
positioning composite.

**Blending it into the S7 composite made things worse** — combined IC 0.070 vs 0.108 at h=96,
decile spread 135 bps vs 390 bps. This is the S2 failure repeating: equal-weight averaging
drags the strongest signal down toward the weakest. It is therefore run as its own sleeve (S28).

## S28 — Market-Wide Flow Breadth as a standalone sleeve — REJECTED out of sample
`flow_breadth96` z-scored, traded on 4h and 12h bars, 24 variants.

| variant | IS CAGR | OOS CAGR | OOS PF | full CAGR / DD |
|---|---|---|---|---|
| thr 0.4, 5d, long-only | +13.0% | **−13.0%** | 0.80 | +2.4% / −34.4% |
| thr 0.7, 5d, long-only | +10.8% | **−7.3%** | 0.87 | +3.5% / −25.3% |
| thr 1.0, 5d, long/short | +10.4% | **−3.4%** | 0.97 | +4.9% / −30.5% |

**Verdict: FAILS.** Every variant is positive in-sample and negative out-of-sample. The
in-sample decile economics were genuinely strong (top decile +131.9 bps over 96h, t = 10.9,
+115.9 bps net of cost) and they did not survive. This is the sharpest reminder in the study
that **in-sample decile economics are not evidence of out-of-sample tradability** — a t-stat of
10.9 on overlapping 96-hour windows is far weaker evidence than it looks.

## S29 — Conditional Gating instead of averaging — REJECTED
Two results pointed the same way: equal-weight *averaging* diluted the strongest signal twice
(S2 PF 1.30→1.19; S28 blend IC 0.108→0.070), while conditional *gating* worked in S1
(PF 1.04→1.30). Gating leaves the primary untouched and only removes trades the confirmer
disputes. Applied to the S7 positioning book with market-wide flow breadth as confirmer
(correlation 0.22).

| variant | CAGR | DD | PF | N | Sharpe | Calmar | IS | OOS |
|---|---|---|---|---|---|---|---|---|
| **ungated baseline** | **22.9%** | −18.7% | 1.49 | 229 | 1.14 | **1.23** | 22.3% | 24.1% |
| gate "agree" 0.0 | 13.7% | −16.0% | 1.40 | 168 | 0.84 | 0.85 | 16.7% | 9.0% |
| gate "strict" 0.3 | 8.0% | −26.4% | 1.23 | 159 | 0.55 | 0.30 | 15.5% | −3.1% |
| gate "veto" 1.0 | 23.3% | −20.1% | 1.51 | 217 | 1.16 | 1.16 | 24.1% | 22.4% |

**Verdict: FAILS.** Every gate that actually filters makes things worse, and the only variant
that matches the baseline is the weakest possible veto — which removes just 12 of 229 trades,
i.e. converges to doing nothing. The contrast with S1 is the point: gating works when the
confirmer is genuinely predictive, and breadth is not (S28 lost money in every out-of-sample
variant). Gating cannot rescue a signal that has no out-of-sample edge.

---
# Round 5

## IS-vs-OOS IC split — the screen I should have been running all along
Every earlier screen measured IC on in-sample data and used decile economics to decide what to
build. S28 showed that is not enough (top decile t=10.9 in sample, lost money out of sample).
The correct filter is to measure the SAME IC separately in both windows.

| feature | IS h=96 | OOS h=96 | verdict |
|---|---|---|---|
| `tt_vs_retail` (1h) | +0.108 | **−0.054** | collapses, flips sign |
| `retail_rng96` | −0.098 | −0.005 | collapses |
| `oi_vol96` | +0.082 | −0.004 | collapses |
| `tt_pos_rng96_z` | +0.057 | −0.041 | collapses, flips |
| `flow_breadth96` | +0.062 | +0.005 | collapses |
| `ttvr_accel96` | +0.085 | +0.016 | partial |
| `disagree` | +0.076 | +0.003 | partial |

**Almost every feature that looked strong in sample is weak or sign-flipped out of sample.**
Applied earlier this screen would have rejected S28 before a line of strategy code was written.

## Audit of my own headline result
The 1h `tt_vs_retail` reading above flips sign out of sample, yet the S7 strategy built on it
reported IS CAGR 22.3% against OOS 24.1%. Both cannot be innocent, so I checked directly.

**The composite the strategy actually uses degrades but does not collapse** (4h bars, which is
what it trades):
| horizon | IS | OOS |
|---|---|---|
| 96h | +0.0819 | +0.0142 |
| 168h | +0.0720 | +0.0336 |

**But the component doing the work out of sample is not the one the strategy is named after.**
At h=96 on 4h bars: `tt_vs_retail` IS +0.104 → OOS **−0.003**; `−retail_acct_z` IS +0.083 → OOS
**+0.017**; `−tt_acct_z` IS +0.126 → OOS **+0.023**. The "smart-money vs retail" ratio is the
component that stops working; what survives is **fading crowded retail and a crowded count of
top-trader accounts**. The strategy's name overstates what is actually carrying it.

**It is not disguised long-bias beta**, which was the other suspicion:
| window | long trades | long P&L | short trades | short P&L | perp buy & hold |
|---|---|---|---|---|---|
| IS | 56 | +9,501 | 65 | +976 | +94.9% |
| OOS | 55 | +5,453 | 53 | +978 | +25.1% |

Net exposure is −0.07 in sample and +0.02 out of sample, and **the short book is profitable in
both windows** — including out of sample, where the perpetual rose 25%. Shorts making money
into a rising market is genuine alpha, not beta. Longs do contribute 85–91% of gross P&L, so
the book is long-driven, but not long-only in disguise.

**Net effect on the study's conclusion: none.** S7 remains the best book at 22.9% net annual.
But the honest characterisation is weaker than the earlier framing: the edge degrades roughly
three- to five-fold out of sample, and it rests on crowd-fading rather than on following
sophisticated positioning.

## Systematic stability screen — the most useful thing in this study
Applied the IS-vs-OOS IC split to **all 91 features** in the 4h panel, at both a 24h and a 96h
horizon. A feature "survives" only if it keeps its sign in both windows at both horizons and
retains more than 40% of its in-sample strength.

* **Sign-stable at both horizons: 41 of 91 (45%)** — barely better than a coin flip.
* Of the 33 with |IS IC| > 0.04, only 18 (55%) are sign-stable, and the **median out-of-sample
  retention is 0.27**: a strong-looking feature keeps about a quarter of its apparent power.

**Survivors (|IS IC| > 0.03, sign-stable, retention > 40%):**
| feature | IS h=24 | OOS h=24 | IS h=96 | OOS h=96 | retention |
|---|---|---|---|---|---|
| **`oi_rank`** (open-interest percentile, faded) | −0.071 | −0.033 | **−0.137** | **−0.073** | 0.53 |
| **`ofi6_res`** (6-bar flow ⟂ past returns) | +0.039 | **+0.058** | +0.085 | +0.044 | 0.68 |
| `ofi6_resz` | +0.043 | +0.056 | +0.080 | +0.032 | 0.70 |
| **`fund`** (funding, faded) | −0.030 | −0.022 | −0.053 | **−0.060** | 1.13 |
| **`fund3`** | −0.017 | −0.023 | −0.041 | **−0.060** | 1.47 |
| `ofi96_res` | +0.008 | +0.010 | +0.040 | +0.034 | 0.85 |
| `ofi24_res` | +0.038 | +0.023 | +0.042 | +0.020 | 0.55 |

**Biggest in-sample mirages (|IS IC| > 0.06, sign flips out of sample):**
| feature | IS h=96 | OOS h=96 |
|---|---|---|
| **`tt_vs_retail`** — what the study's best book was built on | **+0.104** | **−0.003** |
| `ofi96_resz` | +0.080 | −0.009 |
| `mom168` | +0.068 | −0.014 |

**The uncomfortable conclusion:** the feature I selected the best strategy from is the single
biggest mirage in the panel, and the strongest *stable* feature — open-interest crowding,
`oi_rank`, at IS −0.137 / OOS −0.073 — was screened early, scored well, and never had a
strategy built on it. Selecting on in-sample IC, which is what I did for 29 strategies, is the
wrong selection rule. Funding, faded, is genuinely stronger out of sample than in it.

## S30 — Stability-Screened Composite — and the discovery of the study's best book
Built only from features that passed the stability screen: `−oi_rank`, `+ofi6_res`, `−fund`.
Each component was also traded alone.

**4h bars (risk 2.5%):**
| book | CAGR | DD | PF | N | Sharpe | IS | OOS | OOS PF |
|---|---|---|---|---|---|---|---|---|
| `−oi_rank` thr0.7 10d | 8.1% | −32.2% | 1.13 | 366 | 0.48 | 6.5% | 11.2% | 1.15 |
| **`ofi6_res` thr0.7 5d** | **22.9%** | −24.3% | 1.29 | **810** | 0.99 | **23.5%** | **21.9%** | 1.27 |
| `ofi6_res` thr1.0 10d | 17.8% | −25.5% | 1.27 | 553 | 0.83 | **17.7%** | **17.6%** | 1.28 |
| `−fund` thr0.7 5d | 5.7% | −36.0% | 1.08 | 574 | 0.37 | 15.6% | **−8.6%** | 0.92 |
| COMPOSITE thr0.7 5d | 12.2% | −26.4% | 1.18 | 378 | 0.69 | 17.6% | 4.1% | 1.08 |

**12h bars (risk 2.5%):**
| book | CAGR | DD | PF | N | Sharpe | Calmar | IS | OOS | OOS PF |
|---|---|---|---|---|---|---|---|---|---|
| **`ofi6_res` thr0.7 10d** | 16.9% | **−17.4%** | **1.62** | 259 | **1.31** | 0.97 | 20.3% | 11.7% | 1.38 |
| `ofi6_res` thr1.0 10d | 12.0% | −16.7% | 1.58 | 200 | 1.01 | 0.72 | **11.6%** | **13.2%** | 1.56 |
| COMPOSITE thr0.7 10d | 10.9% | −17.3% | 1.68 | 132 | 1.04 | 0.63 | 11.8% | 9.5% | 1.49 |

Three results worth separating:

1. **`ofi6_res` is the best single book in the study.** Sharpe **1.31** on 12h (against S7's 1.14),
   profit factor 1.62, drawdown inside the 20% limit, and the tightest IS/OOS agreement anywhere
   here — 11.6% vs 13.2% at thr1.0, 23.5% vs 21.9% on 4h. It also trades 3.5× more often than S7.

2. **Blending diluted again — the third time.** The equal-weight composite of the three
   stability-screened features is worse than `ofi6_res` alone on every measure (OOS 4.1% vs
   21.9% at 4h). S2, the breadth blend and now this: averaging signals of unequal strength is
   a reliable way to make the strong one worse.

3. **A stable IC does not guarantee a tradable strategy.** `−fund` passed the stability screen
   with the *highest* retention in the panel (1.13–1.47, stronger out of sample than in) and
   still loses money out of sample (−8.6%, PF 0.92). Its IC is real but too small (|0.05|) to
   survive a 16 bps round turn. Stability is necessary, not sufficient — magnitude matters too.

## S31 — Short-Window Orthogonal Flow — **the best book in the study**
Full parameter map over 3 timeframes × 3 thresholds × 3 stop/target pairs × 2 holding caps
(54 configurations). Best by drawdown-penalised Sharpe: **12h, thr 1.0, stop 3.0×ATR, target
2.0R, 7-day time stop.**

| risk | CAGR | MaxDD | PF | N | WR | Sharpe | Calmar | IS | OOS (PF) |
|---|---|---|---|---|---|---|---|---|---|
| 2.5% | 15.9% | −14.3% | 1.61 | 234 | 53.0% | **1.23** | 1.11 | 14.7% | **18.6%** (1.60) |
| **3.5%** | **22.4%** | **−19.6%** | 1.60 | 234 | 53.0% | 1.23 | 1.14 | 20.7% | **26.2%** (1.59) |
| 4.5% | 28.8% | −24.6% | 1.59 | 234 | 53.0% | 1.23 | 1.17 | 26.7% | 33.8% (1.58) |

**Robustness:**
* Cost stress: 15.9% → 14.7% → 13.3% at 1×/1.5×/2× costs; **PF 1.61 → 1.57 → 1.53**.
* Yearly: 2021 +13%, 2022 **−1%**, 2023 +32%, 2024 +6%, 2025 +34%, 2026 +10% — positive in
  five of six years, worst year −1%.
* Bootstrap (risk 2.5%): median DD −15.8%, 5th pct −26.3%, P(DD>20%) 21%.
* The whole 12h/thr-1.0 neighbourhood holds PF 1.55–1.63 with positive OOS in every cell — a
  robust region, not one lucky configuration.

**Against the previous best (S7):** PF 1.60 vs 1.49, Sharpe 1.23 vs 1.14, OOS CAGR 26.2% vs
24.1%, PF at double costs 1.53 vs 1.43, at comparable CAGR and drawdown.

**One honest caveat.** The traded signal's own rank-IC degrades out of sample much as
`tt_vs_retail`'s did: +0.112 in-sample vs +0.013 out, at h=96. The *strategy* nevertheless
does better out of sample. The reason is that it trades only the tails beyond |z|>1 and shapes
the payoff with a stop and a target, and full-distribution rank-IC says nothing about tail
behaviour. So the stability screen earned its keep by **pointing at an under-explored feature
family**, not by predicting strategy performance directly — the claim "stable IC implies stable
strategy" is not supported by this study's own evidence, in either direction.
