# BTCUSDT Systematic Strategy Research

A search for a BTCUSDT strategy meeting: net yearly profit > 300%, at least 100 completed
trades, profit factor > 1.10, maximum drawdown < 20%, realistic risk management, no
look-ahead bias.

Scope: **Binance USDⓈ-M perpetual futures BTCUSDT only.** No spot leg.

**Result: no strategy qualified.** Under a 20% drawdown budget the best futures-only
configuration reaches roughly 20% net annual. Removing the drawdown constraint, the best
single book peaks at 59.6% CAGR (convex trend, growth-optimal size, −70% drawdown) and a
continuous volatility-targeted implementation reaches 100.6% CAGR at −74%.

The binding result is the **growth-optimal leverage curve** (`strategies/s18_kelly.py`):
compound return peaks and then *declines* with size. At 5% risk per trade the convex book
returns +59.6%; at 22% risk the same book returns **−71.6%**. No leverage setting reaches
300%, because compound growth has a maximum that is a property of the return distribution.

* `REPORT.html` — the full write-up: rankings, reproduction specs, and the arithmetic
  showing why the target is not reachable on a single instrument.
* `RESEARCH_LOG.md` — chronological record of all 14 strategies and 4 screen-stage rejections.
* `results/` — raw sweep output for every parameter set tested.

## Layout

| path | what it is |
|---|---|
| `fetch.sh`, `build.py` | download and assemble Binance public archive data |
| `engine/core.py` | bar-by-bar simulator (fills, stops, funding, sizing, liquidation) |
| `engine/weights.py` | continuous-weight simulator for volatility-targeted exposure |
| `engine/indicators.py` | indicator library, all past-only |
| `research/features.py` | 75-feature panel: price, order flow, basis, funding, open interest, positioning |
| `research/ic.py` | rank-IC and decile-economics screens |
| `research/harness.py` | signal→execution alignment with the one-bar lag that prevents look-ahead |
| `research/robust.py` | block bootstrap and cost-stress diagnostics |
| `strategies/` | S01–S22: signal books, convex/pyramided trend, meta-labelling, portfolio construction, growth-optimal leverage curve |

## Reproducing

```sh
pip install pandas numpy pyarrow numba scipy lightgbm scikit-learn
mkdir -p data && cd data
./fetch.sh spot        BTCUSDT 1h  2017-08 2026-08 raw/spot_1h
./fetch.sh futures/um  BTCUSDT 1h  2020-01 2026-08 raw/fut_1h
./fetch.sh futures/um  BTCUSDT 15m 2020-01 2026-08 raw/fut_15m
./fetch.sh futures/um  BTCUSDT 5m  2020-01 2026-08 raw/fut_5m
python build.py
cd .. && python strategies/s07_smart.py
```

Funding rates and the 5-minute positioning metrics come from
`futures/um/monthly/fundingRate/BTCUSDT` and `futures/um/daily/metrics/BTCUSDT`
in the same archive.

## Trading assumptions

Binance USDⓈ-M perpetual BTCUSDT (futures only). 5 bps commission and 3 bps slippage per side (16 bps
round-turn), real historical funding at 00/08/16 UTC, USDT borrow at 8% APR on levered spot
legs. Signals from closed bars only; fills at the open of the next 15-minute execution bar;
stops resolved on 15-minute bars with the stop assumed to fill first when a bar spans both
stop and target. In-sample 2020-01-01→2024-06-30, out-of-sample 2024-07-01→2026-08-31.
