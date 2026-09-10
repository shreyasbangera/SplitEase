# BTCUSDT Systematic Strategy Research

A search for a BTCUSDT strategy meeting: net yearly profit > 300%, at least 100 completed
trades, profit factor > 1.10, maximum drawdown < 20%, realistic risk management, no
look-ahead bias.

**Result: no strategy qualified.** The best configuration found reaches 39.5% net annual
with a 16.2% maximum drawdown, profit factor 1.29 over 622 trades — it clears every gate
except net yearly profit, which it misses by a factor of 7.6.

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
| `strategies/` | S01–S14 plus portfolio construction |

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

Binance USDⓈ-M perpetual BTCUSDT. 5 bps commission and 3 bps slippage per side (16 bps
round-turn), real historical funding at 00/08/16 UTC, USDT borrow at 8% APR on levered spot
legs. Signals from closed bars only; fills at the open of the next 15-minute execution bar;
stops resolved on 15-minute bars with the stop assumed to fill first when a bar spans both
stop and target. In-sample 2020-01-01→2024-06-30, out-of-sample 2024-07-01→2026-08-31.
