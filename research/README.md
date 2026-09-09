# research/ — XAUUSDT & BTCUSDT strategy research

Self-contained quantitative research loop: data preparation, a look-ahead-safe
backtest engine, 13 strategy families, and a validation battery.

**Read [REPORT.md](REPORT.md) for the findings.** Short version: no strategy met
the 500%/yr-at-<20%-drawdown bar (that target implies a Calmar ratio of 25; the
best validated system here reaches 1.9). The best result is a BTC 4-module
portfolio at 34.9%/yr with an 18.7% max drawdown over 2014–2026, and a
well-supported negative result for gold.

## Layout

```
engine/
  prep_data.py   downloads/cleans the public datasets into parquet
  backtest.py    bar-by-bar engine: next-bar-open fills, stops/targets/trailing,
                 fees + slippage + perp funding, fixed-fractional sizing
  indicators.py  ATR, RSI, ADX, Bollinger, Keltner, Donchian, z-score, Hurst...
  scan.py        signal-level edge scanner (excess-over-drift, overlap-adjusted t)
  runner.py      IS/OOS evaluation, parameter sweeps, walk-forward, risk scaling,
                 portfolio combination, and the attempt ledger
  final_eval.py  the final validation battery
strategies/      s01..s13, one file per strategy family, each documenting its
                 research basis in its module docstring
results/
  ledger.jsonl          every logged backtest
  final_modules.json    final validated numbers
  btc_modules_equity.png
```

## Reproducing

```bash
pip install numpy pandas pyarrow matplotlib
python3 engine/prep_data.py     # rebuilds datasets from the public sources
python3 engine/final_eval.py    # module detail, cost sensitivity, portfolio, leverage table
```

`prep_data.py` writes to a scratch directory; adjust `OUT` (and `CLEAN` in
`runner.py`) if you want the parquet files elsewhere.

## Engine validation

The engine reproduces buy-and-hold CAGR exactly for both instruments (gold
10.9%, BTC 44.5%), and random entries at zero cost return a profit factor of
0.91–1.03 — i.e. it leaks no edge of its own.
