# Local trading dashboard

A web app you run **on your own machine**. Dashboard at `http://127.0.0.1:8000`.

## Why it is not a hosted website

Two independent reasons, either of which is decisive:

* **Signing.** Binance requires every private call to be HMAC-signed with your API
  secret. Browser JavaScript can only do that by having the secret *in the
  browser*, where any extension, any bug and anyone with the page can read it.
* **CORS.** Binance does not serve `fapi.binance.com` to browser origins, so a
  hosted page cannot reach it even if you were willing to ship the secret.

So the app runs locally and your keys never leave your machine. This app never
accepts a key through the browser, never writes one to disk, and never logs one
— it reads them from the environment and shows you the last four characters so
you can tell which key is loaded.

## Install and run

```bash
pip install fastapi uvicorn pandas numpy pyarrow
export BOOK_STORE=~/quant/data/live
python live/fetch.py seed --months 36      # once; the app needs the panels
python -m webapp.app
```

Open `http://127.0.0.1:8000`.

## The three modes

| `BOT_MODE` | what it does | needs |
|---|---|---|
| **`paper`** *(default)* | no exchange contact at all; fills simulated at the real price with the backtest's own costs — 5 bps commission, 3 bps adverse slippage | nothing |
| `test` | Binance USD-M futures **testnet**; fake money, real order plumbing | `BINANCE_TEST_KEY`, `BINANCE_TEST_SECRET` |
| `live` | **real money** | `BINANCE_KEY`, `BINANCE_SECRET`, `ALLOW_LIVE=yes`, **and** typing `TRADE REAL MONEY` in the UI |

Paper is the default deliberately. It is the only mode whose numbers are
comparable to the backtest — testnet has its own thin book and its own prices, so
a testnet fill tells you nothing about what a trade would really have cost.

**Live mode has three independent locks:** the environment variable, the
credentials, and the typed confirmation. Changing strategy disarms. The kill
switch cancels every order, closes the position at market, and disarms.

## Making the key safely

When you create the production key: **enable futures trading, disable
withdrawals, and set an IP allowlist.** There is no reason this app ever needs
withdrawal permission, and a key without it cannot move your coins off the
exchange no matter what goes wrong.

## Adding a strategy

Drop a file in `webapp/strategies/`. It is discovered automatically.

```python
from .base import Strategy, Sleeve, Decision

class MyBook(Strategy):
    name = "mybook"
    description = "One line, shown in the dropdown."
    decision_tf = "12h"
    warmup_bars = 200

    def needs(self):
        return ["panel_12h"]

    def decide(self, panels, equity, risk):
        df = panels["panel_12h"]
        px = float(df.close.iloc[-1])
        sma = df.close.rolling(50).mean().iloc[-1]
        if px <= sma:
            return Decision([], note="flat")
        stop_dist = 0.03 * px
        qty = self.size_for(equity, risk, 1.0, stop_dist, px)
        return Decision([Sleeve("above the 50", qty, stop=px - stop_dist)],
                        diagnostics={"close": px, "sma50": float(sma)})
```

A strategy answers exactly one question: *given the data and my risk budget,
what should I be holding?* It never asks what is currently held. The engine sums
the sleeves, compares against the real position, and moves the difference — so
strategies stay stateless and the awkward parts live in one place.

Return several sleeves when the strategy holds several configurations at once, as
V7 does. Each sleeve's stop becomes its own reduce-only order against the single
netted position. `webapp/strategies/v7.py` also contains a deliberately trivial
`BuyAndHold` as a second worked example.

## What is verified and what is not

**Verified:** the strategy interface, the engine's netting and ladder logic, and
the paper broker — including that a stop fires at *its own price* rather than the
ticker (the first version of the paper broker reported a 150 USDT stop-out as a
4 USDT one, which is exactly the kind of error that makes a paper record
worthless).

**Not exercised:** `webapp/broker/binance.py`. Every call in it is written
against Binance's documented API and has never returned a real response, because
the endpoints are unreachable from the environment this was built in. Run it in
`test` mode first and read every reply by hand before going near `live`.

## Risk

V7's own bootstrap puts a drawdown worse than 20% at **34% probability** at 8%
risk and **98%** at the size behind its headline return. The backtest's realised
−19.99% was a favourable draw, not an expectation. Start in `paper`, stay there
for a quarter including one quarterly re-selection, and compare your realised
signals against `strategies/s87_combined.py` before trusting anything.
