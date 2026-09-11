# Putting it on a real URL

## What I cannot do, and why

I cannot host this for you, and no page hosted on claude.ai can trade your
Binance account. An artifact's runtime can store data, sync viewers, call your
claude.ai connectors and ask Claude — there is no capability that makes an
outbound call to an exchange API, and browser JavaScript cannot sign a Binance
request without putting your secret in the browser. So a "website that trades"
always means **a server you control**.

The good news is that the server already exists. `webapp/` is a complete web
app; it has only ever been bound to localhost.

## The architecture that actually works on free tiers

Split the two jobs:

| job | where | why |
|---|---|---|
| **placing trades** | GitHub Actions, twice a day | already built; needs nothing running |
| **the website** | Render / Fly / Railway free tier | if it sleeps, nothing breaks |

This is the trick that makes free hosting viable. Free web services sleep when
idle, which would be fatal if the website were the thing trading — but it is
not. It is a viewer. The bot is `.github/workflows/btcusdt-book.yml`.

## Render — the shortest path

1. Push this repo to GitHub (done).
2. render.com → **New → Blueprint** → pick the repo. It reads `render.yaml`.
3. Render generates `APP_PASSWORD` — open the service's **Environment** tab and
   copy it. That is your login, username `trader`.
4. Paste `BINANCE_TEST_KEY` and `BINANCE_TEST_SECRET` in the same tab.
5. Deploy. You get `https://btcusdt-book-xxxx.onrender.com`.

First load after idle takes ~30s while the service wakes. That is the free tier,
and it does not matter for a dashboard.

## Fly.io

```bash
fly launch --no-deploy          # reads fly.toml
fly volumes create data --size 1
fly secrets set APP_PASSWORD='...' BINANCE_TEST_KEY='...' BINANCE_TEST_SECRET='...'
fly deploy
```

Machines auto-stop when idle and start on the next request.

## Any VPS — the version with no caveats

```bash
docker build -t btcbook -f research/btcusdt-quant/webapp/Dockerfile research/btcusdt-quant
docker run -d --name btcbook --restart unless-stopped \
  -p 127.0.0.1:8000:8000 -v /opt/bookdata:/data \
  -e BOT_MODE=test -e APP_PASSWORD='...' \
  -e BINANCE_TEST_KEY='...' -e BINANCE_TEST_SECRET='...' \
  btcbook
```

Then put Caddy or nginx in front for HTTPS, or reach it over an SSH tunnel and
skip the public URL entirely.

## The lock

The app can arm itself and place orders, so an open port is an open account.

* Every route including the API is behind HTTP Basic, compared with
  `secrets.compare_digest`.
* `APP_PASSWORD` has **no default**. With none set the app **refuses to start**
  on anything but localhost — forgetting it cannot quietly publish an open bot.
* Set `APP_PUBLIC=yes` only if you genuinely intend an unauthenticated bot on a
  public interface. There is no good reason to.

One operator, one password. If several people need access, put a real identity
proxy in front rather than growing this.

## Order of operations

1. Deploy with `BOT_MODE=test` and **no** Binance secrets. The dashboard loads,
   computes decisions from public data, and can place nothing.
2. Add the testnet secrets. Leave it **disarmed** and watch a week of decisions.
3. Arm it, and let GitHub Actions trade the testnet account.
4. Only after a quarter of that, with the signals checked against
   `strategies/s87_combined.py`, consider real money — and then run it on a VPS
   you control, with keys in the environment, never through the browser.
