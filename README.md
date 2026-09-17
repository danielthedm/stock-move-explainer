# Stock Move Explainer

Give it a ticker; it finds the days the stock made a major move and explains them with
**company news** and **competitor / industry news**, backed by what the market, the sector
ETF and the company's peers did the same day. Query it over REST or ask questions in chat.

## Quickstart

Needs **Python 3.11+** (developed and live-tested on 3.13; macOS's stock `python3` is 3.9 and
will not work). Dependencies are pinned to the versions the live smoke test passed against.

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # any python >= 3.11
pip install -r requirements.txt
cp .env.example .env          # add EXA_API_KEY (or NEWSAPI_API_KEY) + ANTHROPIC_API_KEY (or OPENAI_API_KEY)
uvicorn app.main:app --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
./scripts/smoke_test.sh NVDA  # live end-to-end check of every endpoint
```

Keys are optional by design: with no news key you still get prices, movements and market
context (plus a warning); with no LLM key, chat answers in an extractive (non-LLM) mode.

No keys / no network? `python -m scripts.offline_demo` runs the real API on port 8001 against
deterministic fake upstreams (`curl "localhost:8001/stocks/NVDA?start=2026-03-01&end=2026-05-31"`).

Tests: `pytest -q` (41 tests, ~2s, no network).

## API

### `GET /stocks/{ticker}` - all stock + news data
The first call for a ticker/range ingests from upstream (seconds); after that it is served from SQLite (~25 ms).

| Filter | Default | Meaning |
|---|---|---|
| `start`, `end` | last 90 days | Date window (max 730 days) |
| `min_change_pct` | `2.0` | Min absolute close-to-close % change to count as a movement |
| `min_zscore` | - | Also require the move to be >= N x the stock's own trailing 30-day volatility |
| `direction` | `any` | `up` / `down` |
| `news_category` | `all` | `company` / `industry` |
| `min_relevance` | `0` | Drop articles scoring below this (0-1) |
| `max_articles` | `10` | Per movement |
| `include_prices` | `true` | Include the daily OHLCV series |
| `refresh` | `false` | Bypass the cache, re-fetch prices and news |

```bash
curl "localhost:8000/stocks/NVDA?start=2026-01-01&end=2026-06-30&direction=down&min_change_pct=3&include_prices=false"
```
Each movement carries `pct_change`, `zscore`, its `articles`, and a `context` block:
`market_pct` (SPY), `sector_pct` (sector ETF), per-peer moves, `excess_vs_sector_pct` and a
price-based `driver`: `company_specific` | `industry_wide` | `market_wide`.

### `POST /chat`
```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{
  "ticker": "NVDA",
  "message": "Why did it drop in April? Was it just NVDA or all chip stocks?",
  "start": "2026-03-01", "end": "2026-05-31",
  "history": []
}'
```
Returns `answer` with inline `[n]` citations and `sources` (only the articles actually cited).
Stateless: the client sends previous turns back in `history`.

### `GET /stocks/{ticker}/movements/{date}`
One day in detail plus a 2-4 sentence LLM explanation, generated on first request and cached.

## How it works

```
            GET /stocks/{t}        POST /chat        GET .../movements/{date}
                   \                   |                   /
                    +--------- ensure_data() ---------+          <- read-through cache, per-ticker lock
                    |  1. company profile + peers + sector ETF      (yfinance; 30-day TTL)
                    |  2. prices for ticker, peers, ETF, SPY        (one batched download, only missing ranges)
                    |  3. news for un-enriched movement days        (2 searches/day: company + industry, parallel)
                    |     -> dedupe, relevance-rank, categorise
                    +------------------+------------------+
                                    SQLite
                                       |
                    analysis.build_movements()  <- one query layer shared by REST and chat
                       pct change, z-score, market/sector/peer context, driver
                                       |
                        REST JSON   or   numbered context -> LLM -> cited answer
```

`app/providers/` hides every vendor behind a `Protocol` (prices, news, LLM);
`app/services/` holds the logic (`movements` and `ranking` are pure functions);
`app/main.py` is a thin HTTP layer.
