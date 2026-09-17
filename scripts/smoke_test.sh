#!/usr/bin/env bash
# Live smoke test against a running server (real yfinance / news / LLM).
#   uvicorn app.main:app --port 8000 &   then   ./scripts/smoke_test.sh [TICKER] [BASE_URL]
set -euo pipefail
T="${1:-NVDA}"; BASE="${2:-http://127.0.0.1:8000}"
# sed (not head) so the whole response is consumed: head closes the pipe early
# and, under pipefail, a long real response kills the script with SIGPIPE.
pp() { python3 -m json.tool | sed -n "1,${1:-40}p"; }

echo "== health";                curl -fsS "$BASE/health" | pp
echo "== report (first call ingests; may take ~10-30s)"
curl -fsS "$BASE/stocks/$T?include_prices=false&max_articles=2" | pp 80
echo "== filters: down days >= 3%, company news only"
curl -fsS "$BASE/stocks/$T?direction=down&min_change_pct=3&news_category=company&include_prices=false&max_articles=1" | pp 40
echo "== chat"
curl -fsS -X POST "$BASE/chat" -H 'content-type: application/json' \
  -d "{\"ticker\":\"$T\",\"message\":\"What was the biggest move in this period and what likely caused it? Was it company-specific?\"}" | pp 40
echo "== unknown ticker -> 404"
curl -s -o /dev/null -w "%{http_code}\n" "$BASE/stocks/ZZZZZZ"
