#!/usr/bin/env bash
set -euo pipefail
T="${1:-NVDA}"; BASE="${2:-http://127.0.0.1:8000}"; V2="$BASE/v2"; V1="$BASE/v1"
pp() { python3 -m json.tool | sed -n "1,${1:-40}p"; }

echo "== health";                curl -fsS "$BASE/health" | pp
echo "== v2 report (first call ingests; may take ~10-45s)"
curl -fsS "$V2/stocks/$T?include_prices=false&max_articles=3" | pp 80
echo "== v2 movements collection: down days >= 3%, company news only"
curl -fsS "$V2/stocks/$T/movements?direction=down&min_change_pct=3&news_category=company&max_articles=1" | pp 40
echo "== v2 macro/political news only (market-wide and industry-wide days)"
curl -fsS "$V2/stocks/$T/movements?news_category=macro&max_articles=2" \
  | python3 -c 'import sys, json
for m in json.load(sys.stdin)["movements"]:
    print(m["date"], m["pct_change"], m["context"]["driver"], m["macro_status"], [(a["macro_topic"], a["title"][:60]) for a in m["articles"]])'
echo "== v2 chat"
curl -fsS -X POST "$V2/chat" -H 'content-type: application/json' \
  -d "{\"ticker\":\"$T\",\"message\":\"What was the biggest move in this period and what likely caused it? Was it company-specific, industry-wide or macro?\"}" | pp 40
echo "== v1 is frozen: no macro category, macro filter rejected (expect 'company industry' then 422)"
curl -fsS "$V1/stocks/$T/movements" \
  | python3 -c 'import sys, json; print(*sorted({a["category"] for m in json.load(sys.stdin)["movements"] for a in m["articles"]}))'
curl -s -o /dev/null -w "%{http_code}\n" "$V1/stocks/$T/movements?news_category=macro"
echo "== unknown ticker -> 404"
curl -s -o /dev/null -w "%{http_code}\n" "$V2/stocks/ZZZZZZ"
