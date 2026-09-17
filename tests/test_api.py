"""End-to-end through HTTP -> ingest -> SQL -> analysis, with fake upstreams."""
from datetime import date

from tests.fakes import FakeNewsProvider

RANGE = {"start": "2026-03-01", "end": "2026-05-31"}


def _get(ctx, **params):
    r = ctx.client.get("/stocks/nvda", params={**RANGE, **params})
    assert r.status_code == 200, r.text
    return r.json()


def test_full_report(ctx):
    body = _get(ctx)
    assert body["company"]["name"] == "NVIDIA Corporation" and body["company"]["sector_etf"] == "XLK"
    assert [p["symbol"] for p in body["company"]["peers"]] == ["AMD", "AVGO"]
    moves = {m["date"]: m for m in body["movements"]}
    assert list(moves) == ["2026-03-10", "2026-04-15", "2026-05-05", "2026-05-20"]
    assert body["movement_count"] == 4 and body["warnings"] == []

    # prices: only the requested window, first day still has pct_change (lookback buffer worked)
    assert body["prices"][0]["date"] == "2026-03-02" and body["prices"][0]["pct_change"] is not None
    assert body["prices"][-1]["date"] == "2026-05-29"

    # company-specific: earnings the night before is found, deduped, junk dropped
    m = moves["2026-03-10"]
    assert m["pct_change"] == 6.0 and m["direction"] == "up" and m["zscore"] > 5
    assert m["context"]["driver"] == "company_specific" and m["news_status"] == "fetched"
    titles = [a["title"] for a in m["articles"]]
    assert titles[0].startswith("NVIDIA beats earnings") and len(titles) == 2
    assert all(a["category"] == "company" for a in m["articles"])

    # industry-wide: peers + sector fell, industry articles attached
    m = moves["2026-04-15"]
    assert m["context"]["driver"] == "industry_wide"
    assert m["context"]["sector_pct"] == -2.1 and m["context"]["peer_median_pct"] == -4.0
    assert m["context"]["excess_vs_sector_pct"] == -2.9
    assert {a["category"] for a in m["articles"]} == {"industry"}
    assert any("export curbs" in a["title"] for a in m["articles"])

    assert moves["2026-05-05"]["context"]["driver"] == "market_wide"
    assert moves["2026-05-20"]["articles"] == [] and moves["2026-05-20"]["news_status"] == "fetched"

    # two searches (company + industry) per movement day, windowed prev-trading-day .. day+1
    assert len(ctx.news.queries) == 8
    q = next(q for q in ctx.news.queries if q.end == date(2026, 3, 11))
    assert q.start == date(2026, 3, 9)


def test_monday_move_searches_back_over_the_weekend(ctx):
    ctx.client.get("/stocks/NVDA/movements/2026-03-09")  # a Monday
    assert {q.start for q in ctx.news.queries} == {date(2026, 3, 6)}


def test_filters(ctx):
    assert [m["date"] for m in _get(ctx, direction="down")["movements"]] == ["2026-04-15", "2026-05-05"]
    assert [m["date"] for m in _get(ctx, min_change_pct=4)["movements"]] == ["2026-03-10", "2026-04-15"]
    assert _get(ctx, min_change_pct=50)["movements"] == []
    assert "prices" not in _get(ctx, include_prices="false") or _get(ctx, include_prices="false")["prices"] is None

    by_cat = _get(ctx, news_category="industry")
    assert all(a["category"] == "industry" for m in by_cat["movements"] for a in m["articles"])
    assert all(len(m["articles"]) <= 1 for m in _get(ctx, max_articles=1)["movements"])
    strict = _get(ctx, min_relevance=0.9)
    assert all(a["relevance"] >= 0.9 for m in strict["movements"] for a in m["articles"])
    zs = _get(ctx, min_zscore=6)
    assert all(abs(m["zscore"]) >= 6 for m in zs["movements"]) and len(zs["movements"]) < 4


def test_second_call_is_served_from_cache(ctx):
    first = _get(ctx)
    calls = (len(ctx.prices.history_calls), ctx.prices.profile_calls, len(ctx.news.queries))
    second = _get(ctx)
    assert (len(ctx.prices.history_calls), ctx.prices.profile_calls, len(ctx.news.queries)) == calls
    assert first == second
    # narrower window: still cached
    _get(ctx, start="2026-04-01", end="2026-04-30")
    assert len(ctx.prices.history_calls) == calls[0]
    # wider window: one incremental fetch, already-enriched days are not searched again
    ctx.client.get("/stocks/NVDA", params={"start": "2026-01-01", "end": "2026-05-31"})
    assert len(ctx.prices.history_calls) == calls[0] + 1
    # refresh forces upstream
    _get(ctx, refresh="true")
    assert len(ctx.news.queries) == calls[2] + 8


def test_lower_threshold_later_enriches_only_new_days(ctx):
    _get(ctx, min_change_pct=4)
    assert len(ctx.news.queries) == 4
    _get(ctx)  # default 2% -> two more movement days
    assert len(ctx.news.queries) == 8


def test_news_quota_cap_prioritises_largest_moves(ctx):
    from app.config import get_settings
    from app.main import app

    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"max_news_movements": 2})
    body = _get(ctx)
    status = {m["date"]: m["news_status"] for m in body["movements"]}
    assert status == {"2026-03-10": "fetched", "2026-04-15": "fetched", "2026-05-05": "not_fetched", "2026-05-20": "not_fetched"}
    assert "2 smaller movement day(s)" in body["warnings"][0]
    assert all(s == "fetched" for s in (m["news_status"] for m in _get(ctx)["movements"]))  # second call continues


def test_news_outage_degrades_and_retries(ctx):
    from app import deps
    from app.main import app

    flaky = FakeNewsProvider(fail_on={date(2026, 3, 10)})
    app.dependency_overrides[deps.get_news_provider] = lambda: flaky
    body = _get(ctx)
    moves = {m["date"]: m for m in body["movements"]}
    assert moves["2026-03-10"]["news_status"] == "not_fetched" and moves["2026-04-15"]["news_status"] == "fetched"
    assert any("2026-03-10" in w for w in body["warnings"])
    app.dependency_overrides[deps.get_news_provider] = lambda: ctx.news
    assert {m["news_status"] for m in _get(ctx)["movements"]} == {"fetched"}  # healed on next request


def test_no_news_provider_still_serves_prices(ctx):
    from app import deps
    from app.main import app

    app.dependency_overrides[deps.get_news_provider] = lambda: None
    body = _get(ctx)
    assert body["movement_count"] == 4 and "No news provider configured" in body["warnings"][0]


def test_validation_and_errors(ctx):
    assert ctx.client.get("/stocks/ZZZZ", params=RANGE).status_code == 404
    assert ctx.client.get("/stocks/NVDA", params=RANGE).status_code == 200  # a 404 must not poison later requests
    assert ctx.client.get("/stocks/NV$DA").status_code == 422
    assert ctx.client.get("/stocks/NVDA", params={"start": "2026-05-01", "end": "2026-04-01"}).status_code == 422
    assert ctx.client.get("/stocks/NVDA", params={"start": "2020-01-01", "end": "2026-04-01"}).status_code == 422
    assert ctx.client.get("/stocks/NVDA", params={"min_change_pct": -1}).status_code == 422
    assert ctx.client.get("/stocks/NVDA", params={"direction": "sideways"}).status_code == 422
    assert ctx.client.get("/health").json()["status"] == "ok"


def test_upstream_price_failure_is_502(ctx):
    from app.providers.base import ProviderError

    def boom(*a):
        raise ProviderError("yahoo down")

    ctx.prices.get_history = boom
    r = ctx.client.get("/stocks/NVDA", params=RANGE)
    assert r.status_code == 502 and "yahoo down" in r.json()["detail"]


def test_chat_llm_mode(ctx):
    r = ctx.client.post("/chat", json={
        "ticker": "nvda", "message": "Why did it jump in March?", **RANGE,
        "history": [{"role": "assistant", "content": "stray"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "llm" and body["movements_considered"] == 4
    assert [s["ref"] for s in body["sources"]] == [1, 2]  # only the cited ones
    assert body["sources"][0]["url"] == "https://example.com/nvda-earnings"

    system, messages = ctx.llm.calls[0]
    assert "ONLY" in system and messages[0] == {"role": "user", "content": "hi"}  # leading assistant turn dropped
    prompt = messages[-1]["content"]
    assert "Question: Why did it jump in March?" in prompt
    assert "## 2026-03-10: NVDA +6.00%" in prompt and "price-based driver: company_specific" in prompt
    assert "## 2026-04-15: NVDA -5.00%" in prompt and "AMD -4.20%" in prompt and "price-based driver: industry_wide" in prompt
    assert "[1] [company] (2026-03-09, example.com) NVIDIA beats earnings" in prompt
    assert "[industry]" in prompt and "(no relevant articles found)" in prompt


def test_chat_extractive_fallback_and_validation(ctx):
    from app import deps
    from app.main import app

    app.dependency_overrides[deps.get_llm] = lambda: None
    body = ctx.client.post("/chat", json={"ticker": "NVDA", "message": "what happened?", **RANGE}).json()
    assert body["mode"] == "extractive" and "2026-03-10: +6.00%" in body["answer"] and body["sources"]
    assert ctx.client.post("/chat", json={"ticker": "NVDA", "message": ""}).status_code == 422
    assert ctx.client.post("/chat", json={"ticker": "ZZZZ", "message": "hi", **RANGE}).status_code == 404


def test_chat_llm_failure_is_502(ctx):
    from app.providers.base import ProviderError

    def boom(*a, **k):
        raise ProviderError("llm down")

    ctx.llm.complete = boom
    assert ctx.client.post("/chat", json={"ticker": "NVDA", "message": "why?", **RANGE}).status_code == 502


def test_explain_movement_is_cached(ctx):
    r = ctx.client.get("/stocks/NVDA/movements/2026-03-10")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "llm" and body["cached"] is False and "[1]" in body["explanation"]
    assert body["movement"]["pct_change"] == 6.0 and len(body["movement"]["articles"]) == 2
    again = ctx.client.get("/stocks/NVDA/movements/2026-03-10").json()
    assert again["cached"] is True and len(ctx.llm.calls) == 1
    assert ctx.client.get("/stocks/NVDA/movements/2026-03-08").status_code == 404  # Sunday
    assert ctx.client.get("/stocks/NVDA/movements/2999-01-01").status_code == 422
