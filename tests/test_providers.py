import json
from datetime import date

import httpx
import pandas as pd
import pytest

from app.providers.base import ProviderError
from app.providers.llm import AnthropicLLM, OpenAILLM
from app.providers.news import ExaNewsProvider, NewsApiProvider, NewsQuery
from app.providers.prices import YFinancePriceProvider, frame_to_bars

Q = NewsQuery(terms=["NVIDIA", "NVDA"], text="NVIDIA (NVDA) stock news", start=date(2026, 3, 9), end=date(2026, 3, 11), limit=7)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_exa_request_and_parse():
    seen = {}

    def handler(req: httpx.Request):
        seen["headers"], seen["body"], seen["url"] = req.headers, json.loads(req.content), str(req.url)
        return httpx.Response(200, json={"requestId": "r", "results": [
            {"id": "1", "title": " Nvidia\n soars ", "url": "https://www.reuters.com/a", "publishedDate": "2026-03-10T13:00:00.000Z",
             "author": "x", "highlights": ["Shares rose 6%.", "Guidance raised."]},
            {"id": "2", "title": None, "url": "https://bad.example"},
        ]})

    arts = ExaNewsProvider("k", _client(handler)).search(Q)
    assert seen["url"] == "https://api.exa.ai/search" and seen["headers"]["x-api-key"] == "k"
    b = seen["body"]
    assert b["category"] == "news" and b["numResults"] == 7 and b["contents"] == {"highlights": True}
    assert b["startPublishedDate"] == "2026-03-09T00:00:00.000Z" and b["endPublishedDate"] == "2026-03-11T23:59:59.999Z"
    assert len(arts) == 1
    a = arts[0]
    assert (a.title, a.source) == ("Nvidia soars", "reuters.com")
    assert a.published_at.isoformat() == "2026-03-10T13:00:00" and "Guidance raised." in a.snippet


def test_newsapi_request_and_parse():
    seen = {}

    def handler(req: httpx.Request):
        seen["params"], seen["headers"] = dict(req.url.params), req.headers
        return httpx.Response(200, json={"status": "ok", "articles": [
            {"source": {"id": None, "name": "CNBC"}, "title": "Nvidia jumps", "url": "https://cnbc.com/a",
             "publishedAt": "2026-03-10T13:00:00Z", "description": "desc", "content": "..."},
            {"source": {"name": "x"}, "title": "[Removed]", "url": "https://removed.com"},
        ]})

    arts = NewsApiProvider("k", _client(handler)).search(Q)
    p = seen["params"]
    assert p["q"] == '"NVIDIA" OR "NVDA"' and p["from"] == "2026-03-09" and p["to"].startswith("2026-03-11")
    assert p["pageSize"] == "7" and seen["headers"]["x-api-key"] == "k"
    assert [(a.title, a.source, a.snippet) for a in arts] == [("Nvidia jumps", "CNBC", "desc")]


def test_http_errors_and_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        return httpx.Response(429 if calls["n"] < 3 else 200, json={"results": []})

    assert ExaNewsProvider("k", _client(flaky)).search(Q) == [] and calls["n"] == 3
    with pytest.raises(ProviderError, match="401"):
        ExaNewsProvider("k", _client(lambda r: httpx.Response(401, text="bad key"))).search(Q)

    def down(req):
        raise httpx.ConnectError("boom")

    with pytest.raises(ProviderError, match="unreachable"):
        ExaNewsProvider("k", _client(down)).search(Q)


def test_anthropic_and_openai_clients():
    seen = {}

    def anthropic(req):
        seen["a"] = (req.headers, json.loads(req.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": " hello "}]})

    def openai(req):
        seen["o"] = (req.headers, json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    msgs = [{"role": "user", "content": "q"}]
    assert AnthropicLLM("ak", client=_client(anthropic)).complete("sys", msgs) == "hello"
    h, b = seen["a"]
    assert h["x-api-key"] == "ak" and h["anthropic-version"] == "2023-06-01"
    assert b["system"] == "sys" and b["messages"] == msgs and b["max_tokens"] == 900

    assert OpenAILLM("ok", client=_client(openai)).complete("sys", msgs) == "hi"
    h, b = seen["o"]
    assert h["authorization"] == "Bearer ok" and b["messages"][0] == {"role": "system", "content": "sys"}


def _yf_frame(symbols):
    idx = pd.DatetimeIndex(["2026-03-09", "2026-03-10"], name="Date")
    cols = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Price"])
    data = [[1.0, 2.0, 0.5, 1.5, 100] * len(symbols), [1.5, 2.5, 1.0, float("nan"), 0] * len(symbols)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_yfinance_frame_normalisation():
    bars = frame_to_bars(_yf_frame(["NVDA", "SPY"]), ["NVDA", "SPY", "MISSING"])
    assert bars["MISSING"] == [] and len(bars["NVDA"]) == 1
    b = bars["SPY"][0]
    assert (b.date, b.close, b.volume) == (date(2026, 3, 9), 1.5, 100)
    flat = _yf_frame(["NVDA"]).droplevel(0, axis=1)
    assert len(frame_to_bars(flat, ["NVDA"])["NVDA"]) == 1
    assert frame_to_bars(pd.DataFrame(), ["NVDA"]) == {"NVDA": []}


def test_yfinance_provider_wiring(monkeypatch):
    import yfinance as yf

    captured = {}

    def fake_download(**kw):
        captured.update(kw)
        return _yf_frame(["NVDA"])

    class FakeTicker:
        def __init__(self, t): ...
        info = {"longName": "NVIDIA Corporation", "sector": "Technology", "sectorKey": "technology",
                "industry": "Semiconductors", "industryKey": "semiconductors"}

    class FakeIndustry:
        def __init__(self, key): assert key == "semiconductors"
        top_companies = pd.DataFrame(
            {"name": ["NVIDIA", "Broadcom Inc.", float("nan"), "Intel"], "rating": ["Buy"] * 4, "market weight": [0.4, 0.2, 0.1, 0.05]},
            index=pd.Index(["NVDA", "AVGO", "AMD", "INTC"], name="symbol"),
        )

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yf, "Industry", FakeIndustry)

    p = YFinancePriceProvider()
    bars = p.get_history(["NVDA"], date(2026, 3, 9), date(2026, 3, 10))
    assert captured["end"] == "2026-03-11" and captured["auto_adjust"] is True and captured["group_by"] == "ticker"
    assert captured["threads"] is False
    assert len(bars["NVDA"]) == 1

    prof = p.get_profile("NVDA", max_peers=2)
    assert (prof.name, prof.sector_etf) == ("NVIDIA Corporation", "XLK")
    assert prof.peers == [{"symbol": "AVGO", "name": "Broadcom Inc."}, {"symbol": "AMD", "name": "AMD"}]

    class Broken:
        def __init__(self, t): raise RuntimeError("yahoo down")

    monkeypatch.setattr(yf, "Ticker", Broken)
    assert p.get_profile("NVDA", 2).name == "NVDA"


def test_yfinance_retries_silently_dropped_symbols(monkeypatch):
    import yfinance as yf

    calls = []

    def flaky_download(**kw):
        calls.append(kw["tickers"])
        return _yf_frame(["SPY"] if len(calls) == 1 else kw["tickers"])

    monkeypatch.setattr(yf, "download", flaky_download)
    bars = YFinancePriceProvider().get_history(["NVDA", "SPY"], date(2026, 3, 9), date(2026, 3, 10))
    assert calls == [["NVDA", "SPY"], ["NVDA"]]
    assert len(bars["NVDA"]) == 1 and len(bars["SPY"]) == 1
