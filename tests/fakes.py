"""Deterministic in-memory providers. Used by the test-suite and by
scripts/offline_demo.py (run the real API with zero API keys / network)."""
import hashlib
from datetime import date, datetime, timedelta

from app.providers.news import NewsQuery, RawArticle
from app.providers.prices import Bar, CompanyProfile

EPOCH = date(2025, 1, 1)

PROFILES = {
    "NVDA": CompanyProfile(
        ticker="NVDA", name="NVIDIA Corporation", sector="Technology", industry="Semiconductors", sector_etf="XLK",
        peers=[{"symbol": "AMD", "name": "Advanced Micro Devices, Inc."}, {"symbol": "AVGO", "name": "Broadcom Inc."}],
    ),
}

# symbol -> {date: forced % move}. Everything else is small seeded noise.
SHOCKS = {
    # 2026-03-10: company-specific pop (peers/sector/market flat)
    # 2026-04-15: industry-wide sell-off (peers + XLK down, SPY mildly down)
    # 2026-05-05: market-wide drop
    "NVDA": {date(2026, 3, 10): 6.0, date(2026, 4, 15): -5.0, date(2026, 5, 5): -3.0, date(2026, 5, 20): 2.4},
    "AMD": {date(2026, 4, 15): -4.2, date(2026, 5, 5): -2.5},
    "AVGO": {date(2026, 4, 15): -3.8, date(2026, 5, 5): -2.2},
    "XLK": {date(2026, 4, 15): -2.1, date(2026, 5, 5): -1.9},
    "SPY": {date(2026, 4, 15): -0.4, date(2026, 5, 5): -1.8},
}

NEWS = [
    # (published, title, url, snippet)
    (datetime(2026, 3, 9, 21, 30), "NVIDIA beats earnings expectations and raises guidance on data-center demand",
     "https://example.com/nvda-earnings", "NVIDIA reported record quarterly revenue after the bell on Monday."),
    (datetime(2026, 3, 10, 9, 0), "Nvidia beats earnings expectations and raises guidance on data center demand",
     "https://syndicated.example.org/nvda-earnings-copy", "Syndicated duplicate of the same story."),
    (datetime(2026, 3, 10, 14, 0), "Analysts raise NVDA price target after blowout quarter",
     "https://example.com/nvda-pt?utm_source=x", "Several brokers lifted targets."),
    (datetime(2026, 3, 10, 12, 0), "Local bakery wins regional award", "https://example.com/bakery", "Nothing to do with chips."),
    (datetime(2026, 4, 14, 22, 0), "US weighs new export curbs on advanced chips; semiconductors slide",
     "https://example.com/export-curbs", "AMD and Broadcom fell in extended trading as the semiconductors group sold off."),
    (datetime(2026, 4, 15, 13, 0), "AMD shares tumble as chip export rules loom",
     "https://example.com/amd-tumble", "Advanced Micro Devices led chipmakers lower."),
    (datetime(2026, 5, 5, 15, 0), "NVIDIA shares fall with broader market", "https://example.com/nvda-market",
     "Stocks dropped across the board."),
]


def business_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _noise(symbol: str, d: date) -> float:
    h = int(hashlib.sha256(f"{symbol}{d}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF - 0.5) * 1.2  # +/-0.6%


class FakePriceProvider:
    def __init__(self):
        self.history_calls: list[tuple[list[str], date, date]] = []
        self.profile_calls = 0

    def get_profile(self, ticker: str, max_peers: int) -> CompanyProfile:
        self.profile_calls += 1
        return PROFILES.get(ticker, CompanyProfile(ticker=ticker, name=ticker))

    def get_history(self, symbols, start, end):
        self.history_calls.append((list(symbols), start, end))
        known = set(SHOCKS) | set(PROFILES)
        out = {}
        for symbol in symbols:
            bars, close = [], 100.0
            if symbol in known:
                # Walk from a fixed epoch so closes are identical across ranges.
                for d in business_days(EPOCH, end):
                    pct = SHOCKS.get(symbol, {}).get(d, _noise(symbol, d))
                    prev, close = close, close * (1 + pct / 100)
                    if d >= start:
                        bars.append(Bar(date=d, open=prev, high=max(prev, close), low=min(prev, close), close=close, volume=1_000_000))
            out[symbol] = bars
        return out


class FakeNewsProvider:
    name = "fake"

    def __init__(self, fail_on: set[date] | None = None):
        self.queries: list[NewsQuery] = []
        self.fail_on = fail_on or set()

    def search(self, query: NewsQuery):
        from app.providers.base import ProviderError

        self.queries.append(query)
        if any(query.start <= d <= query.end for d in self.fail_on):
            raise ProviderError("simulated outage")
        return [
            RawArticle(title=t, url=u, source="example.com", published_at=p, snippet=s)
            for p, t, u, s in NEWS
            if query.start <= p.date() <= query.end
        ]


class FakeLLM:
    model = "fake-llm"

    def __init__(self):
        self.calls: list[tuple[str, list[dict]]] = []

    def complete(self, system, messages, max_tokens=900):
        self.calls.append((system, messages))
        return "The move coincided with strong earnings [1] and analyst upgrades [2]."
