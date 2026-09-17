"""Run the real API against deterministic fake upstreams: no keys, no network.

    python -m scripts.offline_demo          # http://127.0.0.1:8001/docs
    curl "localhost:8001/v2/stocks/NVDA?start=2026-03-01&end=2026-05-31"

Only NVDA (with peers AMD/AVGO) has scripted data; chat runs in extractive mode
unless an LLM key is set in the environment/.env.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./data/offline_demo.db")

import uvicorn

from app import deps
from app.main import app
from tests.fakes import FakeNewsProvider, FakePriceProvider

prices, news = FakePriceProvider(), FakeNewsProvider()
app.dependency_overrides[deps.get_price_provider] = lambda: prices
app.dependency_overrides[deps.get_news_provider] = lambda: news

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
