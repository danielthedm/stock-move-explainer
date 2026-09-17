"""Run the API on port 8001 against fake upstreams: python -m scripts.offline_demo"""
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
