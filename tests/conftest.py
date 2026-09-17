import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"  # must be set before app import
for key in ("EXA_API_KEY", "NEWSAPI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(key, None)

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import deps
from app.db import Base, get_db
from app.main import app
from tests.fakes import FakeLLM, FakeNewsProvider, FakePriceProvider


@pytest.fixture
def ctx():
    """Real app + real SQL (fresh in-memory DB) + fake upstream providers."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from app import models  # noqa: F401

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    def _db():
        with Session() as s:
            yield s

    class Ctx:
        prices, news, llm = FakePriceProvider(), FakeNewsProvider(), FakeLLM()

    c = Ctx()
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[deps.get_price_provider] = lambda: c.prices
    app.dependency_overrides[deps.get_news_provider] = lambda: c.news
    app.dependency_overrides[deps.get_llm] = lambda: c.llm
    c.client = TestClient(app)
    c.Session = Session
    yield c
    app.dependency_overrides.clear()
