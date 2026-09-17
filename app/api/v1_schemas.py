"""The v1 contract, frozen.

`app/schemas.py` always describes the *current* API (v2) and is what the services
produce. These are the v1 shapes: no macro article category, no `macro_status`, no
`macro_topic`. They are written out rather than derived from the current models
on purpose, so that evolving v2 can never silently change what v1 clients receive.
FastAPI serialises the services' output through them, dropping the v2-only fields.
"""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas import CompanyOut, MarketContext, PricePoint


class ArticleOutV1(BaseModel):
    category: Literal["company", "industry"]
    title: str
    url: str
    source: str | None
    published_at: datetime | None
    snippet: str | None
    relevance: float


class MovementOutV1(BaseModel):
    date: date
    direction: Literal["up", "down"]
    close: float
    prev_close: float
    pct_change: float
    zscore: float | None
    context: MarketContext
    news_status: Literal["fetched", "not_fetched"]
    articles: list[ArticleOutV1]


class MovementQueryV1(BaseModel):
    start: date | None = Field(None, description="Default: end - 90 days")
    end: date | None = Field(None, description="Default: today")
    min_change_pct: float | None = Field(
        None, ge=0, le=100, description="Min absolute daily % change to count as a movement (default 2.0)"
    )
    min_zscore: float | None = Field(
        None, ge=0, description="Also require |move| >= N x the stock's trailing 30-day volatility"
    )
    direction: Literal["any", "up", "down"] = "any"
    news_category: Literal["all", "company", "industry"] = "all"
    min_relevance: float = Field(0.0, ge=0, le=1)
    max_articles: int = Field(10, ge=0, le=20, description="Per movement")
    refresh: bool = Field(False, description="Bypass the cache and re-fetch prices and news")


class ReportQueryV1(MovementQueryV1):
    include_prices: bool = Field(True, description="Include the daily OHLCV series")


class FiltersOutV1(BaseModel):
    min_change_pct: float
    min_zscore: float | None
    direction: Literal["any", "up", "down"]
    news_category: Literal["all", "company", "industry"]
    min_relevance: float
    max_articles: int


class MovementListV1(BaseModel):
    ticker: str
    start: date
    end: date
    filters: FiltersOutV1
    movement_count: int
    movements: list[MovementOutV1]
    warnings: list[str] = []


class StockReportV1(BaseModel):
    company: CompanyOut
    start: date
    end: date
    filters: FiltersOutV1
    movement_count: int
    movements: list[MovementOutV1]
    prices: list[PricePoint] | None = None
    warnings: list[str] = []


class MovementExplanationV1(BaseModel):
    ticker: str
    movement: MovementOutV1
    explanation: str
    mode: Literal["llm", "extractive"]
    model: str | None = None
    cached: bool = False
