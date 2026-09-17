from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class PeerOut(BaseModel):
    symbol: str
    name: str


class CompanyOut(BaseModel):
    ticker: str
    name: str
    sector: str | None
    industry: str | None
    sector_etf: str | None
    peers: list[PeerOut]


class PricePoint(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    pct_change: float | None


class PeerMove(BaseModel):
    symbol: str
    pct_change: float


class MarketContext(BaseModel):
    market_symbol: str
    market_pct_change: float | None
    sector_etf: str | None
    sector_pct_change: float | None
    peers: list[PeerMove]
    peer_median_pct_change: float | None
    excess_vs_sector_pct: float | None
    driver: Literal["company_specific", "industry_wide", "market_wide"]


class ArticleOut(BaseModel):
    category: Literal["company", "industry", "macro"]
    title: str
    url: str
    source: str | None
    published_at: datetime | None
    snippet: str | None
    relevance: float
    macro_topic: Literal["monetary_policy", "economy", "trade", "geopolitics", "regulation"] | None = None


class MovementOut(BaseModel):
    date: date
    direction: Literal["up", "down"]
    close: float
    prev_close: float
    pct_change: float
    zscore: float | None
    context: MarketContext
    news_status: Literal["fetched", "not_fetched"]
    macro_status: Literal["fetched", "not_fetched", "not_applicable"] = "not_applicable"
    articles: list[ArticleOut]


class MovementQuery(BaseModel):
    start: date | None = Field(None, description="Default: end - 90 days")
    end: date | None = Field(None, description="Default: today")
    min_change_pct: float | None = Field(
        None, ge=0, le=100, description="Min absolute daily % change to count as a movement (default 2.0)"
    )
    min_zscore: float | None = Field(
        None, ge=0, description="Also require |move| >= N x the stock's trailing 30-day volatility"
    )
    direction: Literal["any", "up", "down"] = "any"
    news_category: Literal["all", "company", "industry", "macro"] = "all"
    min_relevance: float = Field(0.0, ge=0, le=1)
    max_articles: int = Field(10, ge=0, le=20, description="Per movement")
    refresh: bool = Field(False, description="Bypass the cache and re-fetch prices and news")


class ReportQuery(MovementQuery):
    include_prices: bool = Field(True, description="Include the daily OHLCV series")


class FiltersOut(BaseModel):
    min_change_pct: float
    min_zscore: float | None
    direction: Literal["any", "up", "down"]
    news_category: Literal["all", "company", "industry", "macro"]
    min_relevance: float
    max_articles: int


class MovementList(BaseModel):
    ticker: str
    start: date
    end: date
    filters: FiltersOut
    movement_count: int
    movements: list[MovementOut]
    warnings: list[str] = []


class StockReport(BaseModel):
    company: CompanyOut
    start: date
    end: date
    filters: FiltersOut
    movement_count: int
    movements: list[MovementOut]
    prices: list[PricePoint] | None = None
    warnings: list[str] = []


class MovementExplanation(BaseModel):
    ticker: str
    movement: MovementOut
    explanation: str
    mode: Literal["llm", "extractive"]
    model: str | None = None
    cached: bool = False


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    ticker: str = Field(examples=["NVDA"])
    message: str = Field(min_length=1, max_length=4000, examples=["Why did the stock drop in April?"])
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    start: date | None = None
    end: date | None = None
    min_change_pct: float | None = Field(default=None, ge=0)


class SourceOut(BaseModel):
    ref: int
    title: str
    url: str
    source: str | None
    move_date: date


class ChatResponse(BaseModel):
    ticker: str
    answer: str
    mode: Literal["llm", "extractive"]
    sources: list[SourceOut]
    movements_considered: int
    start: date
    end: date
    warnings: list[str] = []
