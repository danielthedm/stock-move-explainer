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
    """What else moved that day - the price-side evidence for attribution."""

    market_symbol: str
    market_pct: float | None
    sector_etf: str | None
    sector_pct: float | None
    peers: list[PeerMove]
    peer_median_pct: float | None
    excess_vs_sector_pct: float | None
    driver: Literal["company_specific", "industry_wide", "market_wide"]


class ArticleOut(BaseModel):
    id: int
    category: Literal["company", "industry"]
    title: str
    url: str
    source: str | None
    published_at: datetime | None
    snippet: str | None
    relevance: float


class MovementOut(BaseModel):
    date: date
    direction: Literal["up", "down"]
    close: float
    prev_close: float
    pct_change: float
    zscore: float | None
    context: MarketContext
    news_status: Literal["fetched", "not_fetched"]
    articles: list[ArticleOut]


class StockReport(BaseModel):
    company: CompanyOut
    start: date
    end: date
    filters: dict
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
    # Stateless API: the client owns the transcript and sends prior turns back.
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
