"""Storage model.

Only *facts* are stored (bars, articles, explanations). Derived numbers
(% change, z-score, peer/sector context) are computed on read, so the
"major movement" definition is a query-time filter rather than something baked
in at ingest time.
"""
from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Company(Base):
    __tablename__ = "companies"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(100))
    sector_etf: Mapped[str | None] = mapped_column(String(16))
    peers: Mapped[list] = mapped_column(JSON, default=list)  # [{"symbol","name"}]
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class PriceBar(Base):
    """Daily adjusted OHLCV for any symbol (target, peers, sector ETF, SPY)."""

    __tablename__ = "price_bars"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)


class PriceCoverage(Base):
    """Date range already fetched per symbol (the read-through cache index)."""

    __tablename__ = "price_coverage"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    start: Mapped[date] = mapped_column(Date)
    end: Mapped[date] = mapped_column(Date)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class NewsFetch(Base):
    """Marks a movement day whose news search has completed (even if empty)."""

    __tablename__ = "news_fetches"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (UniqueConstraint("ticker", "move_date", "url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    move_date: Mapped[date] = mapped_column(Date, index=True)
    category: Mapped[str] = mapped_column(String(16))  # company | industry
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(200))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    snippet: Mapped[str | None] = mapped_column(Text)
    relevance: Mapped[float] = mapped_column(Float)


class MacroFetch(Base):
    """Marks a (day, scope) whose macro/political news search has completed.
    Keyed by scope, not ticker: every stock in the same industry (or every stock,
    for market-wide days) shares one search."""

    __tablename__ = "macro_fetches"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    scope: Mapped[str] = mapped_column(String(120), primary_key=True)  # "market" | "industry:<name>"
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class MacroArticle(Base):
    __tablename__ = "macro_articles"
    __table_args__ = (UniqueConstraint("date", "scope", "url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    scope: Mapped[str] = mapped_column(String(120), index=True)
    topic: Mapped[str] = mapped_column(String(32))  # monetary_policy | economy | trade | geopolitics | regulation
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(200))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    snippet: Mapped[str | None] = mapped_column(Text)
    relevance: Mapped[float] = mapped_column(Float)


class Explanation(Base):
    """Cached LLM explanation for one movement day. `macro` separates the v1
    explanation from the v2 one, which also sees macro/political articles."""

    __tablename__ = "explanations"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    macro: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=False)
    text: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime)
