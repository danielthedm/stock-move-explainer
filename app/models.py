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
    peers: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class PriceBar(Base):
    __tablename__ = "price_bars"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)


class PriceCoverage(Base):
    __tablename__ = "price_coverage"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    start: Mapped[date] = mapped_column(Date)
    end: Mapped[date] = mapped_column(Date)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class NewsFetch(Base):
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
    category: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(200))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    snippet: Mapped[str | None] = mapped_column(Text)
    relevance: Mapped[float] = mapped_column(Float)


class MacroFetch(Base):
    __tablename__ = "macro_fetches"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    scope: Mapped[str] = mapped_column(String(120), primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)


class MacroArticle(Base):
    __tablename__ = "macro_articles"
    __table_args__ = (UniqueConstraint("date", "scope", "url"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    scope: Mapped[str] = mapped_column(String(120), index=True)
    topic: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(200))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    snippet: Mapped[str | None] = mapped_column(Text)
    relevance: Mapped[float] = mapped_column(Float)


class Explanation(Base):
    __tablename__ = "explanations"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    macro: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=False)
    text: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime)
