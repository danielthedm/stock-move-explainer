import re
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from sqlalchemy.orm import Session

from app import deps
from app.config import Settings, get_settings
from app.db import get_db, init_db
from app.models import Explanation
from app.providers.base import ProviderError, TickerNotFound
from app.schemas import ChatRequest, ChatResponse, MovementExplanation, StockReport
from app.services import chat as chat_service
from app.services.analysis import build_movements, company_out, load_metrics, price_points
from app.services.ingest import ensure_data
from app.timeutil import utcnow


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Stock Move Explainer",
    description="Explains major daily stock moves with company and industry news.",
    version="1.0.0",
    lifespan=lifespan,
)

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
DbDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not TICKER_RE.match(ticker):
        raise HTTPException(422, f"'{raw}' is not a valid ticker symbol")
    return ticker


def _window(settings: Settings, start: date | None, end: date | None) -> tuple[date, date]:
    end = min(end or date.today(), date.today())
    start = start or end - timedelta(days=settings.default_lookback_days)
    if start > end:
        raise HTTPException(422, "start must be on or before end (and not in the future)")
    if (end - start).days > settings.max_range_days:
        raise HTTPException(422, f"date range is limited to {settings.max_range_days} days")
    return start, end


def _ingest(db, settings, prices, news, ticker, start, end, min_change_pct, refresh=False):
    try:
        return ensure_data(db, settings, prices, news, ticker, start, end, min_change_pct, refresh)
    except TickerNotFound:
        raise HTTPException(404, f"No price data found for '{ticker}' (unknown symbol, or Yahoo Finance returned nothing)")
    except ProviderError as exc:
        raise HTTPException(502, f"Upstream data provider failed: {exc}")


def _load(db, settings, prices, news, ticker, start, end, min_change_pct, refresh=False, **filters):
    """The path every endpoint shares: ingest what is missing, then query it."""
    company, warnings = _ingest(db, settings, prices, news, ticker, start, end, min_change_pct, refresh)
    metrics = load_metrics(db, company, settings, start, end)
    movements = build_movements(db, company, settings, metrics, start, end, min_change_pct=min_change_pct, **filters)
    return company, warnings, metrics, movements


def _threshold(settings: Settings, min_change_pct: float | None) -> float:
    return settings.default_min_change_pct if min_change_pct is None else min_change_pct


@app.get("/health")
def health(news=Depends(deps.get_news_provider), llm=Depends(deps.get_llm)):
    return {
        "status": "ok",
        "news_provider": getattr(news, "name", None),
        "llm_model": getattr(llm, "model", None),
    }


@app.get("/stocks/{ticker}", response_model=StockReport)
def get_stock(
    db: DbDep,
    settings: SettingsDep,
    ticker: Annotated[str, Path(examples=["NVDA"])],
    start: Annotated[date | None, Query(description="Default: end - 90 days")] = None,
    end: Annotated[date | None, Query(description="Default: today")] = None,
    min_change_pct: Annotated[
        float | None, Query(ge=0, le=100, description="Min absolute daily % change to count as a movement (default 2.0)")
    ] = None,
    min_zscore: Annotated[
        float | None, Query(ge=0, description="Also require |move| >= N x the stock's trailing 30-day volatility")
    ] = None,
    direction: Literal["any", "up", "down"] = "any",
    news_category: Literal["all", "company", "industry"] = "all",
    min_relevance: Annotated[float, Query(ge=0, le=1)] = 0.0,
    max_articles: Annotated[int, Query(ge=0, le=20, description="Per movement")] = 10,
    include_prices: Annotated[bool, Query(description="Include the daily OHLCV series")] = True,
    refresh: Annotated[bool, Query(description="Bypass the cache and re-fetch prices and news")] = False,
    prices=Depends(deps.get_price_provider),
    news=Depends(deps.get_news_provider),
):
    """All stock + news data for a ticker. First call for a (ticker, range)
    ingests from upstream; later calls are served from the local DB."""
    ticker = _ticker(ticker)
    start, end = _window(settings, start, end)
    threshold = _threshold(settings, min_change_pct)

    company, warnings, metrics, movements = _load(
        db, settings, prices, news, ticker, start, end, threshold, refresh,
        min_zscore=min_zscore, direction=direction,
        news_category=news_category, min_relevance=min_relevance, max_articles=max_articles,
    )
    return StockReport(
        company=company_out(company),
        start=start,
        end=end,
        filters={
            "min_change_pct": threshold, "min_zscore": min_zscore, "direction": direction,
            "news_category": news_category, "min_relevance": min_relevance, "max_articles": max_articles,
        },
        movement_count=len(movements),
        movements=movements,
        prices=price_points(metrics[ticker], start) if include_prices else None,
        warnings=warnings,
    )


@app.get("/stocks/{ticker}/movements/{move_date}", response_model=MovementExplanation)
def explain_movement(
    db: DbDep,
    settings: SettingsDep,
    ticker: str,
    move_date: date,
    refresh: bool = False,
    prices=Depends(deps.get_price_provider),
    news=Depends(deps.get_news_provider),
    llm=Depends(deps.get_llm),
):
    """One day in detail, with a cached natural-language explanation."""
    ticker = _ticker(ticker)
    if move_date > date.today():
        raise HTTPException(422, "move_date is in the future")
    # min_change_pct=0 -> any trading day can be explained, not just >= 2% days.
    company, _, _, movements = _load(db, settings, prices, news, ticker, move_date, move_date, 0.0, refresh)
    if not movements:
        raise HTTPException(404, f"{move_date} is not a trading day for {ticker}")
    movement, comp = movements[0], company_out(company)

    cached = db.get(Explanation, (ticker, move_date))
    if cached and not refresh:
        return MovementExplanation(
            ticker=ticker, movement=movement, explanation=cached.text, mode="llm", model=cached.model, cached=True
        )
    try:
        text, mode, _ = chat_service.answer(
            llm, company=comp, movements=[movement], start=move_date, end=move_date, min_change_pct=0.0,
            message=chat_service.explain_prompt(comp, movement), history=[],
        )
    except ProviderError as exc:
        raise HTTPException(502, f"LLM provider failed: {exc}")
    if mode == "llm":
        db.merge(Explanation(ticker=ticker, date=move_date, text=text, model=llm.model, created_at=utcnow()))
        db.commit()
    return MovementExplanation(
        ticker=ticker, movement=movement, explanation=text, mode=mode, model=getattr(llm, "model", None)
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db: DbDep,
    settings: SettingsDep,
    prices=Depends(deps.get_price_provider),
    news=Depends(deps.get_news_provider),
    llm=Depends(deps.get_llm),
):
    """Ask questions about a ticker's movements. Stateless: send prior turns in
    `history`. Narrow `start`/`end` to focus on a period."""
    ticker = _ticker(req.ticker)
    start, end = _window(settings, req.start, req.end)
    threshold = _threshold(settings, req.min_change_pct)

    company, warnings, _, movements = _load(db, settings, prices, news, ticker, start, end, threshold)
    try:
        text, mode, sources = chat_service.answer(
            llm, company=company_out(company), movements=movements, start=start, end=end,
            min_change_pct=threshold, message=req.message, history=req.history,
        )
    except ProviderError as exc:
        raise HTTPException(502, f"LLM provider failed: {exc}")
    return ChatResponse(
        ticker=ticker, answer=text, mode=mode, sources=sources,
        movements_considered=len(movements), start=start, end=end, warnings=warnings,
    )
