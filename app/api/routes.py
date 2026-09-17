import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import deps, schemas
from app.api import v1_schemas
from app.config import Settings, get_settings
from app.db import get_db
from app.models import Explanation
from app.providers.base import ProviderError, TickerNotFound
from app.services import chat as chat_service
from app.services.analysis import build_movements, company_out, load_metrics, price_points
from app.services.ingest import ensure_data
from app.timeutil import utcnow

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
DbDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@dataclass(frozen=True)
class ApiVersion:
    name: str
    macro: bool
    movement_query: type[BaseModel]
    report_query: type[BaseModel]
    report: type[BaseModel]
    movement_list: type[BaseModel]
    explanation: type[BaseModel]


V1 = ApiVersion(
    name="v1",
    macro=False,
    movement_query=v1_schemas.MovementQueryV1,
    report_query=v1_schemas.ReportQueryV1,
    report=v1_schemas.StockReportV1,
    movement_list=v1_schemas.MovementListV1,
    explanation=v1_schemas.MovementExplanationV1,
)
V2 = ApiVersion(
    name="v2",
    macro=True,
    movement_query=schemas.MovementQuery,
    report_query=schemas.ReportQuery,
    report=schemas.StockReport,
    movement_list=schemas.MovementList,
    explanation=schemas.MovementExplanation,
)
VERSIONS = [V1, V2]
LATEST = V2


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


def _threshold(settings: Settings, min_change_pct: float | None) -> float:
    return settings.default_min_change_pct if min_change_pct is None else min_change_pct


def _load(db, settings, prices, news, ticker, start, end, min_change_pct, *, macro, refresh=False, **filters):
    try:
        company, warnings = ensure_data(db, settings, prices, news, ticker, start, end, min_change_pct, refresh, macro)
    except TickerNotFound:
        raise HTTPException(404, f"No price data found for '{ticker}' (unknown symbol, or Yahoo Finance returned nothing)")
    except ProviderError as exc:
        raise HTTPException(502, f"Upstream data provider failed: {exc}")
    metrics = load_metrics(db, company, settings, start, end)
    movements = build_movements(
        db, company, settings, metrics, start, end, min_change_pct=min_change_pct, include_macro=macro, **filters
    )
    return company, warnings, metrics, movements


def make_router(v: ApiVersion) -> APIRouter:
    router = APIRouter(prefix=f"/{v.name}", tags=[v.name])

    def _movements(db, settings, prices, news, ticker: str, q):
        ticker = _ticker(ticker)
        start, end = _window(settings, q.start, q.end)
        filters = schemas.FiltersOut(
            min_change_pct=_threshold(settings, q.min_change_pct), min_zscore=q.min_zscore, direction=q.direction,
            news_category=q.news_category, min_relevance=q.min_relevance, max_articles=q.max_articles,
        )
        company, warnings, metrics, movements = _load(
            db, settings, prices, news, ticker, start, end, macro=v.macro, refresh=q.refresh, **filters.model_dump()
        )
        return company, warnings, metrics, movements, filters, start, end

    @router.get("/stocks/{ticker}", response_model=v.report)
    def get_stock(
        db: DbDep,
        settings: SettingsDep,
        ticker: Annotated[str, Path(examples=["NVDA"])],
        q: Annotated[v.report_query, Query()],
        prices=Depends(deps.get_price_provider),
        news=Depends(deps.get_news_provider),
    ):
        """All stock and news data for a ticker: company, movements with news and market context, prices."""
        company, warnings, metrics, movements, filters, start, end = _movements(db, settings, prices, news, ticker, q)
        return schemas.StockReport(
            company=company_out(company),
            start=start,
            end=end,
            filters=filters,
            movement_count=len(movements),
            movements=movements,
            prices=price_points(metrics[company.ticker], start) if q.include_prices else None,
            warnings=warnings,
        )

    @router.get("/stocks/{ticker}/movements", response_model=v.movement_list)
    def list_movements(
        db: DbDep,
        settings: SettingsDep,
        ticker: Annotated[str, Path(examples=["NVDA"])],
        q: Annotated[v.movement_query, Query()],
        prices=Depends(deps.get_price_provider),
        news=Depends(deps.get_news_provider),
    ):
        """The ticker's major moves, with news and market context."""
        company, warnings, _, movements, filters, start, end = _movements(db, settings, prices, news, ticker, q)
        return schemas.MovementList(
            ticker=company.ticker, start=start, end=end, filters=filters,
            movement_count=len(movements), movements=movements, warnings=warnings,
        )

    @router.get("/stocks/{ticker}/movements/{move_date}", response_model=v.explanation)
    def explain_movement(
        db: DbDep,
        settings: SettingsDep,
        ticker: Annotated[str, Path(examples=["NVDA"])],
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
        company, _, _, movements = _load(
            db, settings, prices, news, ticker, move_date, move_date, 0.0, macro=v.macro, refresh=refresh
        )
        if not movements:
            raise HTTPException(404, f"{move_date} is not a trading day for {ticker}")
        movement, comp = movements[0], company_out(company)

        cached = db.get(Explanation, (ticker, move_date, v.macro))
        if cached and not refresh:
            return schemas.MovementExplanation(
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
            db.merge(
                Explanation(ticker=ticker, date=move_date, macro=v.macro, text=text, model=llm.model, created_at=utcnow())
            )
            db.commit()
        return schemas.MovementExplanation(
            ticker=ticker, movement=movement, explanation=text, mode=mode, model=getattr(llm, "model", None)
        )

    @router.post("/chat", response_model=schemas.ChatResponse)
    def chat(
        req: schemas.ChatRequest,
        db: DbDep,
        settings: SettingsDep,
        prices=Depends(deps.get_price_provider),
        news=Depends(deps.get_news_provider),
        llm=Depends(deps.get_llm),
    ):
        """Ask questions about a ticker's movements. Stateless: send prior turns in `history`."""
        ticker = _ticker(req.ticker)
        start, end = _window(settings, req.start, req.end)
        threshold = _threshold(settings, req.min_change_pct)
        company, warnings, _, movements = _load(db, settings, prices, news, ticker, start, end, threshold, macro=v.macro)
        try:
            text, mode, sources = chat_service.answer(
                llm, company=company_out(company), movements=movements, start=start, end=end,
                min_change_pct=threshold, message=req.message, history=req.history,
            )
        except ProviderError as exc:
            raise HTTPException(502, f"LLM provider failed: {exc}")
        return schemas.ChatResponse(
            ticker=ticker, answer=text, mode=mode, sources=sources,
            movements_considered=len(movements), start=start, end=end, warnings=warnings,
        )

    return router
