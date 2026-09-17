"""Read-through ingestion: make sure everything needed to answer a request for
(ticker, start, end) is in the DB, fetching only what is missing."""
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Article, Company, NewsFetch, PriceBar, PriceCoverage
from app.providers.base import ProviderError, TickerNotFound
from app.providers.news import NewsProvider, NewsQuery, RawArticle
from app.providers.prices import PriceProvider
from app.services.movements import compute_metrics
from app.timeutil import utcnow
from app.services.ranking import rank_articles, short_name

# Trailing history needed before `start` so the first day has a z-score.
LOOKBACK_BUFFER_DAYS = 70
PROFILE_TTL = timedelta(days=30)
TODAY_BAR_TTL = timedelta(hours=1)

# One ingest per ticker at a time (single-process guard; see README for the
# multi-worker story).
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def symbols_for(company: Company, settings: Settings) -> list[str]:
    symbols = [company.ticker, settings.benchmark_symbol]
    if company.sector_etf:
        symbols.append(company.sector_etf)
    symbols += [p["symbol"] for p in company.peers or []]
    return list(dict.fromkeys(symbols))


def ensure_data(
    db: Session,
    settings: Settings,
    prices: PriceProvider,
    news: NewsProvider | None,
    ticker: str,
    start: date,
    end: date,
    min_change_pct: float,
    refresh: bool = False,
) -> tuple[Company, list[str]]:
    with _locks[ticker]:
        try:
            company = _ensure_company(db, settings, prices, ticker, refresh)
            _ensure_prices(db, settings, prices, company, start, end, refresh)
            warnings = _ensure_news(db, settings, news, company, start, end, min_change_pct, refresh)
            db.commit()
            return company, warnings
        except Exception:
            db.rollback()
            raise


def _ensure_company(db: Session, settings: Settings, prices: PriceProvider, ticker: str, refresh: bool) -> Company:
    company = db.get(Company, ticker)
    now = utcnow()
    if company and not refresh and now - company.updated_at < PROFILE_TTL:
        return company
    profile = prices.get_profile(ticker, settings.max_peers)
    if company is None:
        company = Company(ticker=ticker)
        db.add(company)
    company.name = profile.name
    company.sector = profile.sector
    company.industry = profile.industry
    company.sector_etf = profile.sector_etf
    company.peers = profile.peers
    company.updated_at = now
    db.flush()
    return company


def _ensure_prices(
    db: Session, settings: Settings, prices: PriceProvider, company: Company, start: date, end: date, refresh: bool
) -> None:
    today = date.today()
    want_start = start - timedelta(days=LOOKBACK_BUFFER_DAYS)
    want_end = min(end, today)
    now = utcnow()

    stale: list[str] = []
    fetch_start, fetch_end = want_start, want_end
    for symbol in symbols_for(company, settings):
        cov = db.get(PriceCoverage, symbol)
        covered = cov is not None and cov.start <= want_start and cov.end >= want_end
        today_bar_stale = cov is not None and want_end >= today and now - cov.fetched_at > TODAY_BAR_TTL
        if refresh or not covered or today_bar_stale:
            stale.append(symbol)
            if cov:  # widen so stored coverage stays one contiguous range
                fetch_start, fetch_end = min(fetch_start, cov.start), max(fetch_end, cov.end)

    if stale:
        history = prices.get_history(stale, fetch_start, fetch_end)
        for symbol in stale:
            bars = history.get(symbol, [])
            if not bars:
                continue  # a delisted peer shouldn't fail the request
            db.execute(
                delete(PriceBar).where(
                    PriceBar.symbol == symbol, PriceBar.date >= fetch_start, PriceBar.date <= fetch_end
                )
            )
            db.add_all(PriceBar(symbol=symbol, **vars(b)) for b in bars)
            cov = db.get(PriceCoverage, symbol) or PriceCoverage(symbol=symbol)
            cov.start, cov.end, cov.fetched_at = fetch_start, fetch_end, now
            db.add(cov)
        db.flush()

    has_bars = db.scalar(select(PriceBar.date).where(PriceBar.symbol == company.ticker).limit(1))
    if has_bars is None:
        raise TickerNotFound(company.ticker)


def _news_queries(company: Company, window: tuple[date, date], limit: int) -> dict[str, NewsQuery]:
    name = short_name(company.name)
    start, end = window
    queries = {
        "company": NewsQuery(
            terms=[name, company.ticker],
            text=f"{name} ({company.ticker}) stock news: earnings, guidance, analyst ratings, "
            f"product launches, lawsuits, regulation",
            start=start,
            end=end,
            limit=limit,
        )
    }
    peer_names = [short_name(p["name"]) for p in (company.peers or [])[:3]]
    industry_terms = peer_names + ([company.industry] if company.industry else [])
    if industry_terms:
        queries["industry"] = NewsQuery(
            terms=industry_terms,
            text=f"{company.industry or company.sector or ''} industry news affecting {name} and competitors "
            f"{', '.join(peer_names)}".strip(),
            start=start,
            end=end,
            limit=limit,
        )
    return queries


def _ensure_news(
    db: Session,
    settings: Settings,
    news: NewsProvider | None,
    company: Company,
    start: date,
    end: date,
    min_change_pct: float,
    refresh: bool,
) -> list[str]:
    warnings: list[str] = []
    bars = db.scalars(
        select(PriceBar).where(
            PriceBar.symbol == company.ticker,
            PriceBar.date >= start - timedelta(days=LOOKBACK_BUFFER_DAYS),
            PriceBar.date <= end,
        )
    ).all()
    df = compute_metrics(bars)
    dates = list(df.index)
    moves = df[(df.index >= start) & (df["pct_change"].abs() >= min_change_pct)]

    done = set(
        db.scalars(select(NewsFetch.date).where(NewsFetch.ticker == company.ticker, NewsFetch.date.in_(list(moves.index))))
    )
    pending = [d for d in moves.index if refresh or d not in done]
    if not pending:
        return warnings
    if news is None:
        return [f"No news provider configured: {len(pending)} movement day(s) have no articles. Set EXA_API_KEY or NEWSAPI_API_KEY."]

    # Biggest moves first: they are what users ask about, and quota is finite.
    pending.sort(key=lambda d: abs(moves.loc[d, "pct_change"]), reverse=True)
    batch, deferred = pending[: settings.max_news_movements], pending[settings.max_news_movements :]
    if deferred:
        warnings.append(
            f"{len(deferred)} smaller movement day(s) not yet enriched with news (per-request cap "
            f"{settings.max_news_movements}). Repeat the request to continue."
        )

    # News can land after the previous close (earnings calls, weekend events),
    # so search from the previous trading day through the day after the move.
    windows = {}
    for d in batch:
        i = dates.index(d)
        prev_trading = dates[i - 1] if i > 0 else d - timedelta(days=1)
        windows[d] = (prev_trading, d + timedelta(days=1))

    tasks = [
        (d, kind, q)
        for d in batch
        for kind, q in _news_queries(company, windows[d], settings.articles_per_category * 2).items()
    ]
    results: dict[date, list[RawArticle]] = defaultdict(list)
    failed: set[date] = set()
    with ThreadPoolExecutor(max_workers=settings.news_concurrency) as pool:
        futures = {pool.submit(news.search, q): d for d, _, q in tasks}
        for future, d in futures.items():
            try:
                results[d].extend(future.result())
            except ProviderError as exc:
                failed.add(d)
                warnings.append(f"News search failed for {d}: {exc}")

    name = short_name(company.name)
    peers = company.peers or []
    now = utcnow()
    for d in batch:
        if d in failed:
            continue  # not marked as fetched -> retried on the next request
        ranked = rank_articles(
            results[d],
            company_names=[name],
            company_tickers=[company.ticker],
            industry_names=[short_name(p["name"]) for p in peers] + ([company.industry] if company.industry else []),
            industry_tickers=[p["symbol"] for p in peers],
            move_date=d,
            prev_trading_date=windows[d][0],
            per_category=settings.articles_per_category,
        )
        db.execute(delete(Article).where(Article.ticker == company.ticker, Article.move_date == d))
        db.add_all(
            Article(
                ticker=company.ticker,
                move_date=d,
                category=r.category,
                title=r.raw.title,
                url=r.raw.url,
                source=r.raw.source,
                published_at=r.raw.published_at,
                snippet=r.raw.snippet,
                relevance=r.relevance,
            )
            for r in ranked
        )
        db.merge(NewsFetch(ticker=company.ticker, date=d, fetched_at=now))
    db.flush()
    return warnings
