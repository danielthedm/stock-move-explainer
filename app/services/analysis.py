"""Read side: derive movements + market context + articles from stored facts."""
from datetime import date, timedelta
from statistics import median

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Article, Company, MacroArticle, MacroFetch, NewsFetch, PriceBar
from app.schemas import ArticleOut, CompanyOut, MarketContext, MovementOut, PeerMove, PricePoint
from app.services.movements import classify_driver, compute_metrics, macro_scope
from app.services.ranking import balanced_top, macro_topic, url_key

# Trailing history needed before `start` so the first day has a z-score.
LOOKBACK_BUFFER_DAYS = 70


def _r(value, digits: int = 2) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)


def company_out(company: Company) -> CompanyOut:
    return CompanyOut(
        ticker=company.ticker,
        name=company.name,
        sector=company.sector,
        industry=company.industry,
        sector_etf=company.sector_etf,
        peers=company.peers or [],
    )


def symbols_for(company: Company, settings: Settings) -> list[str]:
    symbols = [company.ticker, settings.benchmark_symbol]
    if company.sector_etf:
        symbols.append(company.sector_etf)
    symbols += [p["symbol"] for p in company.peers or []]
    return list(dict.fromkeys(symbols))


def load_metrics(db: Session, company: Company, settings: Settings, start: date, end: date) -> dict[str, pd.DataFrame]:
    symbols = symbols_for(company, settings)
    bars = db.scalars(
        select(PriceBar).where(
            PriceBar.symbol.in_(symbols),
            PriceBar.date >= start - timedelta(days=LOOKBACK_BUFFER_DAYS),
            PriceBar.date <= end,
        )
    ).all()
    by_symbol: dict[str, list[PriceBar]] = {s: [] for s in symbols}
    for b in bars:
        by_symbol[b.symbol].append(b)
    return {s: compute_metrics(rows) for s, rows in by_symbol.items()}


def price_points(df: pd.DataFrame, start: date) -> list[PricePoint]:
    return [
        PricePoint(
            date=d,
            open=_r(row["open"], 4),
            high=_r(row["high"], 4),
            low=_r(row["low"], 4),
            close=_r(row["close"], 4),
            volume=int(row["volume"]),
            pct_change=_r(row["pct_change"])  # not row.pct_change: that is a Series method,
        )
        for d, row in df[df.index >= start].iterrows()
    ]


def market_context(
    metrics: dict[str, pd.DataFrame], company: Company, settings: Settings, d: date, pct: float
) -> MarketContext:
    """What the market, the sector ETF and the peers did on `d`, and the driver
    label that follows. Used on read and by ingest (to decide which days need a
    macro search)."""

    def pct_on(symbol: str | None) -> float | None:
        frame = metrics.get(symbol) if symbol else None
        if frame is None or d not in frame.index:
            return None
        return _r(frame.loc[d, "pct_change"])

    market, sector = pct_on(settings.benchmark_symbol), pct_on(company.sector_etf)
    peer_moves = [
        PeerMove(symbol=p["symbol"], pct_change=v) for p in company.peers or [] if (v := pct_on(p["symbol"])) is not None
    ]
    peer_values = [p.pct_change for p in peer_moves]
    return MarketContext(
        market_symbol=settings.benchmark_symbol,
        market_pct_change=market,
        sector_etf=company.sector_etf,
        sector_pct_change=sector,
        peers=peer_moves,
        peer_median_pct_change=_r(median(peer_values)) if peer_values else None,
        excess_vs_sector_pct=_r(pct - sector) if sector is not None else None,
        driver=classify_driver(pct, market, sector, peer_values),
    )


def build_movements(
    db: Session,
    company: Company,
    settings: Settings,
    metrics: dict[str, pd.DataFrame],
    start: date,
    end: date,
    *,
    min_change_pct: float,
    min_zscore: float | None = None,
    direction: str = "any",
    news_category: str = "all",
    min_relevance: float = 0.0,
    max_articles: int = 10,
    include_macro: bool = False,
) -> list[MovementOut]:
    """`include_macro` (API v2) also attaches the macro/political articles found
    for each day's scope; v1 never sees them."""
    df = metrics[company.ticker]
    moves = df[(df.index >= start) & (df.index <= end) & (df["pct_change"].abs() >= min_change_pct)]
    if min_zscore is not None:
        moves = moves[moves["zscore"].abs() >= min_zscore]
    if direction == "up":
        moves = moves[moves["pct_change"] > 0]
    elif direction == "down":
        moves = moves[moves["pct_change"] < 0]
    if moves.empty:
        return []

    move_dates = list(moves.index)
    fetched = set(
        db.scalars(select(NewsFetch.date).where(NewsFetch.ticker == company.ticker, NewsFetch.date.in_(move_dates)))
    )
    articles: dict[date, list[ArticleOut]] = {}
    for a in db.scalars(select(Article).where(Article.ticker == company.ticker, Article.move_date.in_(move_dates))):
        out = ArticleOut.model_validate(a, from_attributes=True)
        out.macro_topic = macro_topic(a.title, a.snippet)  # an industry story can still be about tariffs
        articles.setdefault(a.move_date, []).append(out)

    macro: dict[tuple[date, str], list[ArticleOut]] = {}
    macro_fetched: set[tuple[date, str]] = set()
    if include_macro:
        for m in db.scalars(select(MacroArticle).where(MacroArticle.date.in_(move_dates))):
            macro.setdefault((m.date, m.scope), []).append(
                ArticleOut(
                    category="macro", title=m.title, url=m.url, source=m.source, published_at=m.published_at,
                    snippet=m.snippet, relevance=m.relevance, macro_topic=m.topic,
                )
            )
        macro_fetched = {
            (f.date, f.scope) for f in db.scalars(select(MacroFetch).where(MacroFetch.date.in_(move_dates)))
        }

    out = []
    for d, row in moves.iterrows():
        pct = float(row["pct_change"])
        context = market_context(metrics, company, settings, d, pct)
        scope = macro_scope(context.driver, company.industry, company.sector) if include_macro else None

        found = list(articles.get(d, []))
        if scope:
            # The industry search and the macro search overlap (export controls...):
            # the more specific label wins, and the article keeps its macro_topic.
            seen = {url_key(a.url) for a in found}
            found += [a for a in macro.get((d, scope), []) if url_key(a.url) not in seen]
        found = [
            a for a in found if a.relevance >= min_relevance and (news_category == "all" or a.category == news_category)
        ]
        found.sort(key=lambda a: a.relevance, reverse=True)

        out.append(
            MovementOut(
                date=d,
                direction="up" if pct > 0 else "down",
                close=_r(row["close"], 4),
                prev_close=_r(row["prev_close"], 4),
                pct_change=_r(pct),
                zscore=_r(row["zscore"]),
                context=context,
                news_status="fetched" if d in fetched else "not_fetched",
                macro_status=(
                    "not_applicable" if scope is None else "fetched" if (d, scope) in macro_fetched else "not_fetched"
                ),
                articles=balanced_top(found, max_articles),
            )
        )
    return out
