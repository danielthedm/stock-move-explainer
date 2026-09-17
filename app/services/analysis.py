"""Read side: derive movements + market context + articles from stored facts."""
from datetime import date, timedelta
from statistics import median

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Article, Company, NewsFetch, PriceBar
from app.schemas import ArticleOut, CompanyOut, MarketContext, MovementOut, PeerMove, PricePoint
from app.services.ingest import LOOKBACK_BUFFER_DAYS, symbols_for
from app.services.movements import classify_driver, compute_metrics
from app.services.ranking import balanced_top


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
) -> list[MovementOut]:
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
    stmt = select(Article).where(
        Article.ticker == company.ticker, Article.move_date.in_(move_dates), Article.relevance >= min_relevance
    )
    if news_category != "all":
        stmt = stmt.where(Article.category == news_category)
    articles: dict[date, list[Article]] = {}
    for a in db.scalars(stmt.order_by(Article.relevance.desc())):
        articles.setdefault(a.move_date, []).append(a)

    def pct_on(symbol: str | None, d: date) -> float | None:
        frame = metrics.get(symbol) if symbol else None
        if frame is None or d not in frame.index:
            return None
        return _r(frame.loc[d, "pct_change"])

    out = []
    for d, row in moves.iterrows():
        pct = float(row["pct_change"])
        market = pct_on(settings.benchmark_symbol, d)
        sector = pct_on(company.sector_etf, d)
        peer_moves = [
            PeerMove(symbol=p["symbol"], pct_change=v)
            for p in company.peers or []
            if (v := pct_on(p["symbol"], d)) is not None
        ]
        peer_values = [p.pct_change for p in peer_moves]
        out.append(
            MovementOut(
                date=d,
                direction="up" if pct > 0 else "down",
                close=_r(row["close"], 4),
                prev_close=_r(row["prev_close"], 4),
                pct_change=_r(pct),
                zscore=_r(row["zscore"]),
                context=MarketContext(
                    market_symbol=settings.benchmark_symbol,
                    market_pct=market,
                    sector_etf=company.sector_etf,
                    sector_pct=sector,
                    peers=peer_moves,
                    peer_median_pct=_r(median(peer_values)) if peer_values else None,
                    excess_vs_sector_pct=_r(pct - sector) if sector is not None else None,
                    driver=classify_driver(pct, market, sector, peer_values),
                ),
                news_status="fetched" if d in fetched else "not_fetched",
                articles=[
                    ArticleOut.model_validate(a, from_attributes=True)
                    for a in balanced_top(articles.get(d, []), max_articles)
                ],
            )
        )
    return out
