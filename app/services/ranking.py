"""Turn noisy search results into a short, relevant, de-duplicated list.

Deliberately heuristic (no LLM): deterministic, free, fast, unit-testable.
An article must mention the company, a peer, or the industry to survive.
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta

from app.providers.news import RawArticle

_SUFFIXES = re.compile(
    r"[,\s]+(inc\.?|incorporated|corporation|corp\.?|company|co\.?|ltd\.?|limited|plc|n\.v\.|s\.a\.|ag|se|"
    r"holdings?|group|platforms|technologies|technology|systems|enterprises|international|"
    r"class [a-c]|common stock)$",
    re.IGNORECASE,
)
_CATALYSTS = re.compile(
    r"\b(earnings|revenue|guidance|forecast|outlook|profit|results|beats?|miss(es|ed)?|"
    r"upgrad\w*|downgrad\w*|price target|analyst|lawsuit|sue[sd]?|settle\w*|probe|investigat\w*|"
    r"sec|ftc|doj|antitrust|recall|launch\w*|unveil\w*|acqui\w*|merger|deal|layoffs?|ceo|resign\w*|"
    r"tariffs?|export|ban|shares|stock|surg\w*|soar\w*|plung\w*|tumbl\w*|slump\w*|rall(y|ies|ied))\b",
    re.IGNORECASE,
)


def short_name(name: str) -> str:
    """'NVIDIA Corporation' -> 'NVIDIA', 'Amazon.com, Inc.' -> 'Amazon'."""
    out = re.sub(r"^the\s+", "", name.strip(), flags=re.IGNORECASE)
    while True:
        stripped = _SUFFIXES.sub("", out).strip(" ,")
        if stripped == out or not stripped:
            break
        out = stripped
    return re.sub(r"\.com$", "", out, flags=re.IGNORECASE) or name


def _mentions(text: str, names: list[str], tickers: list[str]) -> bool:
    for n in names:
        if n and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.IGNORECASE):
            return True
    for t in tickers:
        # Tickers are case-sensitive whole words; 1-letter tickers are too ambiguous.
        if len(t) >= 2 and re.search(rf"(?<![\w.]){re.escape(t)}(?![\w])", text):
            return True
    return False


@dataclass
class RankedArticle:
    raw: RawArticle
    category: str  # company | industry
    relevance: float


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def balanced_top(articles: list, n: int) -> list:
    """Top `n` of a relevance-sorted list, alternating categories while both have
    articles left. Company articles outscore industry ones by construction (on
    live data nearly all land at 0.95 vs <= 0.80), so a plain [:n] silently drops
    every competitor/industry article whenever n is small."""
    queues: dict[str, list] = {}
    for a in articles:
        queues.setdefault(a.category, []).append(a)
    picked: list = []
    while len(picked) < n and any(queues.values()):
        for q in queues.values():
            if q and len(picked) < n:
                picked.append(q.pop(0))
    return sorted(picked, key=lambda a: a.relevance, reverse=True)


def rank_articles(
    articles: list[RawArticle],
    *,
    company_names: list[str],
    company_tickers: list[str],
    industry_names: list[str],
    industry_tickers: list[str],
    move_date: date,
    prev_trading_date: date,
    per_category: int,
) -> list[RankedArticle]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    ranked: list[RankedArticle] = []

    for a in articles:
        url_key = a.url.split("?")[0].rstrip("/").lower()
        title_key = _norm_title(a.title)
        if url_key in seen_urls or title_key in seen_titles:
            continue  # syndicated copies
        seen_urls.add(url_key)
        seen_titles.add(title_key)

        title, body = a.title, a.snippet or ""
        if _mentions(title, company_names, company_tickers):
            category, score = "company", 0.60
        elif _mentions(body, company_names, company_tickers):
            category, score = "company", 0.35
        elif _mentions(title, industry_names, industry_tickers):
            category, score = "industry", 0.45
        elif _mentions(body, industry_names, industry_tickers):
            category, score = "industry", 0.25
        else:
            continue  # irrelevant to this company and its industry

        if a.published_at:
            pub = a.published_at.date()
            if prev_trading_date <= pub <= move_date:
                score += 0.20  # could have caused the move
            elif pub == move_date + timedelta(days=1):
                score += 0.10  # next-day recap ("shares fell yesterday after...")
        if _CATALYSTS.search(title):
            score += 0.15
        ranked.append(RankedArticle(raw=a, category=category, relevance=round(min(score, 1.0), 3)))

    ranked.sort(key=lambda r: r.relevance, reverse=True)
    out: list[RankedArticle] = []
    counts = {"company": 0, "industry": 0}
    for r in ranked:
        if counts[r.category] < per_category:
            counts[r.category] += 1
            out.append(r)
    return out
