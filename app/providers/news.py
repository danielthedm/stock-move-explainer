"""News providers. Each one renders the same structured NewsQuery in the way
its search engine works best (semantic sentence for Exa, boolean for NewsAPI)."""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

import httpx

from app.providers.base import request_json


@dataclass(frozen=True)
class NewsQuery:
    terms: list[str]  # entities that should appear, e.g. ["NVIDIA", "NVDA"]
    text: str  # natural-language description of what we want
    start: date
    end: date
    limit: int = 10


@dataclass
class RawArticle:
    title: str
    url: str
    source: str | None = None
    published_at: datetime | None = None
    snippet: str | None = None


class NewsProvider(Protocol):
    name: str

    def search(self, query: NewsQuery) -> list[RawArticle]: ...


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Store naive UTC everywhere.
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _host(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.") or None


class ExaNewsProvider:
    """https://exa.ai/docs/reference/search - semantic search with hard
    published-date filters and no history limit, which is what this use case
    needs (explaining a move from 8 months ago)."""

    name = "exa"
    URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=20)
        self._headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    def search(self, query: NewsQuery) -> list[RawArticle]:
        body = {
            "query": query.text,
            "type": "auto",
            "category": "news",
            "numResults": query.limit,
            "startPublishedDate": f"{query.start.isoformat()}T00:00:00.000Z",
            "endPublishedDate": f"{query.end.isoformat()}T23:59:59.999Z",
            "contents": {"highlights": True},  # content options must be nested
        }
        data = request_json(self._client, "POST", self.URL, headers=self._headers, json=body)
        articles = []
        for r in data.get("results", []):
            if not r.get("url") or not r.get("title"):
                continue
            highlights = r.get("highlights") or []
            snippet = " … ".join(highlights)
            articles.append(
                RawArticle(
                    title=r["title"].strip(),
                    url=r["url"],
                    source=_host(r["url"]),
                    published_at=_parse_dt(r.get("publishedDate")),
                    snippet=snippet[:600] or None,
                )
            )
        return articles


class NewsApiProvider:
    """https://newsapi.org/docs/endpoints/everything (free tier: ~last 30 days)."""

    name = "newsapi"
    URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=20)
        self._headers = {"X-Api-Key": api_key}

    def search(self, query: NewsQuery) -> list[RawArticle]:
        q = " OR ".join(f'"{t}"' for t in query.terms)[:500]
        params = {
            "q": q,
            "from": query.start.isoformat(),
            "to": f"{query.end.isoformat()}T23:59:59",
            "language": "en",
            "sortBy": "relevancy",
            "searchIn": "title,description",
            "pageSize": query.limit,
        }
        data = request_json(self._client, "GET", self.URL, headers=self._headers, params=params)
        articles = []
        for a in data.get("articles", []):
            if not a.get("url") or not a.get("title") or a["title"] == "[Removed]":
                continue
            articles.append(
                RawArticle(
                    title=a["title"].strip(),
                    url=a["url"],
                    source=(a.get("source") or {}).get("name") or _host(a["url"]),
                    published_at=_parse_dt(a.get("publishedAt")),
                    snippet=(a.get("description") or a.get("content") or "")[:600] or None,
                )
            )
        return articles
