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
    r"tariffs?|export|ban)\b",
    re.IGNORECASE,
)
_PRICE = re.compile(r"\b(shares?|stocks?)\b", re.IGNORECASE)
_DIRECTION = re.compile(
    r"\b(drop\w*|f[ae]ll\w*|s[ia]nk\w*|plung\w*|tumbl\w*|slid\w*|slump\w*|div(e|es|ing)|lower|down|crash\w*|"
    r"surg\w*|jump\w*|soar\w*|rall(y|ies|ied)|ris(e|es|ing)|rose|higher|up|climb\w*|gain\w*|pop\w*|rebound\w*)\b",
    re.IGNORECASE,
)


def _describes_price_reaction(title: str) -> bool:
    return bool(_PRICE.search(title) and _DIRECTION.search(title))


_MACRO_TOPICS = [
    ("monetary_policy", r"fed|federal reserve|fomc|powell|rate (cut|hike|decision|increase)s?|interest rates?|"
                        r"central banks?|ecb|bank of (england|japan)|(treasury|bond) yields?"),
    ("economy", r"inflation|cpi|ppi|pce|jobs report|payrolls|unemployment|jobless|gdp|recession|"
                r"consumer (confidence|sentiment|spending)|retail sales"),
    ("trade", r"tariffs?|trade (war|deal|talks|tensions|policy)|sanctions?|embargo|"
              r"export (controls?|curbs?|bans?|rules?|restrictions?|licen[cs]es?)"),
    ("geopolitics", r"geopolit\w*|war|invasion|ceasefire|missiles?|military|taiwan strait|middle east|opec|oil prices?"),
    ("regulation", r"regulat\w*|antitrust|legislation|executive order|bans?|banned|doj|ftc|supreme court|"
                   r"congress|senate|white house|government shutdown|debt ceiling|elections?"),
]
_MACRO_TOPICS = [(topic, re.compile(rf"\b({pattern})\b", re.IGNORECASE)) for topic, pattern in _MACRO_TOPICS]
_MARKET_WORDS = re.compile(
    r"\b(stocks?|shares|markets?|wall street|s&p|nasdaq|dow|investors|sell-?off|rall(y|ies|ied)|slides?|tumbles?)\b",
    re.IGNORECASE,
)


def macro_topic(title: str, snippet: str | None = None) -> str | None:
    for text in (title, snippet or ""):
        for topic, pattern in _MACRO_TOPICS:
            if pattern.search(text):
                return topic
    return None


def url_key(url: str) -> str:
    return url.split("?")[0].rstrip("/").lower()


def short_name(name: str) -> str:
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
        if len(t) >= 2 and re.search(rf"(?<![\w.]){re.escape(t)}(?![\w])", text):
            return True
    return False


@dataclass
class RankedArticle:
    raw: RawArticle
    category: str
    relevance: float


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def balanced_top(articles: list, n: int) -> list:
    queues: dict[str, list] = {}
    for a in articles:
        queues.setdefault(a.category, []).append(a)
    picked: list = []
    while len(picked) < n and any(queues.values()):
        for q in queues.values():
            if q and len(picked) < n:
                picked.append(q.pop(0))
    return sorted(picked, key=lambda a: a.relevance, reverse=True)


def _is_duplicate(a: RawArticle, seen_urls: set[str], seen_titles: set[str]) -> bool:
    u, t = url_key(a.url), _norm_title(a.title)
    if u in seen_urls or t in seen_titles:
        return True
    seen_urls.add(u)
    seen_titles.add(t)
    return False


def _timing_bonus(a: RawArticle, move_date: date, prev_trading_date: date) -> float:
    if not a.published_at:
        return 0.0
    pub = a.published_at.date()
    if prev_trading_date <= pub <= move_date:
        return 0.20
    if pub == move_date + timedelta(days=1):
        return 0.10
    return 0.0


def rank_macro(
    articles: list[RawArticle], *, move_date: date, prev_trading_date: date, limit: int
) -> list[tuple[RawArticle, str, float]]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    ranked = []
    for a in articles:
        if _is_duplicate(a, seen_urls, seen_titles):
            continue
        topic = macro_topic(a.title, a.snippet)
        if topic is None:
            continue
        score = 0.45 if macro_topic(a.title) else 0.25
        score += _timing_bonus(a, move_date, prev_trading_date)
        if _MARKET_WORDS.search(a.title):
            score += 0.15
        ranked.append((a, topic, round(min(score, 1.0), 3)))
    ranked.sort(key=lambda r: r[2], reverse=True)
    return ranked[:limit]


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
        if _is_duplicate(a, seen_urls, seen_titles):
            continue

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
            continue

        score += _timing_bonus(a, move_date, prev_trading_date)
        if _CATALYSTS.search(title):
            score += 0.10
        if _describes_price_reaction(title):
            score += 0.10
        ranked.append(RankedArticle(raw=a, category=category, relevance=round(min(score, 1.0), 3)))

    ranked.sort(key=lambda r: r.relevance, reverse=True)
    out: list[RankedArticle] = []
    counts = {"company": 0, "industry": 0}
    for r in ranked:
        if counts[r.category] < per_category:
            counts[r.category] += 1
            out.append(r)
    return out
