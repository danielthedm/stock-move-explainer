import re

from app.providers.llm import LLMProvider
from app.schemas import ChatTurn, CompanyOut, MovementOut, SourceOut
from app.services.ranking import balanced_top

MAX_MOVEMENTS_IN_CONTEXT = 30
ARTICLES_PER_MOVEMENT = 6

SYSTEM_PROMPT = """You are an equity research assistant. You explain stock price movements using ONLY the \
data in the <context> block: daily moves, what the market / sector ETF / peers did the same day, and news \
articles found around each move.

Rules:
- Ground every causal claim in an article and cite it inline as [n], one source per bracket ([1][2], not \
[1, 2]). Never invent news, numbers or dates; quote figures exactly as the context gives them.
- The strongest evidence is an article that itself links an event to the price reaction ("shares fell \
after..."). Prefer it over articles that merely report an event, and name every cause such articles give.
- Use the price context: if the market or sector moved the same way, say the move looks market- or \
industry-driven rather than company-specific, and prefer [industry] articles. If peers were flat, prefer \
[company] articles.
- [macro] articles are macroeconomic or political events (rate decisions, inflation data, tariffs, export \
controls, geopolitics, regulation). Use them to explain market-wide and industry-wide moves, and name the event.
- Article dates are *publication* dates (date only, UTC): coverage often appears the evening of an event or \
the next day. Never infer an event's date from a publication date, and never argue that news "came after" \
the move on that basis.
- The price-based driver is a heuristic. If the numbers contradict it, say so explicitly and show the numbers.
- News explains moves probabilistically. Say "likely" / "coincided with", and say plainly when no article \
in the context explains a move.
- If the question is outside the context (other tickers, dates outside the window, predictions, investment \
advice), say what you can and cannot see. Do not give buy/sell advice.
- Be concise: lead with the answer, use dates and percentages."""


def build_context(company: CompanyOut, movements: list[MovementOut], start, end, min_change_pct: float):
    kept = sorted(movements, key=lambda m: abs(m.pct_change), reverse=True)[:MAX_MOVEMENTS_IN_CONTEXT]
    kept.sort(key=lambda m: m.date)

    peers = ", ".join(f"{p.symbol} ({p.name})" for p in company.peers) or "n/a"
    lines = [
        f"Company: {company.name} ({company.ticker}) | sector: {company.sector or 'n/a'} | "
        f"industry: {company.industry or 'n/a'} | sector ETF: {company.sector_etf or 'n/a'}",
        f"Peers: {peers}",
        f"Window: {start} to {end}. A 'major move' here is a daily close-to-close change of at least "
        f"{min_change_pct}%. {len(movements)} such day(s) found"
        + (f"; showing the {len(kept)} largest." if len(kept) < len(movements) else "."),
        "",
    ]
    sources: list[SourceOut] = []
    for m in kept:
        c = m.context
        peer_txt = ", ".join(f"{p.symbol} {p.pct_change:+.2f}%" for p in c.peers) or "n/a"
        z = f", {m.zscore:+.1f} sigma vs trailing volatility" if m.zscore is not None else ""
        lines.append(
            f"## {m.date}: {company.ticker} {m.pct_change:+.2f}% (close {m.close}{z}) | "
            f"{c.market_symbol} {_fmt(c.market_pct_change)} | sector {c.sector_etf or ''} {_fmt(c.sector_pct_change)} | "
            f"peers: {peer_txt} | price-based driver: {c.driver}"
        )
        if not m.articles:
            lines.append("  (no relevant articles found)" if m.news_status == "fetched" else "  (news not fetched yet)")
        for a in balanced_top(m.articles, ARTICLES_PER_MOVEMENT):
            ref = len(sources) + 1
            sources.append(SourceOut(ref=ref, title=a.title, url=a.url, source=a.source, move_date=m.date))
            published = a.published_at.date().isoformat() if a.published_at else "undated"
            snippet = f" - {a.snippet[:280]}" if a.snippet else ""
            lines.append(f"  [{ref}] [{a.category}] ({published}, {a.source or 'unknown'}) {a.title}{snippet}")
        lines.append("")
    return "\n".join(lines), sources


def cited_refs(text: str) -> set[int]:
    return {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for n in re.findall(r"\d+", group)}


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"


def answer(
    llm: LLMProvider | None,
    *,
    company: CompanyOut,
    movements: list[MovementOut],
    start,
    end,
    min_change_pct: float,
    message: str,
    history: list[ChatTurn],
) -> tuple[str, str, list[SourceOut]]:
    context, sources = build_context(company, movements, start, end, min_change_pct)
    if llm is None:
        return _extractive(company, movements), "extractive", sources[:10]

    messages = [t.model_dump() for t in history]
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {message}"})
    text = llm.complete(SYSTEM_PROMPT, messages)
    cited = cited_refs(text)
    return text, "llm", [s for s in sources if s.ref in cited]


def explain_prompt(company: CompanyOut, movement: MovementOut) -> str:
    return (
        f"Explain why {company.ticker} moved {movement.pct_change:+.2f}% on {movement.date} in 2-4 sentences. "
        "State whether it looks company-specific, industry-wide or market-wide, and cite articles."
    )


def _extractive(company: CompanyOut, movements: list[MovementOut]) -> str:
    if not movements:
        return f"No major movements found for {company.ticker} in this window."
    top = sorted(movements, key=lambda m: abs(m.pct_change), reverse=True)[:5]
    lines = [f"(LLM not configured - showing the raw evidence.) Largest moves for {company.ticker}:"]
    for m in sorted(top, key=lambda m: m.date):
        headline = f'"{m.articles[0].title}" ({m.articles[0].source})' if m.articles else "no relevant article found"
        lines.append(f"- {m.date}: {m.pct_change:+.2f}%, price-based driver: {m.context.driver}. Top article: {headline}")
    return "\n".join(lines)
