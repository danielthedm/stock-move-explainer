"""Live quality evals against golden cases (evals/cases.json). Needs a running server.

    python -m evals.run [--version v1|v2] [--judge] [--refresh] [--only id,id]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

from app.services.chat import cited_refs

CASES = Path(__file__).with_name("cases.json")
ARTICLES_SHOWN_TO_LLM = 6
CHECKS = ["retrieval", "tier", "driver", "explanation", "citations", "faithful"]


def _mentions(text: str, keywords: list[str]) -> bool:
    text = f" {text.lower()} "
    return any(k.lower() in text for k in keywords)


def score_case(case: dict, body: dict) -> dict:
    movement, articles = body["movement"], body["movement"]["articles"]
    hits = [a for a in articles if _mentions(f"{a['title']} {a.get('snippet') or ''}", case["keywords"])]

    tiers = [case["tier"]] if isinstance(case["tier"], str) else case["tier"]
    in_tier = [
        a for a in hits
        if a["category"] in tiers or ("macro" in tiers and a.get("macro_topic"))
    ]

    checks = {
        "retrieval": bool(hits),
        "tier": bool(in_tier),
        "driver": movement["context"]["driver"] in case["drivers"] if case.get("drivers") else None,
        "explanation": None,
        "citations": None,
        "faithful": None,
    }
    if body["mode"] == "llm":
        cited = cited_refs(body["explanation"])
        shown = min(len(articles), ARTICLES_SHOWN_TO_LLM)
        checks["explanation"] = _mentions(body["explanation"], case["keywords"])
        checks["citations"] = bool(cited) and all(1 <= n <= shown for n in cited)
    first_hit = next((i for i, a in enumerate(articles, 1) if a in hits), None)
    return {"checks": checks, "first_hit_rank": first_hit, "driver": movement["context"]["driver"]}


JUDGE_SYSTEM = """You grade a stock-move explanation for faithfulness to its evidence. You get the exact context \
the writer was shown (price header + numbered articles) and the explanation it wrote.

A claim is UNSUPPORTED only if it states a fact (event, number, quote, date) that appears nowhere in the context.
Do NOT penalise: restating or doing arithmetic on header numbers; hedged language; disagreeing with the \
"price-based driver" label when the header numbers support the disagreement; a fact cited to [2] that is actually \
in [1]; treating an article published the day after the move as describing the event that caused it.

Reply with JSON only: {"faithful": true|false, "unsupported": ["claim", ...]}"""


def judge(llm, http: httpx.Client, version: str, body: dict) -> tuple[bool | None, list[str]]:
    from app.schemas import CompanyOut, MovementOut
    from app.services.chat import build_context

    m = body["movement"]
    report = http.get(f"/{version}/stocks/{body['ticker']}",
                      params={"start": m["date"], "end": m["date"], "include_prices": "false"}).json()
    context, _ = build_context(CompanyOut(**report["company"]), [MovementOut(**m)], m["date"], m["date"], 0.0)
    reply = llm.complete(
        JUDGE_SYSTEM, [{"role": "user", "content": f"<context>\n{context}\n</context>\n\nEXPLANATION\n{body['explanation']}"}],
        max_tokens=500,
    )
    try:
        verdict = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
        return bool(verdict["faithful"]), list(verdict.get("unsupported", []))
    except (ValueError, KeyError):
        return None, [f"unparseable judge reply: {reply[:120]}"]


def summarise(results: list[dict]) -> dict[str, tuple[int, int]]:
    out = {}
    for check in CHECKS:
        graded = [r["checks"][check] for r in results if r.get("checks") and r["checks"][check] is not None]
        if graded:
            out[check] = (sum(graded), len(graded))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--refresh", action="store_true", help="re-fetch news and regenerate explanations")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness check")
    ap.add_argument("--judge-model", default="claude-sonnet-5",
                    help="a stronger model than the writer's: a model grading its own output is biased and noisy")
    ap.add_argument("--fail-under", type=float, default=0.0, help="exit 1 if the overall pass rate is below this (0-1)")
    args = ap.parse_args()

    cases = json.loads(CASES.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    judge_llm = None
    if args.judge:
        from app.config import get_settings
        from app.providers.llm import AnthropicLLM

        if not get_settings().anthropic_api_key:
            sys.exit("--judge needs ANTHROPIC_API_KEY")
        judge_llm = AnthropicLLM(get_settings().anthropic_api_key, args.judge_model)

    results = []
    print(f"{'case':<24}{'move':>8}  {'driver':<17}" + "".join(f"{c[:6]:>8}" for c in CHECKS) + "  hit@")
    with httpx.Client(base_url=args.base, timeout=180) as http:
        for case in cases:
            r = http.get(f"/{args.version}/stocks/{case['ticker']}/movements/{case['date']}",
                         params={"refresh": "true"} if args.refresh else {})
            if r.status_code != 200:
                print(f"{case['id']:<24} HTTP {r.status_code}: {r.text[:100]}")
                results.append({"id": case["id"], "error": f"HTTP {r.status_code}"})
                continue
            body = r.json()
            scored = score_case(case, body)
            if judge_llm and body["mode"] == "llm":
                scored["checks"]["faithful"], scored["unsupported"] = judge(judge_llm, http, args.version, body)
            results.append({"id": case["id"], **scored, "pct_change": body["movement"]["pct_change"],
                            "explanation": body["explanation"], "mode": body["mode"]})
            marks = "".join(f"{ {True: 'pass', False: 'FAIL', None: '-'}[scored['checks'][c]]:>8}" for c in CHECKS)
            print(f"{case['id']:<24}{body['movement']['pct_change']:>+7.2f}%  {scored['driver']:<17}{marks}"
                  f"  {scored['first_hit_rank'] or '-'}")

    summary = summarise(results)
    passed, total = sum(p for p, _ in summary.values()), sum(t for _, t in summary.values())
    print("\n" + "  ".join(f"{k} {p}/{t}" for k, (p, t) in summary.items()))
    print(f"overall {passed}/{total} = {passed / total:.0%}" if total else "nothing graded")

    out = Path(__file__).with_name("results") / f"{datetime.now():%Y%m%d-%H%M%S}-{args.version}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"version": args.version, "summary": summary, "results": results}, indent=2))
    print(f"details: {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    return 1 if total and passed / total < args.fail_under else 0


if __name__ == "__main__":
    sys.exit(main())
