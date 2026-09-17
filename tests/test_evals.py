from evals.run import score_case, summarise


def _body(articles, explanation="Tariffs announced by the White House drove the sell-off [1].", mode="llm", driver="market_wide"):
    return {"ticker": "NVDA", "mode": mode, "explanation": explanation,
            "movement": {"pct_change": -7.8, "context": {"driver": driver}, "articles": articles}}


def _art(title, category, macro_topic=None, snippet=None):
    return {"title": title, "category": category, "macro_topic": macro_topic, "snippet": snippet}


CASE = {"id": "x", "ticker": "NVDA", "date": "2025-04-03", "tier": "macro", "keywords": ["tariff"], "drivers": ["market_wide"]}


def test_score_case_passes_when_cause_is_found_in_tier_and_cited():
    scored = score_case(CASE, _body([_art("Nvidia slides", "company"), _art("Trump unveils sweeping tariffs", "macro", "trade")]))
    assert scored["checks"] == {"retrieval": True, "tier": True, "driver": True, "explanation": True,
                                "citations": True, "faithful": None}
    assert scored["first_hit_rank"] == 2


def test_score_case_tier_and_macro_topic_rules():
    v1 = score_case(CASE, _body([_art("Chip stocks sink on tariffs", "industry")]))
    assert v1["checks"]["retrieval"] is True and v1["checks"]["tier"] is False
    assert score_case(CASE, _body([_art("Chip stocks sink on tariffs", "industry", "trade")]))["checks"]["tier"] is True
    case = {**CASE, "tier": ["company", "industry"]}
    assert score_case(case, _body([_art("Shares plunge", "company", snippet="New tariffs hit.")]))["checks"]["tier"] is True


def test_score_case_failures_and_not_applicable():
    miss = score_case(CASE, _body([_art("Nvidia slides", "company")], explanation="Unclear [7].", driver="company_specific"))
    assert miss["checks"] == {"retrieval": False, "tier": False, "driver": False, "explanation": False,
                              "citations": False, "faithful": None}
    assert miss["first_hit_rank"] is None
    extractive = score_case({**CASE, "drivers": None}, _body([], mode="extractive"))
    assert extractive["checks"]["driver"] is None and extractive["checks"]["explanation"] is None
    assert score_case(CASE, _body([_art("Tariffs", "macro")], explanation="Tariffs, no citation."))["checks"]["citations"] is False


def test_summarise_ignores_errors_and_not_applicable():
    results = [{"checks": {"retrieval": True, "tier": None, "driver": False, "explanation": None, "citations": None, "faithful": None}},
               {"id": "boom", "error": "HTTP 500"}]
    assert summarise(results) == {"retrieval": (1, 1), "driver": (0, 1)}
