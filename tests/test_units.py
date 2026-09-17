from datetime import date, datetime

import pytest

from app.providers.news import RawArticle
from app.providers.prices import Bar
from app.services.movements import classify_driver, compute_metrics, macro_scope
from app.services.ranking import balanced_top, macro_topic, rank_articles, rank_macro, short_name
from tests.fakes import business_days


def _bars(closes, start=date(2026, 1, 1)):
    days = list(business_days(start, date(2026, 12, 31)))
    return [Bar(date=d, open=c, high=c, low=c, close=c, volume=1) for d, c in zip(days, closes)]


def test_pct_change_and_zscore():
    closes = [100 * (1.005 if i % 2 else 0.995) ** 1 for i in range(40)]
    closes.append(closes[-1] * 1.05)
    df = compute_metrics(_bars(closes))
    assert df["pct_change"].iloc[0] != df["pct_change"].iloc[0]
    assert df["pct_change"].iloc[-1] == pytest.approx(5.0)
    assert df["zscore"].iloc[-1] > 4
    assert df["zscore"].iloc[-1] == pytest.approx(5.0 / df["pct_change"].iloc[-31:-1].std())


def test_compute_metrics_empty_and_unsorted():
    assert compute_metrics([]).empty
    bars = _bars([100, 110])
    assert compute_metrics(bars[::-1])["pct_change"].iloc[-1] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "pct,market,sector,peers,expected",
    [
        (6.0, 0.1, 0.3, [0.2, -0.1], "company_specific"),
        (-5.0, -0.4, -2.1, [-4.2, -3.8], "industry_wide"),
        (-5.0, -0.4, None, [-4.2, -3.8], "industry_wide"),
        (-3.0, -1.8, -1.9, [-2.5], "market_wide"),
        (-3.0, 1.5, 1.2, [2.0], "company_specific"),
        (3.0, None, None, [], "company_specific"),
    ],
)
def test_classify_driver(pct, market, sector, peers, expected):
    assert classify_driver(pct, market, sector, peers) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NVIDIA Corporation", "NVIDIA"),
        ("Amazon.com, Inc.", "Amazon"),
        ("Meta Platforms, Inc.", "Meta"),
        ("The Coca-Cola Company", "Coca-Cola"),
        ("Advanced Micro Devices, Inc.", "Advanced Micro Devices"),
        ("Micron Technology, Inc.", "Micron"),
        ("Alphabet Inc. Class A", "Alphabet"),
        ("AT&T Inc.", "AT&T"),
    ],
)
def test_short_name(raw, expected):
    assert short_name(raw) == expected


def _rank(articles):
    return rank_articles(
        articles,
        company_names=["NVIDIA"], company_tickers=["NVDA"],
        industry_names=["Advanced Micro Devices", "Semiconductors"], industry_tickers=["AMD"],
        move_date=date(2026, 3, 10), prev_trading_date=date(2026, 3, 9), per_category=2,
    )


def test_ranking_filters_dedupes_and_categorises():
    arts = [
        RawArticle("Bakery wins award", "https://x.com/a", published_at=datetime(2026, 3, 10)),
        RawArticle("Nvidia beats earnings", "https://x.com/b?utm=1", published_at=datetime(2026, 3, 9, 22)),
        RawArticle("NVIDIA beats earnings!", "https://y.com/copy", published_at=datetime(2026, 3, 10)),
        RawArticle("Nvidia beats earnings", "https://x.com/b", published_at=datetime(2026, 3, 9, 22)),
        RawArticle("AMD unveils new GPU", "https://x.com/c", published_at=datetime(2026, 3, 10)),
        RawArticle("Chip sector wrap", "https://x.com/d", snippet="Semiconductors rallied", published_at=datetime(2026, 3, 12)),
        RawArticle("Old news about NVDA", "https://x.com/e", published_at=datetime(2026, 3, 1)),
    ]
    ranked = _rank(arts)
    titles = [r.raw.title for r in ranked]
    assert "Bakery wins award" not in titles
    assert len([t for t in titles if "beats earnings" in t]) == 1
    assert ranked[0].raw.title == "Nvidia beats earnings" and ranked[0].category == "company"
    by_title = {r.raw.title: r for r in ranked}
    assert by_title["AMD unveils new GPU"].category == "industry"
    assert by_title["Nvidia beats earnings"].relevance > by_title["Old news about NVDA"].relevance
    assert by_title["AMD unveils new GPU"].relevance > by_title["Chip sector wrap"].relevance


def test_ranking_ticker_matching_is_strict():
    arts = [RawArticle("Camden council votes on budget", "https://x.com/1"), RawArticle("amd64 packages updated", "https://x.com/2")]
    assert _rank(arts) == []


def test_per_category_cap():
    arts = [RawArticle(f"NVIDIA story {i}", f"https://x.com/{i}") for i in range(5)]
    assert len(_rank(arts)) == 2


def test_balanced_top_keeps_industry_news_when_truncating():
    from types import SimpleNamespace as A

    arts = [A(category="company", relevance=0.95, id=i) for i in range(5)]
    arts += [A(category="industry", relevance=r, id=10 + i) for i, r in enumerate([0.8, 0.65, 0.45])]

    top = balanced_top(arts, 4)
    assert [a.category for a in top] == ["company", "company", "industry", "industry"]
    assert [a.id for a in top] == [0, 1, 10, 11]
    assert len(balanced_top(arts, 20)) == 8 and balanced_top(arts, 0) == [] and balanced_top([], 3) == []
    only_company = balanced_top(arts[:5], 3)
    assert [a.id for a in only_company] == [0, 1, 2]


def test_macro_scope_follows_the_price_driver():
    assert macro_scope("market_wide", "Semiconductors", "Technology") == "market"
    assert macro_scope("industry_wide", "Semiconductors", "Technology") == "industry:Semiconductors"
    assert macro_scope("industry_wide", None, "Technology") == "industry:Technology"
    assert macro_scope("industry_wide", None, None) is None
    assert macro_scope("company_specific", "Semiconductors", "Technology") is None


@pytest.mark.parametrize("title,topic", [
    ("Fed cuts interest rates by a quarter point", "monetary_policy"),
    ("Hot CPI print revives inflation fears", "economy"),
    ("Trump threatens new tariffs as trade war escalates", "trade"),
    ("US tightens export controls on AI chips to China", "trade"),
    ("Oil prices spike after missile strike in the Middle East", "geopolitics"),
    ("EU regulators open antitrust case against cloud providers", "regulation"),
    ("Nvidia unveils new GPU at annual conference", None),
    ("Federated learning startup raises $20M", None),
])
def test_macro_topic(title, topic):
    assert macro_topic(title) == topic


def test_macro_topic_prefers_title_then_snippet():
    assert macro_topic("Chip stocks slide", "Traders blamed new tariffs on imports.") == "trade"
    assert macro_topic("Fed decision looms", "Tariffs also weighed.") == "monetary_policy"


def test_rank_macro_gates_on_vocabulary_and_dedupes():
    def art(title, url, day, snippet=None):
        return RawArticle(title=title, url=url, published_at=datetime(2026, 5, day, 12), snippet=snippet)

    ranked = rank_macro(
        [
            art("Fed holds rates; Wall Street slides", "https://a.com/fed", 5),
            art("Fed holds rates: Wall Street slides", "https://b.com/copy", 5),
            art("Stocks fall for a third day", "https://a.com/stocks", 5, "Investors cited new tariffs."),
            art("Celebrity chef opens restaurant", "https://a.com/chef", 5),
            art("ECB signals rate cut", "https://a.com/ecb", 1),
        ],
        move_date=date(2026, 5, 5), prev_trading_date=date(2026, 5, 4), limit=5,
    )
    assert [(a.url, topic, rel) for a, topic, rel in ranked] == [
        ("https://a.com/fed", "monetary_policy", 0.8),
        ("https://a.com/stocks", "trade", 0.6),
        ("https://a.com/ecb", "monetary_policy", 0.45),
    ]
    many = [art(f"Fed speaker {i} talks rates", f"https://a.com/{i}", 5) for i in range(9)]
    assert len(rank_macro(many, move_date=date(2026, 5, 5), prev_trading_date=date(2026, 5, 4), limit=5)) == 5


@pytest.mark.parametrize("pct,market,sector,peers,expected", [
    (-22.38, 0.14, -0.59, [-2.13], "company_specific"),
    (-14.26, -0.48, -2.52, [-1.12], "company_specific"),
    (8.74, 0.66, 3.16, [0.97], "company_specific"),
    (-16.97, -1.41, -4.90, [-11.71], "industry_wide"),
    (-3.55, -1.54, -2.64, [-4.14], "market_wide"),
    (-7.81, -4.93, -6.82, [-10.51], "market_wide"),
    (-2.0, -1.0, None, [], "market_wide"),
    (-2.0, -0.9, None, [], "company_specific"),
])
def test_driver_needs_a_meaningful_share_of_the_move(pct, market, sector, peers, expected):
    assert classify_driver(pct, market, sector, peers) == expected


def test_headline_about_the_price_reaction_outranks_generic_coverage():
    def art(title, url):
        return RawArticle(title=title, url=url, published_at=datetime(2025, 10, 30, 12))

    ranked = rank_articles(
        [
            art("META earns new Street-high price target from Rosenblatt", "https://a.com/1"),
            art("5 takeaways from Meta's Q3 earnings call", "https://a.com/2"),
            art("Meta's spending outlook is sending the stock lower", "https://a.com/3"),
            art("Meta hosts annual developer picnic", "https://a.com/4"),
        ],
        company_names=["Meta"], company_tickers=["META"], industry_names=[], industry_tickers=[],
        move_date=date(2025, 10, 30), prev_trading_date=date(2025, 10, 29), per_category=5,
    )
    assert [(r.raw.url[-1], r.relevance) for r in ranked] == [("3", 1.0), ("1", 0.9), ("2", 0.9), ("4", 0.8)]


def test_cited_refs_accepts_the_formats_models_actually_write():
    from app.services.chat import cited_refs

    assert cited_refs("Fed held rates [4, 6] while Nvidia news [1][2] was positive [price-based driver].") == {1, 2, 4, 6}
    assert cited_refs("No citations here, and [n] is not one.") == set()
