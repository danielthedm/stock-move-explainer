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
    closes = [100 * (1.005 if i % 2 else 0.995) ** 1 for i in range(40)]  # ~1% alternating noise
    closes.append(closes[-1] * 1.05)  # +5% shock
    df = compute_metrics(_bars(closes))
    assert df["pct_change"].iloc[0] != df["pct_change"].iloc[0]  # NaN: no previous close
    assert df["pct_change"].iloc[-1] == pytest.approx(5.0)
    assert df["zscore"].iloc[-1] > 4  # 5% is huge vs ~1% trailing vol
    # the shock day must not contaminate its own volatility estimate
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
        (-5.0, -0.4, None, [-4.2, -3.8], "industry_wide"),  # no ETF: peers decide
        (-3.0, -1.8, -1.9, [-2.5], "market_wide"),
        (-3.0, 1.5, 1.2, [2.0], "company_specific"),  # everything else went UP
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
        RawArticle("NVIDIA beats earnings!", "https://y.com/copy", published_at=datetime(2026, 3, 10)),  # same title
        RawArticle("Nvidia beats earnings", "https://x.com/b", published_at=datetime(2026, 3, 9, 22)),  # same url
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
    # in-window + catalyst beats stale mention
    assert by_title["Nvidia beats earnings"].relevance > by_title["Old news about NVDA"].relevance
    # title hit beats snippet-only hit
    assert by_title["AMD unveils new GPU"].relevance > by_title["Chip sector wrap"].relevance


def test_ranking_ticker_matching_is_strict():
    # 'AMD' must not match inside other words, and lowercase 'amd' is not a ticker hit
    arts = [RawArticle("Camden council votes on budget", "https://x.com/1"), RawArticle("amd64 packages updated", "https://x.com/2")]
    assert _rank(arts) == []


def test_per_category_cap():
    arts = [RawArticle(f"NVIDIA story {i}", f"https://x.com/{i}") for i in range(5)]
    assert len(_rank(arts)) == 2


def test_balanced_top_keeps_industry_news_when_truncating():
    """Live data: company articles all score ~0.95, industry <= 0.80, so plain
    [:n] truncation returned zero competitor/industry articles for small n."""
    from types import SimpleNamespace as A

    arts = [A(category="company", relevance=0.95, id=i) for i in range(5)]
    arts += [A(category="industry", relevance=r, id=10 + i) for i, r in enumerate([0.8, 0.65, 0.45])]

    top = balanced_top(arts, 4)
    assert [a.category for a in top] == ["company", "company", "industry", "industry"]  # sorted by relevance
    assert [a.id for a in top] == [0, 1, 10, 11]  # best of each category
    assert len(balanced_top(arts, 20)) == 8 and balanced_top(arts, 0) == [] and balanced_top([], 3) == []
    only_company = balanced_top(arts[:5], 3)
    assert [a.id for a in only_company] == [0, 1, 2]  # one category: plain top-n


def test_macro_scope_follows_the_price_driver():
    assert macro_scope("market_wide", "Semiconductors", "Technology") == "market"
    assert macro_scope("industry_wide", "Semiconductors", "Technology") == "industry:Semiconductors"
    assert macro_scope("industry_wide", None, "Technology") == "industry:Technology"
    assert macro_scope("industry_wide", None, None) is None
    assert macro_scope("company_specific", "Semiconductors", "Technology") is None


@pytest.mark.parametrize("title,topic", [
    ("Fed cuts interest rates by a quarter point", "monetary_policy"),
    ("Hot CPI print revives inflation fears", "economy"),
    ("Trump threatens new tariffs as trade war escalates", "trade"),  # "war" alone would be geopolitics
    ("US tightens export controls on AI chips to China", "trade"),
    ("Oil prices spike after missile strike in the Middle East", "geopolitics"),
    ("EU regulators open antitrust case against cloud providers", "regulation"),
    ("Nvidia unveils new GPU at annual conference", None),
    ("Federated learning startup raises $20M", None),  # whole words only: not "Fed"
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
            art("Fed holds rates: Wall Street slides", "https://b.com/copy", 5),  # syndicated copy
            art("Stocks fall for a third day", "https://a.com/stocks", 5, "Investors cited new tariffs."),
            art("Celebrity chef opens restaurant", "https://a.com/chef", 5),  # no macro vocabulary -> dropped
            art("ECB signals rate cut", "https://a.com/ecb", 1),  # outside the window: no timing bonus
        ],
        move_date=date(2026, 5, 5), prev_trading_date=date(2026, 5, 4), limit=5,
    )
    assert [(a.url, topic, rel) for a, topic, rel in ranked] == [
        ("https://a.com/fed", "monetary_policy", 0.8),
        ("https://a.com/stocks", "trade", 0.6),  # 0.25 snippet-only + 0.20 timing + 0.15 market words
        ("https://a.com/ecb", "monetary_policy", 0.45),
    ]
    many = [art(f"Fed speaker {i} talks rates", f"https://a.com/{i}", 5) for i in range(9)]
    assert len(rank_macro(many, move_date=date(2026, 5, 5), prev_trading_date=date(2026, 5, 4), limit=5)) == 5
