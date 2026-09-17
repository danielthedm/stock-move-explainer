"""Pure functions: what counts as a movement and what moved with it."""
from statistics import median

import pandas as pd

ZSCORE_WINDOW = 30  # trading days of trailing volatility
CO_MOVE_PCT = 1.0  # a benchmark "moved with" the stock if >= 1% the same way


def compute_metrics(bars: list) -> pd.DataFrame:
    """bars: objects with .date/.open/.high/.low/.close/.volume, any order.

    Returns a frame indexed by date with pct_change and zscore.
    zscore = today's return / stdev of the *previous* 30 returns, so a 2% day
    means something different for a utility than for a meme stock.
    """
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "prev_close", "pct_change", "zscore"])
    df = pd.DataFrame(
        {
            "date": [b.date for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    ).set_index("date").sort_index()
    df["prev_close"] = df["close"].shift(1)
    df["pct_change"] = (df["close"] / df["prev_close"] - 1.0) * 100.0
    trailing_std = df["pct_change"].shift(1).rolling(ZSCORE_WINDOW, min_periods=10).std()
    df["zscore"] = df["pct_change"] / trailing_std
    return df


def _co_moved(ref: float | None, pct: float) -> bool:
    return ref is not None and ref * pct > 0 and abs(ref) >= CO_MOVE_PCT


def classify_driver(pct: float, market: float | None, sector: float | None, peers: list[float]) -> str:
    """Cheap, explainable attribution from prices alone. It tells the reader
    (and the LLM) which *kind* of news to weight: a stock that fell 3% while
    its sector fell 3% rarely has a company-specific story."""
    if _co_moved(market, pct):
        return "market_wide"
    peer_median = median(peers) if peers else None
    if _co_moved(sector, pct) or _co_moved(peer_median, pct):
        return "industry_wide"
    return "company_specific"
