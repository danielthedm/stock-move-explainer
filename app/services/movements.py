from statistics import median

import pandas as pd

ZSCORE_WINDOW = 30
CO_MOVE_PCT = 1.0
CO_MOVE_SHARE = 0.4


def compute_metrics(bars: list) -> pd.DataFrame:
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
    return ref is not None and ref * pct > 0 and abs(ref) >= max(CO_MOVE_PCT, CO_MOVE_SHARE * abs(pct))


def macro_scope(driver: str, industry: str | None, sector: str | None) -> str | None:
    if driver == "market_wide":
        return "market"
    if driver == "industry_wide" and (industry or sector):
        return f"industry:{industry or sector}"
    return None


def classify_driver(pct: float, market: float | None, sector: float | None, peers: list[float]) -> str:
    if _co_moved(market, pct):
        return "market_wide"
    peer_median = median(peers) if peers else None
    if _co_moved(sector, pct) or _co_moved(peer_median, pct):
        return "industry_wide"
    return "company_specific"
