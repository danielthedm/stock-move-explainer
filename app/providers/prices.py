"""Price + company-profile provider. yfinance is the only implementation, but
everything downstream depends on the Protocol so it is swappable/testable."""
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from app.providers.base import ProviderError

# Yahoo sectorKey -> SPDR sector ETF, used as the "industry" price benchmark.
SECTOR_ETFS = {
    "technology": "XLK",
    "financial-services": "XLF",
    "healthcare": "XLV",
    "consumer-cyclical": "XLY",
    "consumer-defensive": "XLP",
    "energy": "XLE",
    "industrials": "XLI",
    "basic-materials": "XLB",
    "utilities": "XLU",
    "real-estate": "XLRE",
    "communication-services": "XLC",
}


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class CompanyProfile:
    ticker: str
    name: str
    sector: str | None = None
    industry: str | None = None
    sector_etf: str | None = None
    peers: list[dict] = field(default_factory=list)  # [{"symbol","name"}]


class PriceProvider(Protocol):
    def get_profile(self, ticker: str, max_peers: int) -> CompanyProfile: ...

    def get_history(self, symbols: list[str], start: date, end: date) -> dict[str, list[Bar]]: ...


def frame_to_bars(df: pd.DataFrame | None, symbols: list[str]) -> dict[str, list[Bar]]:
    """Normalise a yf.download(group_by="ticker") frame into plain Bars.

    Handles both the MultiIndex (ticker, field) layout and the flat layout some
    yfinance versions return for a single symbol.
    """
    out: dict[str, list[Bar]] = {s: [] for s in symbols}
    if df is None or df.empty:
        return out
    for symbol in symbols:
        if isinstance(df.columns, pd.MultiIndex):
            if symbol not in df.columns.get_level_values(0):
                continue
            sub = df[symbol]
        else:
            sub = df
        sub = sub.dropna(subset=["Close"])
        for ts, row in sub.iterrows():
            out[symbol].append(
                Bar(
                    date=pd.Timestamp(ts).date(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                )
            )
    return out


class YFinancePriceProvider:
    def get_profile(self, ticker: str, max_peers: int) -> CompanyProfile:
        """Best effort: a profile failure degrades to 'no peers', it never
        blocks price data (ticker validity is decided by price history)."""
        import yfinance as yf

        profile = CompanyProfile(ticker=ticker, name=ticker)
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            return profile
        profile.name = info.get("longName") or info.get("shortName") or ticker
        profile.sector = info.get("sector")
        profile.industry = info.get("industry")
        profile.sector_etf = SECTOR_ETFS.get(info.get("sectorKey") or "")

        industry_key = info.get("industryKey")
        if industry_key:
            try:
                top = yf.Industry(industry_key).top_companies  # index=symbol
                if top is not None and not top.empty:
                    if "market weight" in top.columns:
                        top = top.sort_values("market weight", ascending=False)
                    for symbol, row in top.iterrows():
                        if symbol and symbol != ticker and len(profile.peers) < max_peers:
                            profile.peers.append({"symbol": symbol, "name": row.get("name") or symbol})
            except Exception:
                pass
        return profile

    def get_history(self, symbols: list[str], start: date, end: date) -> dict[str, list[Bar]]:
        bars = self._download(symbols, start, end)
        # yfinance swallows per-symbol failures (it only logs them), so a
        # transient error looks exactly like a delisted ticker. Retry empties once
        # so a blip on the primary ticker doesn't become a false 404.
        missing = [s for s in symbols if not bars[s]]
        if missing:
            bars.update(self._download(missing, start, end))
        return bars

    def _download(self, symbols: list[str], start: date, end: date) -> dict[str, list[Bar]]:
        import yfinance as yf

        try:
            df = yf.download(
                tickers=symbols,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
                interval="1d",
                auto_adjust=True,  # split/dividend adjusted -> no fake "movements"
                group_by="ticker",
                progress=False,
                # Sequential: threaded downloads race on yfinance's SQLite tz cache
                # ("database is locked") and silently drop symbols.
                threads=False,
            )
        except Exception as exc:  # yfinance raises a zoo of exception types
            raise ProviderError(f"yfinance download failed: {exc}") from exc
        return frame_to_bars(df, symbols)
