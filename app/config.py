from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"

    news_provider: Literal["exa", "newsapi"] = "exa"
    exa_api_key: str | None = None
    newsapi_api_key: str | None = None

    llm_provider: Literal["anthropic", "openai", "none"] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_model: str | None = None

    benchmark_symbol: str = "SPY"
    default_lookback_days: int = 90
    max_range_days: int = 730
    default_min_change_pct: float = 2.0
    max_news_movements: int = 12
    articles_per_category: int = 5
    news_concurrency: int = 4
    max_peers: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
