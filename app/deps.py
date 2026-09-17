from functools import lru_cache

from app.config import get_settings
from app.providers.llm import AnthropicLLM, LLMProvider, OpenAILLM
from app.providers.news import ExaNewsProvider, NewsApiProvider, NewsProvider
from app.providers.prices import PriceProvider, YFinancePriceProvider


@lru_cache
def get_price_provider() -> PriceProvider:
    return YFinancePriceProvider()


@lru_cache
def get_news_provider() -> NewsProvider | None:
    s = get_settings()
    if s.news_provider == "newsapi" and s.newsapi_api_key:
        return NewsApiProvider(s.newsapi_api_key)
    if s.exa_api_key:
        return ExaNewsProvider(s.exa_api_key)
    if s.newsapi_api_key:
        return NewsApiProvider(s.newsapi_api_key)
    return None


@lru_cache
def get_llm() -> LLMProvider | None:
    s = get_settings()
    if s.llm_provider == "none":
        return None
    if s.llm_provider == "openai" and s.openai_api_key:
        return OpenAILLM(s.openai_api_key, s.llm_model)
    if s.anthropic_api_key:
        return AnthropicLLM(s.anthropic_api_key, s.llm_model)
    if s.openai_api_key:
        return OpenAILLM(s.openai_api_key, s.llm_model)
    return None
