import time

import httpx


class ProviderError(Exception):
    pass


class TickerNotFound(Exception):
    pass


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    attempts: int = 3,
    backoff: float = 0.8,
    **kwargs,
) -> dict:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.request(method, url, **kwargs)
            if resp.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
                time.sleep(backoff * (2**attempt))
                continue
            if resp.status_code >= 400:
                raise ProviderError(f"{url} -> HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        except httpx.TransportError as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (2**attempt))
    raise ProviderError(f"{url} unreachable: {last}")
