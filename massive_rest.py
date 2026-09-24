from __future__ import annotations
import time, requests
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any

class MassiveREST:
    def __init__(self, api_key: str, base_url: str, timeout: int = 30, min_seconds_between_calls: float = 12.5):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_gap = float(min_seconds_between_calls)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "QuantSight-ML-Phase1/2.0",
            "Authorization": f"Bearer {api_key}",
        })
        self._last_call = 0.0

    def _throttle(self):
        wait = self.min_gap - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _get_url(self, url: str, params: dict[str, Any] | None = None, retries: int = 6) -> dict[str, Any]:
        last = None
        for attempt in range(retries):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                self._last_call = time.monotonic()
                if r.status_code == 429:
                    retry = float(r.headers.get("Retry-After", max(self.min_gap, 12.5)))
                    time.sleep(retry)
                    continue
                r.raise_for_status()
                payload = r.json()
                status = str(payload.get("status", "")).upper()
                if status and status not in {"OK", "DELAYED"}:
                    raise RuntimeError(f"Massive status={payload.get('status')} error={payload.get('error')}")
                return payload
            except Exception as e:
                last = e
                time.sleep(min(1.0 * (2 ** attempt), 20))
        raise RuntimeError(f"Massive REST request failed after retries: {url}") from last

    def aggregates(self, ticker: str, multiplier: int, timespan: str, start_ms: int, end_ms: int, limit: int = 50000):
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_ms}/{end_ms}"
        params = {"adjusted": "true", "sort": "asc", "limit": int(limit)}
        rows = []
        while url:
            payload = self._get_url(url, params=params)
            rows.extend(payload.get("results") or [])
            nxt = payload.get("next_url")
            if not nxt:
                break
            # next_url may omit auth; Authorization header remains attached.
            url = nxt
            params = None
        return rows
