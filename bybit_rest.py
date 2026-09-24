from __future__ import annotations

import time
from typing import Any, Iterable
import requests

from common import now_ms, is_closed_candle

class BybitPublicREST:
    def __init__(self, base_url: str, timeout: int = 20, sleep_seconds: float = 0.08):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "QuantSight-ML-Phase1/1.0"})

    def _get(self, path: str, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
        url = self.base_url + path
        last = None
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                payload = r.json()
                if payload.get("retCode") != 0:
                    raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")
                time.sleep(self.sleep_seconds)
                return payload
            except Exception as e:
                last = e
                time.sleep(min(0.5 * (2 ** attempt), 8))
        raise RuntimeError(f"REST request failed after retries: {url} {params}") from last

    def fetch_kline_family(
        self,
        endpoint: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        category: str = "linear",
    ) -> list[list[str]]:
        """Backward paginate kline-like endpoints, de-duplicate, sort ascending."""
        rows: dict[int, list[str]] = {}
        cursor_end = end_ms
        while cursor_end >= start_ms:
            payload = self._get(endpoint, {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "end": cursor_end,
                "limit": 1000,
            })
            batch = payload["result"].get("list", [])
            if not batch:
                break
            earliest = None
            for row in batch:
                ts = int(row[0])
                earliest = ts if earliest is None else min(earliest, ts)
                if start_ms <= ts <= end_ms:
                    rows[ts] = row
            if earliest is None or earliest <= start_ms:
                break
            next_end = earliest - 1
            if next_end >= cursor_end:
                break
            cursor_end = next_end
        return [rows[k] for k in sorted(rows)]

    def trade_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, category="linear"):
        raw = self.fetch_kline_family("/v5/market/kline", symbol, interval, start_ms, end_ms, category)
        at_ms = now_ms()
        return [r for r in raw if is_closed_candle(int(r[0]), interval, at_ms)]

    def mark_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, category="linear"):
        raw = self.fetch_kline_family("/v5/market/mark-price-kline", symbol, interval, start_ms, end_ms, category)
        at_ms = now_ms()
        return [r for r in raw if is_closed_candle(int(r[0]), interval, at_ms)]

    def index_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, category="linear"):
        raw = self.fetch_kline_family("/v5/market/index-price-kline", symbol, interval, start_ms, end_ms, category)
        at_ms = now_ms()
        return [r for r in raw if is_closed_candle(int(r[0]), interval, at_ms)]

    def funding_history(self, symbol: str, start_ms: int, end_ms: int, category="linear") -> list[dict[str, str]]:
        """
        Backward paginate funding history.
        Bybit permits endTime alone; using it avoids the invalid 'startTime-only' case.
        """
        out: dict[int, dict[str, str]] = {}
        cursor_end = end_ms
        while cursor_end >= start_ms:
            payload = self._get("/v5/market/funding/history", {
                "category": category,
                "symbol": symbol,
                "endTime": cursor_end,
                "limit": 200,
            })
            batch = payload["result"].get("list", [])
            if not batch:
                break
            timestamps = []
            for item in batch:
                ts = int(item["fundingRateTimestamp"])
                timestamps.append(ts)
                if start_ms <= ts <= end_ms:
                    out[ts] = item
            earliest = min(timestamps)
            if earliest <= start_ms:
                break
            next_end = earliest - 1
            if next_end >= cursor_end:
                break
            cursor_end = next_end
        return [out[k] for k in sorted(out)]
