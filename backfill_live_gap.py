from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from common import DATA_DIR


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

INTERVAL = "5"
INTERVAL_MS = 5 * 60 * 1000

# 48-bar rolling feature-এর জন্য যথেষ্ট buffer.
FETCH_LIMIT = 120

BYBIT_URL = "https://api.bybit.com/v5/market/kline"


def atomic_write(path: Path, df: pd.DataFrame):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp.parquet"
    )

    df.to_parquet(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )


def fetch_recent_closed(symbol: str) -> pd.DataFrame:
    response = requests.get(
        BYBIT_URL,
        params={
            "category": "linear",
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": FETCH_LIMIT,
        },
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"{symbol}: {payload}"
        )

    rows = payload[
        "result"
    ][
        "list"
    ]

    now_ms = int(
        time.time() * 1000
    )

    current_bucket = (
        now_ms
        // INTERVAL_MS
        * INTERVAL_MS
    )

    output = []

    for item in rows:
        start_ms = int(
            item[0]
        )

        # Current still-open 5m candle বাদ.
        if start_ms >= current_bucket:
            continue

        output.append(
            {
                "symbol": symbol,
                "interval": INTERVAL,
                "start_ms": start_ms,
                "start_utc": pd.to_datetime(
                    start_ms,
                    unit="ms",
                    utc=True,
                ),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "turnover": float(item[6]),
                "source": "bybit_rest_recent_backfill",
            }
        )

    if not output:
        raise RuntimeError(
            f"{symbol}: no closed candles returned"
        )

    df = pd.DataFrame(
        output
    )

    df = (
        df
        .sort_values("start_ms")
        .drop_duplicates(
            subset=["start_ms"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def merge_parquet(
    path: Path,
    fresh: pd.DataFrame,
):
    if path.exists():
        old = pd.read_parquet(
            path
        )

        all_columns = sorted(
            set(old.columns)
            | set(fresh.columns)
        )

        old = old.reindex(
            columns=all_columns
        )

        fresh = fresh.reindex(
            columns=all_columns
        )

        merged = pd.concat(
            [
                old,
                fresh,
            ],
            ignore_index=True,
        )

    else:
        merged = fresh.copy()

    merged["start_ms"] = pd.to_numeric(
        merged["start_ms"],
        errors="coerce",
    )

    merged = merged.dropna(
        subset=["start_ms"]
    ).copy()

    merged["start_ms"] = (
        merged["start_ms"]
        .astype("int64")
    )

    merged = (
        merged
        .sort_values("start_ms")
        .drop_duplicates(
            subset=["start_ms"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    atomic_write(
        path,
        merged,
    )

    return merged


def recent_continuity(df: pd.DataFrame):
    x = (
        df[["start_ms"]]
        .drop_duplicates()
        .sort_values("start_ms")
        .reset_index(drop=True)
    )

    if x.empty:
        return 0

    count = 1

    values = x[
        "start_ms"
    ].tolist()

    for i in range(
        len(values) - 1,
        0,
        -1,
    ):
        if (
            values[i]
            - values[i - 1]
            == INTERVAL_MS
        ):
            count += 1
        else:
            break

    return count


def main():
    print(
        "QuantSight recent 5m backfill"
    )

    print(
        "NO ORDERS"
    )

    for symbol in SYMBOLS:
        print(
            f"\n[{symbol}] fetching..."
        )

        fresh = fetch_recent_closed(
            symbol
        )

        canonical_path = (
            DATA_DIR
            / "raw"
            / "trade_kline"
            / symbol
            / "5.parquet"
        )

        live_path = (
            DATA_DIR
            / "live"
            / "kline"
            / symbol
            / "5.parquet"
        )

        canonical = merge_parquet(
            canonical_path,
            fresh,
        )

        live = merge_parquet(
            live_path,
            fresh,
        )

        canonical_contiguous = (
            recent_continuity(
                canonical
            )
        )

        live_contiguous = (
            recent_continuity(
                live
            )
        )

        print(
            f"[{symbol}] fetched={len(fresh)}"
        )

        print(
            f"[{symbol}] "
            f"latest={int(fresh['start_ms'].max())}"
        )

        print(
            f"[{symbol}] "
            f"canonical recent contiguous="
            f"{canonical_contiguous} bars"
        )

        print(
            f"[{symbol}] "
            f"live recent contiguous="
            f"{live_contiguous} bars"
        )

        if canonical_contiguous >= 48:
            print(
                f"[{symbol}] CONTINUITY PASS"
            )
        else:
            print(
                f"[{symbol}] CONTINUITY FAIL"
            )

        time.sleep(
            0.2
        )

    print(
        "\nRecent 5m backfill complete."
    )


if __name__ == "__main__":
    main()