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
FETCH_LIMIT = 120

BASE = "https://api.bybit.com"

ENDPOINTS = {
    "trade": "/v5/market/kline",
    "mark": "/v5/market/mark-price-kline",
    "index": "/v5/market/index-price-kline",
}


def atomic_merge(
    path: Path,
    fresh: pd.DataFrame,
    keys: list[str],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        old = pd.read_parquet(
            path
        )

        cols = sorted(
            set(old.columns)
            | set(fresh.columns)
        )

        old = old.reindex(
            columns=cols
        )

        fresh = fresh.reindex(
            columns=cols
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
            subset=keys,
            keep="last",
        )
        .reset_index(drop=True)
    )

    tmp = path.with_name(
        path.name + ".tmp.parquet"
    )

    merged.to_parquet(
        tmp,
        index=False,
    )

    os.replace(
        tmp,
        path,
    )

    return merged


def fetch_rows(
    symbol: str,
    kind: str,
):
    endpoint = ENDPOINTS[kind]

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": FETCH_LIMIT,
    }

    response = requests.get(
        BASE + endpoint,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"{symbol} {kind}: {payload}"
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

        if start_ms >= current_bucket:
            continue

        row = {
            "symbol":
                symbol,

            "interval":
                INTERVAL,

            "start_ms":
                start_ms,

            "start_utc":
                pd.to_datetime(
                    start_ms,
                    unit="ms",
                    utc=True,
                ),

            "open":
                float(item[1]),

            "high":
                float(item[2]),

            "low":
                float(item[3]),

            "close":
                float(item[4]),

            "source":
                f"bybit_rest_recent_{kind}_backfill",
        }

        if kind == "trade":
            row["volume"] = float(
                item[5]
            )

            row["turnover"] = float(
                item[6]
            )

        output.append(
            row
        )

    if not output:
        raise RuntimeError(
            f"{symbol} {kind}: "
            f"no closed rows returned"
        )

    return (
        pd.DataFrame(output)
        .sort_values("start_ms")
        .drop_duplicates(
            "start_ms",
            keep="last",
        )
        .reset_index(drop=True)
    )


def recent_contiguous(
    df: pd.DataFrame,
):
    values = (
        df[["start_ms"]]
        .dropna()
        .drop_duplicates()
        .sort_values("start_ms")[
            "start_ms"
        ]
        .astype("int64")
        .tolist()
    )

    if not values:
        return 0

    count = 1

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
        "QuantSight recent 5m context backfill"
    )

    print(
        "NO ORDERS"
    )

    for symbol in SYMBOLS:
        print(
            f"\n[{symbol}]"
        )

        trade = fetch_rows(
            symbol,
            "trade",
        )

        mark = fetch_rows(
            symbol,
            "mark",
        )

        index = fetch_rows(
            symbol,
            "index",
        )

        targets = [
            (
                "trade raw",
                DATA_DIR
                / "raw"
                / "trade_kline"
                / symbol
                / "5.parquet",
                trade,
            ),
            (
                "trade live",
                DATA_DIR
                / "live"
                / "kline"
                / symbol
                / "5.parquet",
                trade,
            ),
            (
                "mark raw",
                DATA_DIR
                / "raw"
                / "mark_kline"
                / symbol
                / "5.parquet",
                mark,
            ),
            (
                "mark live",
                DATA_DIR
                / "live"
                / "mark_kline"
                / symbol
                / "5.parquet",
                mark,
            ),
            (
                "index raw",
                DATA_DIR
                / "raw"
                / "index_kline"
                / symbol
                / "5.parquet",
                index,
            ),
            (
                "index live",
                DATA_DIR
                / "live"
                / "index_kline"
                / symbol
                / "5.parquet",
                index,
            ),
        ]

        for (
            label,
            path,
            fresh,
        ) in targets:

            merged = atomic_merge(
                path,
                fresh,
                [
                    "symbol",
                    "interval",
                    "start_ms",
                ],
            )

            bars = recent_contiguous(
                merged
            )

            print(
                f"  {label}: "
                f"recent_contiguous={bars}"
            )

        print(
            f"  fetched: "
            f"trade={len(trade)} "
            f"mark={len(mark)} "
            f"index={len(index)}"
        )

    print(
        "\nContext backfill complete."
    )


if __name__ == "__main__":
    main()