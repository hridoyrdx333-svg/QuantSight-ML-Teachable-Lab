from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import (
    DATA_DIR,
    load_config,
    massive_api_key,
    now_ms,
    utc_iso,
    file_sha256,
)
from massive_rest import MassiveREST
from bybit_rest import BybitPublicREST
from storage import upsert_parquet


def massive_minute_df(rows, symbol: str, ticker: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    rename = {
        "t": "start_ms",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "n": "transactions",
    }

    df = df.rename(columns=rename)

    keep = [
        c
        for c in [
            "start_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "transactions",
        ]
        if c in df.columns
    ]

    df = df[keep].copy()

    df["start_ms"] = pd.to_numeric(
        df["start_ms"],
        errors="raise",
    ).astype("int64")

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

    df.insert(0, "symbol", symbol)
    df.insert(1, "provider_ticker", ticker)

    df["start_utc"] = pd.to_datetime(
        df["start_ms"],
        unit="ms",
        utc=True,
    )

    df["source"] = "massive_crypto_aggregate_1m"

    return df


def resample_bars(
    one_min: pd.DataFrame,
    interval: str,
) -> pd.DataFrame:

    if one_min.empty:
        return one_min

    mins = int(interval)

    x = (
        one_min
        .copy()
        .set_index("start_utc")
        .sort_index()
    )

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    if "vwap" in x.columns:
        agg["vwap"] = "mean"

    if "transactions" in x.columns:
        agg["transactions"] = "sum"

    y = x.resample(
        f"{mins}min",
        label="left",
        closed="left",
    ).agg(agg)

    counts = x["close"].resample(
        f"{mins}min",
        label="left",
        closed="left",
    ).count()

    # Keep only fully completed resampled bars.
    y = y[counts == mins]

    y = y.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    y["start_ms"] = (
        y.index.astype("int64")
        // 1_000_000
    ).astype("int64")

    y["symbol"] = one_min["symbol"].iloc[0]
    y["provider_ticker"] = one_min["provider_ticker"].iloc[0]
    y["interval"] = interval
    y["source"] = f"massive_1m_resampled_{interval}m"

    cols = [
        "symbol",
        "provider_ticker",
        "interval",
        "start_ms",
        "start_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for c in [
        "vwap",
        "transactions",
    ]:
        if c in y.columns:
            cols.append(c)

    cols.append("source")

    return y.reset_index()[cols]


def bybit_price_df(
    rows,
    symbol,
    interval,
    source,
):

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [r[:5] for r in rows],
        columns=[
            "start_ms",
            "open",
            "high",
            "low",
            "close",
        ],
    )

    df["start_ms"] = pd.to_numeric(
        df["start_ms"],
        errors="raise",
    ).astype("int64")

    for c in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df.insert(
        0,
        "symbol",
        symbol,
    )

    df.insert(
        1,
        "interval",
        interval,
    )

    df["start_utc"] = pd.to_datetime(
        df["start_ms"],
        unit="ms",
        utc=True,
    )

    df["source"] = source

    return df


def funding_df(rows):

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(
        columns={
            "fundingRateTimestamp": "timestamp_ms",
            "fundingRate": "funding_rate",
        }
    )

    df["timestamp_ms"] = pd.to_numeric(
        df["timestamp_ms"],
        errors="raise",
    ).astype("int64")

    df["funding_rate"] = pd.to_numeric(
        df["funding_rate"],
        errors="coerce",
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_ms"],
        unit="ms",
        utc=True,
    )

    df["source"] = "bybit_rest_funding_history"

    return df[
        [
            "symbol",
            "timestamp_ms",
            "timestamp_utc",
            "funding_rate",
            "source",
        ]
    ]


def main():

    cfg = load_config()

    key = massive_api_key(True)

    end_ms = now_ms() - 1

    massive_start = (
        end_ms
        - int(cfg["history_days"])
        * 86_400_000
    )

    mc = cfg["massive"]

    massive = MassiveREST(
        key,
        mc["base_url"],
        int(
            mc.get(
                "request_timeout_seconds",
                30,
            )
        ),
        float(
            mc.get(
                "min_seconds_between_calls",
                12.5,
            )
        ),
    )

    bc = cfg["bybit"]

    bybit = BybitPublicREST(
        bc["rest_base"],
        int(
            bc.get(
                "request_timeout_seconds",
                20,
            )
        ),
        float(
            bc.get(
                "request_sleep_seconds",
                0.08,
            )
        ),
    )

    bybit_start = (
        end_ms
        - int(
            bc.get(
                "history_days",
                365,
            )
        )
        * 86_400_000
    )

    manifest = {
        "phase": 1,
        "collector": "massive+bybit",
        "started_at_ms": now_ms(),
        "massive_target_start_utc": utc_iso(
            massive_start
        ),
        "target_end_utc": utc_iso(
            end_ms
        ),
        "datasets": [],
        "errors": [],
    }

    chunk_ms = (
        int(
            mc.get(
                "chunk_days",
                30,
            )
        )
        * 86_400_000
    )

    for symbol in cfg["symbols"]:

        ticker = cfg["massive_tickers"][symbol]

        print(
            f"\n=== Massive {ticker} -> {symbol} ===",
            flush=True,
        )

        # Raw 1-minute storage path.
        p1 = (
            DATA_DIR
            / "raw"
            / "massive_1m"
            / symbol
            / "1.parquet"
        )

        cur = massive_start

        while cur <= end_ms:

            ce = min(
                cur + chunk_ms - 1,
                end_ms,
            )

            try:

                print(
                    f"[GET] {ticker} 1m "
                    f"{utc_iso(cur)} -> {utc_iso(ce)}",
                    flush=True,
                )

                rows = massive.aggregates(
                    ticker,
                    int(
                        mc.get(
                            "source_multiplier",
                            1,
                        )
                    ),
                    mc.get(
                        "source_timespan",
                        "minute",
                    ),
                    cur,
                    ce,
                    int(
                        mc.get(
                            "limit",
                            50000,
                        )
                    ),
                )

                chunk_df = massive_minute_df(
                    rows,
                    symbol,
                    ticker,
                )

                if not chunk_df.empty:

                    # SAVE EVERY CHUNK IMMEDIATELY.
                    # This means UI can see 1m data
                    # before the full 2-year job finishes.
                    upsert_parquet(
                        p1,
                        chunk_df,
                        [
                            "symbol",
                            "start_ms",
                        ],
                        [
                            "symbol",
                            "start_ms",
                        ],
                    )

                    saved_df = pd.read_parquet(
                        p1
                    )

                    print(
                        f"[SAVE] {symbol} 1m "
                        f"chunk={len(chunk_df):,} "
                        f"total={len(saved_df):,}",
                        flush=True,
                    )

                else:

                    print(
                        f"[EMPTY] {symbol} "
                        f"{utc_iso(cur)} -> "
                        f"{utc_iso(ce)}",
                        flush=True,
                    )

            except Exception as e:

                msg = (
                    f"massive "
                    f"{ticker} "
                    f"{cur}-{ce}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                manifest["errors"].append(
                    msg
                )

                print(
                    "[ERROR]",
                    msg,
                    flush=True,
                )

            cur = ce + 1

        # Reload complete raw 1m dataset
        # from disk after all chunks.
        if p1.exists():
            one = pd.read_parquet(
                p1
            )
        else:
            one = pd.DataFrame()

        manifest["datasets"].append(
            {
                "dataset": "massive_1m",
                "symbol": symbol,
                "rows": int(
                    len(one)
                ),
                "path": str(
                    p1.relative_to(
                        DATA_DIR.parent
                    )
                ),
                "sha256": (
                    file_sha256(p1)
                    if p1.exists()
                    else None
                ),
            }
        )

        print(
            f"[RAW COMPLETE] "
            f"{symbol} 1m rows="
            f"{len(one):,}",
            flush=True,
        )

        # Build 5m / 15m / 60m
        # from the collected 1m source.
        for interval in cfg["intervals"]:

            d = resample_bars(
                one,
                interval,
            )

            p = (
                DATA_DIR
                / "raw"
                / "trade_kline"
                / symbol
                / f"{interval}.parquet"
            )

            if not d.empty:

                upsert_parquet(
                    p,
                    d,
                    [
                        "symbol",
                        "interval",
                        "start_ms",
                    ],
                    [
                        "symbol",
                        "interval",
                        "start_ms",
                    ],
                )

            manifest["datasets"].append(
                {
                    "dataset":
                    "massive_resampled_trade_kline",
                    "symbol":
                    symbol,
                    "interval":
                    interval,
                    "rows":
                    int(
                        len(d)
                    ),
                    "path":
                    str(
                        p.relative_to(
                            DATA_DIR.parent
                        )
                    ),
                    "sha256":
                    (
                        file_sha256(p)
                        if p.exists()
                        else None
                    ),
                }
            )

            print(
                f"[OK] "
                f"{symbol} "
                f"{interval}m: "
                f"{len(d):,} bars",
                flush=True,
            )

        print(
            f"=== Bybit futures context "
            f"{symbol} ===",
            flush=True,
        )

        # Public Bybit mark/index candles.
        for interval in cfg["intervals"]:

            jobs = [
                (
                    "mark_kline",
                    bybit.mark_klines,
                    "bybit_rest_mark_kline",
                ),
                (
                    "index_kline",
                    bybit.index_klines,
                    "bybit_rest_index_kline",
                ),
            ]

            for (
                name,
                fetcher,
                source,
            ) in jobs:

                try:

                    rows = fetcher(
                        symbol,
                        interval,
                        bybit_start,
                        end_ms,
                        bc["category"],
                    )

                    d = bybit_price_df(
                        rows,
                        symbol,
                        interval,
                        source,
                    )

                    p = (
                        DATA_DIR
                        / "raw"
                        / name
                        / symbol
                        / f"{interval}.parquet"
                    )

                    if not d.empty:

                        upsert_parquet(
                            p,
                            d,
                            [
                                "symbol",
                                "interval",
                                "start_ms",
                            ],
                            [
                                "symbol",
                                "interval",
                                "start_ms",
                            ],
                        )

                    manifest["datasets"].append(
                        {
                            "dataset":
                            name,
                            "symbol":
                            symbol,
                            "interval":
                            interval,
                            "rows":
                            int(
                                len(d)
                            ),
                            "path":
                            str(
                                p.relative_to(
                                    DATA_DIR.parent
                                )
                            ),
                            "sha256":
                            (
                                file_sha256(p)
                                if p.exists()
                                else None
                            ),
                        }
                    )

                    print(
                        f"[OK] "
                        f"{name} "
                        f"{symbol} "
                        f"{interval}: "
                        f"{len(d):,}",
                        flush=True,
                    )

                except Exception as e:

                    msg = (
                        f"{name} "
                        f"{symbol} "
                        f"{interval}: "
                        f"{type(e).__name__}: "
                        f"{e}"
                    )

                    manifest["errors"].append(
                        msg
                    )

                    print(
                        "[ERROR]",
                        msg,
                        flush=True,
                    )

        # Public Bybit funding history.
        try:

            rows = bybit.funding_history(
                symbol,
                bybit_start,
                end_ms,
                bc["category"],
            )

            d = funding_df(
                rows
            )

            p = (
                DATA_DIR
                / "raw"
                / "funding"
                / symbol
                / "funding.parquet"
            )

            if not d.empty:

                upsert_parquet(
                    p,
                    d,
                    [
                        "symbol",
                        "timestamp_ms",
                    ],
                    [
                        "symbol",
                        "timestamp_ms",
                    ],
                )

            manifest["datasets"].append(
                {
                    "dataset":
                    "funding",
                    "symbol":
                    symbol,
                    "rows":
                    int(
                        len(d)
                    ),
                    "path":
                    str(
                        p.relative_to(
                            DATA_DIR.parent
                        )
                    ),
                    "sha256":
                    (
                        file_sha256(p)
                        if p.exists()
                        else None
                    ),
                }
            )

            print(
                f"[OK] funding "
                f"{symbol}: "
                f"{len(d):,}",
                flush=True,
            )

        except Exception as e:

            msg = (
                f"funding "
                f"{symbol}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

            manifest["errors"].append(
                msg
            )

            print(
                "[ERROR]",
                msg,
                flush=True,
            )

    manifest["finished_at_ms"] = now_ms()

    out = (
        DATA_DIR
        / "dataset_manifest.json"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nManifest:",
        out,
        flush=True,
    )

    print(
        "Errors:",
        len(
            manifest["errors"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()