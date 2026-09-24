from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["5", "15", "60"]


def resample_bars(one_min: pd.DataFrame, interval: str) -> pd.DataFrame:
    mins = int(interval)

    x = one_min.copy().sort_values("start_ms")

    x["start_utc"] = pd.to_datetime(
        x["start_ms"],
        unit="ms",
        utc=True,
    )

    x = x.set_index("start_utc")

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

    # only full bars
    y = y[counts == mins]

    y = y.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    # IMPORTANT FIX:
    # convert datetime index to epoch milliseconds correctly
    y["start_ms"] = (
        y.index.as_unit("ns").asi8
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
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if "vwap" in y.columns:
        cols.append("vwap")

    if "transactions" in y.columns:
        cols.append("transactions")

    cols.append("source")

    out = y.reset_index()

    return out[
        ["start_utc"] + cols
    ]


def main():
    for symbol in SYMBOLS:
        raw_path = (
            DATA
            / "raw"
            / "massive_1m"
            / symbol
            / "1.parquet"
        )

        if not raw_path.exists():
            print(f"[SKIP] {symbol} raw 1m missing")
            continue

        print(f"\n=== {symbol} ===")

        one = pd.read_parquet(raw_path)

        print(
            f"[RAW] {symbol} "
            f"{len(one):,} rows"
        )

        for interval in INTERVALS:
            out = resample_bars(
                one,
                interval,
            )

            target = (
                DATA
                / "raw"
                / "trade_kline"
                / symbol
                / f"{interval}.parquet"
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # overwrite corrupted resampled file
            out.to_parquet(
                target,
                index=False,
            )

            print(
                f"[REBUILT] {symbol} "
                f"{interval}m = "
                f"{len(out):,} rows"
            )

    print("\nDONE")


if __name__ == "__main__":
    main()