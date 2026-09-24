from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import DATA_DIR, load_config, now_ms


FEATURE_VERSION = "v4_context_plus"

BAR_MS = 300_000
LABEL_HORIZON_BARS = 6


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where(
        (up_move > down_move)
        & (up_move > 0),
        0.0,
    )

    minus_dm = down_move.where(
        (down_move > up_move)
        & (down_move > 0),
        0.0,
    )

    atr_value = atr(
        df,
        period,
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            np.nan,
        )
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def prepare_trade_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = (
        df
        .sort_values("start_ms")
        .drop_duplicates(
            subset=["start_ms"],
            keep="last",
        )
        .reset_index(drop=True)
        .copy()
    )

    gap = (
        x["start_ms"].diff()
        != BAR_MS
    )

    if len(gap):
        gap.iloc[0] = True

    x["segment_id"] = (
        gap.astype(int).cumsum()
    )

    return x


def segment_features(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    x = segment.copy()

    for bars in [
        1,
        3,
        6,
        12,
    ]:
        x[f"ret_{bars}"] = (
            x["close"].pct_change(
                periods=bars,
                fill_method=None,
            )
        )

    ema20 = x["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = x["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    ema200 = x["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    x["ema20_dist"] = (
        x["close"] / ema20 - 1
    )

    x["ema50_dist"] = (
        x["close"] / ema50 - 1
    )

    x["ema200_dist"] = (
        x["close"] / ema200 - 1
    )

    x["ema20_slope_3"] = (
        ema20.pct_change(
            periods=3,
            fill_method=None,
        )
    )

    x["ema50_slope_6"] = (
        ema50.pct_change(
            periods=6,
            fill_method=None,
        )
    )

    x["rsi14"] = rsi(
        x["close"],
        14,
    )

    x["atr14"] = atr(
        x,
        14,
    )

    x["atr_pct"] = (
        x["atr14"]
        / x["close"]
    )

    x["adx14"] = adx(
        x,
        14,
    )

    x["range_pct"] = (
        x["high"]
        - x["low"]
    ) / x["close"]

    x["body_pct"] = (
        x["close"]
        - x["open"]
    ).abs() / x["close"]

    volume_mean = (
        x["volume"]
        .rolling(
            50,
            min_periods=20,
        )
        .mean()
    )

    volume_std = (
        x["volume"]
        .rolling(
            50,
            min_periods=20,
        )
        .std()
    )

    x["volume_z"] = (
        x["volume"]
        - volume_mean
    ) / volume_std.replace(
        0,
        np.nan,
    )

    x["relative_volume_20"] = (
        x["volume"]
        / x["volume"]
        .rolling(
            20,
            min_periods=10,
        )
        .mean()
        .replace(
            0,
            np.nan,
        )
    )

    x["realized_vol_12"] = (
        x["ret_1"]
        .rolling(
            12,
            min_periods=6,
        )
        .std()
    )

    x["realized_vol_48"] = (
        x["ret_1"]
        .rolling(
            48,
            min_periods=24,
        )
        .std()
    )

    x["trend_regime"] = np.select(
        [
            (
                (x["ema20_dist"] > 0)
                & (x["ema50_dist"] > 0)
            ),
            (
                (x["ema20_dist"] < 0)
                & (x["ema50_dist"] < 0)
            ),
        ],
        [
            1.0,
            -1.0,
        ],
        default=0.0,
    )

    x["high_vol_regime"] = np.where(
        (
            x["realized_vol_12"].notna()
            & x["realized_vol_48"].notna()
        ),
        (
            x["realized_vol_12"]
            > x["realized_vol_48"]
        ).astype(float),
        np.nan,
    )

    x["future_ret_6"] = (
        x["close"].shift(
            -LABEL_HORIZON_BARS
        )
        / x["close"]
        - 1
    )

    x["label_up_6"] = np.where(
        x["future_ret_6"].isna(),
        np.nan,
        (
            x["future_ret_6"]
            > 0
        ).astype(float),
    )

    return x


def build_gap_aware_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    x = prepare_trade_data(
        raw
    )

    parts = []

    for segment_id, segment in x.groupby(
        "segment_id",
        sort=True,
    ):

        y = segment_features(
            segment.reset_index(
                drop=True
            )
        )

        y["segment_id"] = (
            segment_id
        )

        parts.append(y)

    if not parts:
        return pd.DataFrame()

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            "start_ms"
        )
        .reset_index(
            drop=True
        )
    )


def load_bybit_price(
    symbol: str,
    dataset_name: str,
) -> pd.DataFrame:

    path = (
        DATA_DIR
        / "raw"
        / dataset_name
        / symbol
        / "5.parquet"
    )

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(
        path
    )

    required = {
        "start_ms",
        "close",
    }

    if not required.issubset(
        df.columns
    ):
        return pd.DataFrame()

    return (
        df[
            [
                "start_ms",
                "close",
            ]
        ]
        .drop_duplicates(
            "start_ms",
            keep="last",
        )
        .sort_values(
            "start_ms"
        )
        .reset_index(
            drop=True
        )
    )


def add_basis_features(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    x = segment.copy()

    basis = x["mark_index_basis"]

    x["basis_change_1"] = (
        basis.diff(1)
    )

    x["basis_change_3"] = (
        basis.diff(3)
    )

    rolling_mean = (
        basis
        .rolling(
            48,
            min_periods=12,
        )
        .mean()
    )

    rolling_std = (
        basis
        .rolling(
            48,
            min_periods=12,
        )
        .std()
    )

    x["basis_z_48"] = (
        basis
        - rolling_mean
    ) / rolling_std.replace(
        0,
        np.nan,
    )

    return x


def attach_mark_index(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:

    out = df.copy()

    mark = load_bybit_price(
        symbol,
        "mark_kline",
    )

    index = load_bybit_price(
        symbol,
        "index_kline",
    )

    if not mark.empty:

        mark = mark.rename(
            columns={
                "close":
                    "mark_close"
            }
        )

        out = out.merge(
            mark,
            on="start_ms",
            how="left",
        )

    else:
        out["mark_close"] = np.nan

    if not index.empty:

        index = index.rename(
            columns={
                "close":
                    "index_close"
            }
        )

        out = out.merge(
            index,
            on="start_ms",
            how="left",
        )

    else:
        out["index_close"] = np.nan

    out["mark_divergence"] = (
        out["mark_close"]
        / out["close"]
        - 1
    )

    out["index_divergence"] = (
        out["index_close"]
        / out["close"]
        - 1
    )

    out["mark_index_basis"] = (
        out["mark_close"]
        / out["index_close"]
        - 1
    )

    parts = []

    for _, segment in out.groupby(
        "segment_id",
        sort=True,
    ):

        parts.append(
            add_basis_features(
                segment
                .sort_values(
                    "start_ms"
                )
                .copy()
            )
        )

    if not parts:
        return out

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            "start_ms"
        )
        .reset_index(
            drop=True
        )
    )


def prepare_funding(
    symbol: str,
) -> pd.DataFrame:

    path = (
        DATA_DIR
        / "raw"
        / "funding"
        / symbol
        / "funding.parquet"
    )

    if not path.exists():
        return pd.DataFrame()

    funding = pd.read_parquet(
        path
    )

    required = {
        "timestamp_ms",
        "funding_rate",
    }

    if not required.issubset(
        funding.columns
    ):
        return pd.DataFrame()

    funding = (
        funding[
            [
                "timestamp_ms",
                "funding_rate",
            ]
        ]
        .sort_values(
            "timestamp_ms"
        )
        .drop_duplicates(
            "timestamp_ms",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    funding["funding_change"] = (
        funding["funding_rate"]
        .diff()
    )

    funding["funding_abs"] = (
        funding["funding_rate"]
        .abs()
    )

    rolling_mean = (
        funding["funding_rate"]
        .rolling(
            30,
            min_periods=5,
        )
        .mean()
    )

    rolling_std = (
        funding["funding_rate"]
        .rolling(
            30,
            min_periods=5,
        )
        .std()
    )

    funding["funding_z_30"] = (
        funding["funding_rate"]
        - rolling_mean
    ) / rolling_std.replace(
        0,
        np.nan,
    )

    funding["funding_regime"] = np.select(
        [
            funding["funding_z_30"] >= 1.0,
            funding["funding_z_30"] <= -1.0,
        ],
        [
            1.0,
            -1.0,
        ],
        default=0.0,
    )

    return funding


def attach_funding(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:

    funding = prepare_funding(
        symbol
    )

    out = (
        df
        .sort_values(
            "start_ms"
        )
        .copy()
    )

    if funding.empty:

        for col in [
            "funding_rate",
            "funding_change",
            "funding_abs",
            "funding_z_30",
            "funding_regime",
        ]:
            out[col] = np.nan

        return out

    out = pd.merge_asof(
        out,
        funding,
        left_on="start_ms",
        right_on="timestamp_ms",
        direction="backward",
        allow_exact_matches=True,
    )

    out = out.drop(
        columns=[
            "timestamp_ms"
        ],
        errors="ignore",
    )

    return out


def build_btc_context() -> pd.DataFrame:

    path = (
        DATA_DIR
        / "raw"
        / "trade_kline"
        / "BTCUSDT"
        / "5.parquet"
    )

    if not path.exists():
        raise RuntimeError(
            "BTCUSDT 5m data missing."
        )

    btc = pd.read_parquet(
        path
    )

    btc = prepare_trade_data(
        btc
    )

    parts = []

    for _, segment in btc.groupby(
        "segment_id",
        sort=True,
    ):

        x = (
            segment
            .sort_values(
                "start_ms"
            )
            .copy()
        )

        x["btc_ret_1"] = (
            x["close"].pct_change(
                periods=1,
                fill_method=None,
            )
        )

        x["btc_ret_6"] = (
            x["close"].pct_change(
                periods=6,
                fill_method=None,
            )
        )

        x["btc_vol_12"] = (
            x["btc_ret_1"]
            .rolling(
                12,
                min_periods=6,
            )
            .std()
        )

        btc_ema50 = (
            x["close"]
            .ewm(
                span=50,
                adjust=False,
            )
            .mean()
        )

        x["btc_ema50_dist"] = (
            x["close"]
            / btc_ema50
            - 1
        )

        x["btc_ret_1_lag1"] = (
            x["btc_ret_1"]
            .shift(1)
        )

        x["btc_ret_1_lag3"] = (
            x["btc_ret_1"]
            .shift(3)
        )

        x["btc_ret_1_lag6"] = (
            x["btc_ret_1"]
            .shift(6)
        )

        x["btc_ret_6_lag1"] = (
            x["btc_ret_6"]
            .shift(1)
        )

        x["btc_ret_6_lag3"] = (
            x["btc_ret_6"]
            .shift(3)
        )

        x["btc_vol_12_lag1"] = (
            x["btc_vol_12"]
            .shift(1)
        )

        parts.append(x)

    btc = pd.concat(
        parts,
        ignore_index=True,
    )

    columns = [
        "start_ms",
        "btc_ret_1",
        "btc_ret_6",
        "btc_vol_12",
        "btc_ema50_dist",
        "btc_ret_1_lag1",
        "btc_ret_1_lag3",
        "btc_ret_1_lag6",
        "btc_ret_6_lag1",
        "btc_ret_6_lag3",
        "btc_vol_12_lag1",
    ]

    return (
        btc[columns]
        .drop_duplicates(
            "start_ms",
            keep="last",
        )
        .sort_values(
            "start_ms"
        )
        .reset_index(
            drop=True
        )
    )


def attach_btc_context(
    df: pd.DataFrame,
    btc_context: pd.DataFrame,
) -> pd.DataFrame:

    return df.merge(
        btc_context,
        on="start_ms",
        how="left",
    )


def attach_symbol_features(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:

    out = df.copy()

    out["symbol_is_btc"] = float(
        symbol == "BTCUSDT"
    )

    out["symbol_is_eth"] = float(
        symbol == "ETHUSDT"
    )

    out["symbol_is_sol"] = float(
        symbol == "SOLUSDT"
    )

    return out


def main():

    cfg = load_config()

    output_dir = (
        ROOT
        / "ml"
        / "features"
        / FEATURE_VERSION
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Building BTC context...",
        flush=True,
    )

    btc_context = (
        build_btc_context()
    )

    manifest = {
        "feature_version":
            FEATURE_VERSION,

        "generated_at_ms":
            now_ms(),

        "source_timeframe":
            "5m",

        "label_horizon_bars":
            LABEL_HORIZON_BARS,

        "gap_policy":
            "features and labels are calculated "
            "inside contiguous 5m segments only",

        "funding_policy":
            "backward asof only",

        "orderbook_in_training":
            False,

        "files":
            [],
    }

    for symbol in cfg["symbols"]:

        path = (
            DATA_DIR
            / "raw"
            / "trade_kline"
            / symbol
            / "5.parquet"
        )

        if not path.exists():

            print(
                f"[SKIP] {symbol}: missing 5m data",
                flush=True,
            )

            continue

        print(
            f"\n=== {symbol} ===",
            flush=True,
        )

        raw = pd.read_parquet(
            path
        )

        features = (
            build_gap_aware_features(
                raw
            )
        )

        features = attach_mark_index(
            features,
            symbol,
        )

        features = attach_funding(
            features,
            symbol,
        )

        features = attach_btc_context(
            features,
            btc_context,
        )

        features = attach_symbol_features(
            features,
            symbol,
        )

        output_path = (
            output_dir
            / f"{symbol}_5m.parquet"
        )

        tmp_output = output_path.with_name(
            output_path.name + ".tmp.parquet"
        )

        features.to_parquet(
            tmp_output,
            index=False,
        )

        os.replace(
            tmp_output,
            output_path,
        )

        segments = int(
            features["segment_id"]
            .nunique()
        )

        funding_rows = int(
            features["funding_rate"]
            .notna()
            .sum()
        )

        basis_rows = int(
            features["mark_index_basis"]
            .notna()
            .sum()
        )

        btc_lag_rows = int(
            features["btc_ret_1_lag6"]
            .notna()
            .sum()
        )

        print(
            f"[OK] {symbol}",
            flush=True,
        )

        print(
            " rows:",
            f"{len(features):,}",
            flush=True,
        )

        print(
            " segments:",
            segments,
            flush=True,
        )

        print(
            " funding:",
            f"{funding_rows:,}",
            flush=True,
        )

        print(
            " basis:",
            f"{basis_rows:,}",
            flush=True,
        )

        print(
            " BTC lag context:",
            f"{btc_lag_rows:,}",
            flush=True,
        )

        manifest["files"].append(
            {
                "symbol":
                    symbol,

                "rows":
                    int(len(features)),

                "segments":
                    segments,

                "funding_non_null":
                    funding_rows,

                "basis_non_null":
                    basis_rows,

                "btc_lag_non_null":
                    btc_lag_rows,

                "path":
                    str(
                        output_path.relative_to(
                            ROOT
                        )
                    ),
            }
        )

    manifest_path = (
        output_dir
        / "feature_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nFeature V4 complete.",
        flush=True,
    )

    print(
        "Manifest:",
        manifest_path,
        flush=True,
    )


if __name__ == "__main__":
    main()