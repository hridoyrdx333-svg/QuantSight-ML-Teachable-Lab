from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import pandas as pd
import websocket

from common import DATA_DIR, load_config, now_ms


WS_URL = "wss://stream.bybit.com/v5/public/linear"

MARK_INTERVAL = "5"
MARK_INTERVAL_MS = 5 * 60 * 1000

FLUSH_SECONDS = 5
PING_SECONDS = 20


# ============================================================
# Atomic parquet storage
# ============================================================

def atomic_upsert_parquet(
    path: Path,
    new_df: pd.DataFrame,
    key_columns: list[str],
):
    if new_df is None or new_df.empty:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        if path.exists():
            old = pd.read_parquet(path)

            all_columns = sorted(
                set(old.columns)
                | set(new_df.columns)
            )

            old = old.reindex(
                columns=all_columns
            )

            new_df = new_df.reindex(
                columns=all_columns
            )

            merged = pd.concat(
                [old, new_df],
                ignore_index=True,
            )

        else:
            merged = new_df.copy()

        merged = (
            merged
            .drop_duplicates(
                subset=key_columns,
                keep="last",
            )
            .sort_values(
                key_columns
            )
            .reset_index(
                drop=True
            )
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

    finally:
        tmp = path.with_name(
            path.name + ".tmp.parquet"
        )

        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# ============================================================
# Local orderbook
# ============================================================

class LocalOrderBook:
    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

        self.last_exchange_ts_ms = 0
        self.last_seq = 0

    @staticmethod
    def _apply_side(
        book: dict[float, float],
        updates,
    ):
        for price, size in updates:
            p = float(price)
            q = float(size)

            if q == 0:
                book.pop(
                    p,
                    None,
                )
            else:
                book[p] = q

    def apply(
        self,
        data: dict,
        message_type: str,
        recv_ms: int,
    ):
        if message_type == "snapshot":
            self.bids.clear()
            self.asks.clear()

        self._apply_side(
            self.bids,
            data.get("b", []),
        )

        self._apply_side(
            self.asks,
            data.get("a", []),
        )

        self.last_exchange_ts_ms = int(
            data.get("cts")
            or recv_ms
        )

        self.last_seq = int(
            data.get("seq")
            or data.get("u")
            or self.last_seq
            or 0
        )

    def feature_row(
        self,
        symbol: str,
        depth_levels: int,
        recv_ms: int,
    ):
        if not self.bids or not self.asks:
            return None

        bid_prices = sorted(
            self.bids.keys(),
            reverse=True,
        )

        ask_prices = sorted(
            self.asks.keys()
        )

        best_bid = bid_prices[0]
        best_ask = ask_prices[0]

        if (
            best_bid <= 0
            or best_ask <= 0
            or best_ask <= best_bid
        ):
            return None

        mid = (
            best_bid + best_ask
        ) / 2.0

        spread = (
            best_ask - best_bid
        )

        spread_bps = (
            spread / mid
        ) * 10000.0

        bid_depth = sum(
            self.bids[p]
            for p in bid_prices[
                :depth_levels
            ]
        )

        ask_depth = sum(
            self.asks[p]
            for p in ask_prices[
                :depth_levels
            ]
        )

        denominator = (
            bid_depth + ask_depth
        )

        imbalance = (
            (
                bid_depth
                - ask_depth
            )
            / denominator
            if denominator > 0
            else 0.0
        )

        stale_ms = max(
            0,
            recv_ms
            - self.last_exchange_ts_ms,
        )

        return {
            "timestamp_ms":
                recv_ms,

            "timestamp_utc":
                pd.to_datetime(
                    recv_ms,
                    unit="ms",
                    utc=True,
                ),

            "symbol":
                symbol,

            "best_bid":
                best_bid,

            "best_ask":
                best_ask,

            "mid":
                mid,

            "spread":
                spread,

            "spread_bps":
                spread_bps,

            "bid_depth":
                bid_depth,

            "ask_depth":
                ask_depth,

            "imbalance":
                imbalance,

            "depth_levels":
                depth_levels,

            "last_exchange_ts_ms":
                self.last_exchange_ts_ms,

            "last_seq":
                self.last_seq,

            "stale_ms":
                stale_ms,

            "source":
                "bybit_ws_orderbook",
        }


# ============================================================
# 5m mark/index candle builder
# ============================================================

class PriceCandle:
    def __init__(
        self,
        symbol: str,
        source_name: str,
    ):
        self.symbol = symbol
        self.source_name = source_name

        self.bucket_ms = None
        self.open = None
        self.high = None
        self.low = None
        self.close = None

    def update(
        self,
        timestamp_ms: int,
        price: float,
    ):
        bucket = (
            timestamp_ms
            // MARK_INTERVAL_MS
            * MARK_INTERVAL_MS
        )

        completed = None

        if self.bucket_ms is None:
            self._start(
                bucket,
                price,
            )

            return None

        if bucket > self.bucket_ms:
            completed = (
                self.to_row()
            )

            self._start(
                bucket,
                price,
            )

            return completed

        if bucket < self.bucket_ms:
            return None

        self.high = max(
            self.high,
            price,
        )

        self.low = min(
            self.low,
            price,
        )

        self.close = price

        return None

    def _start(
        self,
        bucket_ms: int,
        price: float,
    ):
        self.bucket_ms = bucket_ms
        self.open = price
        self.high = price
        self.low = price
        self.close = price

    def to_row(self):
        if self.bucket_ms is None:
            return None

        return {
            "symbol":
                self.symbol,

            "interval":
                MARK_INTERVAL,

            "start_ms":
                int(
                    self.bucket_ms
                ),

            "start_utc":
                pd.to_datetime(
                    self.bucket_ms,
                    unit="ms",
                    utc=True,
                ),

            "open":
                float(
                    self.open
                ),

            "high":
                float(
                    self.high
                ),

            "low":
                float(
                    self.low
                ),

            "close":
                float(
                    self.close
                ),

            "source":
                self.source_name,
        }


# ============================================================
# Collector
# ============================================================

class LiveCollector:
    def __init__(self):
        self.cfg = load_config()

        self.stop = threading.Event()

        self.symbols = list(
            self.cfg["symbols"]
        )

        self.intervals = [
            str(x)
            for x in self.cfg[
                "intervals"
            ]
        ]

        self.orderbook_depth = int(
            self.cfg[
                "bybit"
            ][
                "orderbook_depth"
            ]
        )

        self.books = {
            symbol:
                LocalOrderBook()
            for symbol
            in self.symbols
        }

        self.mark_candles = {
            symbol:
                PriceCandle(
                    symbol,
                    "bybit_ws_mark_price",
                )
            for symbol
            in self.symbols
        }

        self.index_candles = {
            symbol:
                PriceCandle(
                    symbol,
                    "bybit_ws_index_price",
                )
            for symbol
            in self.symbols
        }

        self.ticker_state = {
            symbol: {
                "markPrice": None,
                "indexPrice": None,
            }
            for symbol
            in self.symbols
        }

        self.pending_kline = []
        self.pending_ob = []
        self.pending_mark = []
        self.pending_index = []

        self.buf_lock = (
            threading.Lock()
        )

        self.ws = None

        self.last_flush = (
            time.time()
        )

        self.last_ping = (
            time.time()
        )

    # --------------------------------------------------------

    def subscription_topics(self):
        args = []

        for symbol in self.symbols:
            for interval in self.intervals:
                args.append(
                    f"kline.{interval}.{symbol}"
                )

            args.append(
                (
                    f"orderbook."
                    f"{self.orderbook_depth}."
                    f"{symbol}"
                )
            )

            args.append(
                f"tickers.{symbol}"
            )

        return args

    # --------------------------------------------------------

    def on_open(
        self,
        ws,
    ):
        args = (
            self.subscription_topics()
        )

        ws.send(
            json.dumps(
                {
                    "op":
                        "subscribe",

                    "args":
                        args,
                }
            )
        )

        print(
            "[WS] subscribed:",
            len(args),
            "topics",
            flush=True,
        )

    # --------------------------------------------------------

    def on_message(
        self,
        ws,
        raw_message,
    ):
        try:
            msg = json.loads(
                raw_message
            )

        except Exception:
            return

        topic = msg.get(
            "topic",
            "",
        )

        recv_ms = now_ms()

        # ---------------------------------------------
        # CLOSED TRADE KLINES
        # ---------------------------------------------

        if topic.startswith(
            "kline."
        ):
            parts = topic.split(
                "."
            )

            if len(parts) < 3:
                return

            interval = parts[1]
            symbol = parts[2]

            for k in msg.get(
                "data",
                [],
            ):
                if not k.get(
                    "confirm",
                    False,
                ):
                    continue

                row = {
                    "symbol":
                        symbol,

                    "interval":
                        str(
                            interval
                        ),

                    "start_ms":
                        int(
                            k["start"]
                        ),

                    "start_utc":
                        pd.to_datetime(
                            int(
                                k["start"]
                            ),
                            unit="ms",
                            utc=True,
                        ),

                    "open":
                        float(
                            k["open"]
                        ),

                    "high":
                        float(
                            k["high"]
                        ),

                    "low":
                        float(
                            k["low"]
                        ),

                    "close":
                        float(
                            k["close"]
                        ),

                    "volume":
                        float(
                            k["volume"]
                        ),

                    "turnover":
                        float(
                            k.get(
                                "turnover",
                                0,
                            )
                        ),

                    "exchange_timestamp_ms":
                        int(
                            k.get(
                                "timestamp"
                            )
                            or recv_ms
                        ),

                    "received_ms":
                        recv_ms,

                    "source":
                        "bybit_ws_kline",
                }

                with self.buf_lock:
                    self.pending_kline.append(
                        row
                    )

            return

        # ---------------------------------------------
        # ORDERBOOK
        # ---------------------------------------------

        if topic.startswith(
            "orderbook."
        ):
            data = msg.get(
                "data",
                {},
            )

            symbol = data.get(
                "s"
            )

            if (
                not symbol
                or symbol
                not in self.books
            ):
                return

            self.books[
                symbol
            ].apply(
                data,
                msg.get(
                    "type",
                    "delta",
                ),
                recv_ms,
            )

            row = self.books[
                symbol
            ].feature_row(
                symbol,
                self.orderbook_depth,
                recv_ms,
            )

            if row is not None:
                with self.buf_lock:
                    self.pending_ob.append(
                        row
                    )

            return

        # ---------------------------------------------
        # TICKERS: MARK + INDEX
        # ---------------------------------------------

        if topic.startswith(
            "tickers."
        ):
            symbol = topic.split(
                ".",
                1,
            )[1]

            if symbol not in self.ticker_state:
                return

            payload = msg.get(
                "data",
                {},
            )

            if isinstance(
                payload,
                list,
            ):
                if not payload:
                    return

                payload = payload[0]

            if not isinstance(
                payload,
                dict,
            ):
                return

            state = self.ticker_state[
                symbol
            ]

            if (
                payload.get(
                    "markPrice"
                )
                not in (
                    None,
                    "",
                )
            ):
                state[
                    "markPrice"
                ] = float(
                    payload[
                        "markPrice"
                    ]
                )

            if (
                payload.get(
                    "indexPrice"
                )
                not in (
                    None,
                    "",
                )
            ):
                state[
                    "indexPrice"
                ] = float(
                    payload[
                        "indexPrice"
                    ]
                )

            ticker_ts = int(
                msg.get(
                    "ts"
                )
                or recv_ms
            )

            mark_price = state.get(
                "markPrice"
            )

            index_price = state.get(
                "indexPrice"
            )

            if mark_price is not None:
                completed = (
                    self.mark_candles[
                        symbol
                    ].update(
                        ticker_ts,
                        mark_price,
                    )
                )

                if completed:
                    with self.buf_lock:
                        self.pending_mark.append(
                            completed
                        )

            if index_price is not None:
                completed = (
                    self.index_candles[
                        symbol
                    ].update(
                        ticker_ts,
                        index_price,
                    )
                )

                if completed:
                    with self.buf_lock:
                        self.pending_index.append(
                            completed
                        )

            return

    # --------------------------------------------------------

    def on_error(
        self,
        ws,
        error,
    ):
        print(
            "[WS ERROR]",
            error,
            flush=True,
        )

    # --------------------------------------------------------

    def on_close(
        self,
        ws,
        status_code,
        message,
    ):
        print(
            "[WS CLOSED]",
            status_code,
            message,
            flush=True,
        )

    # --------------------------------------------------------

    def flush(self):
        with self.buf_lock:
            klines = (
                self.pending_kline
            )

            obs = (
                self.pending_ob
            )

            marks = (
                self.pending_mark
            )

            indexes = (
                self.pending_index
            )

            self.pending_kline = []
            self.pending_ob = []
            self.pending_mark = []
            self.pending_index = []

        # ---------------------------------------------
        # Trade kline
        # ---------------------------------------------

        if klines:
            df = pd.DataFrame(
                klines
            )

            for (
                symbol,
                interval,
            ), group in df.groupby(
                [
                    "symbol",
                    "interval",
                ]
            ):
                path = (
                    DATA_DIR
                    / "live"
                    / "kline"
                    / symbol
                    / f"{interval}.parquet"
                )

                atomic_upsert_parquet(
                    path,
                    group,
                    [
                        "symbol",
                        "interval",
                        "start_ms",
                    ],
                )

            print(
                f"[FLUSH] "
                f"{len(klines)} "
                f"closed kline row(s)",
                flush=True,
            )

        # ---------------------------------------------
        # Orderbook
        # ---------------------------------------------

        if obs:
            df = pd.DataFrame(
                obs
            )

            for (
                symbol,
                group,
            ) in df.groupby(
                "symbol"
            ):
                path = (
                    DATA_DIR
                    / "live"
                    / "orderbook_features"
                    / symbol
                    / "features.parquet"
                )

                atomic_upsert_parquet(
                    path,
                    group,
                    [
                        "symbol",
                        "timestamp_ms",
                    ],
                )

            print(
                f"[FLUSH] "
                f"{len(obs)} "
                f"orderbook row(s)",
                flush=True,
            )

        # ---------------------------------------------
        # Mark 5m
        # Write directly to RAW canonical context
        # so build_features.py sees it immediately.
        # ---------------------------------------------

        if marks:
            df = pd.DataFrame(
                marks
            )

            for (
                symbol,
                group,
            ) in df.groupby(
                "symbol"
            ):
                raw_path = (
                    DATA_DIR
                    / "raw"
                    / "mark_kline"
                    / symbol
                    / "5.parquet"
                )

                live_path = (
                    DATA_DIR
                    / "live"
                    / "mark_kline"
                    / symbol
                    / "5.parquet"
                )

                keys = [
                    "symbol",
                    "interval",
                    "start_ms",
                ]

                atomic_upsert_parquet(
                    raw_path,
                    group,
                    keys,
                )

                atomic_upsert_parquet(
                    live_path,
                    group,
                    keys,
                )

            print(
                f"[FLUSH] "
                f"{len(marks)} "
                f"closed mark 5m row(s)",
                flush=True,
            )

        # ---------------------------------------------
        # Index 5m
        # ---------------------------------------------

        if indexes:
            df = pd.DataFrame(
                indexes
            )

            for (
                symbol,
                group,
            ) in df.groupby(
                "symbol"
            ):
                raw_path = (
                    DATA_DIR
                    / "raw"
                    / "index_kline"
                    / symbol
                    / "5.parquet"
                )

                live_path = (
                    DATA_DIR
                    / "live"
                    / "index_kline"
                    / symbol
                    / "5.parquet"
                )

                keys = [
                    "symbol",
                    "interval",
                    "start_ms",
                ]

                atomic_upsert_parquet(
                    raw_path,
                    group,
                    keys,
                )

                atomic_upsert_parquet(
                    live_path,
                    group,
                    keys,
                )

            print(
                f"[FLUSH] "
                f"{len(indexes)} "
                f"closed index 5m row(s)",
                flush=True,
            )

        self.last_flush = (
            time.time()
        )

    # --------------------------------------------------------

    def heartbeat(self):
        if self.ws is None:
            return

        if (
            time.time()
            - self.last_ping
            < PING_SECONDS
        ):
            return

        try:
            self.ws.send(
                json.dumps(
                    {
                        "op":
                            "ping"
                    }
                )
            )

            self.last_ping = (
                time.time()
            )

        except Exception:
            pass

    # --------------------------------------------------------

    def run_connection(self):
        self.ws = websocket.WebSocketApp(
            WS_URL,

            on_open=
                self.on_open,

            on_message=
                self.on_message,

            on_error=
                self.on_error,

            on_close=
                self.on_close,
        )

        thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={
                "ping_interval":
                    None,
            },
            daemon=True,
        )

        thread.start()

        while (
            thread.is_alive()
            and not self.stop.is_set()
        ):
            if (
                time.time()
                - self.last_flush
                >= FLUSH_SECONDS
            ):
                self.flush()

            self.heartbeat()

            time.sleep(
                0.5
            )

        self.flush()

        try:
            self.ws.close()
        except Exception:
            pass

    # --------------------------------------------------------

    def run(self):
        print(
            "QuantSight Live Collector",
            flush=True,
        )

        print(
            "Public market data only.",
            flush=True,
        )

        print(
            "NO ORDERS.",
            flush=True,
        )

        print(
            "WS:",
            WS_URL,
            flush=True,
        )

        print(
            "Symbols:",
            self.symbols,
            flush=True,
        )

        print(
            "Intervals:",
            self.intervals,
            flush=True,
        )

        print(
            "Orderbook depth:",
            self.orderbook_depth,
            flush=True,
        )

        print(
            "Ticker mark/index aggregation:",
            "5m",
            flush=True,
        )

        while not self.stop.is_set():
            try:
                self.run_connection()

            except Exception as e:
                print(
                    "[COLLECTOR ERROR]",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

            if self.stop.is_set():
                break

            print(
                "[WS] reconnecting in 3 seconds...",
                flush=True,
            )

            time.sleep(
                3
            )

    # --------------------------------------------------------

    def request_stop(self):
        self.stop.set()

        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass


def main():
    collector = LiveCollector()

    def stop_handler(
        signum,
        frame,
    ):
        print(
            "\nStopping collector...",
            flush=True,
        )

        collector.request_stop()

    signal.signal(
        signal.SIGINT,
        stop_handler,
    )

    if hasattr(
        signal,
        "SIGTERM",
    ):
        signal.signal(
            signal.SIGTERM,
            stop_handler,
        )

    collector.run()


if __name__ == "__main__":
    main()