from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

POLL_SECONDS = 2
INTERVAL_MS = 5 * 60 * 1000

DATA_DIR = ROOT / "data"

LIVE_TRADE_DIR = DATA_DIR / "live" / "kline"
LIVE_MARK_DIR = DATA_DIR / "live" / "mark_kline"
LIVE_INDEX_DIR = DATA_DIR / "live" / "index_kline"

RAW_TRADE_DIR = DATA_DIR / "raw" / "trade_kline"
RAW_MARK_DIR = DATA_DIR / "raw" / "mark_kline"
RAW_INDEX_DIR = DATA_DIR / "raw" / "index_kline"

FEATURE_DIR = ROOT / "ml" / "features" / "v4_context_plus"
RUNTIME_DIR = ROOT / "ml" / "shadow"

STATE_PATH = RUNTIME_DIR / "live_feature_builder_state.json"
STATUS_PATH = RUNTIME_DIR / "live_feature_builder_status.json"

BACKFILL_SCRIPT = ROOT / "backfill_live_context.py"
BUILD_SCRIPT = ROOT / "ml" / "build_features.py"


def python_exe() -> Path:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else Path(sys.executable)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for _ in range(10):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            time.sleep(0.1)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_latest_ms(path: Path):
    if not path.exists():
        return None

    for _ in range(3):
        try:
            df = pd.read_parquet(path, columns=["start_ms"])
            if df.empty:
                return None
            values = pd.to_numeric(df["start_ms"], errors="coerce").dropna()
            if values.empty:
                return None
            return int(values.max())
        except Exception:
            time.sleep(0.2)

    return None


def path_for(kind: str, root_kind: str, symbol: str) -> Path:
    if root_kind == "live":
        base = {
            "trade": LIVE_TRADE_DIR,
            "mark": LIVE_MARK_DIR,
            "index": LIVE_INDEX_DIR,
        }[kind]
    else:
        base = {
            "trade": RAW_TRADE_DIR,
            "mark": RAW_MARK_DIR,
            "index": RAW_INDEX_DIR,
        }[kind]

    return base / symbol / "5.parquet"


def feature_path(symbol: str) -> Path:
    return FEATURE_DIR / f"{symbol}_5m.parquet"


def latest_snapshot() -> dict:
    result = {}

    for symbol in SYMBOLS:
        result[symbol] = {
            "live_trade": safe_latest_ms(path_for("trade", "live", symbol)),
            "live_mark": safe_latest_ms(path_for("mark", "live", symbol)),
            "live_index": safe_latest_ms(path_for("index", "live", symbol)),
            "raw_trade": safe_latest_ms(path_for("trade", "raw", symbol)),
            "raw_mark": safe_latest_ms(path_for("mark", "raw", symbol)),
            "raw_index": safe_latest_ms(path_for("index", "raw", symbol)),
            "feature": safe_latest_ms(feature_path(symbol)),
        }

    return result


def aligned_ready_ms(snapshot: dict):
    symbol_ready = {}

    for symbol in SYMBOLS:
        row = snapshot[symbol]
        values = [
            row["live_trade"],
            row["live_mark"],
            row["live_index"],
        ]

        if any(v is None for v in values):
            return None, symbol_ready

        # Build only when trade + mark + index for this symbol
        # point at the exact same CLOSED 5m candle.
        if not (values[0] == values[1] == values[2]):
            return None, symbol_ready

        symbol_ready[symbol] = values[0]

    # Build only when all 3 symbols are aligned to the same candle.
    ready_values = list(symbol_ready.values())
    if len(set(ready_values)) != 1:
        return None, symbol_ready

    return ready_values[0], symbol_ready


def run_process(label: str, script: Path):
    print(f"[{label}] {script.name}", flush=True)

    proc = subprocess.Popen(
        [str(python_exe()), str(script)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output = []

    if proc.stdout:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)
                output.append(line)
                if len(output) > 100:
                    output = output[-100:]

    code = proc.wait()
    return code, output


def run_backfill():
    if not BACKFILL_SCRIPT.exists():
        print(
            "[BACKFILL] backfill_live_context.py not found; "
            "continuing with existing data.",
            flush=True,
        )
        return True

    code, _ = run_process("BACKFILL", BACKFILL_SCRIPT)

    if code != 0:
        print(f"[BACKFILL ERROR] exit_code={code}", flush=True)
        return False

    print("[BACKFILL OK] recent trade/mark/index context refreshed.", flush=True)
    return True


def run_feature_build():
    code, output = run_process("BUILD", BUILD_SCRIPT)
    return code, output


def raw_context_ready(ready_ms: int):
    snap = latest_snapshot()

    for symbol in SYMBOLS:
        row = snap[symbol]
        for key in ("raw_trade", "raw_mark", "raw_index"):
            value = row[key]
            if value is None or value < ready_ms:
                return False, snap

    return True, snap


def wait_for_raw_context(ready_ms: int, timeout_seconds: int = 15):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        ok, snap = raw_context_ready(ready_ms)
        if ok:
            return True, snap
        time.sleep(0.5)

    return raw_context_ready(ready_ms)


def main():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    state = read_json(
        STATE_PATH,
        {
            "last_ready_ms": None,
            "last_successful_build_ms": None,
        },
    )

    print("=" * 64, flush=True)
    print("QuantSight Live V4 Feature Builder - Aligned Context", flush=True)
    print("NO ORDERS", flush=True)
    print(
        "Rule: build only after BTC/ETH/SOL trade + mark + index "
        "share the same closed 5m timestamp.",
        flush=True,
    )
    print("=" * 64, flush=True)

    write_json(
        STATUS_PATH,
        {
            "status": "starting",
            "stage": "startup_backfill",
            "last_error": None,
        },
    )

    # Permanent recovery for PC shutdown / overnight / websocket downtime.
    backfill_ok = run_backfill()

    if not backfill_ok:
        write_json(
            STATUS_PATH,
            {
                "status": "error",
                "stage": "startup_backfill",
                "last_error": "backfill failed",
            },
        )
        raise SystemExit(1)

    # Force one clean aligned build after upgrading from the old builder.
    # The old state file did not have this field, so first startup rebuilds.
    last_ready_ms = state.get("last_ready_ms")

    write_json(
        STATUS_PATH,
        {
            "status": "running",
            "stage": "waiting_for_aligned_5m",
            "last_error": None,
            "last_ready_ms": last_ready_ms,
        },
    )

    try:
        while True:
            snap = latest_snapshot()
            ready_ms, per_symbol = aligned_ready_ms(snap)

            if ready_ms is None:
                write_json(
                    STATUS_PATH,
                    {
                        "status": "running",
                        "stage": "waiting_for_aligned_5m",
                        "snapshot": snap,
                        "last_ready_ms": last_ready_ms,
                        "last_error": None,
                    },
                )
                time.sleep(POLL_SECONDS)
                continue

            if ready_ms == last_ready_ms:
                write_json(
                    STATUS_PATH,
                    {
                        "status": "running",
                        "stage": "waiting_for_next_5m",
                        "ready_ms": ready_ms,
                        "snapshot": snap,
                        "last_error": None,
                    },
                )
                time.sleep(POLL_SECONDS)
                continue

            # If we missed one or more 5m bars while offline, repair first.
            if (
                last_ready_ms is not None
                and ready_ms - last_ready_ms > INTERVAL_MS
            ):
                print(
                    f"[GAP] last={last_ready_ms} current={ready_ms}; "
                    "running context backfill before build.",
                    flush=True,
                )

                if not run_backfill():
                    write_json(
                        STATUS_PATH,
                        {
                            "status": "error",
                            "stage": "gap_backfill",
                            "ready_ms": ready_ms,
                            "last_error": "gap backfill failed",
                        },
                    )
                    time.sleep(10)
                    continue

                snap = latest_snapshot()
                ready_ms, per_symbol = aligned_ready_ms(snap)

                if ready_ms is None:
                    time.sleep(POLL_SECONDS)
                    continue

            # Collector writes trade, then mark/index. Even after alignment,
            # allow filesystem writes to settle before the full feature build.
            print(
                f"[READY] aligned closed 5m candle = {ready_ms}",
                flush=True,
            )
            time.sleep(1.0)

            raw_ok, raw_snap = wait_for_raw_context(ready_ms)

            if not raw_ok:
                print(
                    "[WAIT] raw trade/mark/index context not fully visible yet.",
                    flush=True,
                )
                time.sleep(POLL_SECONDS)
                continue

            write_json(
                STATUS_PATH,
                {
                    "status": "running",
                    "stage": "building_v4",
                    "ready_ms": ready_ms,
                    "snapshot": raw_snap,
                    "last_error": None,
                },
            )

            code, output = run_feature_build()

            if code != 0:
                write_json(
                    STATUS_PATH,
                    {
                        "status": "error",
                        "stage": "build_failed",
                        "ready_ms": ready_ms,
                        "exit_code": code,
                        "last_output": output,
                        "last_error": "build_features.py failed",
                    },
                )
                print(f"[BUILD ERROR] exit_code={code}", flush=True)
                time.sleep(10)
                continue

            post = latest_snapshot()

            # Do not mark success unless all feature files reached this candle.
            feature_ok = all(
                post[s]["feature"] is not None
                and post[s]["feature"] >= ready_ms
                for s in SYMBOLS
            )

            if not feature_ok:
                write_json(
                    STATUS_PATH,
                    {
                        "status": "error",
                        "stage": "feature_freshness_check",
                        "ready_ms": ready_ms,
                        "snapshot": post,
                        "last_error": "feature files did not reach ready_ms",
                    },
                )
                print(
                    "[FEATURE ERROR] build ended but feature freshness "
                    "check failed.",
                    flush=True,
                )
                time.sleep(5)
                continue

            last_ready_ms = ready_ms

            state = {
                "last_ready_ms": last_ready_ms,
                "last_successful_build_ms": int(time.time() * 1000),
            }

            write_json(STATE_PATH, state)

            write_json(
                STATUS_PATH,
                {
                    "status": "running",
                    "stage": "waiting_for_next_5m",
                    "ready_ms": ready_ms,
                    "snapshot": post,
                    "last_error": None,
                },
            )

            print(
                f"[OK] V4 aligned live features refreshed through {ready_ms}.",
                flush=True,
            )

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nLive feature builder stopped.", flush=True)
        write_json(
            STATUS_PATH,
            {
                "status": "stopped",
                "last_ready_ms": last_ready_ms,
                "snapshot": latest_snapshot(),
            },
        )


if __name__ == "__main__":
    main()
