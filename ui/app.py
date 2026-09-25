from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parents[1]

print("="*60, flush=True)
print("REQUIRED CHECKS FOR USER:", flush=True)
print(f"1. Exact ROOT used by ui/app.py: {ROOT}", flush=True)
print(f"2. Exact paths used:", flush=True)
paths_to_check = {
    "ml/features/v4_context_plus": ROOT / "ml" / "features" / "v4_context_plus",
    "ml/reports/training_report_v4_context_plus.json": ROOT / "ml" / "reports" / "training_report_v4_context_plus.json",
    "ml/reports/training_report.json": ROOT / "ml" / "reports" / "training_report.json",
    "data/raw": ROOT / "data" / "raw",
    "data/live": ROOT / "data" / "live",
    "ui/runtime": ROOT / "ui" / "runtime"
}
for name, p in paths_to_check.items():
    print(f"   - {name}: {p} (Exists: {p.exists()})", flush=True)
print("="*60, flush=True)

RUNTIME_DIR = ROOT / "ui" / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

TASK_LOCK = RUNTIME_DIR / "heavy_task.lock"
TASK_STATE = RUNTIME_DIR / "task_state.json"
LIVE_LOCK = RUNTIME_DIR / "live_collector.lock"
BUILDER_LOCK = RUNTIME_DIR / "builder.lock"
SHADOW_LOCK = RUNTIME_DIR / "shadow.lock"

SHADOW_DIR = ROOT / "ml" / "shadow"
SHADOW_STATUS = SHADOW_DIR / "shadow_status.json"
SHADOW_LOG = SHADOW_DIR / "shadow_predictions.jsonl"
LIVE_FEATURE_STATUS = SHADOW_DIR / "live_feature_builder_status.json"

load_dotenv(ROOT / ".env")

app = Flask(
    __name__,
    template_folder=str(ROOT / "ui" / "templates"),
    static_folder=str(ROOT / "ui" / "static"),
)

STATE_LOCK = threading.Lock()


# ---------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------

def read_json(path: Path, default):
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return default


def write_json(path: Path, payload):
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    for _ in range(10):
        try:
            temp.replace(path)
            break
        except PermissionError:
            time.sleep(0.1)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# ---------------------------------------------------------
# Windows process helpers
# ---------------------------------------------------------

def pid_running(pid: int | None) -> bool:
    if not pid:
        return False

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"if (Get-Process -Id {int(pid)} "
                    "-ErrorAction SilentlyContinue) "
                    "{ exit 0 } else { exit 1 }"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return result.returncode == 0

    except Exception:
        return False


def stop_pid(pid: int | None) -> bool:
    if not pid:
        return False

    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"Stop-Process -Id {int(pid)} "
                    "-Force "
                    "-ErrorAction SilentlyContinue"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return True

    except Exception:
        return False


# ---------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------

def read_lock_pid(path: Path) -> int | None:
    if not path.exists():
        return None

    try:
        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            return None

        return int(text)

    except Exception:
        return None


def remove_lock(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def task_is_running() -> bool:
    if not TASK_LOCK.exists():
        return False

    pid = read_lock_pid(
        TASK_LOCK
    )

    # Lock exists but process has not yet written PID.
    if pid is None:
        return True

    if pid_running(pid):
        return True

    remove_lock(
        TASK_LOCK
    )

    return False


def live_is_running() -> bool:
    if not LIVE_LOCK.exists():
        return False

    pid = read_lock_pid(
        LIVE_LOCK
    )

    if pid is None:
        return True

    if pid_running(pid):
        return True

    remove_lock(
        LIVE_LOCK
    )

    return False

def builder_is_running() -> bool:
    if not BUILDER_LOCK.exists():
        return False
    pid = read_lock_pid(BUILDER_LOCK)
    if pid is None:
        return True
    if pid_running(pid):
        return True
    remove_lock(BUILDER_LOCK)
    return False

def shadow_is_running() -> bool:
    if not SHADOW_LOCK.exists():
        return False
    pid = read_lock_pid(SHADOW_LOCK)
    if pid is None:
        return True
    if pid_running(pid):
        return True
    remove_lock(SHADOW_LOCK)
    return False


# ---------------------------------------------------------
# Task state
# ---------------------------------------------------------

def default_task_state():
    return {
        "status": "idle",
        "label": None,
        "pid": None,
        "stage": None,
        "started_at": None,
        "ended_at": None,
        "last_output_at": None,
        "exit_code": None,
        "last_message": "Ready",
    }


def detect_stage(line: str):
    text = line.strip()

    mappings = [
        ("Building BTC context", "Preparing BTC context"),
        ("=== BTCUSDT ===", "BTCUSDT"),
        ("=== ETHUSDT ===", "ETHUSDT"),
        ("=== SOLUSDT ===", "SOLUSDT"),
        ("Training logistic", "Training Logistic"),
        ("Training random_forest", "Training Random Forest"),
        (
            "Training hist_gradient_boosting",
            "Training HistGradientBoosting",
        ),
        ("Feature V4 complete", "V4 complete"),
        ("Feature V3 complete", "V3 complete"),
        ("Training complete", "Training complete"),
    ]

    for needle, stage in mappings:
        if needle in text:
            return stage

    return None


def get_task_state():
    state = read_json(
        TASK_STATE,
        default_task_state(),
    )

    running = task_is_running()

    if running:
        if state.get("status") not in {
            "starting",
            "running",
        }:
            state["status"] = "running"

    elif state.get("status") in {
        "starting",
        "running",
    }:
        state["status"] = "stopped"

    now = time.time()

    started_at = state.get(
        "started_at"
    )

    last_output_at = state.get(
        "last_output_at"
    )

    state["elapsed_seconds"] = (
        max(
            0,
            int(now - started_at),
        )
        if started_at
        else 0
    )

    state["silent_seconds"] = (
        max(
            0,
            int(now - last_output_at),
        )
        if last_output_at
        else 0
    )

    state["running"] = running

    silent = state[
        "silent_seconds"
    ]

    if running and silent >= 1200:
        state["stall_level"] = (
            "critical"
        )

    elif running and silent >= 600:
        state["stall_level"] = (
            "warning"
        )

    else:
        state["stall_level"] = (
            "normal"
        )

    return state


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

def info(path: Path):
    if not path.exists():
        return {
            "exists": False,
            "rows": 0,
            "start": None,
            "end": None,
            "coverage_hours": 0,
        }

    try:
        df = pd.read_parquet(
            path
        )

        result = {
            "exists": True,
            "rows": int(len(df)),
            "start": None,
            "end": None,
            "coverage_hours": 0,
        }

        if df.empty:
            return result

        if "start_ms" in df.columns:
            col = "start_ms"

        elif "timestamp_ms" in df.columns:
            col = "timestamp_ms"

        else:
            col = None

        if col:
            start_ms = int(
                df[col].min()
            )

            end_ms = int(
                df[col].max()
            )

            result["start"] = (
                pd.to_datetime(
                    start_ms,
                    unit="ms",
                    utc=True,
                ).isoformat()
            )

            result["end"] = (
                pd.to_datetime(
                    end_ms,
                    unit="ms",
                    utc=True,
                ).isoformat()
            )

            result[
                "coverage_hours"
            ] = round(
                max(
                    0,
                    end_ms - start_ms,
                )
                / 3_600_000,
                2,
            )

        return result

    except Exception as e:
        return {
            "exists": True,
            "rows": 0,
            "start": None,
            "end": None,
            "coverage_hours": 0,
            "error":
                f"{type(e).__name__}: {e}",
        }


def load_config_file():
    path = ROOT / "config.json"

    if not path.exists():
        path = (
            ROOT
            / "config.example.json"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ---------------------------------------------------------
# Feature versions
# ---------------------------------------------------------

def feature_versions():
    root = (
        ROOT
        / "ml"
        / "features"
    )

    versions = []

    if not root.exists():
        return versions

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        manifest_path = (
            directory
            / "feature_manifest.json"
        )

        if not manifest_path.exists():
            continue

        manifest = read_json(
            manifest_path,
            {},
        )

        versions.append(
            {
                "version":
                    manifest.get(
                        "feature_version",
                        directory.name,
                    ),

                "generated_at_ms":
                    manifest.get(
                        "generated_at_ms"
                    ),

                "source_timeframe":
                    manifest.get(
                        "source_timeframe"
                    ),

                "orderbook_in_training":
                    manifest.get(
                        "orderbook_in_training"
                    ),

                "files":
                    len(
                        manifest.get(
                            "files",
                            [],
                        )
                    ),

                "modified":
                    manifest_path.stat().st_mtime,
            }
        )

    versions.sort(
        key=lambda x:
            x.get("modified", 0),
        reverse=True,
    )

    return versions


def current_feature_version():
    versions = feature_versions()

    if not versions:
        return None

    return versions[0].get(
        "version"
    )


# ---------------------------------------------------------
# Training report dashboard
# ---------------------------------------------------------

def normalize_metrics(metrics):
    metrics = metrics or {}

    return {
        "precision":
            safe_float(
                metrics.get(
                    "precision"
                )
            ),

        "recall":
            safe_float(
                metrics.get(
                    "recall"
                )
            ),

        "roc_auc":
            safe_float(
                metrics.get(
                    "roc_auc"
                )
            ),

        "brier":
            safe_float(
                metrics.get(
                    "brier"
                )
            ),

        "log_loss":
            safe_float(
                metrics.get(
                    "log_loss"
                )
            ),
    }


def parse_training_report(
    path: Path,
):
    payload = read_json(
        path,
        {},
    )

    if not payload:
        return None

    models = {}

    for name, values in (
        payload.get(
            "models",
            {}
        )
        or {}
    ).items():

        models[name] = {
            "validation":
                normalize_metrics(
                    values.get(
                        "validation"
                    )
                ),

            "test":
                normalize_metrics(
                    values.get(
                        "test"
                    )
                ),
        }

    return {
        "feature_version":
            payload.get(
                "feature_version"
            ),

        "target":
            payload.get(
                "target"
            ),

        "label_horizon_bars":
            payload.get(
                "label_horizon_bars"
            ),

        "features_count":
            len(
                payload.get(
                    "features",
                    []
                )
            ),

        "split":
            payload.get(
                "split",
                {},
            ),

        "models":
            models,

        "filename":
            path.name,

        "modified":
            path.stat().st_mtime,
    }


def training_reports():
    reports_dir = (
        ROOT
        / "ml"
        / "reports"
    )

    if not reports_dir.exists():
        return []

    results = []

    for path in reports_dir.glob(
        "*training_report*.json"
    ):
        report = (
            parse_training_report(
                path
            )
        )

        if report:
            results.append(
                report
            )

    # Current legacy filename support.
    legacy = (
        reports_dir
        / "training_report.json"
    )

    if (
        legacy.exists()
        and not any(
            x["filename"]
            == legacy.name
            for x in results
        )
    ):
        report = (
            parse_training_report(
                legacy
            )
        )

        if report:
            results.append(
                report
            )

    # Avoid showing duplicate same-version reports.
    deduped = {}

    for report in results:
        key = (
            report.get(
                "feature_version"
            )
            or report.get(
                "filename"
            )
        )

        previous = (
            deduped.get(
                key
            )
        )

        if (
            previous is None
            or report["modified"]
            > previous["modified"]
        ):
            deduped[key] = report

    results = list(
        deduped.values()
    )

    results.sort(
        key=lambda x:
            x.get("modified", 0),
        reverse=True,
    )

    return results


def model_dashboard():
    reports = (
        training_reports()
    )

    return {
        "latest":
            reports[0]
            if reports
            else None,

        "reports":
            reports,
    }


# ---------------------------------------------------------
# Shadow / live-feature dashboard
# ---------------------------------------------------------

def shadow_dashboard():
    status = read_json(
        SHADOW_STATUS,
        {},
    )

    feature_builder = read_json(
        LIVE_FEATURE_STATUS,
        {},
    )

    latest = {}

    if SHADOW_LOG.exists():
        try:
            with SHADOW_LOG.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as handle:
                for line in handle:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(
                            line
                        )
                    except Exception:
                        continue

                    symbol = record.get(
                        "symbol"
                    )

                    if symbol:
                        latest[
                            symbol
                        ] = record
        except Exception:
            latest = {}

    now = time.time()

    for item in latest.values():
        logged_at = item.get(
            "logged_at"
        )

        age = None

        if logged_at:
            try:
                age = max(
                    0.0,
                    now
                    - pd.Timestamp(
                        logged_at
                    ).timestamp(),
                )
            except Exception:
                age = None

        item[
            "log_age_seconds"
        ] = age

    decisions = {
        symbol:
            item.get(
                "decision"
            )
        for symbol, item
        in latest.items()
    }

    all_symbols_scored = bool(
        latest
        and all(
            symbol in latest
            for symbol in (
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
            )
        )
    )

    no_errors = bool(
        all_symbols_scored
        and all(
            decisions.get(
                symbol
            )
            in {
                "PASS",
                "REJECT",
                "STALE",
            }
            for symbol in (
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
            )
        )
    )

    fresh = bool(
        no_errors
        and all(
            float(
                latest[
                    symbol
                ].get(
                    "feature_age_seconds",
                    10**12,
                )
            )
            <= 15 * 60
            for symbol in (
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT",
            )
        )
    )

    return {
        "status":
            status,

        "feature_builder":
            feature_builder,

        "latest":
            latest,

        "all_symbols_scored":
            all_symbols_scored,

        "no_errors":
            no_errors,

        "fresh":
            fresh,
    }


# ---------------------------------------------------------
# Readiness
# ---------------------------------------------------------

def readiness_status(
    rows,
    validation,
    features,
    model_data,
    approved,
    shadow,
):
    symbols = list(
        rows.keys()
    )

    historical_ready = bool(
        symbols
        and all(
            rows[s]["5m"].get(
                "rows",
                0,
            ) > 0
            for s in symbols
        )
    )

    funding_ready = bool(
        symbols
        and all(
            rows[s]["funding"].get(
                "rows",
                0,
            ) > 0
            for s in symbols
        )
    )

    orderbook_started = bool(
        symbols
        and all(
            rows[s]["orderbook"].get(
                "rows",
                0,
            ) > 0
            for s in symbols
        )
    )

    feature_ready = bool(
        features
    )

    training_ready = bool(
        model_data.get(
            "latest"
        )
    )

    validation_available = bool(
        validation
    )

    approved_ready = bool(
        approved
    )

    shadow_ready = bool(
        shadow.get(
            "fresh"
        )
    )

    demo_auth_ready = bool(os.getenv("BYBIT_DEMO_API_KEY") and os.getenv("BYBIT_DEMO_API_SECRET"))
    
    place_orders_false = False
    demo_trader_path = ROOT / "ml" / "demo_trader.py"
    if demo_trader_path.exists():
        dt_text = demo_trader_path.read_text()
        if "PLACE_ORDERS = False" in dt_text:
            place_orders_false = True
            
    demo_paper_ready = bool(
        historical_ready
        and funding_ready
        and feature_ready
        and training_ready
        and validation_available
        and shadow.get("status", {}).get("thresholds")
        and shadow_ready
        and demo_auth_ready
        and place_orders_false
    )


    return {
        "historical_data":
            historical_ready,

        "funding_data":
            funding_ready,

        "orderbook_collection_started":
            orderbook_started,

        "feature_build":
            feature_ready,

        "training_report":
            training_ready,

        "validation_report":
            validation_available,

        "approved_model":
            approved_ready,

        "live_shadow_fresh":
            shadow_ready,

        # Deliberately factual:
        # approval remains a separate required step.
        "demo_paper_ready": demo_paper_ready,
    }


# ---------------------------------------------------------
# Main status API
# ---------------------------------------------------------

def get_status():
    cfg = load_config_file()

    rows = {}

    for symbol in cfg[
        "symbols"
    ]:

        rows[symbol] = {
            "1m":
                info(
                    ROOT
                    / "data"
                    / "raw"
                    / "massive_1m"
                    / symbol
                    / "1.parquet"
                ),

            "5m":
                info(
                    ROOT
                    / "data"
                    / "raw"
                    / "trade_kline"
                    / symbol
                    / "5.parquet"
                ),

            "15m":
                info(
                    ROOT
                    / "data"
                    / "raw"
                    / "trade_kline"
                    / symbol
                    / "15.parquet"
                ),

            "1h":
                info(
                    ROOT
                    / "data"
                    / "raw"
                    / "trade_kline"
                    / symbol
                    / "60.parquet"
                ),

            "funding":
                info(
                    ROOT
                    / "data"
                    / "raw"
                    / "funding"
                    / symbol
                    / "funding.parquet"
                ),

            "orderbook":
                info(
                    ROOT
                    / "data"
                    / "live"
                    / "orderbook_features"
                    / symbol
                    / "features.parquet"
                ),
        }

    validation = {}

    validation_path = (
        ROOT
        / "data"
        / "validation_report.json"
    )

    if validation_path.exists():
        payload = read_json(
            validation_path,
            {},
        )

        validation = (
            payload.get(
                "summary",
                {},
            )
        )

    approved = {}

    approved_path = (
        ROOT
        / "ml"
        / "models"
        / "approved"
        / "model_contract.json"
    )

    if approved_path.exists():
        approved = read_json(
            approved_path,
            {},
        )

    task = get_task_state()

    features = feature_versions()
    models = model_dashboard()
    shadow = shadow_dashboard()

    readiness = (
        readiness_status(
            rows,
            validation,
            features,
            models,
            approved,
            shadow,
        )
    )

    progress_flags = {
        "historical_data":
            readiness.get(
                "historical_data",
                False,
            ),

        "funding_data":
            readiness.get(
                "funding_data",
                False,
            ),

        "feature_build":
            readiness.get(
                "feature_build",
                False,
            ),

        "validation_report":
            readiness.get(
                "validation_report",
                False,
            ),

        "training_report":
            readiness.get(
                "training_report",
                False,
            ),

        "orderbook_collection":
            readiness.get(
                "orderbook_collection_started",
                False,
            ),

        "live_shadow":
            shadow.get(
                "fresh",
                False,
            ),

        "approved_model":
            readiness.get(
                "approved_model",
                False,
            ),
            
        "demo_paper_ready":
            readiness.get(
                "demo_paper_ready",
                False,
            ),
    }

    progress_done = sum(
        bool(value)
        for value in progress_flags.values()
    )

    progress = {
        "completed":
            progress_done,

        "total":
            len(
                progress_flags
            ),

        "percent":
            round(
                100
                * progress_done
                / max(
                    1,
                    len(
                        progress_flags
                    ),
                )
            ),

        "milestones":
            progress_flags,
    }

    state = {
        "running":
            (
                task.get("label")
                if task.get(
                    "running"
                )
                else None
            ),

        "last_message":
            task.get(
                "last_message",
                "Ready",
            ),
    }

    return {
        "massive_api_key":
            bool(
                os.getenv(
                    "MASSIVE_API_KEY",
                    "",
                ).strip()
            ),

        "rows":
            rows,

        "validation":
            validation,

        "approved":
            approved,

        "state":
            state,

        "task":
            task,

        "live": {
            "running":
                live_is_running(),

            "pid":
                read_lock_pid(
                    LIVE_LOCK
                ),
        },

        "builder": {
            "running": builder_is_running(),
            "pid": read_lock_pid(BUILDER_LOCK),
        },

        "shadow_proc": {
            "running": shadow_is_running(),
            "pid": read_lock_pid(SHADOW_LOCK),
        },

        "features": {
            "current_version":
                current_feature_version(),

            "versions":
                features,
        },

        "models":
            models,

        "shadow":
            shadow,

        "progress":
            progress,

        "readiness":
            readiness,
    }


# ---------------------------------------------------------
# Heavy job control
# ---------------------------------------------------------

def start_job(
    label: str,
    args: list[str],
):
    with STATE_LOCK:

        if task_is_running():
            state = get_task_state()

            return (
                False,
                (
                    "Another heavy task is already running: "
                    f"{state.get('label') or 'unknown'}"
                ),
            )

        try:
            fd = os.open(
                TASK_LOCK,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY,
            )

            os.write(
                fd,
                b"",
            )

            os.close(fd)

        except FileExistsError:
            return (
                False,
                "Heavy task lock already exists.",
            )

        started = time.time()

        state = default_task_state()

        state.update(
            {
                "status":
                    "starting",

                "label":
                    label,

                "started_at":
                    started,

                "last_output_at":
                    started,

                "last_message":
                    f"{label} starting...",
            }
        )

        write_json(
            TASK_STATE,
            state,
        )

    def worker():
        process = None

        try:
            python_exe = (
                ROOT
                / ".venv"
                / "Scripts"
                / "python.exe"
            )

            env = os.environ.copy()

            # Critical for live UI logs.
            env[
                "PYTHONUNBUFFERED"
            ] = "1"

            process = (
                subprocess.Popen(
                    [
                        str(
                            python_exe
                        ),
                        *args,
                    ],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            )

            TASK_LOCK.write_text(
                str(
                    process.pid
                ),
                encoding="utf-8",
            )

            state = read_json(
                TASK_STATE,
                default_task_state(),
            )

            state.update(
                {
                    "status":
                        "running",

                    "pid":
                        process.pid,

                    "last_output_at":
                        time.time(),
                }
            )

            write_json(
                TASK_STATE,
                state,
            )

            output_lines = []

            if process.stdout:

                for line in process.stdout:
                    line = (
                        line.rstrip()
                    )

                    if not line:
                        continue

                    output_lines.append(
                        line
                    )

                    output_lines = (
                        output_lines[-120:]
                    )

                    state = read_json(
                        TASK_STATE,
                        default_task_state(),
                    )

                    stage = detect_stage(
                        line
                    )

                    state.update(
                        {
                            "status":
                                "running",

                            "pid":
                                process.pid,

                            "last_output_at":
                                time.time(),

                            "last_message":
                                "\n".join(
                                    output_lines
                                ),
                        }
                    )

                    if stage:
                        state["stage"] = (
                            stage
                        )

                    write_json(
                        TASK_STATE,
                        state,
                    )

            return_code = (
                process.wait()
            )

            state = read_json(
                TASK_STATE,
                default_task_state(),
            )

            if (
                state.get("status")
                != "cancelled"
            ):

                state.update(
                    {
                        "status":
                            (
                                "completed"
                                if return_code == 0
                                else "failed"
                            ),

                        "exit_code":
                            return_code,

                        "ended_at":
                            time.time(),

                        "pid":
                            None,

                        "last_message":
                            (
                                state.get(
                                    "last_message",
                                    "",
                                )
                                + "\n\nExit code: "
                                + str(
                                    return_code
                                )
                            ).strip(),
                    }
                )

                write_json(
                    TASK_STATE,
                    state,
                )

        except Exception as e:
            state = read_json(
                TASK_STATE,
                default_task_state(),
            )

            state.update(
                {
                    "status":
                        "failed",

                    "ended_at":
                        time.time(),

                    "pid":
                        None,

                    "last_message":
                        (
                            f"{type(e).__name__}: "
                            f"{e}"
                        ),
                }
            )

            write_json(
                TASK_STATE,
                state,
            )

        finally:
            remove_lock(
                TASK_LOCK
            )

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()

    return True, "Started"


def cancel_current_task():
    state = get_task_state()

    pid = state.get(
        "pid"
    )

    if not state.get(
        "running"
    ):
        return (
            False,
            "No heavy task is running.",
        )

    if pid:
        stop_pid(
            int(pid)
        )

    state.update(
        {
            "status":
                "cancelled",

            "ended_at":
                time.time(),

            "pid":
                None,

            "last_message":
                (
                    state.get(
                        "last_message",
                        "",
                    )
                    + "\n\nTask cancelled by user."
                ).strip(),
        }
    )

    write_json(
        TASK_STATE,
        state,
    )

    remove_lock(
        TASK_LOCK
    )

    return True, "Task cancelled."


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def home():
    return render_template(
        "index.html"
    )


@app.get("/api/status")
def status():
    return jsonify(
        get_status()
    )


@app.post(
    "/api/action/<name>"
)
def action(name):
    actions = {
        "collect": (
            "Historical collection",
            [
                "collect_historical.py",
            ],
        ),

        "validate": (
            "Validation",
            [
                "validate_data.py",
            ],
        ),

        "features": (
            "Feature build",
            [
                "ml/build_features.py",
            ],
        ),

        "train": (
            "Model training",
            [
                "ml/train_models.py",
            ],
        ),

        "export": (
            "QuantSight export",
            [
                "ml/export_for_quantsight.py",
            ],
        ),
    }

    if name not in actions:
        return jsonify(
            {
                "ok": False,
                "message":
                    "Unknown action",
            }
        ), 400

    ok, message = start_job(
        *actions[name]
    )

    return jsonify(
        {
            "ok": ok,
            "message": message,
        }
    ), (
        200
        if ok
        else 409
    )


@app.post("/api/task/cancel")
def cancel_task():
    ok, message = (
        cancel_current_task()
    )

    return jsonify(
        {
            "ok": ok,
            "message": message,
        }
    ), (
        200
        if ok
        else 409
    )


# ---------------------------------------------------------
# Live collector
# ---------------------------------------------------------

@app.post("/api/live/start")
def live_start():
    with STATE_LOCK:

        if live_is_running():
            return jsonify(
                {
                    "ok": True,
                    "running": True,
                    "message":
                        "Live collector already running.",
                }
            )

        try:
            fd = os.open(
                LIVE_LOCK,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY,
            )

            os.write(
                fd,
                b"",
            )

            os.close(fd)

        except FileExistsError:
            return jsonify(
                {
                    "ok": False,
                    "running": False,
                    "message":
                        "Live collector lock exists.",
                }
            ), 409

        try:
            python_exe = (
                ROOT
                / ".venv"
                / "Scripts"
                / "python.exe"
            )

            process = (
                subprocess.Popen(
                    [
                        str(
                            python_exe
                        ),
                        str(
                            ROOT
                            / "collect_live.py"
                        ),
                    ],
                    cwd=str(ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

            LIVE_LOCK.write_text(
                str(
                    process.pid
                ),
                encoding="utf-8",
            )

        except Exception as e:
            remove_lock(
                LIVE_LOCK
            )

            return jsonify(
                {
                    "ok": False,
                    "running": False,
                    "message":
                        (
                            f"{type(e).__name__}: "
                            f"{e}"
                        ),
                }
            ), 500

    return jsonify(
        {
            "ok": True,
            "running": True,
            "pid": process.pid,
            "message":
                "Live collector started.",
        }
    )


@app.post("/api/live/stop")
def live_stop():
    pid = read_lock_pid(
        LIVE_LOCK
    )

    if not live_is_running():
        remove_lock(
            LIVE_LOCK
        )

        return jsonify(
            {
                "ok": True,
                "running": False,
                "message":
                    "Live collector already stopped.",
            }
        )

    if pid:
        stop_pid(pid)

    remove_lock(
        LIVE_LOCK
    )

    return jsonify(
        {
            "ok": True,
            "running": False,
            "message":
                "Live collector stopped.",
        }
    )


# ---------------------------------------------------------
# Promotion
# ---------------------------------------------------------

@app.post("/api/promote")
def promote():
    payload = request.json or {}

    model = payload.get(
        "model",
        "",
    )

    allowed = {
        "logistic",
        "random_forest",
        "hist_gradient_boosting",
    }

    if model not in allowed:
        return jsonify(
            {
                "ok": False,
                "message":
                    "Invalid model",
            }
        ), 400

    ok, message = start_job(
        f"Promote {model}",
        [
            "ml/promote_model.py",
            model,
        ],
    )

    return jsonify(
        {
            "ok": ok,
            "message": message,
        }
    ), (
        200
        if ok
        else 409
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8765,
        debug=False,
        use_reloader=False,
        threaded=True,
    )