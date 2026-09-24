from __future__ import annotations

from pathlib import Path
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_VERSION = "v4_context_plus"
MODEL_NAME = "random_forest"

FEATURE_DIR = (
    ROOT
    / "ml"
    / "features"
    / FEATURE_VERSION
)

MODEL_PATH = (
    ROOT
    / "ml"
    / "models"
    / "candidate"
    / f"{FEATURE_VERSION}__{MODEL_NAME}.joblib"
)

TRAIN_REPORT = (
    ROOT
    / "ml"
    / "reports"
    / "training_report_v4_context_plus.json"
)

THRESHOLD_REPORT = (
    ROOT
    / "ml"
    / "reports"
    / "v43_locked_thresholds.json"
)

SHADOW_DIR = (
    ROOT
    / "ml"
    / "shadow"
)

SHADOW_LOG = (
    SHADOW_DIR
    / "shadow_predictions.jsonl"
)

SHADOW_STATE = (
    SHADOW_DIR
    / "shadow_state.json"
)

SHADOW_STATUS = (
    SHADOW_DIR
    / "shadow_status.json"
)

POLL_SECONDS = 5

# A feature row older than this is reported as STALE.
MAX_FEATURE_AGE_SECONDS = 15 * 60


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
    default=None,
):
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return default


def write_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    for _ in range(10):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            time.sleep(0.1)


def append_jsonl(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                payload,
                separators=(",", ":"),
            )
            + "\n"
        )


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model missing: {MODEL_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise RuntimeError(
            "Unexpected model artifact format."
        )

    model = artifact.get(
        "model"
    )

    features = artifact.get(
        "features"
    )

    if model is None:
        raise RuntimeError(
            "Model object missing."
        )

    if not features:
        report = read_json(
            TRAIN_REPORT,
            {},
        )

        features = report.get(
            "features"
        )

    if not features:
        raise RuntimeError(
            "Feature contract missing."
        )

    artifact_version = artifact.get(
        "feature_version"
    )

    if (
        artifact_version
        and artifact_version
        != FEATURE_VERSION
    ):
        raise RuntimeError(
            "Feature version mismatch: "
            f"{artifact_version} != "
            f"{FEATURE_VERSION}"
        )

    return (
        model,
        list(features),
    )


def load_thresholds():
    report = read_json(
        THRESHOLD_REPORT,
        {},
    )

    symbols = report.get(
        "symbols",
        {}
    )

    thresholds = {}

    for symbol, payload in (
        symbols.items()
    ):
        threshold = payload.get(
            "selected_threshold"
        )

        if threshold is not None:
            thresholds[
                symbol
            ] = float(
                threshold
            )

    if not thresholds:
        raise RuntimeError(
            "No locked thresholds found."
        )

    return thresholds


def load_state():
    state = read_json(
        SHADOW_STATE,
        {},
    )

    if not isinstance(
        state,
        dict,
    ):
        state = {}

    return state


def latest_feature_row(
    symbol: str,
):
    path = (
        FEATURE_DIR
        / f"{symbol}_5m.parquet"
    )

    if not path.exists():
        return (
            None,
            f"Feature file missing: {path}",
        )

    last_error = None

    for attempt in range(3):
        try:
            df = pd.read_parquet(
                path
            )
            last_error = None
            break

        except Exception as e:
            last_error = e
            time.sleep(0.15 * (attempt + 1))

    if last_error is not None:
        return (
            None,
            (
                f"{type(last_error).__name__}: "
                f"{last_error}"
            ),
        )

    if df.empty:
        return (
            None,
            "Feature file empty.",
        )

    if "start_ms" not in df.columns:
        return (
            None,
            "start_ms missing.",
        )

    df = df.sort_values(
        "start_ms"
    )

    return (
        df.iloc[-1].copy(),
        None,
    )


def row_age_seconds(
    start_ms: int,
):
    now_ms = int(
        time.time() * 1000
    )

    return max(
        0.0,
        (
            now_ms
            - int(start_ms)
        )
        / 1000.0,
    )


def validate_feature_row(
    row: pd.Series,
    features: list[str],
):
    missing_columns = [
        feature
        for feature in features
        if feature not in row.index
    ]

    if missing_columns:
        return (
            False,
            "Missing columns: "
            + ", ".join(
                missing_columns
            ),
        )

    values = pd.to_numeric(
        row[
            features
        ],
        errors="coerce",
    )

    bad_features = [
        feature
        for feature, value
        in values.items()
        if (
            pd.isna(value)
            or not np.isfinite(
                float(value)
            )
        )
    ]

    if bad_features:
        return (
            False,
            "Invalid feature values: "
            + ", ".join(
                bad_features
            ),
        )

    return (
        True,
        None,
    )


def score_row(
    model,
    features,
    row,
):
    X = pd.DataFrame(
        [
            {
                feature:
                    float(
                        row[
                            feature
                        ]
                    )
                for feature
                in features
            }
        ]
    )

    probability = float(
        model.predict_proba(
            X
        )[0, 1]
    )

    return probability


def write_status(
    **kwargs,
):
    payload = {
        "updated_at":
            utc_now_iso(),

        "mode":
            "SHADOW_ONLY",

        "orders_enabled":
            False,

        "place_orders":
            False,

        "feature_version":
            FEATURE_VERSION,

        "model":
            MODEL_NAME,
    }

    payload.update(
        kwargs
    )

    write_json(
        SHADOW_STATUS,
        payload,
    )


def process_symbol(
    symbol,
    threshold,
    model,
    features,
    state,
    force=False,
):
    row, error = (
        latest_feature_row(
            symbol
        )
    )

    if error:
        print(
            f"[{symbol}] ERROR: "
            f"{error}",
            flush=True,
        )

        return False

    start_ms = int(
        row[
            "start_ms"
        ]
    )

    previous = state.get(
        symbol
    )

    if (
        not force
        and previous is not None
        and int(previous)
        >= start_ms
    ):
        return False

    age_seconds = (
        row_age_seconds(
            start_ms
        )
    )

    valid, error = (
        validate_feature_row(
            row,
            features,
        )
    )

    if not valid:
        record = {
            "logged_at":
                utc_now_iso(),

            "symbol":
                symbol,

            "start_ms":
                start_ms,

            "feature_version":
                FEATURE_VERSION,

            "model":
                MODEL_NAME,

            "decision":
                "ERROR",

            "reason":
                error,

            "orders_sent":
                False,
        }

        append_jsonl(
            SHADOW_LOG,
            record,
        )

        print(
            f"[{symbol}] ERROR: "
            f"{error}",
            flush=True,
        )

        # Invalid rows are not marked as processed.
        # The same bar will be retried after live features are repaired.
        return True

    probability = score_row(
        model,
        features,
        row,
    )

    stale = (
        age_seconds
        > MAX_FEATURE_AGE_SECONDS
    )

    if stale:
        decision = "STALE"

    elif probability >= threshold:
        decision = "PASS"

    else:
        decision = "REJECT"

    record = {
        "logged_at":
            utc_now_iso(),

        "symbol":
            symbol,

        "start_ms":
            start_ms,

        "start_utc":
            pd.to_datetime(
                start_ms,
                unit="ms",
                utc=True,
            ).isoformat(),

        "feature_age_seconds":
            round(
                age_seconds,
                2,
            ),

        "feature_version":
            FEATURE_VERSION,

        "model":
            MODEL_NAME,

        "probability_up":
            probability,

        "locked_threshold":
            threshold,

        "decision":
            decision,

        "orders_sent":
            False,

        "mode":
            "SHADOW_ONLY",
    }

    append_jsonl(
        SHADOW_LOG,
        record,
    )

    print(
        (
            f"[{symbol}] "
            f"{decision} | "
            f"p={probability:.4f} | "
            f"threshold={threshold:.2f} | "
            f"age={age_seconds:.0f}s | "
            f"start_ms={start_ms}"
        ),
        flush=True,
    )

    state[
        symbol
    ] = start_ms

    return True


def main():
    SHADOW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    print(
        "QuantSight V4 Shadow Runner",
        flush=True,
    )

    print(
        "NO ORDERS WILL BE SENT",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    model, features = (
        load_model()
    )

    thresholds = (
        load_thresholds()
    )

    state = load_state()

    print(
        "Feature version:",
        FEATURE_VERSION,
        flush=True,
    )

    print(
        "Model:",
        MODEL_NAME,
        flush=True,
    )

    print(
        "Features:",
        len(features),
        flush=True,
    )

    print(
        "Thresholds:",
        thresholds,
        flush=True,
    )

    print(
        "Poll seconds:",
        POLL_SECONDS,
        flush=True,
    )

    print(
        "Shadow log:",
        SHADOW_LOG,
        flush=True,
    )

    print(
        flush=True,
    )

    write_status(
        status="running",
        thresholds=thresholds,
        features_count=len(
            features
        ),
    )

    try:
        first_cycle = True

        while True:
            processed = False

            for (
                symbol,
                threshold,
            ) in thresholds.items():

                changed = (
                    process_symbol(
                        symbol,
                        threshold,
                        model,
                        features,
                        state,
                        force=first_cycle,
                    )
                )

                processed = (
                    processed
                    or changed
                )

            first_cycle = False

            if processed:
                write_json(
                    SHADOW_STATE,
                    state,
                )

            write_status(
                status="running",
                thresholds=thresholds,
                features_count=len(
                    features
                ),
                last_processed=state,
                waiting_for_new_bar=not processed,
            )

            time.sleep(
                POLL_SECONDS
            )

    except KeyboardInterrupt:
        print(
            "\nShadow runner stopped.",
            flush=True,
        )

        write_status(
            status="stopped",
            thresholds=thresholds,
            features_count=len(
                features
            ),
            last_processed=state,
        )


if __name__ == "__main__":
    main()