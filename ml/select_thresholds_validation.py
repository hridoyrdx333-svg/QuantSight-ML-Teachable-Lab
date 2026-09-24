from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_VERSION = "v4_context_plus"

REPORT_PATH = (
    ROOT
    / "ml"
    / "reports"
    / "training_report_v4_context_plus.json"
)

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
    / f"{FEATURE_VERSION}__random_forest.joblib"
)

OUTPUT_PATH = (
    ROOT
    / "ml"
    / "reports"
    / "v43_locked_thresholds.json"
)

TARGET = "label_up_6"

BAR_MS = 300_000
PURGE_MS = 6 * BAR_MS


THRESHOLDS = [
    0.50,
    0.52,
    0.54,
    0.56,
    0.58,
    0.60,
    0.62,
    0.65,
]


MIN_SELECTED_VALIDATION = 200


def load_json(path: Path):
    if not path.exists():
        raise RuntimeError(
            f"Missing file: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_data(features):
    frames = []

    for path in sorted(
        FEATURE_DIR.glob(
            "*_5m.parquet"
        )
    ):
        df = pd.read_parquet(path)

        if "symbol" not in df.columns:
            df["symbol"] = (
                path.stem
                .replace(
                    "_5m",
                    "",
                )
            )

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No V4 feature parquet files found."
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    required = (
        list(features)
        + [
            TARGET,
            "start_ms",
            "symbol",
        ]
    )

    missing = [
        col
        for col in required
        if col not in data.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(missing)
        )

    data = (
        data[
            required
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=(
                list(features)
                + [
                    TARGET,
                    "start_ms",
                ]
            )
        )
        .copy()
    )

    data[TARGET] = (
        data[TARGET]
        .astype(int)
    )

    data["start_ms"] = (
        data["start_ms"]
        .astype("int64")
    )

    return (
        data
        .sort_values(
            [
                "start_ms",
                "symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def split_data(data):
    times = np.array(
        sorted(
            data[
                "start_ms"
            ]
            .unique()
        )
    )

    train_index = int(
        len(times) * 0.70
    )

    validation_index = int(
        len(times) * 0.85
    )

    train_boundary = int(
        times[train_index]
    )

    validation_boundary = int(
        times[validation_index]
    )

    validation = data[
        (
            data[
                "start_ms"
            ]
            >= train_boundary
        )
        &
        (
            data[
                "start_ms"
            ]
            < (
                validation_boundary
                - PURGE_MS
            )
        )
    ].copy()

    test = data[
        data[
            "start_ms"
        ]
        >= validation_boundary
    ].copy()

    return (
        validation,
        test,
    )


def threshold_stats(
    y,
    probability,
    threshold,
):
    selected = (
        probability
        >= threshold
    )

    selected_count = int(
        selected.sum()
    )

    total = int(
        len(y)
    )

    if selected_count == 0:
        return {
            "threshold": threshold,
            "selected_rows": 0,
            "coverage": 0.0,
            "precision": None,
            "recall": 0.0,
        }

    selected_y = y[
        selected
    ]

    precision = float(
        selected_y.mean()
    )

    recall = float(
        selected_y.sum()
        / max(
            1,
            y.sum(),
        )
    )

    return {
        "threshold":
            threshold,

        "selected_rows":
            selected_count,

        "coverage":
            float(
                selected_count
                / total
            ),

        "precision":
            precision,

        "recall":
            recall,
    }


def choose_threshold(
    y,
    probability,
):
    rows = [
        threshold_stats(
            y,
            probability,
            threshold,
        )
        for threshold
        in THRESHOLDS
    ]

    eligible = [
        row
        for row in rows
        if (
            row[
                "selected_rows"
            ]
            >= MIN_SELECTED_VALIDATION
            and
            row[
                "precision"
            ]
            is not None
        )
    ]

    if not eligible:
        return (
            None,
            rows,
        )

    eligible.sort(
        key=lambda x: (
            x["precision"],
            x["coverage"],
        ),
        reverse=True,
    )

    return (
        eligible[0],
        rows,
    )


def evaluate_fixed_threshold(
    y,
    probability,
    threshold,
):
    selected = (
        probability
        >= threshold
    )

    selected_count = int(
        selected.sum()
    )

    total = int(
        len(y)
    )

    if selected_count == 0:
        return {
            "threshold":
                threshold,

            "selected_rows":
                0,

            "coverage":
                0.0,

            "precision":
                None,

            "recall":
                0.0,
        }

    selected_y = y[
        selected
    ]

    return {
        "threshold":
            threshold,

        "selected_rows":
            selected_count,

        "coverage":
            float(
                selected_count
                / total
            ),

        "precision":
            float(
                selected_y.mean()
            ),

        "recall":
            float(
                selected_y.sum()
                / max(
                    1,
                    y.sum(),
                )
            ),
    }


def evaluate_model_quality(
    y,
    probability,
):
    result = {
        "rows":
            int(len(y)),

        "positive_rate":
            float(
                y.mean()
            ),

        "brier":
            float(
                brier_score_loss(
                    y,
                    probability,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y,
                    probability,
                    labels=[0, 1],
                )
            ),
    }

    try:
        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y,
                probability,
            )
        )
    except Exception:
        result[
            "roc_auc"
        ] = None

    return result


def main():
    report = load_json(
        REPORT_PATH
    )

    features = report[
        "features"
    ]

    artifact = joblib.load(
        MODEL_PATH
    )

    model = artifact[
        "model"
    ]

    data = load_data(
        features
    )

    validation, test = (
        split_data(
            data
        )
    )

    print(
        "Validation rows:",
        f"{len(validation):,}",
        flush=True,
    )

    print(
        "Test rows:",
        f"{len(test):,}",
        flush=True,
    )

    output = {
        "feature_version":
            FEATURE_VERSION,

        "model":
            "random_forest",

        "selection_rule":
            {
                "minimum_validation_selected_rows":
                    MIN_SELECTED_VALIDATION,

                "objective":
                    (
                        "highest validation precision "
                        "subject to minimum sample size"
                    ),
            },

        "symbols":
            {},
    }

    for symbol in sorted(
        validation[
            "symbol"
        ].unique()
    ):
        validation_subset = validation[
            validation[
                "symbol"
            ]
            == symbol
        ]

        test_subset = test[
            test[
                "symbol"
            ]
            == symbol
        ]

        validation_y = (
            validation_subset[
                TARGET
            ]
            .to_numpy()
        )

        test_y = (
            test_subset[
                TARGET
            ]
            .to_numpy()
        )

        validation_probability = (
            model.predict_proba(
                validation_subset[
                    features
                ]
            )[:, 1]
        )

        test_probability = (
            model.predict_proba(
                test_subset[
                    features
                ]
            )[:, 1]
        )

        (
            chosen,
            threshold_table,
        ) = choose_threshold(
            validation_y,
            validation_probability,
        )

        print()
        print(
            "=" * 60,
            flush=True,
        )

        print(
            symbol,
            flush=True,
        )

        print(
            "=" * 60,
            flush=True,
        )

        print(
            "Validation ROC-AUC:",
            evaluate_model_quality(
                validation_y,
                validation_probability,
            )[
                "roc_auc"
            ],
            flush=True,
        )

        for row in threshold_table:
            print(
                (
                    f"VAL "
                    f"{row['threshold']:.2f} | "
                    f"selected="
                    f"{row['selected_rows']:,} | "
                    f"coverage="
                    f"{row['coverage']:.4f} | "
                    f"precision="
                    f"{row['precision']}"
                ),
                flush=True,
            )

        if chosen is None:
            print(
                "No eligible threshold.",
                flush=True,
            )

            output[
                "symbols"
            ][
                symbol
            ] = {
                "selected_threshold":
                    None,

                "validation_thresholds":
                    threshold_table,

                "test_at_locked_threshold":
                    None,
            }

            continue

        locked_threshold = (
            chosen[
                "threshold"
            ]
        )

        test_result = (
            evaluate_fixed_threshold(
                test_y,
                test_probability,
                locked_threshold,
            )
        )

        print()
        print(
            "LOCKED THRESHOLD:",
            locked_threshold,
            flush=True,
        )

        print(
            "Validation selected:",
            chosen,
            flush=True,
        )

        print(
            "FINAL TEST:",
            test_result,
            flush=True,
        )

        output[
            "symbols"
        ][
                symbol
            ] = {
                "selected_threshold":
                    locked_threshold,

                "validation_model_quality":
                    evaluate_model_quality(
                        validation_y,
                        validation_probability,
                    ),

                "test_model_quality":
                    evaluate_model_quality(
                        test_y,
                        test_probability,
                    ),

                "validation_thresholds":
                    threshold_table,

                "validation_selected":
                    chosen,

                "test_at_locked_threshold":
                    test_result,
            }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "V4.3 locked threshold evaluation complete.",
        flush=True,
    )

    print(
        "Report:",
        OUTPUT_PATH,
        flush=True,
    )


if __name__ == "__main__":
    main()