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
    / "v42_threshold_evaluation.json"
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


def load_json(path):
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

    data = (
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

    return data


def make_test_split(data):
    times = np.array(
        sorted(
            data[
                "start_ms"
            ]
            .unique()
        )
    )

    validation_index = int(
        len(times) * 0.85
    )

    validation_boundary = int(
        times[
            validation_index
        ]
    )

    return data[
        data[
            "start_ms"
        ]
        >= validation_boundary
    ].copy()


def threshold_metrics(
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

            "total_rows":
                total,

            "selected_rows":
                0,

            "coverage":
                0.0,

            "precision":
                None,

            "recall":
                0.0,

            "win_rate":
                None,
        }

    selected_y = y[
        selected
    ]

    selected_prediction = np.ones(
        selected_count,
        dtype=int,
    )

    precision = float(
        precision_score(
            selected_y,
            selected_prediction,
            zero_division=0,
        )
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

        "total_rows":
            total,

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

        "win_rate":
            float(
                selected_y.mean()
            ),
    }


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

    test = make_test_split(
        data
    )

    X = test[
        features
    ]

    y = (
        test[
            TARGET
        ]
        .to_numpy()
    )

    probability = (
        model.predict_proba(
            X
        )[:, 1]
    )

    overall = {
        "rows":
            int(len(test)),

        "positive_rate":
            float(
                y.mean()
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    probability,
                )
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

    thresholds = [
        threshold_metrics(
            y,
            probability,
            threshold,
        )
        for threshold
        in THRESHOLDS
    ]

    per_symbol = {}

    for symbol in sorted(
        test[
            "symbol"
        ].unique()
    ):
        subset = test[
            test[
                "symbol"
            ]
            == symbol
        ]

        sx = subset[
            features
        ]

        sy = (
            subset[
                TARGET
            ]
            .to_numpy()
        )

        sp = (
            model.predict_proba(
                sx
            )[:, 1]
        )

        per_symbol[
            symbol
        ] = {
            "rows":
                int(
                    len(subset)
                ),

            "roc_auc":
                float(
                    roc_auc_score(
                        sy,
                        sp,
                    )
                ),

            "thresholds":
                [
                    threshold_metrics(
                        sy,
                        sp,
                        threshold,
                    )
                    for threshold
                    in THRESHOLDS
                ],
        }

    output = {
        "feature_version":
            FEATURE_VERSION,

        "model":
            "random_forest",

        "overall":
            overall,

        "thresholds":
            thresholds,

        "per_symbol":
            per_symbol,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== OVERALL ==="
    )

    print(
        overall
    )

    print()

    print(
        "=== THRESHOLDS ==="
    )

    for row in thresholds:
        print(
            f"threshold={row['threshold']:.2f} | "
            f"selected={row['selected_rows']:,} | "
            f"coverage={row['coverage']:.4f} | "
            f"precision={row['precision']} | "
            f"recall={row['recall']:.4f}"
        )

    print()

    print(
        "=== PER SYMBOL ==="
    )

    for symbol, payload in per_symbol.items():
        print()
        print(
            symbol,
            "ROC-AUC=",
            payload[
                "roc_auc"
            ],
        )

        for row in payload[
            "thresholds"
        ]:
            print(
                f"  {row['threshold']:.2f} | "
                f"selected={row['selected_rows']:,} | "
                f"coverage={row['coverage']:.4f} | "
                f"precision={row['precision']}"
            )

    print()

    print(
        "V4.2 threshold evaluation complete."
    )

    print(
        "Report:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()