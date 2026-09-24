from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_VERSION = "v4_context_plus"

FDIR = (
    ROOT
    / "ml"
    / "features"
    / FEATURE_VERSION
)

MDIR = (
    ROOT
    / "ml"
    / "models"
    / "candidate"
)

RDIR = (
    ROOT
    / "ml"
    / "reports"
)

UI_STATIC = (
    ROOT
    / "ui"
    / "static"
)

RDIR.mkdir(
    parents=True,
    exist_ok=True,
)

UI_STATIC.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_REPORT = (
    RDIR
    / "training_report_v4_context_plus.json"
)

OUTPUT_REPORT = (
    RDIR
    / "v4_diagnostics.json"
)

UI_REPORT = (
    UI_STATIC
    / "v4_diagnostics.json"
)

TARGET = "label_up_6"

BAR_MS = 300_000
LABEL_HORIZON_BARS = 6
PURGE_MS = (
    LABEL_HORIZON_BARS
    * BAR_MS
)


def metric_dict(
    y,
    probability,
):

    y = np.asarray(y)
    probability = np.asarray(
        probability
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    result = {
        "rows":
            int(len(y)),

        "positive_rate":
            float(
                np.mean(y)
            ),

        "precision":
            float(
                precision_score(
                    y,
                    prediction,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y,
                    prediction,
                    zero_division=0,
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    y,
                    probability,
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

    try:
        result[
            "log_loss"
        ] = float(
            log_loss(
                y,
                probability,
                labels=[0, 1],
            )
        )

    except Exception:
        result[
            "log_loss"
        ] = None

    return result


def load_report():

    if not TRAIN_REPORT.exists():
        raise RuntimeError(
            "V4 training report missing: "
            f"{TRAIN_REPORT}"
        )

    return json.loads(
        TRAIN_REPORT.read_text(
            encoding="utf-8"
        )
    )


def load_data(
    features,
):

    files = sorted(
        FDIR.glob(
            "*_5m.parquet"
        )
    )

    if not files:
        raise RuntimeError(
            "No V4 feature parquet files."
        )

    frames = []

    for path in files:

        df = pd.read_parquet(
            path
        )

        if "symbol" not in df.columns:
            df[
                "symbol"
            ] = (
                path.stem
                .replace(
                    "_5m",
                    "",
                )
            )

        frames.append(
            df
        )

        print(
            f"[LOAD] {path.name}: "
            f"{len(df):,}",
            flush=True,
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
            + ", ".join(
                missing
            )
        )

    clean = (
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

    clean[
        TARGET
    ] = (
        clean[
            TARGET
        ]
        .astype(int)
    )

    clean[
        "start_ms"
    ] = (
        clean[
            "start_ms"
        ]
        .astype(
            "int64"
        )
    )

    clean = (
        clean
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

    return clean


def split_data(
    data,
):

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
        times[
            train_index
        ]
    )

    validation_boundary = int(
        times[
            validation_index
        ]
    )

    train = data[
        data[
            "start_ms"
        ]
        < (
            train_boundary
            - PURGE_MS
        )
    ].copy()

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
        train,
        validation,
        test,
    )


def load_models():

    models = {}

    for name in [
        "logistic",
        "random_forest",
        "hist_gradient_boosting",
    ]:

        path = (
            MDIR
            / (
                f"{FEATURE_VERSION}"
                f"__{name}.joblib"
            )
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing model artifact: {path}"
            )

        artifact = joblib.load(
            path
        )

        models[
            name
        ] = artifact[
            "model"
        ]

    return models


def per_symbol_results(
    models,
    test,
    features,
):

    output = {}

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

        X = subset[
            features
        ]

        y = subset[
            TARGET
        ]

        output[
            symbol
        ] = {}

        for name, model in (
            models.items()
        ):

            probability = (
                model.predict_proba(
                    X
                )[:, 1]
            )

            output[
                symbol
            ][
                name
            ] = metric_dict(
                y,
                probability,
            )

    return output


def baseline_result(
    train,
    test,
):

    train_probability = float(
        train[
            TARGET
        ].mean()
    )

    probability = np.full(
        len(test),
        train_probability,
        dtype=float,
    )

    result = metric_dict(
        test[
            TARGET
        ],
        probability,
    )

    result[
        "constant_probability"
    ] = train_probability

    return result


def random_forest_importance(
    models,
    features,
):

    pipeline = models[
        "random_forest"
    ]

    estimator = (
        pipeline.named_steps[
            "model"
        ]
    )

    values = getattr(
        estimator,
        "feature_importances_",
        None,
    )

    if values is None:
        return []

    pairs = [
        {
            "feature":
                feature,

            "importance":
                float(value),
        }
        for feature, value
        in zip(
            features,
            values,
        )
    ]

    pairs.sort(
        key=lambda x:
            x[
                "importance"
            ],
        reverse=True,
    )

    return pairs


def logistic_importance(
    models,
    features,
):

    pipeline = models[
        "logistic"
    ]

    estimator = (
        pipeline.named_steps[
            "model"
        ]
    )

    coefficients = (
        estimator.coef_[0]
    )

    pairs = [
        {
            "feature":
                feature,

            "coefficient":
                float(value),

            "absolute_importance":
                float(
                    abs(value)
                ),
        }
        for feature, value
        in zip(
            features,
            coefficients,
        )
    ]

    pairs.sort(
        key=lambda x:
            x[
                "absolute_importance"
            ],
        reverse=True,
    )

    return pairs


def walk_forward_rf(
    base_model,
    data,
    features,
):

    times = np.array(
        sorted(
            data[
                "start_ms"
            ]
            .unique()
        )
    )

    # Expanding-window:
    # 55->70, 70->85, 85->100
    windows = [
        (
            0.55,
            0.70,
        ),
        (
            0.70,
            0.85,
        ),
        (
            0.85,
            1.00,
        ),
    ]

    results = []

    for fold_number, (
        train_fraction,
        test_fraction,
    ) in enumerate(
        windows,
        start=1,
    ):

        train_end_index = int(
            len(times)
            * train_fraction
        )

        test_end_index = min(
            len(times) - 1,
            int(
                len(times)
                * test_fraction
            ),
        )

        train_boundary = int(
            times[
                train_end_index
            ]
        )

        test_boundary = int(
            times[
                test_end_index
            ]
        )

        train = data[
            data[
                "start_ms"
            ]
            < (
                train_boundary
                - PURGE_MS
            )
        ]

        test = data[
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
                <= test_boundary
            )
        ]

        if (
            train.empty
            or test.empty
        ):
            continue

        model = clone(
            base_model
        )

        print(
            f"[WF] fold {fold_number}: "
            f"train={len(train):,} "
            f"test={len(test):,}",
            flush=True,
        )

        model.fit(
            train[
                features
            ],
            train[
                TARGET
            ],
        )

        probability = (
            model.predict_proba(
                test[
                    features
                ]
            )[:, 1]
        )

        fold_metrics = metric_dict(
            test[
                TARGET
            ],
            probability,
        )

        fold_metrics[
            "fold"
        ] = fold_number

        results.append(
            fold_metrics
        )

    auc_values = [
        row[
            "roc_auc"
        ]
        for row in results
        if row.get(
            "roc_auc"
        )
        is not None
    ]

    summary = {
        "folds":
            results,

        "mean_roc_auc":
            (
                float(
                    np.mean(
                        auc_values
                    )
                )
                if auc_values
                else None
            ),

        "min_roc_auc":
            (
                float(
                    np.min(
                        auc_values
                    )
                )
                if auc_values
                else None
            ),

        "max_roc_auc":
            (
                float(
                    np.max(
                        auc_values
                    )
                )
                if auc_values
                else None
            ),
    }

    return summary


def main():

    started = time.time()

    report = load_report()

    features = report[
        "features"
    ]

    print(
        "Feature version:",
        FEATURE_VERSION,
        flush=True,
    )

    print(
        "Features:",
        len(features),
        flush=True,
    )

    data = load_data(
        features
    )

    (
        train,
        validation,
        test,
    ) = split_data(
        data
    )

    print(
        "Diagnostic train rows:",
        f"{len(train):,}",
        flush=True,
    )

    print(
        "Diagnostic test rows:",
        f"{len(test):,}",
        flush=True,
    )

    models = load_models()

    baseline = baseline_result(
        train,
        test,
    )

    print(
        "\nBaseline:",
        baseline,
        flush=True,
    )

    overall = {}

    for name, model in (
        models.items()
    ):

        probability = (
            model.predict_proba(
                test[
                    features
                ]
            )[:, 1]
        )

        overall[
            name
        ] = metric_dict(
            test[
                TARGET
            ],
            probability,
        )

        print(
            name,
            overall[
                name
            ],
            flush=True,
        )

    print(
        "\nPer-symbol diagnostics...",
        flush=True,
    )

    per_symbol = (
        per_symbol_results(
            models,
            test,
            features,
        )
    )

    print(
        "Feature importance...",
        flush=True,
    )

    rf_importance = (
        random_forest_importance(
            models,
            features,
        )
    )

    logistic_weights = (
        logistic_importance(
            models,
            features,
        )
    )

    print(
        "Walk-forward Random Forest...",
        flush=True,
    )

    walk_forward = (
        walk_forward_rf(
            models[
                "random_forest"
            ],
            data,
            features,
        )
    )

    output = {
        "feature_version":
            FEATURE_VERSION,

        "generated_at_unix":
            time.time(),

        "features_count":
            len(features),

        "rows": {
            "all":
                int(
                    len(data)
                ),

            "train":
                int(
                    len(train)
                ),

            "validation":
                int(
                    len(validation)
                ),

            "test":
                int(
                    len(test)
                ),
        },

        "baseline":
            baseline,

        "overall":
            overall,

        "per_symbol":
            per_symbol,

        "random_forest_feature_importance":
            rf_importance,

        "logistic_feature_weights":
            logistic_weights,

        "walk_forward_random_forest":
            walk_forward,

        "elapsed_seconds":
            round(
                time.time()
                - started,
                2,
            ),
    }

    text = json.dumps(
        output,
        indent=2,
    )

    OUTPUT_REPORT.write_text(
        text,
        encoding="utf-8",
    )

    UI_REPORT.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "\nV4 diagnostics complete.",
        flush=True,
    )

    print(
        "Report:",
        OUTPUT_REPORT,
        flush=True,
    )

    print(
        "Frontend copy:",
        UI_REPORT,
        flush=True,
    )


if __name__ == "__main__":
    main()