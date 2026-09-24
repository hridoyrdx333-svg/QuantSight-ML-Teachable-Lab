from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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

RDIR = (
    ROOT
    / "ml"
    / "reports"
)

RDIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_REPORT = (
    RDIR
    / "training_report_v4_context_plus.json"
)

DIAG_REPORT = (
    RDIR
    / "v4_diagnostics.json"
)

OUTPUT_REPORT = (
    RDIR
    / "v41_feature_set_evaluation.json"
)

TARGET = "label_up_6"

BAR_MS = 300_000
LABEL_HORIZON_BARS = 6
PURGE_MS = (
    LABEL_HORIZON_BARS
    * BAR_MS
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

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


def metrics(
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


def make_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ---------------------------------------------------------
# Load V4 data
# ---------------------------------------------------------

def load_dataset(
    all_features,
):

    files = sorted(
        FDIR.glob(
            "*_5m.parquet"
        )
    )

    if not files:
        raise RuntimeError(
            f"No feature files in {FDIR}"
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
            f"{len(df):,} rows",
            flush=True,
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    required = (
        list(all_features)
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
            "Missing required columns: "
            + ", ".join(
                missing
            )
        )

    data = (
        data[
            required
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .copy()
    )

    data[
        TARGET
    ] = pd.to_numeric(
        data[
            TARGET
        ],
        errors="coerce",
    )

    data[
        "start_ms"
    ] = pd.to_numeric(
        data[
            "start_ms"
        ],
        errors="coerce",
    )

    data = (
        data
        .dropna(
            subset=[
                TARGET,
                "start_ms",
            ]
        )
        .copy()
    )

    data[
        TARGET
    ] = (
        data[
            TARGET
        ]
        .astype(int)
    )

    data[
        "start_ms"
    ] = (
        data[
            "start_ms"
        ]
        .astype(
            "int64"
        )
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


# ---------------------------------------------------------
# Feature sets
# ---------------------------------------------------------

def build_feature_sets(
    training_report,
    diagnostics,
):

    full = list(
        training_report[
            "features"
        ]
    )

    importance_rows = (
        diagnostics[
            "random_forest_feature_importance"
        ]
    )

    ranked = [
        row[
            "feature"
        ]
        for row in importance_rows
    ]

    top15 = ranked[:15]
    top20 = ranked[:20]

    context_candidates = [

        # momentum
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",

        # trend
        "ema20_dist",
        "ema50_dist",
        "ema200_dist",
        "rsi14",
        "adx14",

        # volatility
        "atr_pct",
        "realized_vol_12",
        "realized_vol_48",
        "volume_z",

        # funding
        "funding_rate",
        "funding_change",
        "funding_abs",
        "funding_z_30",
        "funding_regime",

        # basis / divergence
        "mark_divergence",
        "index_divergence",
        "mark_index_basis",
        "basis_change_1",
        "basis_change_3",
        "basis_z_48",

        # BTC context
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

        # symbol identity
        "symbol_is_btc",
        "symbol_is_eth",
        "symbol_is_sol",
    ]

    context = [
        feature
        for feature in context_candidates
        if feature in full
    ]

    return {
        "full_41":
            full,

        "top_15":
            top15,

        "top_20":
            top20,

        "context_focused":
            context,
    }


# ---------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------

def walk_forward(
    data,
    features,
):

    usable = (
        data[
            features
            + [
                TARGET,
                "start_ms",
                "symbol",
            ]
        ]
        .dropna(
            subset=features
        )
        .copy()
    )

    times = np.array(
        sorted(
            usable[
                "start_ms"
            ]
            .unique()
        )
    )

    if len(times) < 100:
        raise RuntimeError(
            "Not enough timestamps."
        )

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

    folds = []

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

        train = usable[
            usable[
                "start_ms"
            ]
            < (
                train_boundary
                - PURGE_MS
            )
        ].copy()

        test = usable[
            (
                usable[
                    "start_ms"
                ]
                >= train_boundary
            )
            &
            (
                usable[
                    "start_ms"
                ]
                <= test_boundary
            )
        ].copy()

        if (
            train.empty
            or test.empty
        ):
            continue

        print(
            f"  Fold {fold_number}: "
            f"train={len(train):,} "
            f"test={len(test):,}",
            flush=True,
        )

        model = make_model()

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

        fold_metrics = metrics(
            test[
                TARGET
            ],
            probability,
        )

        fold_metrics[
            "fold"
        ] = fold_number

        # per-symbol performance
        fold_metrics[
            "per_symbol"
        ] = {}

        for symbol in sorted(
            test[
                "symbol"
            ]
            .unique()
        ):

            subset = test[
                test[
                    "symbol"
                ]
                == symbol
            ]

            symbol_probability = (
                model.predict_proba(
                    subset[
                        features
                    ]
                )[:, 1]
            )

            fold_metrics[
                "per_symbol"
            ][
                symbol
            ] = metrics(
                subset[
                    TARGET
                ],
                symbol_probability,
            )

        folds.append(
            fold_metrics
        )

    auc_values = [
        fold[
            "roc_auc"
        ]
        for fold in folds
        if fold.get(
            "roc_auc"
        )
        is not None
    ]

    brier_values = [
        fold[
            "brier"
        ]
        for fold in folds
        if fold.get(
            "brier"
        )
        is not None
    ]

    logloss_values = [
        fold[
            "log_loss"
        ]
        for fold in folds
        if fold.get(
            "log_loss"
        )
        is not None
    ]

    summary = {
        "feature_count":
            len(features),

        "rows":
            int(
                len(usable)
            ),

        "folds":
            folds,

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

        "std_roc_auc":
            (
                float(
                    np.std(
                        auc_values
                    )
                )
                if auc_values
                else None
            ),

        "mean_brier":
            (
                float(
                    np.mean(
                        brier_values
                    )
                )
                if brier_values
                else None
            ),

        "mean_log_loss":
            (
                float(
                    np.mean(
                        logloss_values
                    )
                )
                if logloss_values
                else None
            ),
    }

    return summary


# ---------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------

def make_comparison(
    evaluations,
):

    rows = []

    for name, result in (
        evaluations.items()
    ):

        rows.append(
            {
                "feature_set":
                    name,

                "feature_count":
                    result.get(
                        "feature_count"
                    ),

                "rows":
                    result.get(
                        "rows"
                    ),

                "mean_roc_auc":
                    result.get(
                        "mean_roc_auc"
                    ),

                "min_roc_auc":
                    result.get(
                        "min_roc_auc"
                    ),

                "max_roc_auc":
                    result.get(
                        "max_roc_auc"
                    ),

                "std_roc_auc":
                    result.get(
                        "std_roc_auc"
                    ),

                "mean_brier":
                    result.get(
                        "mean_brier"
                    ),

                "mean_log_loss":
                    result.get(
                        "mean_log_loss"
                    ),
            }
        )

    rows.sort(
        key=lambda x: (
            x[
                "mean_roc_auc"
            ]
            if x[
                "mean_roc_auc"
            ]
            is not None
            else -1
        ),
        reverse=True,
    )

    return rows


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    started = time.time()

    print(
        "Loading V4 reports...",
        flush=True,
    )

    training_report = load_json(
        TRAIN_REPORT
    )

    diagnostics = load_json(
        DIAG_REPORT
    )

    feature_sets = (
        build_feature_sets(
            training_report,
            diagnostics,
        )
    )

    all_features = list(
        training_report[
            "features"
        ]
    )

    print(
        "Loading dataset...",
        flush=True,
    )

    data = load_dataset(
        all_features
    )

    print(
        f"Dataset rows: "
        f"{len(data):,}",
        flush=True,
    )

    evaluations = {}

    for name, features in (
        feature_sets.items()
    ):

        print()
        print(
            "=" * 60,
            flush=True,
        )

        print(
            f"Evaluating: {name}",
            flush=True,
        )

        print(
            f"Features: {len(features)}",
            flush=True,
        )

        print(
            "=" * 60,
            flush=True,
        )

        result = walk_forward(
            data,
            features,
        )

        evaluations[
            name
        ] = result

        print(
            f"Mean ROC-AUC: "
            f"{result['mean_roc_auc']}",
            flush=True,
        )

        print(
            f"Min ROC-AUC: "
            f"{result['min_roc_auc']}",
            flush=True,
        )

        print(
            f"Mean Brier: "
            f"{result['mean_brier']}",
            flush=True,
        )

        print(
            f"Mean Log-loss: "
            f"{result['mean_log_loss']}",
            flush=True,
        )

    comparison = (
        make_comparison(
            evaluations
        )
    )

    output = {
        "source_feature_version":
            FEATURE_VERSION,

        "generated_at_unix":
            time.time(),

        "feature_sets": {
            name:
                features
            for name, features
            in feature_sets.items()
        },

        "evaluations":
            evaluations,

        "comparison":
            comparison,

        "elapsed_seconds":
            round(
                time.time()
                - started,
                2,
            ),
    }

    OUTPUT_REPORT.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 60,
        flush=True,
    )

    print(
        "V4.1 FEATURE SET COMPARISON",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    for row in comparison:

        print(
            (
                f"{row['feature_set']}: "
                f"features={row['feature_count']} | "
                f"mean_auc={row['mean_roc_auc']:.6f} | "
                f"min_auc={row['min_roc_auc']:.6f} | "
                f"std={row['std_roc_auc']:.6f} | "
                f"brier={row['mean_brier']:.6f} | "
                f"logloss={row['mean_log_loss']:.6f}"
            ),
            flush=True,
        )

    print()
    print(
        "V4.1 evaluation complete.",
        flush=True,
    )

    print(
        "Report:",
        OUTPUT_REPORT,
        flush=True,
    )


if __name__ == "__main__":
    main()