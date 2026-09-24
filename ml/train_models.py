from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
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

MDIR.mkdir(
    parents=True,
    exist_ok=True,
)

RDIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "label_up_6"

LABEL_HORIZON_BARS = 6
BAR_MS = 300_000
PURGE_MS = LABEL_HORIZON_BARS * BAR_MS


# ---------------------------------------------------------
# V4 features
# ---------------------------------------------------------

FEATURE_CANDIDATES = [

    # Price / momentum
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",

    # EMA distance
    "ema20_dist",
    "ema50_dist",
    "ema200_dist",

    # Technical context
    "rsi14",
    "atr_pct",
    "adx14",

    "range_pct",
    "body_pct",
    "volume_z",

    "realized_vol_12",
    "realized_vol_48",

    # Regime
    "trend_regime",
    "high_vol_regime",

    # Funding
    "funding_rate",
    "funding_change",
    "funding_abs",
    "funding_z_30",
    "funding_regime",

    # Mark / index / basis
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

    # Lagged BTC context
    "btc_ret_1_lag1",
    "btc_ret_1_lag3",
    "btc_ret_1_lag6",

    "btc_ret_6_lag1",
    "btc_ret_6_lag3",

    "btc_vol_12_lag1",

    # Symbol identity
    "symbol_is_btc",
    "symbol_is_eth",
    "symbol_is_sol",
]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def metrics(
    y,
    probability,
):

    prediction = (
        probability >= 0.5
    ).astype(int)

    result = {
        "precision": float(
            precision_score(
                y,
                prediction,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y,
                prediction,
                zero_division=0,
            )
        ),

        "brier": float(
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


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def load_dataset():

    files = sorted(
        FDIR.glob(
            "*_5m.parquet"
        )
    )

    if not files:
        raise RuntimeError(
            f"No {FEATURE_VERSION} feature files found in {FDIR}"
        )

    frames = []

    print(
        f"Feature version: {FEATURE_VERSION}",
        flush=True,
    )

    print(
        f"Feature files: {len(files)}",
        flush=True,
    )

    for path in files:

        df = pd.read_parquet(
            path
        )

        if "symbol" not in df.columns:
            symbol = (
                path.stem
                .replace(
                    "_5m",
                    "",
                )
            )

            df[
                "symbol"
            ] = symbol

        frames.append(
            df
        )

        print(
            f"[LOAD] {path.name}: {len(df):,} rows",
            flush=True,
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    if TARGET not in data.columns:
        raise RuntimeError(
            f"Missing target column: {TARGET}"
        )

    if "start_ms" not in data.columns:
        raise RuntimeError(
            "Missing start_ms column."
        )

    available_features = [
        feature
        for feature
        in FEATURE_CANDIDATES
        if feature in data.columns
    ]

    missing_features = [
        feature
        for feature
        in FEATURE_CANDIDATES
        if feature not in data.columns
    ]

    if not available_features:
        raise RuntimeError(
            "No V4 training features found."
        )

    print(
        "\nV4 features used:",
        len(
            available_features
        ),
        flush=True,
    )

    for feature in available_features:
        print(
            " +",
            feature,
            flush=True,
        )

    if missing_features:

        print(
            "\nOptional features not present:",
            flush=True,
        )

        for feature in missing_features:
            print(
                " -",
                feature,
                flush=True,
            )

    print(
        "\nRows before required-feature filter:",
        f"{len(data):,}",
        flush=True,
    )

    required = (
        available_features
        + [
            TARGET,
            "start_ms",
        ]
    )

    clean = (
        data[
            required
            + (
                ["symbol"]
                if "symbol" in data.columns
                else []
            )
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=required
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
            if "symbol" in clean.columns
            else [
                "start_ms"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "Rows after required-feature filter:",
        f"{len(clean):,}",
        flush=True,
    )

    return (
        clean,
        available_features,
        missing_features,
    )


# ---------------------------------------------------------
# Chronological split + purge
# ---------------------------------------------------------

def chronological_split(
    data,
):

    unique_times = np.array(
        sorted(
            data[
                "start_ms"
            ]
            .unique()
        )
    )

    if len(
        unique_times
    ) < 100:
        raise RuntimeError(
            "Not enough unique timestamps."
        )

    train_index = int(
        len(
            unique_times
        )
        * 0.70
    )

    validation_index = int(
        len(
            unique_times
        )
        * 0.85
    )

    train_boundary = int(
        unique_times[
            train_index
        ]
    )

    validation_boundary = int(
        unique_times[
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
        train_boundary,
        validation_boundary,
    )


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

def make_models():

    logistic = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
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

    hist_gradient_boosting = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    max_iter=200,
                    learning_rate=0.05,
                    random_state=42,
                ),
            ),
        ]
    )

    return {
        "logistic":
            logistic,

        "random_forest":
            random_forest,

        "hist_gradient_boosting":
            hist_gradient_boosting,
    }


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

def main():

    started_at = time.time()

    (
        data,
        features,
        missing_features,
    ) = load_dataset()

    (
        train,
        validation,
        test,
        train_boundary,
        validation_boundary,
    ) = chronological_split(
        data
    )

    print(
        "\nTraining rows:",
        f"{len(train):,}",
        flush=True,
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

    print(
        "Purged horizon:",
        LABEL_HORIZON_BARS,
        "bars",
        flush=True,
    )

    if (
        train.empty
        or validation.empty
        or test.empty
    ):
        raise RuntimeError(
            "Train/validation/test split produced empty dataset."
        )

    X_train = train[
        features
    ]

    y_train = train[
        TARGET
    ]

    X_validation = (
        validation[
            features
        ]
    )

    y_validation = (
        validation[
            TARGET
        ]
    )

    X_test = test[
        features
    ]

    y_test = test[
        TARGET
    ]

    models = make_models()

    report_models = {}

    for (
        model_name,
        model,
    ) in models.items():

        print(
            f"\nTraining {model_name}...",
            flush=True,
        )

        model.fit(
            X_train,
            y_train,
        )

        validation_probability = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        test_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        validation_metrics = (
            metrics(
                y_validation,
                validation_probability,
            )
        )

        test_metrics = (
            metrics(
                y_test,
                test_probability,
            )
        )

        print(
            "Validation:",
            validation_metrics,
            flush=True,
        )

        print(
            "Test:",
            test_metrics,
            flush=True,
        )

        artifact = {
            "feature_version":
                FEATURE_VERSION,

            "model_name":
                model_name,

            "features":
                features,

            "target":
                TARGET,

            "label_horizon_bars":
                LABEL_HORIZON_BARS,

            "bar_ms":
                BAR_MS,

            "model":
                model,
        }

        model_path = (
            MDIR
            / (
                f"{FEATURE_VERSION}"
                f"__{model_name}.joblib"
            )
        )

        joblib.dump(
            artifact,
            model_path,
        )

        report_models[
            model_name
        ] = {
            "validation":
                validation_metrics,

            "test":
                test_metrics,

            "artifact":
                str(
                    model_path
                ),
        }

    elapsed = (
        time.time()
        - started_at
    )

    report = {
        "feature_version":
            FEATURE_VERSION,

        "target":
            TARGET,

        "label_horizon_bars":
            LABEL_HORIZON_BARS,

        "bar_ms":
            BAR_MS,

        "purge_ms":
            PURGE_MS,

        "features":
            features,

        "missing_optional_features":
            missing_features,

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

        "split": {
            "method":
                "chronological_70_15_15_with_purge",

            "train_boundary_ms":
                train_boundary,

            "validation_boundary_ms":
                validation_boundary,

            "purge_bars":
                LABEL_HORIZON_BARS,
        },

        "models":
            report_models,

        "elapsed_seconds":
            round(
                elapsed,
                2,
            ),

        "auto_promoted":
            False,
    }

    # IMPORTANT:
    # Version-specific report so V3 is not overwritten.
    version_report_path = (
        RDIR
        / (
            "training_report_"
            f"{FEATURE_VERSION}.json"
        )
    )

    version_report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Latest report pointer/file for compatibility.
    latest_report_path = (
        RDIR
        / "training_report.json"
    )

    latest_report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nTraining complete.",
        flush=True,
    )

    print(
        "Candidate models saved.",
        flush=True,
    )

    print(
        "NO model auto-promoted.",
        flush=True,
    )

    print(
        "Version report:",
        version_report_path,
        flush=True,
    )

    print(
        "Latest report:",
        latest_report_path,
        flush=True,
    )


if __name__ == "__main__":
    main()