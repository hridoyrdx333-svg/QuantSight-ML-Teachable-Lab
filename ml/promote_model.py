from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

import joblib


ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = "v4_context_plus"

CANDIDATE_DIR = ROOT / "ml" / "models" / "candidate"
APPROVED_DIR = ROOT / "ml" / "models" / "approved"
THRESHOLD_REPORT = ROOT / "ml" / "reports" / "v43_locked_thresholds.json"

ALLOWED_MODELS = {
    "logistic",
    "random_forest",
    "hist_gradient_boosting",
}


def load_thresholds() -> dict[str, float]:
    if not THRESHOLD_REPORT.exists():
        return {}

    payload = json.loads(
        THRESHOLD_REPORT.read_text(encoding="utf-8")
    )

    out = {}

    for symbol, item in payload.get("symbols", {}).items():
        value = item.get("selected_threshold")

        if value is not None:
            out[str(symbol)] = float(value)

    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python ml/promote_model.py "
            "logistic|random_forest|hist_gradient_boosting"
        )

    model_name = sys.argv[1].strip()

    if model_name not in ALLOWED_MODELS:
        raise SystemExit(
            f"Unsupported model: {model_name}"
        )

    src = (
        CANDIDATE_DIR
        / f"{FEATURE_VERSION}__{model_name}.joblib"
    )

    if not src.exists():
        raise FileNotFoundError(
            f"Candidate artifact not found: {src}"
        )

    artifact = joblib.load(src)

    artifact_version = artifact.get("feature_version")

    if artifact_version != FEATURE_VERSION:
        raise RuntimeError(
            "Candidate feature version mismatch: "
            f"{artifact_version} != {FEATURE_VERSION}"
        )

    model_object = artifact.get("model")

    if model_object is None:
        raise RuntimeError(
            "Candidate artifact does not contain 'model'."
        )

    features = artifact.get("features") or []

    if not features:
        raise RuntimeError(
            "Candidate artifact has no feature contract."
        )

    APPROVED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dst_model = (
        APPROVED_DIR
        / "approved_model.joblib"
    )

    shutil.copy2(
        src,
        dst_model,
    )

    contract = {
        "model_name":
            artifact.get(
                "model_name",
                model_name,
            ),

        "feature_version":
            FEATURE_VERSION,

        "features":
            list(features),

        "target":
            artifact.get(
                "target",
                "label_up_6",
            ),

        "label_horizon_bars":
            artifact.get(
                "label_horizon_bars",
                6,
            ),

        "bar_ms":
            artifact.get(
                "bar_ms",
                300000,
            ),

        "locked_thresholds":
            load_thresholds(),

        "runtime_role":
            "FILTER_RANK_CONTEXT_ONLY",

        "can_create_trade":
            False,

        "can_bypass_risk":
            False,

        "can_enable_orders":
            False,

        "fallback":
            "RULE_BASED",

        "source_candidate":
            str(src),
    }

    (
        APPROVED_DIR
        / "model_contract.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Approved model ready:",
        dst_model,
    )

    print(
        "Feature version:",
        FEATURE_VERSION,
    )

    print(
        "Feature count:",
        len(features),
    )

    print(
        "Runtime role:",
        contract["runtime_role"],
    )

    print(
        "Orders enabled:",
        False,
    )


if __name__ == "__main__":
    main()
