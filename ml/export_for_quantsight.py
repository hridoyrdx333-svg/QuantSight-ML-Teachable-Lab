from __future__ import annotations

from pathlib import Path
import json
import shutil

import joblib


ROOT = Path(__file__).resolve().parents[1]

APPROVED_DIR = (
    ROOT
    / "ml"
    / "models"
    / "approved"
)

EXPORT_DIR = (
    ROOT
    / "ml"
    / "exports"
    / "quantsight_ml_bundle"
)


RUNTIME_LOADER = 'from __future__ import annotations\n\nfrom pathlib import Path\nimport json\nimport math\n\nimport joblib\nimport pandas as pd\n\n\nclass QuantSightMLRuntime:\n    def __init__(self, bundle_dir):\n        path = Path(bundle_dir)\n\n        self.contract = json.loads(\n            (\n                path\n                / "model_contract.json"\n            ).read_text(\n                encoding="utf-8"\n            )\n        )\n\n        self.artifact = joblib.load(\n            path\n            / "approved_model.joblib"\n        )\n\n        self.model = self.artifact.get(\n            "model"\n        )\n\n        if self.model is None:\n            raise RuntimeError(\n                "Approved artifact does not contain \'model\'."\n            )\n\n        self.features = list(\n            self.contract.get(\n                "features",\n                []\n            )\n        )\n\n        if not self.features:\n            raise RuntimeError(\n                "Model feature contract is empty."\n            )\n\n        artifact_version = (\n            self.artifact.get(\n                "feature_version"\n            )\n        )\n\n        contract_version = (\n            self.contract.get(\n                "feature_version"\n            )\n        )\n\n        if artifact_version != contract_version:\n            raise RuntimeError(\n                "Feature-version mismatch: "\n                f"{artifact_version} != "\n                f"{contract_version}"\n            )\n\n        self.thresholds = dict(\n            self.contract.get(\n                "locked_thresholds",\n                {}\n            )\n        )\n\n    def score(self, feature_row, symbol=None):\n        missing = [\n            feature\n            for feature in self.features\n            if feature not in feature_row\n        ]\n\n        if missing:\n            return {\n                "ml_available": False,\n                "reason": "missing_features",\n                "missing": missing,\n                "fallback": "RULE_BASED",\n            }\n\n        values = {}\n        invalid = []\n\n        for feature in self.features:\n            try:\n                value = float(\n                    feature_row[\n                        feature\n                    ]\n                )\n            except Exception:\n                invalid.append(\n                    feature\n                )\n                continue\n\n            if not math.isfinite(\n                value\n            ):\n                invalid.append(\n                    feature\n                )\n                continue\n\n            values[\n                feature\n            ] = value\n\n        if invalid:\n            return {\n                "ml_available": False,\n                "reason": "invalid_features",\n                "invalid": invalid,\n                "fallback": "RULE_BASED",\n            }\n\n        x = pd.DataFrame(\n            [values],\n            columns=self.features,\n        )\n\n        score = float(\n            self.model.predict_proba(\n                x\n            )[0, 1]\n        )\n\n        result = {\n            "ml_available": True,\n            "score": score,\n            "model":\n                self.contract.get(\n                    "model_name"\n                ),\n            "feature_version":\n                self.contract.get(\n                    "feature_version"\n                ),\n            "role":\n                self.contract.get(\n                    "runtime_role"\n                ),\n            "fallback": "RULE_BASED",\n            "orders_enabled": False,\n        }\n\n        if (\n            symbol\n            and symbol\n            in self.thresholds\n        ):\n            threshold = float(\n                self.thresholds[\n                    symbol\n                ]\n            )\n\n            result[\n                "threshold"\n            ] = threshold\n\n            result[\n                "passes_filter"\n            ] = (\n                score\n                >= threshold\n            )\n\n        return result\n'


def main():
    approved_model = (
        APPROVED_DIR
        / "approved_model.joblib"
    )

    contract_path = (
        APPROVED_DIR
        / "model_contract.json"
    )

    if (
        not approved_model.exists()
        or not contract_path.exists()
    ):
        raise RuntimeError(
            "No approved model. "
            "Promote one explicitly first."
        )

    contract = json.loads(
        contract_path.read_text(
            encoding="utf-8"
        )
    )

    artifact = joblib.load(
        approved_model
    )

    if artifact.get("model") is None:
        raise RuntimeError(
            "Approved artifact missing 'model'."
        )

    if (
        artifact.get(
            "feature_version"
        )
        != contract.get(
            "feature_version"
        )
    ):
        raise RuntimeError(
            "Approved artifact/contract "
            "feature-version mismatch."
        )

    shutil.rmtree(
        EXPORT_DIR,
        ignore_errors=True,
    )

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        approved_model,
        EXPORT_DIR
        / "approved_model.joblib",
    )

    shutil.copy2(
        contract_path,
        EXPORT_DIR
        / "model_contract.json",
    )

    (
        EXPORT_DIR
        / "runtime_loader.py"
    ).write_text(
        RUNTIME_LOADER,
        encoding="utf-8",
    )

    (
        EXPORT_DIR
        / "README.txt"
    ).write_text(
        (
            "QuantSight ML bundle\n"
            "Role: filter/rank/context only.\n"
            "The ML runtime cannot create trades, "
            "bypass risk, or enable orders.\n"
            "Keep rule-based fallback enabled.\n"
        ),
        encoding="utf-8",
    )

    print(
        "Exported:",
        EXPORT_DIR,
    )


if __name__ == "__main__":
    main()
