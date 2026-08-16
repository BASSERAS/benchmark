from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from path_dt_experiments.heston_mixture import (  # noqa: E402
    extract_price_path_features,
    label_bits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and validate the independent Heston-mixture oracle.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--minimum-accuracy", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_prices = np.load(args.data_dir / "oracle_train.npy")
    train_labels = np.load(args.data_dir / "oracle_train_labels.npy")
    validation_prices = np.load(args.data_dir / "oracle_validation.npy")
    validation_labels = np.load(args.data_dir / "oracle_validation_labels.npy")
    train_features = extract_price_path_features(train_prices)
    validation_features = extract_price_path_features(validation_prices)
    classifier = ExtraTreesClassifier(
        n_estimators=int(args.trees),
        min_samples_leaf=2,
        max_features=0.7,
        n_jobs=int(args.workers),
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(train_features, train_labels)
    prediction = classifier.predict(validation_features)
    bit_truth = label_bits(validation_labels)
    bit_prediction = label_bits(prediction)
    report = {
        "eight_regime_accuracy": float(accuracy_score(validation_labels, prediction)),
        "parameter_state_accuracy": {
            name: float(accuracy_score(bit_truth[name], bit_prediction[name]))
            for name in bit_truth
        },
        "confusion_matrix": confusion_matrix(
            validation_labels,
            prediction,
            labels=np.arange(8),
        ).tolist(),
        "validation_true_counts": np.bincount(validation_labels, minlength=8).tolist(),
        "validation_predicted_counts": np.bincount(prediction, minlength=8).tolist(),
        "train_paths": int(len(train_labels)),
        "validation_paths": int(len(validation_labels)),
        "num_features": int(train_features.shape[1]),
        "configuration": {
            "trees": int(args.trees),
            "minimum_accuracy": float(args.minimum_accuracy),
            "random_state": 42,
            "min_samples_leaf": 2,
            "max_features": 0.7,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.output_dir / "oracle.joblib")
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["eight_regime_accuracy"] < float(args.minimum_accuracy):
        raise RuntimeError(
            f"oracle gate failed: {report['eight_regime_accuracy']:.4f} "
            f"< {float(args.minimum_accuracy):.4f}"
        )


if __name__ == "__main__":
    main()
