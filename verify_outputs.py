#!/usr/bin/env python3
"""Check real/synthetic baseline artifacts without re-training or reading the held-out test labels for scoring.

This is an integrity verifier, not evidence that a dataset came from UCI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from run_baseline import LABELS, score

MODELS = ("multinomial_naive_bayes", "logistic_regression")
SPLITS = {"train", "validation", "test"}


def verify(root: Path) -> str:
    processed, models_dir, results = root / "data" / "processed", root / "models", root / "results"
    splits = pd.read_csv(processed / "sms_splits.csv", keep_default_na=False)
    assert {"message_id", "label", "text", "split"} <= set(splits.columns), "Missing split columns"
    assert len(splits) and splits["message_id"].is_unique, "Duplicate or empty message IDs"
    assert set(splits["split"]) == SPLITS, "Unexpected splits"
    assert set(splits["label"]) == set(LABELS), "Unexpected labels"
    assert not splits["text"].duplicated().any(), "Exact duplicate text across splits"
    for name, target in (("train", .6), ("validation", .2), ("test", .2)):
        assert abs((splits["split"] == name).mean() - target) < .01, f"Wrong split fraction for {name}"
    manifest = json.loads((processed / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["random_seed"] == 42
    assert manifest["data_summary"]["deduplicated_message_count"] == len(splits)
    for split in SPLITS:
        expected = {k: int(v) for k, v in splits.loc[splits["split"] == split, "label"].value_counts().items()}
        assert manifest["split_counts"][split] == expected, f"Manifest counts don't match {split}"
    validations = splits.loc[splits["split"] == "validation", ["message_id", "text", "label"]]
    recorded = pd.read_csv(results / "predictions_validation.csv", keep_default_na=False)
    metrics = pd.read_csv(results / "metrics_validation.csv")
    assert set(recorded["model"]) == set(MODELS), "Unexpected recorded models"
    assert len(recorded) == len(validations) * len(MODELS), "Wrong prediction count"
    assert set(recorded["split"]) == {"validation"}, "Test/train labels appear in prediction file"
    assert set(metrics["model"]) == set(MODELS) and len(metrics) == 2
    for name in MODELS:
        pred = recorded[recorded["model"] == name].set_index("message_id")
        assert pred.index.is_unique and set(pred.index) == set(validations["message_id"]), f"Wrong IDs for {name}"
        model = joblib.load(models_dir / f"baseline_{name}.joblib")  # local trusted files only
        assert list(model.classes_) == LABELS, "Unexpected model classes"
        calculated = model.predict(validations["text"])
        aligned = pred.loc[validations["message_id"]]
        assert (calculated == aligned["prediction"].to_numpy()).all(), f"Prediction mismatch for {name}"
        probabilities = model.predict_proba(validations["text"])
        assert ((aligned["probability_spam"].astype(float).to_numpy() - probabilities[:, list(model.classes_).index("spam")]) ** 2).max() < 1e-10
        measured = score(validations["label"], calculated, name)
        claimed = metrics.set_index("model").loc[name]
        for metric in ("accuracy", "spam_precision", "spam_recall", "spam_f1", "false_positive_rate"):
            assert abs(float(claimed[metric]) - float(measured[metric])) < 1e-10, f"Incorrect {metric} for {name}"
        chart = results / f"confusion_matrix_{name}.png"
        if chart.exists():
            assert chart.stat().st_size > 0, f"Empty figure for {name}"
    status = results / "RUN_STATUS.txt"
    if status.exists():
        assert status.read_text(encoding="utf-8").startswith("BASELINE_VALIDATION_COMPLETE")
    return f"PASS: {len(splits)} unique texts, models={len(MODELS)}, validation={len(validations)}, final-test metrics untouched"


def main():
    parser = argparse.ArgumentParser(description="Verify saved baseline outputs without retraining")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(verify(args.root))


if __name__ == "__main__":
    main()