#!/usr/bin/env python3
"""Khaled's baseline only: UCI SMS data, stable split, two models, validation metrics.

No text-evasion attacks, defense training, or final held-out test evaluation here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SEED = 42
LABELS = ["ham", "spam"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_messages(data: bytes, *, depth: int = 0) -> bytes:
    """Read UCI outer zip (including a possible nested zip) without extracting paths."""
    if depth > 3:
        raise ValueError("Nested archive depth exceeded")
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        direct = [n for n in names if Path(n).name.lower() == "smsspamcollection"]
        if direct:
            return archive.read(direct[0])
        nested = [n for n in names if n.lower().endswith(".zip")]
        for name in nested:
            try:
                return extract_messages(archive.read(name), depth=depth + 1)
            except (KeyError, ValueError, zipfile.BadZipFile):
                continue
    raise ValueError("Cannot locate SMSSpamCollection inside archive")


def read_sms_source(raw: bytes) -> pd.DataFrame:
    payload = extract_messages(raw)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = payload.decode("latin-1")
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if "\t" not in line:
            raise ValueError(f"Row {line_no}: missing tab separator")
        label, content = line.split("\t", 1)
        label = label.strip().lower()
        if label not in LABELS:
            raise ValueError(f"Row {line_no}: unknown class {label!r}")
        if not content.strip():
            continue
        rows.append({"label": label, "text": content})
    if not rows:
        raise ValueError("Dataset is empty")
    return pd.DataFrame(rows)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_count = len(df)
    disagreements = df.groupby("text")["label"].nunique()
    n_conflicts = int((disagreements > 1).sum())
    if n_conflicts:
        raise ValueError(f"Found {n_conflicts} exact messages with conflicting spam/ham labels. Resolve before training.")
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    df["message_id"] = df["text"].map(lambda x: "sms_" + hashlib.sha256(x.encode("utf-8")).hexdigest()[:20])
    if df["message_id"].duplicated().any():
        raise ValueError("Hash collision in message_id")
    df = df[["message_id", "label", "text"]].reset_index(drop=True)
    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both spam and ham")
    summary = {
        "raw_message_count": original_count,
        "deduplicated_message_count": len(df),
        "exact_duplicate_rows_removed": original_count - len(df),
        "class_counts_after_deduplication": {k: int(v) for k, v in df["label"].value_counts().items()},
    }
    return df, summary


def stratified_splits(df: pd.DataFrame) -> pd.DataFrame:
    remaining, test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["label"])
    train, validation = train_test_split(remaining, test_size=0.25, random_state=SEED, stratify=remaining["label"])
    assigned = pd.concat([
        train.assign(split="train"),
        validation.assign(split="validation"),
        test.assign(split="test"),
    ], ignore_index=True)
    assert len(assigned) == len(df)
    assert assigned["message_id"].nunique() == len(df)
    assert set(assigned["split"]) == {"train", "validation", "test"}
    return assigned.sort_values(["split", "message_id"]).reset_index(drop=True)


def make_pipelines() -> dict[str, Pipeline]:
    return {
        "multinomial_naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", MultinomialNB(alpha=1.0)),
        ]),
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
    }


def score(y_true, y_pred, model_name: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["spam"], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=LABELS).ravel()
    return {
        "model": model_name,
        "split": "validation",
        "n_evaluated": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spam_precision": float(precision[0]),
        "spam_recall": float(recall[0]),
        "spam_f1": float(f1[0]),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "true_ham": int(tn),
        "ham_wrongly_flagged": int(fp),
        "spam_missed": int(fn),
        "spam_detected": int(tp),
    }


def run(raw: bytes, *, out_root: Path, source_label: str) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    processed = out_root / "data" / "processed"
    models_dir = out_root / "models"
    results_dir = out_root / "results"
    for folder in (processed, models_dir, results_dir):
        folder.mkdir(parents=True, exist_ok=True)

    df, data_summary = prepare(read_sms_source(raw))
    df = stratified_splits(df)
    df.to_csv(processed / "sms_splits.csv", index=False, encoding="utf-8")
    split_counts = pd.crosstab(df["split"], df["label"]).to_dict("index")
    provenance = {
        "dataset": "UCI SMS Spam Collection (Almeida & Hidalgo, 2011)",
        "official_page": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "license": "CC BY 4.0; attribution required",
        "source": source_label,
        "source_sha256": sha256_bytes(raw),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": SEED,
        "split_fraction": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "split_counts": split_counts,
        "data_summary": data_summary,
        "rule": "Drop exact-text duplicates BEFORE stratified splitting; conflicting labels cause an error.",
        "leakage_prevention": "Pipelines and TF-IDF are fit on train only. Validation for development. Test withheld.",
    }
    (processed / "dataset_manifest.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    all_metrics = []
    preds = []
    for name, model in make_pipelines().items():
        model.fit(train["text"], train["label"])
        joblib.dump(model, models_dir / f"baseline_{name}.joblib")
        predicted = model.predict(val["text"])
        probs = model.predict_proba(val["text"])
        spam_i = list(model.classes_).index("spam")
        frame = pd.DataFrame({
            "message_id": val["message_id"].to_numpy(),
            "split": "validation",
            "model": name,
            "true_label": val["label"].to_numpy(),
            "prediction": predicted,
            "probability_spam": probs[:, spam_i],
        })
        preds.append(frame)
        all_metrics.append(score(val["label"], predicted, name))
        cm = confusion_matrix(val["label"], predicted, labels=LABELS)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
        display.plot(cmap="Blues", values_format="d", colorbar=False)
        plt.title(f"Validation only: {name.replace('_', ' ')}")
        plt.tight_layout()
        plt.savefig(results_dir / f"confusion_matrix_{name}.png", dpi=150)
        plt.close()
    pd.concat(preds, ignore_index=True).to_csv(results_dir / "predictions_validation.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(results_dir / "metrics_validation.csv", index=False)
    (results_dir / "metrics_validation.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (results_dir / "RUN_STATUS.txt").write_text(
        "BASELINE_VALIDATION_COMPLETE\n"
        "This is not a final-test evaluation.\n"
        "No attack or defense implemented in this package.\n", encoding="utf-8"
    )
    print("\nBASELINE_VALIDATION_COMPLETE")
    print(pd.DataFrame(all_metrics).to_string(index=False))
    print(f"\nOutputs saved in: {out_root.resolve()}")
    return {"data": df, "metrics": all_metrics, "manifest": provenance}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, help="Path to UCI zip or SMSSpamCollection TSV, if already downloaded")
    ap.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--no-download", action="store_true", help="Fail if no local dataset; do not fetch via network")
    args = ap.parse_args(argv)
    raw_dir = args.out_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.source:
        raw_path = args.source
    elif (raw_dir / "sms_spam_collection.zip").exists():
        raw_path = raw_dir / "sms_spam_collection.zip"
    elif (raw_dir / "SMSSpamCollection").exists():
        raw_path = raw_dir / "SMSSpamCollection"
    else:
        if args.no_download:
            ap.error("Dataset not available locally. Download official zip to data/raw/sms_spam_collection.zip")
        raw_path = raw_dir / "sms_spam_collection.zip"
        print("Downloading dataset from official UCI URL...")
        try:
            request = urllib.request.Request(DATA_URL, headers={"User-Agent": "University-SMS-Spam-Research/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
            if not zipfile.is_zipfile(io.BytesIO(raw)):
                raise ValueError("Downloaded response was not a zip archive")
            raw_path.write_bytes(raw)
        except Exception as exc:
            ap.error(f"Download failed ({exc}). Download manually: {DATA_URL}")
    if not raw_path.exists():
        ap.error(f"No source file at {raw_path}")
    run(raw_path.read_bytes(), out_root=args.out_root, source_label=str(raw_path))


if __name__ == "__main__":
    main()