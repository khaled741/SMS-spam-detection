#!/usr/bin/env python3
"""Run two fixed, offline text-evasion tests on validation spam only.

The trigger list was selected from frequent words in TRAIN spam only. This
script never trains models, changes saved baselines, or evaluates the test set.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd

# Selected by inspecting word frequencies in the training-spam partition only.
# Fixed for the reported validation experiments; do not tune using test data.
TRIGGERS = ("free", "win", "prize", "claim", "cash", "urgent", "call", "reply", "mobile", "won")
LETTER_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
ATTACKS = ("character_substitution", "punctuation_insertion")
MODELS = ("multinomial_naive_bayes", "logistic_regression")
WORD_PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, TRIGGERS)) + r")\b", re.IGNORECASE)


def transform(text: str, attack: str) -> tuple[str, str, bool]:
    """Change at most the first whole-word trigger; leave all other text untouched."""
    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack: {attack}")
    match = WORD_PATTERN.search(text)
    if match is None:
        return text, "", False
    word = match.group(0)
    if attack == "character_substitution":
        new_word = "".join(LETTER_MAP.get(ch.lower(), ch) for ch in word)
    else:
        new_word = ".".join(word)
    if new_word == word:
        return text, word.lower(), False
    changed_text = text[:match.start()] + new_word + text[match.end():]
    return changed_text, word.lower(), True


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = pd.read_csv(root / "data/processed/sms_splits.csv", keep_default_na=False)
    expected = {"message_id", "label", "text", "split"}
    if not expected.issubset(splits.columns):
        raise ValueError(f"Missing columns: {expected - set(splits.columns)}")
    if splits.message_id.duplicated().any() or splits.text.duplicated().any():
        raise ValueError("Duplicate message ID or text; use the existing cleaned split file")
    train_spam = splits.loc[(splits.split == "train") & (splits.label == "spam")]
    val_spam = splits.loc[(splits.split == "validation") & (splits.label == "spam")].sort_values("message_id")
    if train_spam.empty or val_spam.empty:
        raise ValueError("Training or validation spam is empty")

    # Check that the fixed triggers are actually present in TRAIN spam messages.
    trigger_counts = {
        word: int(train_spam.text.str.contains(r"\b" + re.escape(word) + r"\b", case=False, regex=True).sum())
        for word in TRIGGERS
    }
    if any(count == 0 for count in trigger_counts.values()):
        raise ValueError("Trigger absent from training spam; review the shared dataset")

    examples = []
    for attack in ATTACKS:
        for row in val_spam.itertuples(index=False):
            modified, trigger, changed = transform(row.text, attack)
            examples.append({
                "message_id": row.message_id,
                "split": "validation",
                "label": row.label,
                "attack": attack,
                "trigger": trigger,
                "original_text": row.text,
                "modified_text": modified,
                "changed": changed,
            })
    variants = pd.DataFrame(examples)
    metrics = []
    all_predictions = []
    for model_name in MODELS:
        model = joblib.load(root / "models" / f"baseline_{model_name}.joblib")
        original_predictions = model.predict(val_spam.text)
        original_by_id = dict(zip(val_spam.message_id, original_predictions))
        for attack in ATTACKS:
            group = variants.loc[variants.attack == attack].copy()
            modified_predictions = model.predict(group.modified_text)
            probabilities = model.predict_proba(group.modified_text)
            spam_col = list(model.classes_).index("spam")
            group_predictions = pd.DataFrame({
                "message_id": group.message_id.to_numpy(),
                "split": "validation",
                "label": group.label.to_numpy(),
                "model": model_name,
                "attack": attack,
                "trigger": group.trigger.to_numpy(),
                "changed": group.changed.to_numpy(),
                "original_prediction": group.message_id.map(original_by_id).to_numpy(),
                "modified_prediction": modified_predictions,
                "modified_probability_spam": probabilities[:, spam_col],
            })
            all_predictions.append(group_predictions)
            changed_rows = group_predictions.loc[group_predictions.changed]
            considered = len(group_predictions)
            changed = len(changed_rows)
            clean_detected = int((changed_rows.original_prediction == "spam").sum())
            robust_detected = int((changed_rows.modified_prediction == "spam").sum())
            evaded = int(((changed_rows.original_prediction == "spam") &
                          (changed_rows.modified_prediction == "ham")).sum())
            metrics.append({
                "model": model_name,
                "attack": attack,
                "split": "validation",
                "spam_considered": considered,
                "messages_changed": changed,
                "structurally_valid_changed": changed,
                "attack_coverage": changed / considered if considered else None,
                "original_spam_detected_on_changed": clean_detected,
                "modified_spam_detected": robust_detected,
                "robust_recall_on_changed": robust_detected / changed if changed else None,
                "baseline_detected_now_missed": evaded,
                "conditional_evasion_rate": evaded / clean_detected if clean_detected else None,
            })
    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    variants.to_csv(out / "attack_variants_validation.csv", index=False, encoding="utf-8")
    predictions_df.to_csv(out / "attack_predictions_validation.csv", index=False, encoding="utf-8")
    metrics_df.to_csv(out / "attack_metrics_validation.csv", index=False, encoding="utf-8")
    (out / "attack_settings.json").write_text(json.dumps({
        "dataset_partition_used_for_variants": "validation spam only",
        "trigger_selection_partition": "train spam only",
        "fixed_triggers_in_order": list(TRIGGERS),
        "training_spam_message_counts_with_each_trigger": trigger_counts,
        "character_substitution_map": LETTER_MAP,
        "word_matching": "case-insensitive whole words, first trigger occurrence in message",
        "punctuation_rule": "insert '.' between adjacent characters in first trigger",
        "unchanged_messages": "retained and counted as not changed; excluded from robust recall",
        "structurally_valid_changed": "automatic mechanical rule check only, not a human semantic review",
        "test_evaluation": "not performed",
    }, indent=2), encoding="utf-8")
    print("Validation spam examined:", len(val_spam))
    print(metrics_df.to_string(index=False))
    print("Saved: results/attack_variants_validation.csv, results/attack_predictions_validation.csv, results/attack_metrics_validation.csv, results/attack_settings.json")
    return variants, metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()