#!/usr/bin/env python3
"""Two fixed text-evasion experiments for the SMS spam baseline.

Only validation spam is modified; training texts are used solely to confirm
that the predefined trigger words occur in training spam. No retraining or
final-test evaluation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import joblib
import pandas as pd

ATTACKS = ("character_substitution", "punctuation_insertion")
MODEL_NAMES = ("multinomial_naive_bayes", "logistic_regression")
SEED = 42
# Fixed, interpretable keyword candidates; only those found in TRAIN spam
# are enabled. No tokens are selected by looking at validation or test labels.
CANDIDATE_TRIGGERS = (
    "free", "win", "prize", "claim", "cash", "urgent", "call", "reply",
    "won", "award", "selected", "guaranteed", "mobile", "collect",
    "voucher", "offer", "awarded", "winner", "bonus", "gift", "stop",
    "text", "contact",
)
CHAR_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def active_triggers(training_spam: pd.Series) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for message in training_spam:
        counts.update(re.findall(r"\b[a-z]+\b", message.lower()))
    candidates = [token for token in CANDIDATE_TRIGGERS if counts[token] > 0]
    # Guarantee that any selected trigger can actually be altered by substitution.
    return tuple(token for token in candidates if any(char in CHAR_MAP for char in token))


def build_pattern(triggers: tuple[str, ...]) -> re.Pattern:
    if not triggers:
        raise ValueError("No trigger words found in training spam. Cannot run attacks.")
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in triggers) + r")\b", re.IGNORECASE)


def transform(message: str, attack: str, pattern: re.Pattern) -> tuple[str, bool, str]:
    if attack not in ATTACKS:
        raise ValueError(f"Unsupported attack: {attack}")
    match = pattern.search(message)
    if match is None:
        return message, False, ""
    original = match.group(0)
    if attack == "character_substitution":
        replacement = "".join(CHAR_MAP.get(ch.lower(), ch) for ch in original)
    else:
        replacement = ".".join(original)
    if replacement == original:
        return message, False, original.lower()
    modified = message[:match.start()] + replacement + message[match.end():]
    assert modified != message
    assert modified.replace(replacement, original, 1) == message
    return modified, True, original.lower()


def unit_checks() -> None:
    p = build_pattern(("free", "prize", "win"))
    assert transform("Claim a FREE prize now", "character_substitution", p) == (
        "Claim a FR33 prize now", True, "free"
    )
    assert transform("Claim a FREE prize now", "punctuation_insertion", p) == (
        "Claim a F.R.E.E prize now", True, "free"
    )
    assert transform("This is carefree", "character_substitution", p) == (
        "This is carefree", False, ""
    )  # whole-word only
    assert transform("Hi, see you soon", "punctuation_insertion", p) == (
        "Hi, see you soon", False, ""
    )
    assert transform("win win", "character_substitution", p) == (
        "w1n win", True, "win"
    )  # first matching keyword only
    print("PASS: 5 transformation self-checks")


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    return path


def prepare_variants(splits: pd.DataFrame, triggers: tuple[str, ...]) -> pd.DataFrame:
    pattern = build_pattern(triggers)
    spam = splits[(splits["split"] == "validation") & (splits["label"] == "spam")].copy()
    spam = spam.sort_values("message_id")
    if len(spam) == 0:
        raise ValueError("Validation spam is empty")
    output = []
    for attack in ATTACKS:
        for item in spam.itertuples(index=False):
            modified, changed, trigger = transform(item.text, attack, pattern)
            output.append({
                "source_id": item.message_id,
                "split": "validation", "label": "spam",
                "attack_type": attack,
                "trigger": trigger,
                "changed": changed,
                "original_text": item.text,
                "modified_text": modified,
            })
    variants = pd.DataFrame(output)
    assert len(variants) == len(spam) * len(ATTACKS)
    assert variants.groupby("attack_type").size().eq(len(spam)).all()
    assert not variants[variants["changed"]]["original_text"].equals(variants[variants["changed"]]["modified_text"])
    assert (variants.loc[~variants.changed, "original_text"].to_numpy() == variants.loc[~variants.changed, "modified_text"].to_numpy()).all()
    assert not variants.duplicated(subset=["source_id", "attack_type"]).any()
    return variants


def load_models(root: Path):
    loaded = {}
    hashes = {}
    for name in MODEL_NAMES:
        path = safe_path(root, f"models/baseline_{name}.joblib")
        model = joblib.load(path)  # trusted project-created model files only
        if set(model.classes_) != {"ham", "spam"}:
            raise ValueError(f"Unexpected model classes for {name}: {model.classes_}")
        loaded[name] = model
        hashes[name] = digest(path)
    return loaded, hashes


def evaluate(variants: pd.DataFrame, models: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    metrics = []
    for model_name, model in models.items():
        spam_idx = list(model.classes_).index("spam")
        for attack in ATTACKS:
            sub = variants[variants["attack_type"] == attack].reset_index(drop=True)
            before = model.predict(sub["original_text"])
            after = model.predict(sub["modified_text"])
            before_proba = model.predict_proba(sub["original_text"])[:, spam_idx]
            after_proba = model.predict_proba(sub["modified_text"])[:, spam_idx]
            rows = sub[["source_id", "attack_type", "changed", "trigger"]].copy()
            rows.insert(1, "model", model_name)
            rows["original_prediction"] = before
            rows["modified_prediction"] = after
            rows["original_spam_probability"] = before_proba
            rows["modified_spam_probability"] = after_proba
            predictions.append(rows)

            changed = sub["changed"].to_numpy(dtype=bool)
            considered = len(sub)
            changed_n = int(changed.sum())
            clean_detected = int(((before == "spam") & changed).sum())
            attacked_detected = int(((after == "spam") & changed).sum())
            evaded = int(((before == "spam") & (after == "ham") & changed).sum())
            gained = int(((before == "ham") & (after == "spam") & changed).sum())
            metrics.append({
                "model": model_name,
                "attack_type": attack,
                "split": "validation",
                "spam_considered": considered,
                "messages_changed": changed_n,
                "messages_unchanged": considered - changed_n,
                "attack_coverage": changed_n / considered,
                "baseline_detected_on_changed": clean_detected,
                "attacked_detected_on_changed": attacked_detected,
                "baseline_recall_on_changed": clean_detected / changed_n if changed_n else None,
                "robust_recall_on_changed": attacked_detected / changed_n if changed_n else None,
                "newly_evaded": evaded,
                "newly_detected": gained,
                "conditional_evasion_rate": evaded / clean_detected if clean_detected else None,
            })
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def chart_metrics(metrics: pd.DataFrame, output: Path) -> None:
    # Local figure only; intentionally no plots of held-out test data.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [
        f"{'NB' if r.model == 'multinomial_naive_bayes' else 'LR'} / "
        f"{'chars' if r.attack_type == 'character_substitution' else 'dots'}"
        for r in metrics.itertuples(index=False)
    ]
    import numpy as np
    x = np.arange(len(labels))
    width = .38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    before = metrics["baseline_recall_on_changed"].fillna(0) * 100
    after = metrics["robust_recall_on_changed"].fillna(0) * 100
    ax.bar(x - width / 2, before, width, label="Before text changes")
    ax.bar(x + width / 2, after, width, label="After text changes")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Spam detected on changed messages (%)")
    ax.set_title("Validation: same changed-message pool before vs. after")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def execute(project: Path, output: Path) -> None:
    data_file = safe_path(project, "data/processed/sms_splits.csv")
    splits = pd.read_csv(data_file, keep_default_na=False)
    if not {"message_id", "label", "text", "split"}.issubset(splits.columns):
        raise ValueError("sms_splits.csv is missing required columns")
    if splits["message_id"].duplicated().any():
        raise ValueError("Repeated message IDs in dataset")
    if set(splits["split"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected data partitions")
    train_spam = splits.loc[(splits.split == "train") & (splits.label == "spam"), "text"]
    triggers = active_triggers(train_spam)
    variants = prepare_variants(splits, triggers)
    models, model_digests = load_models(project)
    predictions, metrics = evaluate(variants, models)
    # Cross-check all untouched predictions against the baseline CSV from GitHub.
    saved_predictions = safe_path(project, "results/predictions_validation.csv")
    recorded = pd.read_csv(saved_predictions, keep_default_na=False)
    for model_name in MODEL_NAMES:
        expected = recorded.loc[
            (recorded["model"] == model_name) &
            (recorded["true_label"] == "spam"),
            ["message_id", "prediction"],
        ].set_index("message_id")
        actual = predictions.loc[
            (predictions["model"] == model_name) &
            (predictions["attack_type"] == ATTACKS[0]),
            ["source_id", "original_prediction"],
        ].set_index("source_id")
        if (not expected.index.is_unique or
                set(expected.index) != set(actual.index) or
                not (expected.loc[actual.index, "prediction"].to_numpy() ==
                     actual["original_prediction"].to_numpy()).all()):
            raise ValueError(f"Original predictions do not match saved baseline for {model_name}")
    output.mkdir(parents=True, exist_ok=True)
    variants.to_csv(output / "attack_variants_validation.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "attack_predictions_validation.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "attack_metrics_validation.csv", index=False, encoding="utf-8-sig")
    # Predetermined first ten changed examples per attack (ordered by stable ID).
    review = (variants.loc[variants.changed]
              .groupby("attack_type", sort=True, group_keys=False)
              .head(10)[["source_id", "attack_type", "trigger", "original_text", "modified_text"]]
              .copy())
    review["preserves_spam_meaning_review"] = ""  # to be filled by a human (yes/no)
    review.to_csv(output / "attack_review_samples.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "experiment": "two fixed text-evasion rules on validation spam (no defense training)",
        "random_seed_used_for_existing_splits": SEED,
        "input_split_sha256": digest(data_file),
        "baseline_predictions_sha256": digest(saved_predictions),
        "baseline_model_sha256": model_digests,
        "models": list(MODEL_NAMES),
        "attack_types": list(ATTACKS),
        "candidate_triggers_predeclared": list(CANDIDATE_TRIGGERS),
        "enabled_triggers_confirmed_in_training_spam": list(triggers),
        "trigger_matching": "First whole word found left to right, case insensitive; preserves all other message text",
        "substitution_map": CHAR_MAP,
        "punctuation_rule": "Insert a dot between each character of the first matching trigger word",
        "modified_message_semantics": "Rule-based changes only; manual review of fixed sample still required",
        "test_partition_used": False,
        "data_partitions_for_model_prediction": ["validation"],
        "notes": "Robust recall is computed on CHANGED validation spam only; cannot measure ham false positives from spam-only attack pool.",
    }
    (output / "attack_config.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    chart_metrics(metrics, output / "attack_recall_comparison.png")
    msg = [
        "SMS SPAM — VALIDATION TEXT-EVASION EXPERIMENTS",
        "Only validation spam was modified. No training, defense, or final-test scoring performed.",
        f"Triggers confirmed in training spam: {', '.join(triggers)}",
        "",
    ]
    for r in metrics.itertuples(index=False):
        baseline = "N/A" if pd.isna(r.baseline_recall_on_changed) else f"{r.baseline_recall_on_changed:.1%}"
        robust = "N/A" if pd.isna(r.robust_recall_on_changed) else f"{r.robust_recall_on_changed:.1%}"
        evasion = "N/A" if pd.isna(r.conditional_evasion_rate) else f"{r.conditional_evasion_rate:.1%}"
        msg.extend([
            f"{r.model} / {r.attack_type}",
            f"  changed: {r.messages_changed} / {r.spam_considered} ({r.attack_coverage:.1%})",
            f"  detection on changed pool: before={baseline}, after={robust}",
            f"  newly evaded={r.newly_evaded}; conditional evasion={evasion}",
            "",
        ])
    msg.append("Review attack_review_samples.csv manually before interpreting these as accepted meaningful attacks.")
    msg.append("Outputs: results/attacks/ (baseline models and CSVs unchanged).")
    text = "\n".join(msg) + "\n"
    (output / "ATTACK_SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    unit_checks()
    if args.self_test:
        return
    project = args.project.resolve()
    out = args.out.resolve() if args.out else (project / "results" / "attacks")
    execute(project, out)


if __name__ == "__main__":
    main()