# SMS spam detection — baseline

This repository contains the baseline for our SMS spam robustness project. The two classifiers are already trained, and their predictions on the validation set are included so the attack and defense experiments can use the same starting point.

## Files

| File | What it's for |
| --- | --- |
| `data/processed/sms_splits.csv` | The SMS texts, labels (`spam` or `ham`), message IDs, and fixed train/validation/test assignments. |
| `data/processed/dataset_manifest.json` | Dataset source, class counts, split details, and random seed. |
| `models/baseline_multinomial_naive_bayes.joblib` | Trained word-level TF-IDF + Multinomial Naive Bayes pipeline. |
| `models/baseline_logistic_regression.joblib` | Trained word-level TF-IDF + Logistic Regression pipeline. |
| `results/predictions_validation.csv` | Both models' predictions and spam probabilities on validation messages. |
| `results/metrics_validation.csv` | Baseline validation scores for the two models. |
| `run_baseline.py` | Script to reproduce the baseline from the original UCI data. |
| `verify_outputs.py` | Checks the saved outputs against the models and the dataset split. |
| `requirements.txt` | Python packages used for this run. |

## Results so far

The original dataset was cleaned to **5,171 unique messages**. The split is **60% training / 20% validation / 20% test** (seed 42). Results below are from the **1,034 validation messages**, not from the final test set.

| Model | Accuracy | Spam recall | Spam F1 | False positives (ham) |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 95.16% | 61.54% | 76.19% | 0 |
| Logistic Regression | 96.71% | 74.62% | 85.09% | 1 |

Logistic Regression caught more spam in this run (97 messages versus 80). These are baseline results only; we haven't tested text-evasion attacks or a defense yet.

## Using the files

Use **Python 3.13** and install the packages with:

```bash
python -m pip install -r requirements.txt
```

The trained pipelines can be loaded with `joblib.load`, but only load model files from this trusted project repository. You can check the handoff without retraining:

```bash
python verify_outputs.py
```

To reproduce everything from the original data:

```bash
python run_baseline.py
```

That script downloads the public UCI corpus if it isn't already in `data/raw/`. It also regenerates the saved models and validation results.

**For the next experiments:** use the `validation` messages to develop and compare attacks and defenses. Any modified copies of training messages used for defense training must stay in `train`. Keep the `test` partition untouched until we've agreed on the final settings; its scores have not been calculated here. Refer to messages by `message_id` so results can be compared across experiments.

## Dataset

UCI SMS Spam Collection — Almeida & Hidalgo (2011). [Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), [DOI](https://doi.org/10.24432/C5CC84). Licensed under **CC BY 4.0**; the data is included here with attribution. This is an **SMS** dataset, not an email dataset.