# SMS spam detection — baseline

This repository contains the baseline for the SMS spam detection project: the prepared SMS data, two trained classifiers, and their validation results. Both models are saved in `models/`. An initial set of text-evasion experiments is also available below; defense and final-test results have not been added yet.

## Files

| File | Description |
| --- | --- |
| `data/processed/sms_splits.csv` | The SMS texts, true labels (`spam` / `ham`), message IDs, and fixed `train`, `validation`, and `test` assignments. |
| `data/processed/dataset_manifest.json` | Dataset source, cleaning summary, split counts, and random seed. |
| `models/baseline_multinomial_naive_bayes.joblib` | Trained TF-IDF + Naive Bayes model. |
| `models/baseline_logistic_regression.joblib` | Trained TF-IDF + Logistic Regression model. |
| `results/predictions_validation.csv` | Predictions and spam probabilities for each validation message from both models. |
| `results/metrics_validation.csv` | Summary of the baseline validation results. |
| `run_baseline.py` | Recreates the baseline from scratch, including data preparation, model training, results, and charts. There is no need to run it for the next experiments, as both trained models are already available in `models/`. |
| `verify_outputs.py` | Checks that the saved data, models, predictions, and metrics are consistent, without retraining. |
| `requirements.txt` | Python packages and versions used for the baseline. |
| `.gitignore` | Lists local files that Git should ignore, such as temporary files and Python environments. |

## Baseline results

The UCI SMS Spam Collection contains 5,574 rows. After removing 403 exact duplicates, 5,171 unique messages remained. They were divided into 3,102 training, 1,034 validation, and 1,035 test messages, with a fixed random seed of 42.

The figures below are from the **validation set only**, before any attacks or defenses:

| Model | Accuracy | Spam recall | Spam F1 | False positives |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 95.16% | 61.54% | 76.19% | 0 |
| Logistic Regression | 96.71% | 74.62% | 85.09% | 1 |

**Spam recall** is the share of actual spam messages detected by the model. Out of 130 spam messages in validation, Naive Bayes detected 80 and Logistic Regression detected 97. Accuracy measures correct predictions across both spam and legitimate messages.

## Opening and checking the project on Windows

Download the repository using **Code → Download ZIP** on GitHub, then extract it anywhere on your computer. Open the extracted folder containing `README.md` and `requirements.txt`. In Windows File Explorer, type `cmd` in the address bar and press Enter to open Command Prompt in that folder.

With **Python 3.13** installed, the following commands install the required packages and check the saved files:

```cmd
py -3.13 -m pip install -r requirements.txt
py -3.13 verify_outputs.py
```

The two trained models are already included under `models/`. The check does not retrain them or evaluate the final test set.

## Text-evasion experiments (validation only)

The `attacks.py` script applies two fixed changes to validation spam: character substitution (for example, `free` → `fr33`) and punctuation insertion (`free` → `f.r.e.e`). It uses the already trained models and does not retrain them.

Files under `results/attacks/`:

| File | Content |
| --- | --- |
| `attack_variants_validation.csv` | Original and modified messages, their source IDs, attack names, and whether a change was made. |
| `attack_predictions_validation.csv` | Model predictions and spam probabilities before and after each change. |
| `attack_metrics_validation.csv` | Detection rates and counts on the changed-message pool. |
| `attack_config.json` | The fixed attack rules, keywords, and details of the run. |
| `attack_review_samples.csv` | A small set of messages for manual review; the review column has not been filled in yet. |

Each attack changed **113 of 130** validation spam messages. On those 113 messages, Naive Bayes detected **78 before** and **69 after** the changes; Logistic Regression detected **94 before** and **76 after**. Both attack types produced the same predictions and probabilities with the current word-level TF-IDF setup. This is an observed result, not two independently different improvements.

These are **preliminary results**. The modified-message sample still needs to be checked for meaning, especially words used in opt-out instructions such as `STOP`. No final test data or defended models were used.

## For the next experiments

The saved models, fixed data splits, and validation predictions provide a common starting point for the text-evasion and defense work. The `train` split is available for defense training, while the `validation` split can be used for comparing experimental results. The `test` split has not been evaluated and is reserved for the final evaluation.

New experimental outputs can be saved separately, keeping the original baseline models and results available for comparison.

## Data source

UCI SMS Spam Collection — Almeida & Hidalgo (2011). [Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) · [DOI](https://doi.org/10.24432/C5CC84). Dataset license: **CC BY 4.0**. The data contains SMS messages, not emails.
