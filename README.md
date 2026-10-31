# SMS spam detection — baseline and validation attack tests

This repository contains the prepared SMS dataset, two trained spam classifiers, and their validation results for the SMS spam detection project. It also includes preliminary tests of two text-evasion methods. The saved models are ready to use; retraining the baseline is not necessary to work with them.

The attack results are **preliminary**. Semantic review, defense training and final-test evaluation are not yet complete.

## Files in the repository

### Data and baseline

| File | Contents |
| --- | --- |
| `data/processed/sms_splits.csv` | Cleaned messages, labels, stable message IDs, and the fixed `train`, `validation`, and `test` assignments. |
| `data/processed/dataset_manifest.json` | Dataset source, cleaning and split counts, and seed 42. |
| `models/baseline_multinomial_naive_bayes.joblib` | Saved TF-IDF + Naive Bayes classifier. |
| `models/baseline_logistic_regression.joblib` | Saved TF-IDF + Logistic Regression classifier. |
| `results/predictions_validation.csv` | Original predictions and spam probabilities for both models on validation messages. |
| `results/metrics_validation.csv` | Baseline validation scores. |
| `run_baseline.py` | Recreates the baseline from scratch: cleaning, splitting, training, metrics and charts. The trained models are already saved in `models/`, so this script is not needed to use them. |
| `verify_outputs.py` | Checks that the saved baseline data, models, predictions and metrics agree, without retraining. |
| `requirements.txt` | Python dependencies and versions. |
| `.gitignore` | Lists local files and folders that Git should ignore. |

### Text-evasion experiments

The **reviewed preliminary run** uses `attacks.py` and the files in `results/attacks/`. It changes spam messages in the **validation** split only. The two fixed methods are character substitution (for example, `free` → `fr33`) and punctuation insertion (`free` → `f.r.e.e`).

| File | Contents |
| --- | --- |
| `attacks.py` | Runs both text modifications against the two saved baseline models and generates the validation results. No retraining is performed. |
| `results/attacks/attack_variants_validation.csv` | Original and modified messages, source IDs, attack type and change status. |
| `results/attacks/attack_predictions_validation.csv` | Predictions and spam probabilities before and after modification. |
| `results/attacks/attack_metrics_validation.csv` | Counts, coverage, detection rates and conditional evasion rates. |
| `results/attacks/attack_config.json` | Keywords, substitution rules and run details. |
| `results/attacks/attack_review_samples.csv` | 20 examples prepared for a manual check of whether the spam meaning is preserved; the review column is still blank. |

**Separate earlier implementation:** `run_attacks.py`, `results/attack_variants_validation.csv`, and `results/attack_settings.json` belong to a different version of the text-evasion experiment, with a shorter keyword list and a different character map. They are not the source of the results below. Please **do not mix** the two versions' generated files or metrics; one ruleset needs to be agreed upon for the final comparison.

## Results available so far

The UCI dataset started with 5,574 rows. Removing 403 exact duplicates left **5,171 unique SMS messages**: 3,102 training, 1,034 validation and 1,035 test.

Baseline performance on **all 1,034 validation messages**, before modification:

| Model | Accuracy | Spam recall | Spam F1 | False positives |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 95.16% | 61.54% | 76.19% | 0 |
| Logistic Regression | 96.71% | 74.62% | 85.09% | 1 |

Spam recall is the share of actual spam messages detected. Out of 130 validation spam messages, Naive Bayes detected 80 and Logistic Regression detected 97.

In the **reviewed preliminary attack run**, each attack changed **113 of the 130** validation spam messages. Results below are restricted to those **113 changed messages**, not the whole validation set:

| Model | Detected before change | Detected after change | Newly missed |
| --- | ---: | ---: | ---: |
| Naive Bayes | 78 / 113 | 69 / 113 | 9 |
| Logistic Regression | 94 / 113 | 76 / 113 | 18 |

Both attack methods produced the same predictions and probabilities in this run, despite changing the text differently. These results describe the observed model behavior; they do not establish that all modified messages are valid attacks. The manual sample review is still open, especially for edits affecting words such as `STOP` or `reply`.

## Opening the saved project on Windows

Download the repository using **Code → Download ZIP**, extract it, and open the extracted project folder (the one containing `README.md` and `requirements.txt`). In Windows File Explorer, type `cmd` in the address bar and press Enter.

With Python 3.13 installed:

```cmd
py -3.13 -m pip install -r requirements.txt
py -3.13 verify_outputs.py
```

These commands install the required packages and check the existing baseline files. They do not retrain models or run a final-test evaluation.

## Current handoff status

**Available now:** fixed data splits, both baseline models, baseline validation predictions and metrics, two preliminary text-evasion implementations, and results from the reviewed validation attack run.

**Not completed yet:** manual acceptance review of attack examples, agreement on a single attack configuration, augmented-training defense models, defended-model comparisons, and frozen final-test evaluation. The `test` split has not been scored and remains reserved for the final evaluation. New results can be saved separately so the baseline files remain available for comparison.

## Dataset source

UCI SMS Spam Collection — Almeida & Hidalgo (2011). [Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) · [DOI](https://doi.org/10.24432/C5CC84). Licensed under **CC BY 4.0**. This dataset contains SMS messages, not emails.
