# SMS spam detection — baseline

This is the starting point for our SMS spam robustness project. I trained two classifiers on the UCI SMS Spam Collection and saved the models, the dataset split, and their validation results. The text-evasion and defense experiments have not been done yet.

**You do not need to retrain the models to start the next experiments.** The saved pipelines in `models/` are ready to use.

## What's in the repository

| File | Purpose |
| --- | --- |
| `data/processed/sms_splits.csv` | The message text, true label (`spam` or `ham`), `message_id`, and fixed `train` / `validation` / `test` assignment. |
| `data/processed/dataset_manifest.json` | Dataset source, cleaning counts, split counts and random seed. |
| `models/baseline_multinomial_naive_bayes.joblib` | Trained word-level TF-IDF + Naive Bayes pipeline. |
| `models/baseline_logistic_regression.joblib` | Trained word-level TF-IDF + Logistic Regression pipeline. |
| `results/predictions_validation.csv` | Predictions and spam probabilities for each validation message, for both models. |
| `results/metrics_validation.csv` | Summary scores from the baseline validation run. |
| `run_baseline.py` | Rebuilds the dataset split, trains both models, and exports results and charts. You normally don't need to run it. |
| `verify_outputs.py` | Checks whether the saved data, models, predictions and metrics match, without retraining. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Keeps local environments, raw downloads and temporary files out of Git. |

## Results so far

The downloaded UCI dataset had 5,574 rows. After removing 403 exact duplicate messages, 5,171 unique messages remained. The fixed split (seed 42) contains 3,102 training, 1,034 validation, and 1,035 final-test messages.

These results come from **validation only**, before any attacks or defenses:

| Model | Accuracy | Spam recall | Spam F1 | False positives (ham) |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes | 95.16% | 61.54% | 76.19% | 0 |
| Logistic Regression | 96.71% | 74.62% | 85.09% | 1 |

Of the 130 spam messages in validation, Naive Bayes detected 80 and Logistic Regression detected 97. Accuracy is high partly because there are many more legitimate messages than spam, so spam recall is useful to look at separately.

## Getting started

Download or clone the repository, open a terminal in the project folder, and use Python 3.13:

```bash
python -m pip install -r requirements.txt
python verify_outputs.py
```

The verification checks the files already included in the repository. It does **not** train the models again or score the final test set.

To load one of the saved models, you can use:

```python
import joblib
import pandas as pd

data = pd.read_csv("data/processed/sms_splits.csv")
model = joblib.load("models/baseline_logistic_regression.joblib")
validation_spam = data[(data["split"] == "validation") & (data["label"] == "spam")]
print(model.predict(validation_spam["text"].head(5)))
```

Only load the `.joblib` files from this trusted repository, as the joblib format should not be opened from unknown sources.

## Next steps: attacks and defense

**Text-evasion tests:** use `validation` messages to select spam examples, make controlled text changes, and compare the same saved model's predictions before and after the edits. Keep the original `message_id` for each changed message so results can be matched.

**Defense training:** use **only** `train` messages and modified copies of training spam to retrain a separate model. Compare the original and defended models on the same validation messages and fixed edited versions. Also check false alarms on ordinary (`ham`) messages.

Please keep the `test` partition untouched until the group has finalized the attack rules and defense setup. Don't overwrite the baseline model files or validation results when adding new experiments; save new outputs separately.

## Optional: reproduce the baseline and view charts

If you want to recreate the baseline from the original dataset, run:

```bash
python run_baseline.py
```

The script downloads the public UCI dataset if it's not already in `data/raw/`, then rebuilds the saved models and outputs. It also creates two confusion-matrix images under `results/`:

- `confusion_matrix_multinomial_naive_bayes.png`
- `confusion_matrix_logistic_regression.png`

Those PNGs are **generated locally**; they are not required for the attack and defense work and aren't committed here. Running the script regenerates the baseline artifacts, so keep a copy of any separate experimental changes.

## Dataset

UCI SMS Spam Collection — Almeida & Hidalgo (2011). [Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) · [DOI](https://doi.org/10.24432/C5CC84). The dataset is licensed under **CC BY 4.0** and is included with attribution. It contains **SMS messages**, not emails.
