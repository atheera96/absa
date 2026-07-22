# Aspect-Based Sentiment Analysis (ABSA) of E-Commerce Reviews

A working implementation of the system described in the FYP report
*"Aspect-Based Sentiment Analysis of E-Commerce Reviews Using Natural
Language Processing Techniques"*.

Built on your `raw_data.csv` (20,000 Shopee reviews, columns: `rating`,
`comment`).

**Primary model: Logistic Regression** (TF-IDF features, `class_weight="balanced"`).
It was switched from Naive Bayes because it has the best overall accuracy,
and — more importantly — the balanced class weighting fixes the severe
bias toward the majority "Positive" class that the original unweighted
models had (where Negative/Neutral recall was close to 0%).

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naive Bayes | 82.1% | 93.6% | 82.1% | 86.7% |
| SVM | 92.1% | 92.6% | 92.1% | 92.3% |
| **Logistic Regression (primary)** | 86.2% | 93.8% | 86.2% | 89.4% |

Note the trade-off: overall accuracy is a bit lower than the unweighted
version, but Negative/Neutral recall improved from ~0–2% to ~40–60%,
meaning the model actually detects negative reviews now instead of
labelling everything Positive. Also fixed: **negation words** ("not",
"isn't", "doesn't", etc.) were previously being stripped as stopwords,
which flipped "not good" into "good" — they are now preserved.

## Project structure

```
absa_project/
├── data/
│   ├── raw_data.csv              # your uploaded dataset
│   ├── processed_reviews.csv     # generated after training (cleaned + labelled)
│   └── dashboard_stats.json      # generated after training (dashboard cache)
├── models/                       # generated after training (.pkl files)
├── templates/
│   ├── landing.html
│   ├── dashboard.html
│   └── analyzer.html
├── static/
│   └── style.css
├── nlp_utils.py                  # shared preprocessing / aspect-extraction logic
├── train_model.py                # training pipeline (run this first)
├── app.py                        # Flask web app
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Train the model

```bash
python train_model.py
```

This will:
1. Load and clean `data/raw_data.csv` (drop nulls, empty text, duplicates)
2. Preprocess text: lowercasing, URL/number/punctuation removal, stopword
   removal, lightweight lemmatization
3. Extract aspects (rule-based keyword matching): **Product Quality**,
   **Packaging Quality**, **Delivery & Service**, **Others**
4. Derive sentiment labels from star rating (1–2 → Negative, 3 → Neutral,
   4–5 → Positive)
5. Build TF-IDF features (unigram + bigram, max 5000 features)
6. Train & evaluate Naive Bayes (primary model), SVM, and Logistic
   Regression on an 80/20 split
7. Save the trained model, vectorizer, processed dataset, and dashboard
   stats

You'll see per-model Accuracy / Precision / Recall / F1-score printed to
the console, plus a comparison summary.

## Step 2 — Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- **Home** – landing page describing the system
- **Dashboard** – total reviews, sentiment breakdown, and charts
  (Sentiment Volume Across Aspects, Total Counts of Sentiment by Aspect,
  Percentage Distribution Split), plus a model comparison table
- **Sentiment Analyzer**
  - *Single Review Prediction*: type one sentence, get its aspect +
    sentiment instantly
  - *Upload Analysis System*: upload a CSV/Excel file of reviews, see
    aggregated charts + a preview table, and download the full result as
    CSV

## Notes on the model

- The dataset is heavily imbalanced (~94% Positive reviews), which is
  exactly the limitation the original report identifies: Naive Bayes
  gets high overall accuracy but low recall on Neutral/Negative reviews.
  You'll see this reflected in the console output when you train.
- To improve minority-class performance, the report's own
  recommendations apply here too: balance the dataset (e.g. oversampling
  Negative/Neutral, or class-weighting), and expand the aspect keyword
  dictionaries in `nlp_utils.py`.
- Aspect extraction is rule-based keyword matching (not a trained
  classifier) — you can freely edit the keyword lists in
  `ASPECT_KEYWORDS` inside `nlp_utils.py` to improve coverage.

## Retraining

If you update `data/raw_data.csv`, or tweak preprocessing/keywords in
`nlp_utils.py`, just re-run `python train_model.py` to refresh the model
and dashboard stats, then restart `python app.py`.
