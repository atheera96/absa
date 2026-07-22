
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from nlp_utils import (
    LABEL_NAMES,
    clean_for_aspect,
    extract_aspect,
    preprocess,
    rating_to_label,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed_reviews.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATS_PATH = os.path.join(BASE_DIR, "data", "dashboard_stats.json")

os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_clean(path):
    print(f"[1/6] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["comment", "rating"])
    df = df[df["comment"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["comment"])
    after = len(df)
    print(f"      Loaded {before} rows -> {after} rows after removing "
          f"nulls/empty/duplicate reviews.")
    return df


def build_features(df):
    print("[2/6] Pre-processing text (clean -> tokenize -> stopwords -> "
          "lemmatize) ...")
    df["cleaned_review"] = df["comment"].apply(preprocess)

    print("[3/6] Extracting aspects (Product Quality / Packaging Quality / "
          "Delivery & Service / Others) ...")
    df["aspect"] = df["comment"].apply(lambda t: extract_aspect(clean_for_aspect(t)))

    print("[4/6] Deriving sentiment labels from star rating "
          "(1-2=Negative, 3=Neutral, 4-5=Positive) ...")
    df["sentiment_label"] = df["rating"].apply(rating_to_label)
    df["sentiment"] = df["sentiment_label"].map(LABEL_NAMES)

    # drop rows that became empty after cleaning
    df = df[df["cleaned_review"].str.strip() != ""]
    return df


def train_and_evaluate(df):
    print("[5/6] Splitting data (80% train / 20% test) and extracting "
          "TF-IDF features (unigram + bigram) ...")
    X_text = df["cleaned_review"]
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Support Vector Machine": LinearSVC(
            random_state=42, max_iter=5000, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
    }

    # Naive Bayes has no class_weight param, so approximate the same effect
    # with inverse-frequency sample weights (down-weights the majority
    # Positive class, up-weights minority Negative/Neutral classes).
    class_counts = y_train.value_counts()
    inv_freq = {cls: len(y_train) / count for cls, count in class_counts.items()}
    nb_sample_weight = y_train.map(inv_freq).values

    print("[6/6] Training models and evaluating on the held-out test set ...")
    results = {}
    fitted_models = {}
    for name, model in models.items():
        if name == "Naive Bayes":
            model.fit(X_train_tfidf, y_train, sample_weight=nb_sample_weight)
        else:
            model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "support": int(len(y_test)),
        }
        fitted_models[name] = model

        print(f"\n===== {name} =====")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(classification_report(
            y_test, y_pred, target_names=[LABEL_NAMES[k] for k in sorted(LABEL_NAMES)],
            zero_division=0,
        ))

    return vectorizer, fitted_models, results


def save_artifacts(vectorizer, fitted_models, df, results):
    # Primary model used by the web app.
    # Logistic Regression (class-weight balanced) is used here: it has the
    # highest overall accuracy among the three models, and with balanced
    # class weights it is far less biased toward the majority Positive class
    # than the unweighted version reported in the original study.
    # Primary model used by the web app.
    primary_model = fitted_models["Support Vector Machine"]

    joblib.dump(primary_model, os.path.join(MODELS_DIR, "primary_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    for name, model in fitted_models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, fname))

    df.to_csv(PROCESSED_PATH, index=False)

    # Precomputed stats for the dashboard (avoids recomputing on every request)
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    aspect_sentiment = (
        df.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
    )
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in aspect_sentiment.columns:
            aspect_sentiment[col] = 0
    aspect_sentiment = aspect_sentiment[["Positive", "Neutral", "Negative"]]

    stats = {
        "total_reviews": int(len(df)),
        "positive_count": int(sentiment_counts.get("Positive", 0)),
        "neutral_count": int(sentiment_counts.get("Neutral", 0)),
        "negative_count": int(sentiment_counts.get("Negative", 0)),
        "aspect_labels": aspect_sentiment.index.tolist(),
        "aspect_positive": aspect_sentiment["Positive"].tolist(),
        "aspect_neutral": aspect_sentiment["Neutral"].tolist(),
        "aspect_negative": aspect_sentiment["Negative"].tolist(),
        "model_comparison": results,
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved model artifacts to: {MODELS_DIR}")
    print(f"Saved processed dataset to: {PROCESSED_PATH}")
    print(f"Saved dashboard stats to: {STATS_PATH}")


def main():
    df = load_and_clean(DATA_PATH)
    df = build_features(df)
    vectorizer, fitted_models, results = train_and_evaluate(df)
    save_artifacts(vectorizer, fitted_models, df, results)

    print("\n================ MODEL COMPARISON SUMMARY ================")
    print(f"{'Model':<25}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}")
    for name, r in results.items():
        print(f"{name:<25}{r['accuracy']:<10}{r['precision']:<11}"
              f"{r['recall']:<9}{r['f1_score']:<8}")
    print("============================================================")
    print("\nTraining complete. Run 'python app.py' to launch the web app.")


if __name__ == "__main__":
    main()
