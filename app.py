import io
import json
import os

import joblib
import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)

from nlp_utils import (
    LABEL_NAMES,
    analyze_clauses,
    clean_for_aspect,
    extract_aspect,
    preprocess,
    rule_based_sentiment,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATS_PATH = os.path.join(BASE_DIR, "data", "dashboard_stats.json")

MODEL_PATH = os.path.join(MODELS_DIR, "primary_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

app = Flask(__name__)

model = None
vectorizer = None


def load_artifacts():
    global model, vectorizer
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        raise RuntimeError(
            "Model artifacts not found. Please run 'python train_model.py' "
            "first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)


def _ml_predict(cleaned_text):
    """ML fallback used per-clause when the rule-based lexicon has no
    signal for that clause."""
    X = vectorizer.transform([cleaned_text])
    pred_label = int(model.predict(X)[0])
    return LABEL_NAMES.get(pred_label, "Neutral")


def predict_sentiment(raw_text):
    """Single overall (aspect, sentiment) for a review — used for the
    batch CSV summary columns. See `predict_multi_aspect` for the
    multi-aspect breakdown used by Single Review Prediction."""
    cleaned = preprocess(raw_text)
    aspect = extract_aspect(clean_for_aspect(raw_text))

    if cleaned.strip() == "":
        return cleaned, aspect, "Neutral"

    rule_sentiment = rule_based_sentiment(raw_text)
    if rule_sentiment is not None:
        return cleaned, aspect, rule_sentiment

    sentiment = _ml_predict(cleaned)
    return cleaned, aspect, sentiment


def predict_multi_aspect(raw_text):
    """Return a list of {"aspect", "sentiment"} pairs — one per distinct
    aspect/clause detected in the review, so a review mentioning more
    than one aspect (e.g. "product not good but delivery fast") surfaces
    all of them instead of collapsing into a single dominant aspect."""
    return analyze_clauses(raw_text, ml_predict_fn=_ml_predict)


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def dashboard():
    if not os.path.exists(STATS_PATH):
        return render_template(
            "dashboard.html", stats=None,
            error="No stats found. Please run 'python train_model.py' first."
        )
    with open(STATS_PATH) as f:
        stats = json.load(f)
    return render_template("dashboard.html", stats=stats, error=None)


@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html")


@app.route("/api/predict-single", methods=["POST"])
def api_predict_single():
    data = request.get_json(force=True)
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter a review sentence."}), 400

    cleaned = preprocess(text)
    results = predict_multi_aspect(text)

    return jsonify({
        "original": text,
        "cleaned_text": cleaned,
        "results": results,
        # backward-compatible single aspect/sentiment (first detected pair)
        "aspect": results[0]["aspect"],
        "sentiment": results[0]["sentiment"],
    })


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    filename = file.filename or ""

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({
                "error": "Unsupported file format. Please upload a .csv, "
                         ".xlsx or .xls file."
            }), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to read file: {exc}"}), 400

    # try to find the review text column
    text_col = None
    for candidate in ["comment", "review", "text", "Review", "Comment", "Text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        text_col = df.columns[0]

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)

    cleaned_list = []
    combined_aspect_list = []
    combined_sentiment_list = []
    exploded_rows = []  # one row per (review, aspect, sentiment) pair, for accurate per-aspect chart counts

    for row_idx, text in enumerate(df[text_col]):
        cleaned_list.append(preprocess(text))
        multi_results = predict_multi_aspect(text)

        combined_aspect_list.append(
            "; ".join(dict.fromkeys(r["aspect"] for r in multi_results))
        )
        sentiments = [r["sentiment"] for r in multi_results]


        if "Positive" in sentiments and "Negative" in sentiments:
            final_sentiment = "Neutral"
        elif "Negative" in sentiments:
            final_sentiment = "Negative"
        elif "Positive" in sentiments:
            final_sentiment = "Positive"
        else:
            final_sentiment = "Neutral"

        combined_sentiment_list.append(final_sentiment)
        
        for r in multi_results:
            exploded_rows.append({
                "aspect": r["aspect"],
                "sentiment": r["sentiment"],
            })

    df["Cleaned Text"] = cleaned_list
    df["Extracted Aspect"] = combined_aspect_list
    df["Predicted Sentiment"] = combined_sentiment_list

    exploded_df = pd.DataFrame(exploded_rows)

    # aggregate for charts (per aspect-sentiment pair, across all detected
    # aspects in every review, not just one dominant aspect per review)
    aspect_sentiment = (
        exploded_df.groupby(["aspect", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in aspect_sentiment.columns:
            aspect_sentiment[col] = 0
    aspect_sentiment = aspect_sentiment[["Positive", "Neutral", "Negative"]]

    sentiment_counts = exploded_df["sentiment"].value_counts()

    # stash the results csv in-memory keyed by a token so it can be downloaded
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    app.config.setdefault("_LAST_RESULT_CSV", {})
    app.config["_LAST_RESULT_CSV"]["latest"] = csv_content

    preview = []

    for i, row in enumerate(df.head(15).to_dict(orient="records"), start=1):

     preview.append({
        "No": i,
        "Comment": row.get(text_col, ""),
        "Cleaned Text": row.get("Cleaned Text", ""),
        "Aspect": row.get("Extracted Aspect", ""),
        "Sentiment": row.get("Predicted Sentiment", "")
    })

    return jsonify({
        "total_reviews": int(len(df)),
        "aspect_labels": aspect_sentiment.index.tolist(),
        "aspect_positive": aspect_sentiment["Positive"].tolist(),
        "aspect_neutral": aspect_sentiment["Neutral"].tolist(),
        "aspect_negative": aspect_sentiment["Negative"].tolist(),
        "positive_count": int(sentiment_counts.get("Positive", 0)),
        "neutral_count": int(sentiment_counts.get("Neutral", 0)),
        "negative_count": int(sentiment_counts.get("Negative", 0)),
        "preview": preview,
        "text_column": text_col,
    })


@app.route("/api/download-result")
def api_download_result():
    csv_content = app.config.get("_LAST_RESULT_CSV", {}).get("latest")
    if not csv_content:
        return "No result available. Please run an analysis first.", 404
    buffer = io.BytesIO(csv_content.encode("utf-8"))
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="absa_analysis_result.csv",
    )


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    # also load when imported (e.g. by a WSGI server)
    try:
        load_artifacts()
    except RuntimeError:
        pass
