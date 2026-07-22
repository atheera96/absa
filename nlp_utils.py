
import re
import string

# ---------------------------------------------------------------------------
# 1. Stopwords (English) - kept as a plain Python list so the project has no
#    external dependency on NLTK data downloads (this environment has no
#    internet access to fetch NLTK corpora).
#
#    IMPORTANT: negation words (not, no, never, don't, isn't, etc.) are
#    deliberately EXCLUDED from this list. Removing them would flip the
#    meaning of reviews like "not good" -> "good" and silently corrupt
#    sentiment classification.
# ---------------------------------------------------------------------------
STOPWORDS_EN = set("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and if because as until while of at by for with about
against between into through during before after above below to from up down
in out on off over under again further then once here there when where how
all any both each more most other some such only own same
so than too very s t will just now d ll m o re
ve y ain
""".split())

# ---------------------------------------------------------------------------
# 2. Text cleaning
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"http\S+|www\.\S+")
_NUM_RE = re.compile(r"\d+")
_MULTISPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 

def clean_text(text):
    """Lowercase, strip URLs/numbers/punctuation/extra whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    try:
      text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
     pass
    text = _URL_RE.sub("", text)
    text = _NUM_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def tokenize_remove_stopwords(text):
    """Tokenize on whitespace and drop stopwords."""
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS_EN and len(t) > 1]


def preprocess(text):
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.

    Returns the final processed string (ready for TF-IDF vectorization).
    Note: this output is intentionally stemmed/lemmatized, so it should be
    used for TF-IDF feature extraction, NOT for keyword-based aspect
    extraction (use `clean_for_aspect` + `extract_aspect` for that instead,
    since stemming can mangle keywords, e.g. "packaging" -> "packag").
    """
    cleaned = clean_text(text)
    tokens = tokenize_remove_stopwords(cleaned)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


def clean_for_aspect(text):
    """Lightweight cleaning (no stemming) so aspect keywords match reliably."""
    cleaned = clean_text(text)
    tokens = tokenize_remove_stopwords(cleaned)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 3. Aspect extraction (rule-based keyword matching)
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS = {
   "Product Quality": [
        "quality", "durable", "durability", "broken", "defect", "defective",
        "material", "genuine", "fake", "authentic", "sturdy", "functional",
        "works well", "work well", "stopped working", "works", "faulty",
        "damaged item", "as described", "not as described", "cheap material",
        "well made", "strong", "weak", "poor quality", "good quality",
        "excellent quality", "size", "fit", "fits", "fitting", "colour",
        "color", "design", "comfortable", "comfort", "soft", "texture",
        "smell", "effective", "powerful", "compact", "lightweight",
        "value for money", "worth it", "worth the price", "for the price",
        "worth buying", "worth every", "recommend", "satisfied", "strength",
        "performance", "feature", "features", "flaw", "flaws", "match", "matches", "description", "good"
    ],
     "Packaging Quality": [
        "packaging", "package", "packaged", "box", "bubble wrap", "wrapped",
        "wrapping", "seal", "sealed", "carton", "box", "crushed box",
        "parcel", "envelope", "poorly packed", "well packed", "nicely packed",
        "packed well", "bubble envelope", "plastic wrap", "packed nicely",
        "box damaged", "package damaged", "bubble", "wrap",
    ],
    "Delivery & Service": [
        "delivery", "deliver", "shipping", "ship", "courier", "arrived",
        "arrival", "arrive", "arrives", "received", "receive", "late",
        "delay", "delayed", "fast delivery", "slow delivery", "seller",
        "response", "responsive", "customer service", "chat", "reply",
        "communication", "tracking", "logistics", "sent", "condition",
        "days", "shipment", "shipping time", "dispatch", "dispatched",
    ],
}


# Pre-compile a word-boundary regex per keyword so overlapping substrings
# (e.g. "deliver" inside "delivery") are not double-counted.
_ASPECT_PATTERNS = {
    aspect: [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in keywords]
    for aspect, keywords in ASPECT_KEYWORDS.items()
}


def extract_aspect(text):
    """Return the aspect with the most keyword hits in the given text.

    IMPORTANT: pass text produced by `clean_for_aspect` (not the fully
    lemmatized `preprocess` output), otherwise stemming can distort
    keywords (e.g. "packaging" -> "packag") and cause false matches.

    Falls back to 'Others' when no keyword from any aspect is found.
    """
    scores = {aspect: 0 for aspect in ASPECT_KEYWORDS}
    for aspect, patterns in _ASPECT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                scores[aspect] += 1

    best_aspect = max(scores, key=scores.get)
    if scores[best_aspect] == 0:
        return "Others"
    return best_aspect


# ---------------------------------------------------------------------------
# 4. Rule-based sentiment layer (hybrid rule-based + statistical approach,
#    matches Objective ii: "hybrid NLP approach combining rule-based
#    preprocessing and statistical feature extraction techniques")
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    "good", "great", "excellent", "perfect", "nice", "fast", "recommend",
    "recommended", "worth", "love", "loved", "amazing", "satisfied",
    "happy", "genuine", "durable", "affordable", "quick", "smooth",
    "friendly", "sturdy", "beautiful", "awesome", "superb", "responsive",
    "helpful", "authentic", "impressed", "comfortable", "comfy",  "respond",  "responded",
    "quickly",  "polite",  "politely",
}

NEGATIVE_WORDS = {
    "slow", "late", "broken", "damaged", "poor", "fake", "delay",
    "delayed", "refund", "defect", "defective", "faulty", "crushed",
    "cracked", "missing", "wrong", "scam", "complain", "complaint",
    "awful", "horrible", "terrible", "worst", "bad", "disappoint",
    "disappointed", "disappointing", "regret", "waste", "torn", "leak",
    "leaking", "dirty", "smelly", "rude", "flimsy", "cheap", "damage", "Damaged", 
    "crack", "dent",  "dented", "slow", "missing",  "wrong item", "defect",
    "defective", "torn", "leaking", "longer", "waiting", "stop", "stopped", "working", 
    "malfunction", "not working",
}

# words that flip the polarity of the sentiment word(s) that follow them
NEGATION_WORDS = {
    "not", "no", "never", "cant", "cannot", "dont", "doesnt", "didnt",
    "isnt", "wasnt", "arent", "werent", "hasnt", "havent", "hadnt",
    "wont", "wouldnt", "shouldnt", "without",
}

# how many tokens ahead a negation word affects (e.g. "not very good")
_NEGATION_WINDOW = 3

# Split on contrast connectors (but/however/although/yet) as well as
# clause-ending punctuation, so a review like "late delivery but the
# product itself is great" is scored per-clause rather than as one blob
# (this is what turns a genuinely mixed review into "Neutral" instead of
# whichever polarity happens to have more words).
_CLAUSE_SPLIT_RE = re.compile(
    r"\bbut\b|\bhowever\b|\balthough\b|\byet\b|[.,;:!?]+",
    flags=re.IGNORECASE,
)


def split_into_clauses(text):
    parts = _CLAUSE_SPLIT_RE.split(str(text))
    return [p.strip() for p in parts if p.strip()]


def _clause_polarity_hits(clause_text):
    """Return (pos_hits, neg_hits) for one clause, with negation handling
    scoped to that clause only (so negation never leaks across a 'but' or
    a comma into an unrelated clause)."""
    clause = clause_text.lower()
    clause = _URL_RE.sub("", clause)
    clause = _NUM_RE.sub("", clause)
    clause = clause.translate(_PUNCT_TABLE)
    tokens = [t for t in clause.split() if len(t) > 1]

    pos_hits, neg_hits = 0, 0
    negate_countdown = 0
    for token in tokens:
        if token in NEGATION_WORDS:
            negate_countdown = _NEGATION_WINDOW
            continue

        is_positive = token in POSITIVE_WORDS
        is_negative = token in NEGATIVE_WORDS

        if negate_countdown > 0 and (is_positive or is_negative):
            is_positive, is_negative = is_negative, is_positive

        pos_hits += int(is_positive)
        neg_hits += int(is_negative)

        if negate_countdown > 0:
            negate_countdown -= 1

    return pos_hits, neg_hits


def rule_based_sentiment(raw_text):
    """Rule-based sentiment lexicon with negation + clause-aware scoring.

    Returns "Positive", "Negative", "Neutral" (mixed signal found), or
    None when no lexicon word is found at all (caller should fall back
    to the statistical model in that case).
    """
    total_pos, total_neg = 0, 0
    for clause in split_into_clauses(raw_text):
        p, n = _clause_polarity_hits(clause)
        total_pos += p
        total_neg += n

    if total_pos > 0 and total_neg > 0:
        return "Neutral"
    if total_pos > 0:
        return "Positive"
    if total_neg > 0:
        return "Negative"
    return None


def analyze_clauses(raw_text, ml_predict_fn=None):
    """Split a review into clauses and return one (aspect, sentiment) pair
    per clause, so a review that discusses more than one aspect (e.g.
    "product not good but delivery fast") yields multiple results instead
    of collapsing everything into a single dominant aspect.

    `ml_predict_fn`: optional callable(cleaned_text) -> "Positive"/
    "Negative"/"Neutral", used as a fallback when a clause has no
    rule-based lexicon signal. If not provided, clauses with no lexicon
    signal default to "Neutral".

    Returns a list of dicts: [{"clause": str, "aspect": str,
    "sentiment": str}, ...], with consecutive duplicate
    (aspect, sentiment) pairs merged, and uninformative fragments (no
    specific aspect AND no lexicon sentiment signal — typically small
    filler clauses produced by splitting on every comma) dropped, unless
    that would remove every result for the review.
    """
    clauses = split_into_clauses(raw_text) or [str(raw_text)]

    raw_results = []
    for clause in clauses:
        aspect = extract_aspect(clean_for_aspect(clause))
        lexicon_sentiment = rule_based_sentiment(clause)
        has_signal = aspect != "Others" or lexicon_sentiment is not None

        if lexicon_sentiment is not None:
            sentiment = lexicon_sentiment
        elif ml_predict_fn is not None:
            cleaned_clause = preprocess(clause)
            sentiment = (
                ml_predict_fn(cleaned_clause)
                if cleaned_clause.strip() != ""
                else "Neutral"
            )
        else:
            sentiment = "Neutral"

        raw_results.append({
            "clause": clause,
            "aspect": aspect,
            "sentiment": sentiment,
            "has_signal": has_signal,
        })

    # drop uninformative fragments (no specific aspect, no lexicon signal)
    # unless doing so would leave nothing at all
    informative = [r for r in raw_results if r["has_signal"]]
    kept = informative if informative else raw_results

    # merge consecutive clauses that landed on the exact same
    # (aspect, sentiment) pair, e.g. two "Others" clauses in a row
    merged = []
    for r in kept:
        if (merged and merged[-1]["aspect"] == r["aspect"]
                and merged[-1]["sentiment"] == r["sentiment"]):
            merged[-1]["clause"] += "; " + r["clause"]
        else:
            merged.append({
                "clause": r["clause"],
                "aspect": r["aspect"],
                "sentiment": r["sentiment"],
            })

    return merged


def overall_sentiment(results):
    """Combine a list of per-aspect sentiment results (from
    `analyze_clauses`) into one overall verdict.

    Rule: if the review has both a Positive-leaning and a Negative-leaning
    aspect, the overall sentiment is "Neutral" (genuinely mixed review).
    Otherwise it follows whichever polarity is present (Neutral results
    don't override a clear Positive or Negative elsewhere in the review).
    """
    sentiments = [r["sentiment"] for r in results]
    pos = sentiments.count("Positive")
    neg = sentiments.count("Negative")

    if pos > 0 and neg > 0:
        return "Neutral"
    if pos > 0:
        return "Positive"
    if neg > 0:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# 4. Sentiment labelling from star rating (used only during training, since
#    the dataset provides a numeric rating rather than a sentiment label)
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}


def rating_to_label(rating):
    """rating 1-2 -> Negative(0), 3 -> Neutral(1), 4-5 -> Positive(2)."""
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2
