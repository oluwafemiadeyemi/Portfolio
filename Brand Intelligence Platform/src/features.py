"""
Feature engineering for the Enterprise Brand Intelligence Platform.
TF-IDF, VADER sentiment, linguistic, and aspect-based features.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VADER initialisation (lazy, to avoid import errors)
# ---------------------------------------------------------------------------

_vader_analyzer: Optional[Any] = None


def _get_vader():
    """Return a cached SentimentIntensityAnalyzer instance."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                logger.info("Downloading NLTK vader_lexicon")
                nltk.download("vader_lexicon", quiet=True)
            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER analyzer initialized")
        except ImportError as exc:
            logger.error("NLTK not installed — VADER unavailable: %s", exc)
    return _vader_analyzer


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: Optional[str]) -> str:
    """Lowercase, strip HTML, remove extra whitespace, handle None."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Lowercase
    text = text.lower()
    # Replace newlines/tabs with space
    text = re.sub(r"[\n\r\t]+", " ", text)
    # Remove non-ASCII
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# TF-IDF Features
# ---------------------------------------------------------------------------

def extract_tfidf_features(
    texts: List[str],
    max_features: int = 10000,
    fitted_vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[TfidfVectorizer, Any]:
    """Return fitted TF-IDF vectorizer and sparse feature matrix."""
    logger.info("Extracting TF-IDF features (max_features=%d, n_texts=%d)", max_features, len(texts))
    preprocessed = [preprocess_text(t) for t in texts]

    if fitted_vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(preprocessed)
        logger.info("TF-IDF matrix shape: %s", matrix.shape)
        return vectorizer, matrix
    else:
        matrix = fitted_vectorizer.transform(preprocessed)
        return fitted_vectorizer, matrix


# ---------------------------------------------------------------------------
# VADER Lexicon Sentiment
# ---------------------------------------------------------------------------

def extract_sentiment_lexicon(text: str) -> Dict[str, float]:
    """Return VADER sentiment scores: pos, neg, neu, compound."""
    analyzer = _get_vader()
    if analyzer is None:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
    scores = analyzer.polarity_scores(str(text))
    return {
        "pos": scores.get("pos", 0.0),
        "neg": scores.get("neg", 0.0),
        "neu": scores.get("neu", 1.0),
        "compound": scores.get("compound", 0.0),
    }


# ---------------------------------------------------------------------------
# Linguistic Features
# ---------------------------------------------------------------------------

def extract_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add review_length, exclamation_count, question_count, caps_ratio, avg_word_length."""
    logger.info("Extracting linguistic features for %d rows", len(df))
    texts = df["text"].fillna("").astype(str)

    df = df.copy()
    df["review_length"] = texts.str.len()
    df["word_count"] = texts.str.split().str.len().fillna(0)
    df["exclamation_count"] = texts.str.count(r"!")
    df["question_count"] = texts.str.count(r"\?")
    df["caps_ratio"] = texts.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    df["avg_word_length"] = texts.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.strip() else 0.0
    )
    df["sentence_count"] = texts.str.count(r"[.!?]+")
    df["unique_word_ratio"] = texts.apply(
        lambda t: len(set(t.lower().split())) / max(len(t.split()), 1) if t.strip() else 0.0
    )
    return df


# ---------------------------------------------------------------------------
# Aspect Keywords
# ---------------------------------------------------------------------------

def get_aspect_keywords() -> Dict[str, List[str]]:
    """Return dict of aspect → keyword list."""
    return {
        "cleanliness": [
            "clean", "dirty", "tidy", "spotless", "filthy", "hygiene", "hygienic",
            "sanitized", "dusty", "stained", "smell", "odor", "mold", "mould",
            "grime", "grimy", "immaculate", "pristine", "messy", "clutter",
        ],
        "staff": [
            "staff", "rude", "friendly", "helpful", "service", "employee",
            "receptionist", "concierge", "manager", "attentive", "professional",
            "courteous", "polite", "unhelpful", "dismissive", "responsive",
            "welcoming", "warm", "incompetent", "negligent", "kind",
        ],
        "value": [
            "price", "expensive", "cheap", "affordable", "overpriced", "value",
            "worth", "costly", "reasonable", "budget", "rate", "fee", "charge",
            "deal", "bargain", "money", "cost", "pricing", "competitive",
        ],
        "location": [
            "location", "central", "far", "convenient", "accessible", "transport",
            "walk", "distance", "neighborhood", "area", "nearby", "close",
            "remote", "isolated", "downtown", "suburb", "transit", "parking",
            "commute", "quiet", "noisy",
        ],
        "amenities": [
            "pool", "wifi", "parking", "gym", "fitness", "breakfast", "bar",
            "restaurant", "spa", "internet", "tv", "air conditioning", "ac",
            "heating", "elevator", "laundry", "minibar", "jacuzzi", "sauna",
            "concierge", "business center", "conference", "pet",
        ],
    }


# ---------------------------------------------------------------------------
# Aspect Sentiment Extraction
# ---------------------------------------------------------------------------

def extract_aspect_sentiment(text: str, vader_analyzer: Optional[Any] = None) -> Dict[str, Dict[str, float]]:
    """For each aspect, score sentences mentioning aspect keywords using VADER."""
    if vader_analyzer is None:
        vader_analyzer = _get_vader()

    aspect_keywords = get_aspect_keywords()
    results: Dict[str, Dict[str, float]] = {}
    default_score = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "mentioned": False}

    if not text or not isinstance(text, str):
        return {aspect: dict(default_score) for aspect in aspect_keywords}

    # Split into sentences
    sentences = re.split(r"[.!?]+", text.lower())
    sentences = [s.strip() for s in sentences if s.strip()]

    for aspect, keywords in aspect_keywords.items():
        relevant_sentences = [
            s for s in sentences
            if any(kw in s for kw in keywords)
        ]
        if not relevant_sentences or vader_analyzer is None:
            results[aspect] = dict(default_score)
        else:
            scores_list = [vader_analyzer.polarity_scores(s) for s in relevant_sentences]
            results[aspect] = {
                "compound": float(np.mean([s["compound"] for s in scores_list])),
                "pos": float(np.mean([s["pos"] for s in scores_list])),
                "neg": float(np.mean([s["neg"] for s in scores_list])),
                "neu": float(np.mean([s["neu"] for s in scores_list])),
                "mentioned": True,
            }

    return results


# ---------------------------------------------------------------------------
# Full Feature Pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Orchestrate all feature extraction; return feature matrix and enriched metadata df."""
    logger.info("Building feature pipeline for %d rows", len(df))

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)

    # 1. Linguistic features
    df = extract_linguistic_features(df)

    # 2. VADER sentiment
    logger.info("Running VADER sentiment extraction")
    vader = _get_vader()
    vader_scores = df["text"].apply(lambda t: extract_sentiment_lexicon(t))
    df["compound_sentiment"] = vader_scores.apply(lambda s: s["compound"])
    df["pos_sentiment"] = vader_scores.apply(lambda s: s["pos"])
    df["neg_sentiment"] = vader_scores.apply(lambda s: s["neg"])
    df["neu_sentiment"] = vader_scores.apply(lambda s: s["neu"])

    # 3. Sentiment label from compound score
    def _label(c: float) -> str:
        if c >= 0.05:
            return "positive"
        elif c <= -0.05:
            return "negative"
        return "neutral"

    df["sentiment_label"] = df["compound_sentiment"].apply(_label)

    # 4. Aspect sentiment
    logger.info("Running aspect sentiment extraction")
    aspect_results = df["text"].apply(lambda t: extract_aspect_sentiment(t, vader))
    for aspect in get_aspect_keywords():
        df[f"aspect_{aspect}_compound"] = aspect_results.apply(
            lambda r: r.get(aspect, {}).get("compound", 0.0)
        )
        df[f"aspect_{aspect}_mentioned"] = aspect_results.apply(
            lambda r: r.get(aspect, {}).get("mentioned", False)
        )

    # 5. TF-IDF features — keep sparse, reduce via TruncatedSVD to 100 components
    logger.info("Building TF-IDF matrix")
    preprocessed_texts = df["text"].apply(preprocess_text).tolist()
    vectorizer, tfidf_matrix = extract_tfidf_features(preprocessed_texts, max_features=2000)

    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    logger.info("Reducing TF-IDF with TruncatedSVD (100 components)")
    svd = TruncatedSVD(n_components=100, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)  # (n_samples, 100) — fits in RAM

    # 6. Combine numeric features
    numeric_cols = [
        "review_length", "word_count", "exclamation_count", "question_count",
        "caps_ratio", "avg_word_length", "sentence_count", "unique_word_ratio",
        "compound_sentiment", "pos_sentiment", "neg_sentiment", "neu_sentiment",
    ]
    aspect_cols = [c for c in df.columns if c.startswith("aspect_") and c.endswith("_compound")]
    all_numeric_cols = numeric_cols + aspect_cols

    numeric_matrix = df[all_numeric_cols].fillna(0.0).values

    feature_matrix = np.hstack([tfidf_reduced, numeric_matrix])
    logger.info("Feature matrix shape: %s", feature_matrix.shape)

    return feature_matrix, df
