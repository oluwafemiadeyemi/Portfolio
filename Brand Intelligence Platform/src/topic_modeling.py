"""
Topic modeling utilities for the Enterprise Brand Intelligence Platform.
Uses NMF as the primary engine; optionally BERTopic if available.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMF Topic Modeling
# ---------------------------------------------------------------------------

def fit_nmf_topics(
    tfidf_matrix: Any,
    feature_names: List[str],
    n_topics: int = 10,
) -> List[Tuple[str, List[str], np.ndarray]]:
    """Fit NMF; return list of (topic_label, top_words, doc_assignments)."""
    logger.info("Fitting NMF with %d topics on matrix shape %s", n_topics, tfidf_matrix.shape)

    n_topics = min(n_topics, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1])
    n_topics = max(n_topics, 1)

    model = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=400,
        init="nndsvda",
    )
    W = model.fit_transform(tfidf_matrix)  # doc-topic
    H = model.components_               # topic-term

    topics = []
    for topic_idx in range(n_topics):
        top_indices = H[topic_idx].argsort()[::-1][:15]
        top_words = [feature_names[i] for i in top_indices]
        doc_assignments = W[:, topic_idx]
        label = label_topics({f"topic_{topic_idx}": top_words})[f"topic_{topic_idx}"]
        topics.append((label, top_words, doc_assignments))
        logger.debug("Topic %d (%s): %s", topic_idx, label, top_words[:5])

    logger.info("NMF fitting complete: %d topics", len(topics))
    return topics


# ---------------------------------------------------------------------------
# BERTopic (with graceful fallback)
# ---------------------------------------------------------------------------

def fit_bertopic_simple(
    texts: List[str],
    n_topics: int = 10,
) -> Tuple[Dict[str, List[str]], List[int]]:
    """Fit BERTopic or NMF fallback; return topics dict and doc-topic assignments."""
    logger.info("Attempting BERTopic on %d texts", len(texts))

    try:
        from bertopic import BERTopic
        topic_model = BERTopic(nr_topics=n_topics, calculate_probabilities=False, verbose=False)
        topics_list, _ = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()
        topics_dict: Dict[str, List[str]] = {}
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:
                continue
            label = f"topic_{row['Topic']}"
            words = [w for w, _ in topic_model.get_topic(row["Topic"])[:10]]
            topics_dict[label] = words
        logger.info("BERTopic fitted successfully: %d topics", len(topics_dict))
        return topics_dict, list(topics_list)

    except ImportError:
        logger.info("BERTopic not available — falling back to NMF")
    except Exception as exc:
        logger.warning("BERTopic failed (%s) — falling back to NMF", exc)

    # NMF fallback
    logger.info("Using NMF fallback for topic modeling")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    from src.features import preprocess_text
    preprocessed = [preprocess_text(t) for t in texts]

    try:
        tfidf = vectorizer.fit_transform(preprocessed)
    except ValueError as exc:
        logger.error("TF-IDF vectorization failed: %s", exc)
        return {}, []

    feature_names = vectorizer.get_feature_names_out().tolist()
    nmf_topics = fit_nmf_topics(tfidf, feature_names, n_topics=n_topics)

    topics_dict = {}
    all_assignments: np.ndarray = np.zeros(len(texts), dtype=int)

    for i, (label, words, doc_weights) in enumerate(nmf_topics):
        topics_dict[label] = words
        # Each document gets assigned to the topic with highest weight
        update_mask = doc_weights > all_assignments.astype(float)
        # Use argmax approach instead
    # Proper argmax assignment
    W_matrix = np.column_stack([w for _, _, w in nmf_topics]) if nmf_topics else np.zeros((len(texts), 1))
    doc_topic_assignments = W_matrix.argmax(axis=1).tolist()

    return topics_dict, doc_topic_assignments


# ---------------------------------------------------------------------------
# Trending Topics
# ---------------------------------------------------------------------------

def get_trending_topics(
    df: pd.DataFrame,
    topic_assignments: List[int],
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Find topics gaining frequency in the lookback window vs prior period."""
    logger.info("Computing trending topics (lookback=%d days)", lookback_days)

    df = df.copy()
    df["_topic"] = topic_assignments

    if "date" not in df.columns:
        logger.warning("No date column — cannot compute trending topics")
        topic_counts = pd.Series(topic_assignments).value_counts().reset_index()
        topic_counts.columns = ["topic_id", "count"]
        topic_counts["trend_score"] = 0.0
        return topic_counts

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    max_date = df["date"].max()
    if pd.isna(max_date):
        max_date = pd.Timestamp.now()

    cutoff_recent = max_date - pd.Timedelta(days=lookback_days)
    cutoff_prior = max_date - pd.Timedelta(days=lookback_days * 2)

    recent = df[df["date"] >= cutoff_recent]
    prior = df[(df["date"] >= cutoff_prior) & (df["date"] < cutoff_recent)]

    total_recent = max(len(recent), 1)
    total_prior = max(len(prior), 1)

    recent_counts = recent["_topic"].value_counts() / total_recent
    prior_counts = prior["_topic"].value_counts() / total_prior

    all_topics = set(recent_counts.index) | set(prior_counts.index)
    records = []
    for topic_id in all_topics:
        r = recent_counts.get(topic_id, 0.0)
        p = prior_counts.get(topic_id, 0.0)
        trend = r - p
        records.append({
            "topic_id": topic_id,
            "recent_freq": round(float(r), 4),
            "prior_freq": round(float(p), 4),
            "trend_score": round(float(trend), 4),
            "recent_count": int(recent["_topic"].eq(topic_id).sum()),
        })

    result = pd.DataFrame(records).sort_values("trend_score", ascending=False)
    logger.info("Trending topics computed: top topic_id=%s (trend=%.4f)",
                result.iloc[0]["topic_id"] if len(result) > 0 else "N/A",
                result.iloc[0]["trend_score"] if len(result) > 0 else 0.0)
    return result


# ---------------------------------------------------------------------------
# Topic Auto-Labeling
# ---------------------------------------------------------------------------

_LABEL_KEYWORDS: Dict[str, List[str]] = {
    "Cleanliness & Hygiene": ["clean", "dirty", "tidy", "spotless", "hygiene", "filthy", "dust", "stain"],
    "Staff & Service": ["staff", "service", "friendly", "rude", "helpful", "employee", "receptionist", "attentive"],
    "Food & Dining": ["food", "meal", "breakfast", "lunch", "dinner", "restaurant", "eat", "taste", "delicious"],
    "Location & Access": ["location", "central", "transport", "walk", "nearby", "area", "distance", "close"],
    "Value & Pricing": ["price", "expensive", "cheap", "affordable", "value", "worth", "cost", "overpriced"],
    "Amenities & Facilities": ["pool", "wifi", "gym", "parking", "spa", "fitness", "bar", "internet"],
    "Room Quality": ["room", "bed", "bathroom", "shower", "comfortable", "spacious", "noise", "view"],
    "Check-in Experience": ["check", "checkin", "checkout", "arrival", "reception", "desk", "booking"],
    "Overall Experience": ["experience", "overall", "recommend", "amazing", "terrible", "great", "awful"],
    "Management & Operations": ["management", "manager", "policy", "complaint", "respond", "issue", "refund"],
}


def label_topics(topics: Dict[str, List[str]]) -> Dict[str, str]:
    """Auto-label topics based on top words using keyword matching."""
    labels: Dict[str, str] = {}
    for topic_key, top_words in topics.items():
        words_lower = set(w.lower() for w in top_words)
        best_label = topic_key
        best_score = 0
        for label, keywords in _LABEL_KEYWORDS.items():
            score = len(words_lower & set(keywords))
            if score > best_score:
                best_score = score
                best_label = label
        labels[topic_key] = best_label
    return labels


# ---------------------------------------------------------------------------
# Full Pipeline Helper
# ---------------------------------------------------------------------------

def run_topic_pipeline(
    df: pd.DataFrame,
    text_col: str = "text",
    n_topics: int = 10,
) -> Tuple[Dict[str, List[str]], List[int], pd.DataFrame]:
    """Run full topic modeling pipeline; return topics dict, assignments, trending df."""
    logger.info("Running topic pipeline on %d documents", len(df))
    texts = df[text_col].fillna("").astype(str).tolist()

    topics_dict, assignments = fit_bertopic_simple(texts, n_topics=n_topics)
    trending = get_trending_topics(df, assignments, lookback_days=30)

    return topics_dict, assignments, trending
