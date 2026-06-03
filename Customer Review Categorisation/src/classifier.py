"""
Generative AI review classifier.
Provider is selected via PROVIDER env var (default: ollama).

  PROVIDER=ollama      → llama3.2:3b running locally via Ollama (FREE, no API key)
  PROVIDER=anthropic   → claude-sonnet-4-6 (requires ANTHROPIC_API_KEY)
  PROVIDER=groq        → llama-3.3-70b via Groq free tier (requires GROQ_API_KEY)

Features:
  - Structured JSON outputs with few-shot examples
  - Batch processing for scale
  - BERTopic for unsupervised topic discovery
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

def _make_client():
    if PROVIDER == "anthropic":
        from anthropic import Anthropic
        return None, "anthropic"   # handled separately below
    if PROVIDER == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        ), "groq"
    # default: ollama
    return OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    ), "ollama"

PROVIDER_MODELS = {
    "ollama":    os.getenv("OLLAMA_MODEL", "llama3.2:4k"),
    "groq":      "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-4-6",
}

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"

REVIEW_CATEGORIES = [
    "Product Quality", "Delivery & Shipping", "Customer Service",
    "Value for Money", "Product Description Accuracy",
    "Packaging", "Return/Refund Process", "Technical Support",
]

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]

ASPECTS = ["Quality", "Price", "Shipping", "Service", "Durability",
           "Functionality", "Design", "Packaging"]

SYSTEM_PROMPT = """You are an expert customer review analyst for a Fortune 500 e-commerce company.
Your job is to analyse customer reviews and extract structured insights.

For each review, you must output a JSON object with exactly these fields:
{
  "category": one of ["Product Quality", "Delivery & Shipping", "Customer Service",
                       "Value for Money", "Product Description Accuracy",
                       "Packaging", "Return/Refund Process", "Technical Support"],
  "sentiment": one of ["Positive", "Neutral", "Negative"],
  "sentiment_score": float between -1.0 (very negative) and 1.0 (very positive),
  "aspects": list of affected aspects from ["Quality", "Price", "Shipping", "Service",
              "Durability", "Functionality", "Design", "Packaging"],
  "key_issues": list of 1-3 specific issues mentioned,
  "action_required": boolean - true if requires follow-up,
  "priority": one of ["Low", "Medium", "High", "Critical"],
  "summary": one sentence summary of the review
}

Output ONLY valid JSON. No explanation. No markdown code blocks."""

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Review: "The battery died after 2 days. Completely useless product."
Output: {"category":"Product Quality","sentiment":"Negative","sentiment_score":-0.9,"aspects":["Quality","Durability"],"key_issues":["battery failure","product durability"],"action_required":true,"priority":"High","summary":"Battery failed within 2 days of use, product deemed unusable."}

EXAMPLE 2:
Review: "Arrived quickly and exactly as described. Great value!"
Output: {"category":"Delivery & Shipping","sentiment":"Positive","sentiment_score":0.85,"aspects":["Shipping","Price"],"key_issues":[],"action_required":false,"priority":"Low","summary":"Fast delivery, accurate description, and good value for money."}

EXAMPLE 3:
Review: "Package was damaged but product works fine."
Output: {"category":"Packaging","sentiment":"Neutral","sentiment_score":0.1,"aspects":["Packaging","Quality"],"key_issues":["damaged packaging"],"action_required":true,"priority":"Medium","summary":"Packaging damaged during delivery, but product itself is functional."}
"""


class ReviewClassifier:
    """
    Multi-provider review classifier.
    Set PROVIDER env var to: ollama (default, free), anthropic, or groq.
    """

    def __init__(self):
        self.provider = PROVIDER
        self.model = PROVIDER_MODELS[PROVIDER]
        self.client, _ = _make_client()
        self.total_requests = 0
        print(f"ReviewClassifier using provider={self.provider}, model={self.model}")

    def classify_single(self, review_text: str) -> dict:
        """Classify a single review via the configured provider."""
        if self.provider == "anthropic":
            return self._classify_anthropic(review_text)
        return self._classify_openai_compat(review_text)

    def _classify_openai_compat(self, review_text: str) -> dict:
        """OpenAI-compatible call — works for Ollama and Groq."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES},
                    {"role": "user",   "content": f"Review: {review_text}\nOutput:"},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if model wraps in ```json
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            self.total_requests += 1
            return json.loads(raw)
        except json.JSONDecodeError:
            return _fallback_classification(review_text)
        except Exception as e:
            return _fallback_classification(review_text, error=str(e))

    def _classify_anthropic(self, review_text: str) -> dict:
        """Anthropic SDK call with prompt caching."""
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=[{"type": "text",
                          "text": SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES,
                          "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"Review: {review_text}\nOutput:"}],
            )
            self.total_requests += 1
            return json.loads(response.content[0].text.strip())
        except json.JSONDecodeError:
            return _fallback_classification(review_text)
        except Exception as e:
            return _fallback_classification(review_text, error=str(e))

    def classify_batch(self, reviews: List[str], save_path: Path = None) -> pd.DataFrame:
        """Classify a list of reviews. Returns structured DataFrame."""
        results = []
        for i, text in enumerate(tqdm(reviews, desc="Classifying")):
            result = self.classify_single(text)
            result["review_text"] = text[:200]
            results.append(result)
            # Small delay only for cloud providers to respect rate limits
            if self.provider in ("anthropic", "groq") and (i + 1) % 50 == 0:
                time.sleep(1)

        df = pd.DataFrame(results)
        if save_path:
            df.to_parquet(save_path, index=False)
            print(f"Saved {len(df):,} classified reviews")
        print(f"Provider={self.provider} | Total requests: {self.total_requests:,}")
        return df

    def generate_executive_summary(self, reviews: List[str], product_name: str = "Product") -> str:
        """Generate an executive VOC (Voice of Customer) summary."""
        sample = reviews[:50] if len(reviews) > 50 else reviews
        combined = "\n".join(f"- {r[:150]}" for r in sample)
        prompt = (f"Analyse these {len(sample)} customer reviews for {product_name} "
                  f"and write a 3-paragraph executive summary covering:\n"
                  f"1. Overall sentiment and key themes\n"
                  f"2. Top 3 issues requiring immediate action\n"
                  f"3. Strategic recommendations\n\nReviews:\n{combined}")

        if self.provider == "anthropic":
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            resp = client.messages.create(
                model=self.model, max_tokens=1000,
                system=[{"type": "text",
                          "text": "You are a senior customer insights analyst.",
                          "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=1000, temperature=0.3,
                messages=[
                    {"role": "system", "content": "You are a senior customer insights analyst. Write concise executive summaries."},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content


def _fallback_classification(text: str, error: str = None) -> dict:
    """Rule-based fallback when API is unavailable."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["love", "great", "excellent", "perfect", "amazing"]):
        sentiment, score = "Positive", 0.8
    elif any(w in text_lower for w in ["terrible", "broke", "awful", "worst", "garbage"]):
        sentiment, score = "Negative", -0.8
    else:
        sentiment, score = "Neutral", 0.0

    return {
        "category":        "Product Quality",
        "sentiment":       sentiment,
        "sentiment_score": score,
        "aspects":         ["Quality"],
        "key_issues":      [],
        "action_required": sentiment == "Negative",
        "priority":        "High" if sentiment == "Negative" else "Low",
        "summary":         text[:100],
        "source":          "fallback" + (f"_error:{error[:50]}" if error else ""),
    }


# ─── BERTopic Topic Discovery ─────────────────────────────────────────────────

def discover_topics(reviews: pd.DataFrame, n_topics: int = 20) -> dict:
    """
    Unsupervised topic discovery using BERTopic on review text.
    Complements GenAI classification with data-driven theme extraction.
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        texts = reviews["review_text"].dropna().tolist()[:10_000]  # cap for speed
        print(f"Running BERTopic on {len(texts):,} reviews ...")

        topic_model = BERTopic(
            nr_topics=n_topics,
            min_topic_size=20,
            verbose=True,
        )
        topics, probs = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()

        topic_model.save(str(BASE_DIR / "models" / "bertopic_model"))
        print(f"BERTopic discovered {topic_info['Topic'].nunique()} topics")
        return {"topics": topics, "topic_info": topic_info, "model": topic_model}

    except ImportError:
        print("BERTopic not installed. Skipping topic discovery.")
        return {}


if __name__ == "__main__":
    # Demo classification (requires ANTHROPIC_API_KEY)
    clf = ReviewClassifier()
    sample_reviews = [
        "The battery died after just 2 days of use. Extremely disappointed.",
        "Shipping was super fast! Product arrived in perfect condition.",
        "Average quality for the price. Nothing special but it works.",
    ]
    results = clf.classify_batch(sample_reviews)
    print(results.to_string())
