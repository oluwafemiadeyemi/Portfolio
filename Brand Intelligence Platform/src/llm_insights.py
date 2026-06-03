"""
LLM-powered brand intelligence using Ollama (llama3.2:4k, local, free).
Adds structured aspect extraction and executive narrative generation
on top of the existing sklearn sentiment models.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:4k")

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)
    return _client


ASPECT_SYSTEM_PROMPT = """You are an enterprise brand intelligence analyst.
Analyse customer reviews and extract structured insights.
Output ONLY valid JSON — no markdown, no explanation.

Schema:
{
  "overall_sentiment": "Positive" | "Neutral" | "Negative",
  "sentiment_score": float -1.0 to 1.0,
  "aspects": [{"aspect": str, "sentiment": "Positive"|"Neutral"|"Negative", "quote": str}],
  "key_themes": [str],
  "competitive_signals": [str],
  "action_items": [str],
  "executive_summary": str (1 sentence)
}"""

NARRATIVE_SYSTEM_PROMPT = """You are a Chief Customer Officer at a Fortune 500 company.
Given brand performance metrics, write a concise 3-paragraph executive briefing.
Output ONLY the briefing text — no headers, no JSON."""


def analyze_review(review_text: str) -> dict:
    """Extract structured brand intelligence from a single review using Llama."""
    try:
        resp = _get_client().chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": ASPECT_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Review: {review_text[:1500]}"},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_analysis(review_text)
    except Exception as e:
        return _fallback_analysis(review_text, error=str(e))


def generate_brand_narrative(metrics: dict) -> str:
    """Generate an executive brand health narrative from aggregated metrics."""
    metrics_text = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
    try:
        resp = _get_client().chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Brand metrics:\n{metrics_text}"},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Narrative unavailable — Ollama error: {e}]"


def _fallback_analysis(text: str, error: str = "") -> dict:
    words = text.lower().split()
    pos = sum(1 for w in words if w in {"great", "excellent", "love", "best", "amazing", "good"})
    neg = sum(1 for w in words if w in {"bad", "terrible", "awful", "worst", "horrible", "poor"})
    score = (pos - neg) / max(len(words), 1) * 5
    sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
    return {
        "overall_sentiment": sentiment,
        "sentiment_score": round(float(score), 3),
        "aspects": [],
        "key_themes": [],
        "competitive_signals": [],
        "action_items": [],
        "executive_summary": text[:100] + "...",
        "_fallback": True,
        "_error": error,
    }
