"""
LLM-powered supply chain risk analysis using Ollama (llama3.2:4k, local, free).
Adds plain-English risk narratives and SEC filing text analysis on top of
the existing XGBoost/LightGBM financial distress models.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:4k")

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)
    return _client


RISK_NARRATIVE_PROMPT = """You are a Chief Risk Officer at a top investment bank.
Given a company's financial distress indicators, write a concise, actionable 2-paragraph
risk assessment for the credit committee. Use specific numbers. Output plain text only."""

FILING_ANALYSIS_PROMPT = """You are a financial analyst specialising in supply chain risk.
Extract the key risk factors and supply chain dependencies from this text.
Output ONLY valid JSON:
{
  "risk_factors": [str],
  "supply_chain_dependencies": [str],
  "geographic_concentration": [str],
  "key_customers": [str],
  "financial_covenants": [str],
  "overall_risk_level": "Low" | "Medium" | "High" | "Critical",
  "summary": str (1 sentence)
}"""


def generate_risk_narrative(company_data: dict) -> str:
    """Generate plain-English risk narrative from financial ratios and distress scores."""
    data_text = "\n".join(f"  {k}: {v}" for k, v in company_data.items())
    try:
        resp = _get_client().chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": RISK_NARRATIVE_PROMPT},
                {"role": "user",   "content": f"Company financial data:\n{data_text}"},
            ],
            max_tokens=400,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"Risk assessment based on quantitative models: "
            f"distress probability derived from Altman Z-Score and ML ensemble. "
            f"[LLM narrative unavailable: {e}]"
        )


def analyze_filing_text(filing_text: str) -> dict:
    """Extract structured risk intelligence from SEC filing or news text."""
    text_snippet = filing_text[:2000]
    try:
        resp = _get_client().chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": FILING_ANALYSIS_PROMPT},
                {"role": "user",   "content": f"Filing excerpt:\n{text_snippet}"},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "risk_factors": ["Unable to parse LLM output"],
            "supply_chain_dependencies": [],
            "geographic_concentration": [],
            "key_customers": [],
            "financial_covenants": [],
            "overall_risk_level": "Medium",
            "summary": "Filing analysis failed — see quantitative scores.",
            "_error": "json_parse_failed",
        }
    except Exception as e:
        return {
            "risk_factors": [],
            "supply_chain_dependencies": [],
            "geographic_concentration": [],
            "key_customers": [],
            "financial_covenants": [],
            "overall_risk_level": "Unknown",
            "summary": f"LLM unavailable: {e}",
            "_error": str(e),
        }
