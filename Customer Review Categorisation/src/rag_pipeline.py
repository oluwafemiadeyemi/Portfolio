"""
RAG (Retrieval-Augmented Generation) pipeline:
- ChromaDB vector store for semantic review search
- sentence-transformers embeddings
- Claude API for Q&A over reviews
- Prompt caching for efficiency
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
CHROMA_PATH = BASE_DIR / "models" / "chromadb"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embed_model = None
_chroma_collection = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = client.get_or_create_collection(
            name="customer_reviews",
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def build_vector_store(reviews: pd.DataFrame, batch_size: int = 500):
    """Embed reviews and store in ChromaDB."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    collection = _get_chroma_collection()

    # Check if already populated
    if collection.count() > 100:
        print(f"ChromaDB already has {collection.count():,} documents. Skipping rebuild.")
        return collection

    embed_model = _get_embed_model()
    texts = reviews["review_text"].fillna("").tolist()
    ids = [str(i) for i in range(len(texts))]
    metadatas = []
    for _, row in reviews.iterrows():
        metadatas.append({
            "category": str(row.get("category", "Unknown")),
            "rating":   int(row.get("rating", 3)),
            "sentiment": str(row.get("sentiment", "Neutral")),
        })

    print(f"Embedding {len(texts):,} reviews into ChromaDB ...")
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]
        embeddings = embed_model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        if (start // batch_size) % 10 == 0:
            print(f"  Indexed {end:,} / {len(texts):,}")

    print(f"ChromaDB built: {collection.count():,} documents")
    return collection


def semantic_search(query: str, n: int = 10, filter_category: Optional[str] = None,
                    filter_sentiment: Optional[str] = None) -> List[dict]:
    """Retrieve semantically similar reviews for a query."""
    embed_model = _get_embed_model()
    collection = _get_chroma_collection()

    if collection.count() == 0:
        return _demo_search_results(query, n)

    query_emb = embed_model.encode([query]).tolist()
    where_clause = {}
    if filter_category:
        where_clause["category"] = filter_category
    if filter_sentiment:
        where_clause["sentiment"] = filter_sentiment

    kwargs = {"query_embeddings": query_emb, "n_results": n}
    if where_clause:
        kwargs["where"] = where_clause

    results = collection.query(**kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return [{"text": d, "metadata": m, "score": round(1 - dist, 4)}
            for d, m, dist in zip(docs, metas, distances)]


def answer_question(question: str, context_reviews: List[dict] = None,
                    n_context: int = 8) -> dict:
    """
    RAG Q&A: retrieve relevant reviews → pass to LLM as context → generate answer.
    Provider is selected via PROVIDER env var (default: ollama).
    """
    from openai import OpenAI
    provider = os.getenv("PROVIDER", "ollama").lower()
    provider_models = {"ollama": "llama3.2:3b", "groq": "llama-3.3-70b-versatile", "anthropic": "claude-sonnet-4-6"}

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"answer": "ANTHROPIC_API_KEY not configured.", "sources": []}
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
    elif provider == "groq":
        client = OpenAI(api_key=os.getenv("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
    else:
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

    if context_reviews is None:
        context_reviews = semantic_search(question, n=n_context)

    context_text = "\n\n".join([
        f"[Review {i+1}] Rating: {r['metadata'].get('rating', '?')}/5 | "
        f"Category: {r['metadata'].get('category', '?')} | "
        f"Sentiment: {r['metadata'].get('sentiment', '?')}\n{r['text']}"
        for i, r in enumerate(context_reviews)
    ])

    system_text = ("You are a customer insights analyst. Answer questions using the provided "
                   "customer reviews. Be specific, cite review numbers, and provide actionable insights.")

    if provider == "anthropic":
        response = client.messages.create(
            model=provider_models["anthropic"],
            max_tokens=800,
            system=[
                {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": f"Customer Reviews Context:\n{context_text}", "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": question}],
        )
        answer = response.content[0].text
        cache_tokens = getattr(response.usage, "cache_read_input_tokens", 0)
        input_tokens = response.usage.input_tokens
    else:
        response = client.chat.completions.create(
            model=provider_models.get(provider, "llama3.2:3b"),
            max_tokens=800,
            temperature=0.3,
            messages=[
                {"role": "system", "content": f"{system_text}\n\nCustomer Reviews Context:\n{context_text}"},
                {"role": "user",   "content": question},
            ],
        )
        answer = response.choices[0].message.content
        cache_tokens = 0
        input_tokens = getattr(response.usage, "prompt_tokens", 0)

    return {
        "answer":       answer,
        "sources":      [r["text"][:150] for r in context_reviews[:3]],
        "provider":     provider,
        "input_tokens": input_tokens,
        "cache_tokens": cache_tokens,
    }


def _demo_search_results(query: str, n: int) -> List[dict]:
    """Demo results when ChromaDB is empty."""
    templates = [
        "Great product! Exactly what I needed.",
        "Shipping was fast and product was well packaged.",
        "Quality could be better for the price.",
        "Customer service resolved my issue quickly.",
        "Product stopped working after 2 weeks.",
    ]
    return [{"text": t, "metadata": {"category": "Product Quality",
                                      "rating": 4, "sentiment": "Positive"},
             "score": round(0.9 - i * 0.05, 3)} for i, t in enumerate(templates[:n])]


if __name__ == "__main__":
    from data_pipeline import prepare_all
    data = prepare_all(n=100_000, sample_for_rag=10_000)
    build_vector_store(data["rag_sample"])

    # Test search
    results = semantic_search("battery problems with electronics")
    for r in results[:3]:
        print(f"  [{r['score']:.3f}] {r['text'][:80]}")
