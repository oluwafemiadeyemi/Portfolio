"""
FastAPI: music recommendation endpoints.
/recommend/user — personalised recommendations for user
/recommend/similar — tracks similar to a seed track
/recommend/hybrid — blended collaborative + content-based
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Intelligent Music Discovery & Recommendation Platform",
    description="ALS + Content-Based + BERT4Rec hybrid recommendation engine (50M+ events)",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_tracks = None
_als = None
_content = None


def _load_tracks():
    global _tracks
    if _tracks is None:
        p = DATA_PROC / "tracks.parquet"
        if p.exists():
            _tracks = pd.read_parquet(p)
    return _tracks


def _load_als():
    global _als
    if _als is None:
        p = MODELS_DIR / "als_recommender.pkl"
        if p.exists():
            from recommender import ALSRecommender
            _als = ALSRecommender.load()
    return _als


def _load_content():
    global _content
    if _content is None:
        p = MODELS_DIR / "content_features.pkl"
        if p.exists():
            from recommender import ContentBasedRecommender
            _content = ContentBasedRecommender.load()
    return _content


def _demo_recommendations(n: int = 20) -> List[dict]:
    tracks = _load_tracks()
    if tracks is None or len(tracks) == 0:
        return [{"track_id": i, "title": f"Track_{i}", "artist": f"Artist_{i%100}",
                 "genre": "Pop", "score": round(1 - i/n, 3)} for i in range(n)]
    sample = tracks.sample(min(n, len(tracks)), random_state=np.random.randint(0, 100))
    return [{"track_id": int(r["track_id"]) if "track_id" in r else i,
             "title": r.get("title", f"Track_{i}"),
             "artist": r.get("artist", f"Artist_{i}"),
             "genre": r.get("genre", "Unknown"),
             "score": round(1 - i/n, 3)}
            for i, (_, r) in enumerate(sample.iterrows())]


class TrackInfo(BaseModel):
    track_id: int
    title: str
    artist: str
    genre: str
    score: float


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    n: int = Field(default=20, ge=1, le=100)
    seed_track_id: Optional[int] = None


@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "tracks_loaded":  (DATA_PROC / "tracks.parquet").exists(),
        "als_loaded":     (MODELS_DIR / "als_recommender.pkl").exists(),
        "content_loaded": (MODELS_DIR / "content_faiss.index").exists(),
    }


@app.post("/recommend/user", response_model=List[TrackInfo])
def recommend_for_user(req: RecommendRequest):
    """Personalised recommendations via ALS collaborative filtering."""
    als = _load_als()
    tracks = _load_tracks()

    if als is None or als.item_factors is None:
        return _demo_recommendations(req.n)

    recs = als.recommend(req.user_id, n=req.n)
    return _enrich_recommendations(recs, tracks)


@app.get("/recommend/similar/{track_id}", response_model=List[TrackInfo])
def recommend_similar(track_id: int, n: int = 20):
    """Content-based similar tracks using FAISS ANN search."""
    content = _load_content()
    tracks = _load_tracks()

    if content is None:
        return _demo_recommendations(n)

    recs = content.recommend_similar(track_id, n=n)
    return _enrich_recommendations(recs, tracks)


@app.post("/recommend/hybrid", response_model=List[TrackInfo])
def recommend_hybrid(req: RecommendRequest):
    """Blended collaborative (60%) + content-based (40%) recommendations."""
    from recommender import HybridRecommender
    als = _load_als()
    content = _load_content()
    tracks = _load_tracks()

    if als is None or content is None:
        return _demo_recommendations(req.n)

    hybrid = HybridRecommender(als, content, alpha=0.6)
    recs = hybrid.recommend(req.user_id, req.seed_track_id, n=req.n)
    return _enrich_recommendations(recs, tracks)


@app.get("/catalog/search")
def search_catalog(genre: Optional[str] = None, mood: Optional[str] = None,
                   min_energy: float = 0, min_popularity: int = 0, limit: int = 50):
    """Search track catalog by audio features."""
    tracks = _load_tracks()
    if tracks is None:
        raise HTTPException(503, "Track catalog not loaded.")
    df = tracks.copy()
    if genre and "genre" in df.columns:
        df = df[df["genre"].str.lower() == genre.lower()]
    if mood and "mood" in df.columns:
        df = df[df["mood"].str.lower() == mood.lower()]
    if "energy" in df.columns:
        df = df[df["energy"] >= min_energy]
    if "popularity" in df.columns:
        df = df[df["popularity"] >= min_popularity]
    sample = df.head(limit)
    return sample[["track_id", "title", "genre", "mood", "energy",
                    "danceability", "popularity"]].fillna("").to_dict(orient="records")


@app.get("/catalog/stats")
def catalog_stats():
    tracks = _load_tracks()
    if tracks is None:
        raise HTTPException(503, "Catalog not loaded.")
    return {
        "total_tracks": len(tracks),
        "genres": tracks["genre"].value_counts().to_dict() if "genre" in tracks.columns else {},
        "avg_energy": round(float(tracks["energy"].mean()), 3) if "energy" in tracks.columns else None,
        "avg_popularity": round(float(tracks["popularity"].mean()), 1) if "popularity" in tracks.columns else None,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _enrich_recommendations(track_ids: List[int], tracks: pd.DataFrame) -> List[dict]:
    if tracks is None or len(tracks) == 0:
        return [{"track_id": tid, "title": f"Track_{tid}", "artist": "Unknown",
                 "genre": "Unknown", "score": round(1/(i+1), 3)}
                for i, tid in enumerate(track_ids)]

    id_col = "track_id" if "track_id" in tracks.columns else tracks.columns[0]
    result = []
    for i, tid in enumerate(track_ids):
        row = tracks[tracks[id_col] == tid]
        if len(row) > 0:
            r = row.iloc[0]
            result.append(TrackInfo(
                track_id=int(tid),
                title=str(r.get("title", f"Track_{tid}")),
                artist=str(r.get("artist", r.get("artist_name", f"Artist_{tid%100}"))),
                genre=str(r.get("genre", "Unknown")),
                score=round(1 / (i + 1), 4),
            ))
        else:
            result.append(TrackInfo(
                track_id=int(tid), title=f"Track_{tid}",
                artist="Unknown", genre="Unknown", score=round(1/(i+1), 4),
            ))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8006, reload=True)
