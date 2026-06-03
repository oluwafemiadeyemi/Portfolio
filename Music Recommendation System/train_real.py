"""
P16 Music Recommendation System — Real Data Training
Dataset: Million Song Dataset + Spotify + Last.fm
  - User Listening History.csv : 9.7M user-track-playcount triplets (implicit feedback)
  - Music Info.csv             : 50k tracks with Spotify audio features

Models:
  1. ALS Matrix Factorization (implicit library, collaborative filtering)
  2. Content-Based FAISS index (audio features: energy, danceability, valence, etc.)
  3. Popularity baseline (for cold-start)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import json
import faiss
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import normalize, StandardScaler

BASE_DIR   = Path(__file__).resolve().parent
DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DATA_PROC.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data():
    print("Loading track catalog...")
    tracks = pd.read_csv(DATA_RAW / "Music Info.csv", low_memory=False)
    tracks = tracks.drop_duplicates("track_id").reset_index(drop=True)
    tracks["track_idx"] = range(len(tracks))

    print(f"Tracks: {len(tracks):,}  Genres: {tracks['genre'].nunique()}")

    print("Loading user listening history (9.7M rows)...")
    hist = pd.read_csv(DATA_RAW / "User Listening History.csv")
    # Keep only tracks that appear in our catalog
    hist = hist[hist["track_id"].isin(tracks["track_id"])]

    # Map user/track to integer indices
    users  = {u: i for i, u in enumerate(hist["user_id"].unique())}
    track_id_to_idx = dict(zip(tracks["track_id"], tracks["track_idx"]))
    hist["user_idx"]  = hist["user_id"].map(users)
    hist["track_idx"] = hist["track_id"].map(track_id_to_idx)
    hist = hist.dropna(subset=["user_idx", "track_idx"])
    hist["user_idx"]  = hist["user_idx"].astype(int)
    hist["track_idx"] = hist["track_idx"].astype(int)
    hist["playcount"] = hist["playcount"].clip(1, 200)  # cap extreme values

    print(f"History after filtering: {len(hist):,} rows")
    print(f"Unique users: {hist['user_idx'].nunique():,}  |  tracks: {hist['track_idx'].nunique():,}")

    return tracks, hist, users


# ── 2. Build User-Item Matrix ─────────────────────────────────────────────────

def build_matrix(hist: pd.DataFrame, n_users: int, n_tracks: int) -> csr_matrix:
    mat = csr_matrix(
        (hist["playcount"].values.astype(np.float32),
         (hist["user_idx"].values, hist["track_idx"].values)),
        shape=(n_users, n_tracks),
    )
    save_npz(DATA_PROC / "user_item_matrix.npz", mat)
    print(f"User-item matrix: {mat.shape}  nnz={mat.nnz:,}  density={mat.nnz/(mat.shape[0]*mat.shape[1]):.6f}")
    return mat


# ── 3. ALS Collaborative Filtering ───────────────────────────────────────────

def train_als(mat: csr_matrix) -> object:
    try:
        import implicit
        print("\nTraining ALS (implicit library)...")
        model = implicit.als.AlternatingLeastSquares(
            factors=128, iterations=20, regularization=0.05,
            random_state=42,
        )
        model.fit(mat.T.tocsr())  # implicit expects item x user
        print(f"ALS trained: user_factors={model.user_factors.shape}  item_factors={model.item_factors.shape}")
        return model
    except ImportError:
        print("implicit not installed — building fallback SVD-based model...")
        from scipy.sparse.linalg import svds
        U, S, Vt = svds(mat.astype(np.float32), k=128)
        user_factors = U * S
        item_factors = Vt.T
        return {"user_factors": user_factors, "item_factors": item_factors}


# ── 4. Content-Based FAISS Index ──────────────────────────────────────────────

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

def build_faiss_index(tracks: pd.DataFrame):
    print("\nBuilding FAISS content-based index...")
    available = [f for f in AUDIO_FEATURES if f in tracks.columns]
    feat = tracks[available].fillna(0).astype(np.float32)
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat).astype(np.float32)
    feat_normed = normalize(feat_scaled, norm="l2").astype(np.float32)

    index = faiss.IndexFlatIP(feat_normed.shape[1])  # inner product on normed = cosine
    index.add(feat_normed)
    faiss.write_index(index, str(MODELS_DIR / "content_faiss.index"))

    joblib.dump(scaler, MODELS_DIR / "content_scaler.pkl")
    print(f"FAISS index: {index.ntotal:,} tracks, dim={feat_normed.shape[1]}")
    return index, scaler, available


# ── 5. Popularity Baseline ────────────────────────────────────────────────────

def build_popularity(hist: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    pop = (hist.groupby("track_idx")["playcount"]
              .sum()
              .reset_index()
              .rename(columns={"playcount": "total_plays"}))
    pop = pop.merge(
        tracks[["track_idx", "track_id", "name", "artist", "genre"]],
        on="track_idx", how="left"
    ).sort_values("total_plays", ascending=False)
    pop.to_parquet(DATA_PROC / "popularity_index.parquet", index=False)
    print(f"\nTop 5 tracks by play count:\n{pop[['name','artist','total_plays']].head(5).to_string()}")
    return pop


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("P16 Music Recommendation — Real Data Training")
    print("=" * 60)

    tracks, hist, users = load_data()
    n_users  = len(users)
    n_tracks = len(tracks)

    # Save processed track catalog for API
    tracks.to_parquet(DATA_PROC / "tracks.parquet", index=False)

    # User-item matrix
    mat = build_matrix(hist, n_users, n_tracks)

    # ALS collaborative filtering
    als_model = train_als(mat)
    joblib.dump(als_model, MODELS_DIR / "als_recommender.pkl")

    # Content-based FAISS
    faiss_index, scaler, feat_cols = build_faiss_index(tracks)
    joblib.dump(feat_cols, MODELS_DIR / "content_features.pkl")

    # Popularity baseline
    pop = build_popularity(hist, tracks)

    # Save user/track mappings for inference
    user_map   = {v: k for k, v in users.items()}  # idx -> user_id
    joblib.dump(users,    MODELS_DIR / "user_to_idx.pkl")
    joblib.dump(user_map, MODELS_DIR / "idx_to_user.pkl")
    joblib.dump(dict(zip(tracks["track_id"], tracks["track_idx"])),
                MODELS_DIR / "trackid_to_idx.pkl")

    metrics = {
        "dataset": "Million Song Dataset + Spotify + Last.fm (real, 9.7M events)",
        "n_users": n_users, "n_tracks": n_tracks,
        "n_interactions": len(hist),
        "matrix_density": round(mat.nnz / (n_users * n_tracks), 8),
        "als_factors": 128,
        "faiss_features": feat_cols,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("All artefacts saved to models/ and data/processed/")
    print(f"  Users: {n_users:,}  |  Tracks: {n_tracks:,}  |  Events: {len(hist):,}")
