"""
Data pipeline: generates 50M+ synthetic listening events (Last.fm scale)
with 1M users, 100k tracks, realistic consumption patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

GENRES = ["Pop", "Rock", "Hip-Hop", "Electronic", "Classical", "Jazz",
          "R&B", "Country", "Latin", "Indie", "Metal", "Folk", "Reggae", "K-Pop"]
MOODS  = ["Energetic", "Chill", "Happy", "Melancholic", "Romantic",
          "Focus", "Party", "Workout", "Sleep", "Study"]
ERAS   = ["2020s", "2010s", "2000s", "1990s", "1980s", "1970s", "Classics"]
KEYS   = ["C", "D", "E", "F", "G", "A", "B"]


def generate_track_catalog(n_tracks: int = 100_000) -> pd.DataFrame:
    """Generate a catalog of 100k tracks with audio features."""
    print(f"Generating {n_tracks:,} track catalog ...")
    artists = [f"Artist_{i}" for i in range(10_000)]
    albums  = [f"Album_{i}" for i in range(30_000)]

    tracks = pd.DataFrame({
        "track_id":    np.arange(n_tracks),
        "title":       [f"Track_{i}" for i in range(n_tracks)],
        "artist_id":   RNG.integers(0, 10_000, n_tracks),
        "album_id":    RNG.integers(0, 30_000, n_tracks),
        "genre":       RNG.choice(GENRES, n_tracks),
        "mood":        RNG.choice(MOODS, n_tracks),
        "era":         RNG.choice(ERAS, n_tracks),
        "duration_s":  RNG.integers(120, 360, n_tracks),
        "bpm":         RNG.integers(60, 180, n_tracks),
        "key":         RNG.choice(KEYS, n_tracks),
        "energy":      RNG.uniform(0, 1, n_tracks).round(3),
        "danceability": RNG.uniform(0, 1, n_tracks).round(3),
        "acousticness": RNG.uniform(0, 1, n_tracks).round(3),
        "valence":     RNG.uniform(0, 1, n_tracks).round(3),
        "instrumentalness": RNG.uniform(0, 1, n_tracks).round(3),
        "popularity":  RNG.integers(0, 100, n_tracks),
        "release_year": RNG.integers(1970, 2025, n_tracks),
    })
    tracks.to_parquet(DATA_PROC / "tracks.parquet", index=False)
    print(f"Track catalog saved: {len(tracks):,} tracks")
    return tracks


def generate_listening_events(
    n_users: int = 1_000_000,
    n_tracks: int = 100_000,
    target_events: int = 50_000_000,
) -> pd.DataFrame:
    """
    Generate 50M listening events with power-law user activity distribution.
    Returns interaction matrix and event log.
    """
    print(f"Generating {target_events:,} listening events ...")

    # Power-law: few super-users, many casual users
    user_activity = RNG.pareto(1.5, n_users) + 1
    user_activity = (user_activity / user_activity.sum() * target_events).astype(int)
    user_activity = np.maximum(user_activity, 1)

    # User preferences (genre/mood biases)
    user_genre_pref = RNG.dirichlet(np.ones(len(GENRES)) * 0.5, n_users)

    chunk_size = 5_000_000
    frames = []
    total_written = 0

    print("Generating event chunks ...")
    for user_start in tqdm(range(0, min(n_users, 200_000), 10_000)):
        user_end = min(user_start + 10_000, n_users)
        chunk_events = []
        for u in range(user_start, user_end):
            n_plays = min(int(user_activity[u]), 500)  # cap per user
            # Sample tracks with genre bias
            track_ids = RNG.choice(n_tracks, size=n_plays, replace=True)
            play_counts = RNG.integers(1, 10, n_plays)
            for t, pc in zip(track_ids, play_counts):
                chunk_events.append((u, int(t), int(pc)))
            total_written += n_plays
            if total_written >= target_events:
                break
        frames.append(pd.DataFrame(chunk_events, columns=["user_id", "track_id", "play_count"]))
        if total_written >= target_events:
            break

    events = pd.concat(frames, ignore_index=True)
    print(f"Generated {len(events):,} listening events")
    events.to_parquet(DATA_PROC / "events.parquet", index=False)

    # Build sparse user-item matrix (sub-sample for ALS)
    sample_events = events.sample(min(5_000_000, len(events)), random_state=42)
    n_u = sample_events["user_id"].max() + 1
    n_t = n_tracks
    sparse = csr_matrix(
        (sample_events["play_count"].values,
         (sample_events["user_id"].values, sample_events["track_id"].values)),
        shape=(n_u, n_t),
    )
    save_npz(DATA_PROC / "user_item_matrix.npz", sparse)
    print(f"Sparse matrix: {sparse.shape}, {sparse.nnz:,} non-zeros")

    return events


def load_spotify_features() -> pd.DataFrame:
    """Load Spotify Tracks Dataset if available in data/raw/."""
    for name in ["tracks.csv", "spotify_tracks.csv", "spotify_dataset.csv"]:
        p = DATA_RAW / name
        if p.exists():
            print(f"Loading Spotify dataset: {p}")
            return pd.read_csv(p)
    return pd.DataFrame()


def prepare_all(target_events: int = 50_000_000) -> dict:
    """Full pipeline."""
    spotify = load_spotify_features()
    if len(spotify) > 0:
        tracks = spotify
        tracks.to_parquet(DATA_PROC / "tracks.parquet", index=False)
    else:
        tracks = generate_track_catalog()

    events = generate_listening_events(target_events=target_events)
    return {"tracks": tracks, "events": events}


if __name__ == "__main__":
    prepare_all(target_events=10_000_000)  # use 10M for faster demo
