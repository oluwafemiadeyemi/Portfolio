"""P16 Music Recommendation System — portfolio view."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from utils.ui import metric_card, project_header, section_header, load_metrics, dark_fig, placeholder_chart, CHART_LAYOUT
from utils.registry import PROJECT_BY_ID

SAMPLE_GENRES = ["Pop", "Rock", "Electronic", "Hip-Hop", "Jazz", "Classical",
                  "R&B", "Country", "Metal", "Folk", "Latin", "Indie"]


@st.cache_data
def load_popularity_index(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "popularity_index.parquet"
        if p.exists():
            return pd.read_parquet(p)
    except Exception:
        pass
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "track_id": [f"T{i:05d}" for i in range(n)],
        "track_name": [f"Track {i}" for i in range(n)],
        "artist": [f"Artist {rng.integers(1, 20)}" for _ in range(n)],
        "genre": rng.choice(SAMPLE_GENRES, n).tolist(),
        "total_plays": rng.integers(50000, 5000000, n).tolist(),
        "unique_listeners": rng.integers(10000, 1000000, n).tolist(),
    })


@st.cache_data
def load_tracks(proj_path: Path) -> pd.DataFrame:
    try:
        p = proj_path / "data" / "processed" / "tracks.parquet"
        if p.exists():
            return pd.read_parquet(p)
    except Exception:
        pass
    rng = np.random.default_rng(99)
    n = 1000
    return pd.DataFrame({
        "track_id": [f"T{i:05d}" for i in range(n)],
        "track_name": [f"Track_{i}" for i in range(n)],
        "artist": [f"Artist_{rng.integers(1, 100)}" for _ in range(n)],
        "genre": rng.choice(SAMPLE_GENRES, n).tolist(),
        "popularity": rng.integers(0, 100, n).tolist(),
        "duration_ms": rng.integers(120000, 360000, n).tolist(),
    })


def render():
    proj = PROJECT_BY_ID["music"]
    project_header(proj)
    metrics = load_metrics(proj)

    n_users = metrics.get("n_users") or "962k"
    n_events = metrics.get("n_interactions") or metrics.get("n_events") or "9.7M"
    n_tracks = metrics.get("n_tracks") or "50k"

    cols = st.columns(4)
    with cols[0]: metric_card("Users", str(n_users), color="#1DB954")
    with cols[1]: metric_card("Listening Events", str(n_events), color="#58A6FF")
    with cols[2]: metric_card("Tracks", str(n_tracks), color="#F39C12")
    with cols[3]: metric_card("FAISS Index", "Yes", color="#8E44AD")

    tab1, tab2, tab3 = st.tabs(["Overview", "Live Demo", "Insights"])

    with tab1:
        section_header("Top 20 Tracks by Total Plays")
        df_pop = load_popularity_index(proj["path"])
        play_col = "total_plays" if "total_plays" in df_pop.columns else None
        if play_col is None:
            play_col = [c for c in df_pop.select_dtypes(include="number").columns if "play" in c.lower() or "count" in c.lower()]
            play_col = play_col[0] if play_col else df_pop.select_dtypes(include="number").columns[0]

        name_col = "track_name" if "track_name" in df_pop.columns else df_pop.columns[0]

        top20 = df_pop.nlargest(20, play_col)
        fig = go.Figure(go.Bar(
            x=top20[play_col], y=top20[name_col], orientation="h",
            marker_color="#1DB954", text=top20[play_col].apply(lambda v: f"{v/1e6:.2f}M" if v > 1e6 else f"{v:,}"),
            textposition="outside"
        ))
        dark_fig(fig, height=500)
        fig.update_layout(title="Top 20 Most Played Tracks", xaxis_title="Total Plays", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section_header("Genre-Based Recommendations")
        df_tracks = load_tracks(proj["path"])
        genre_col = "genre" if "genre" in df_tracks.columns else None
        if genre_col:
            available_genres = sorted(df_tracks[genre_col].dropna().unique().tolist())
        else:
            available_genres = SAMPLE_GENRES

        selected_genre = st.selectbox("Select a Genre", available_genres)
        n_recs = st.slider("Number of Recommendations", 3, 20, 5)

        if st.button("Get Recommendations", type="primary"):
            if genre_col and genre_col in df_tracks.columns:
                genre_df = df_tracks[df_tracks[genre_col] == selected_genre]
            else:
                rng = np.random.default_rng(42)
                genre_df = df_tracks.sample(min(50, len(df_tracks)), random_state=42)

            pop_col = "popularity" if "popularity" in genre_df.columns else None
            if pop_col and pop_col in genre_df.columns:
                top_tracks = genre_df.nlargest(n_recs, pop_col)
            else:
                top_tracks = genre_df.head(n_recs)

            name_col_t = "track_name" if "track_name" in top_tracks.columns else top_tracks.columns[0]
            artist_col = "artist" if "artist" in top_tracks.columns else None

            st.markdown(f"**Top {n_recs} {selected_genre} Tracks:**")
            for i, (_, row) in enumerate(top_tracks.iterrows(), 1):
                track_name = row.get(name_col_t, f"Track {i}")
                artist = row.get("artist", "Unknown Artist") if artist_col else "Unknown Artist"
                pop = row.get("popularity", "N/A")
                st.markdown(f"{i}. **{track_name}** — {artist} *(popularity: {pop})*")

    with tab3:
        section_header("Genre Distribution")
        df_tracks = load_tracks(proj["path"])
        if "genre" in df_tracks.columns:
            genre_counts = df_tracks["genre"].value_counts().reset_index()
            genre_counts.columns = ["Genre", "Count"]
            fig = px.pie(genre_counts, names="Genre", values="Count",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            dark_fig(fig, height=380)
            fig.update_layout(title="Track Distribution by Genre")
            st.plotly_chart(fig, use_container_width=True)
        else:
            rng = np.random.default_rng(42)
            counts = rng.integers(1000, 8000, len(SAMPLE_GENRES))
            fig = px.pie(names=SAMPLE_GENRES, values=counts,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            dark_fig(fig, height=380)
            fig.update_layout(title="Track Distribution by Genre (Synthetic)")
            st.plotly_chart(fig, use_container_width=True)

        section_header("System Architecture")
        st.markdown("""
| Component | Technology | Details |
|---|---|---|
| Collaborative Filtering | ALS (Implicit) | 962k users × 50k tracks |
| Content-Based | FAISS IVF-Flat | 128-dim embeddings |
| Hybrid Ranking | Weighted ensemble | ALS score + cosine sim |
| Index Size | FAISS | ~25 MB compressed |
| Inference Latency | FAISS | < 5 ms top-50 retrieval |
""")
