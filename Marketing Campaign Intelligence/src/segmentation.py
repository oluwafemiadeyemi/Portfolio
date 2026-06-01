"""
Customer segmentation: UMAP dimensionality reduction + HDBSCAN density-based
clustering + Gaussian Mixture Model probabilistic assignment.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import hdbscan

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_NAMES = {
    0: "Champions",
    1: "Loyal Customers",
    2: "Potential Loyalists",
    3: "At-Risk Customers",
    4: "Lost Customers",
    5: "New Customers",
    6: "High-Value Prospects",
    -1: "Noise / Unclustered",
}

SEGMENT_COLORS = {
    "Champions": "#2ECC71",
    "Loyal Customers": "#3498DB",
    "Potential Loyalists": "#9B59B6",
    "At-Risk Customers": "#E74C3C",
    "Lost Customers": "#95A5A6",
    "New Customers": "#F39C12",
    "High-Value Prospects": "#1ABC9C",
    "Noise / Unclustered": "#BDC3C7",
}


class CustomerSegmentationPipeline:
    """
    Full segmentation pipeline:
      1. UMAP: 15-dim feature space → 2-D embedding
      2. HDBSCAN: density-based cluster labels (noise-aware)
      3. GMM: soft probabilistic membership scores
    """

    def __init__(
        self,
        umap_n_components: int = 2,
        umap_n_neighbors: int = 30,
        umap_min_dist: float = 0.1,
        hdbscan_min_cluster_size: int = 500,
        hdbscan_min_samples: int = 50,
        gmm_n_components: int = 6,
    ):
        self.umap_reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=42,
            n_jobs=-1,
        )
        self.hdbscan_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        self.gmm = GaussianMixture(
            n_components=gmm_n_components,
            covariance_type="full",
            random_state=42,
            max_iter=200,
        )
        self.embedding_ = None
        self.labels_ = None
        self.probabilities_ = None
        self.gmm_labels_ = None
        self.gmm_proba_ = None
        self.metrics_ = {}

    def fit_transform(self, X: np.ndarray) -> dict:
        """Fit all stages, return dict of embeddings, labels, metrics."""
        print(f"Running UMAP on {X.shape[0]:,} samples × {X.shape[1]} features ...")
        self.embedding_ = self.umap_reducer.fit_transform(X)

        print("Running HDBSCAN ...")
        self.hdbscan_clusterer.fit(self.embedding_)
        self.labels_ = self.hdbscan_clusterer.labels_
        self.probabilities_ = self.hdbscan_clusterer.probabilities_

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        noise_pct = (self.labels_ == -1).mean() * 100
        print(f"HDBSCAN: {n_clusters} clusters, {noise_pct:.1f}% noise")

        # Silhouette on non-noise points
        mask = self.labels_ != -1
        if mask.sum() > 1000:
            sil = silhouette_score(self.embedding_[mask], self.labels_[mask], sample_size=10000, random_state=42)
            db = davies_bouldin_score(self.embedding_[mask], self.labels_[mask])
            self.metrics_["silhouette"] = round(float(sil), 4)
            self.metrics_["davies_bouldin"] = round(float(db), 4)
            print(f"Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f}")

        print("Fitting Gaussian Mixture Model ...")
        self.gmm.fit(self.embedding_)
        self.gmm_labels_ = self.gmm.predict(self.embedding_)
        self.gmm_proba_ = self.gmm.predict_proba(self.embedding_)

        self.metrics_["n_clusters_hdbscan"] = int(n_clusters)
        self.metrics_["noise_pct"] = round(float(noise_pct), 2)
        self.metrics_["gmm_bic"] = round(float(self.gmm.bic(self.embedding_)), 2)
        self.metrics_["gmm_aic"] = round(float(self.gmm.aic(self.embedding_)), 2)

        return self._package_results()

    def predict(self, X: np.ndarray) -> dict:
        """Predict on new data using fitted UMAP + approximate HDBSCAN + GMM."""
        emb = self.umap_reducer.transform(X)
        hdb_labels, hdb_proba = hdbscan.approximate_predict(self.hdbscan_clusterer, emb)
        gmm_labels = self.gmm.predict(emb)
        gmm_proba = self.gmm.predict_proba(emb)
        return {
            "embedding": emb,
            "hdbscan_labels": hdb_labels,
            "hdbscan_proba": hdb_proba,
            "gmm_labels": gmm_labels,
            "gmm_proba": gmm_proba,
            "segment_names": [_map_label(l) for l in gmm_labels],
        }

    def _package_results(self) -> dict:
        return {
            "embedding": self.embedding_,
            "hdbscan_labels": self.labels_,
            "hdbscan_proba": self.probabilities_,
            "gmm_labels": self.gmm_labels_,
            "gmm_proba": self.gmm_proba_,
            "metrics": self.metrics_,
        }

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "segmentation_pipeline.pkl"
        joblib.dump(self, path)
        print(f"Segmentation pipeline saved to {path}")

    @staticmethod
    def load(path: Path = None) -> "CustomerSegmentationPipeline":
        path = path or MODELS_DIR / "segmentation_pipeline.pkl"
        return joblib.load(path)


def _map_label(label: int) -> str:
    return SEGMENT_NAMES.get(int(label), f"Segment {label}")


def build_segment_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels to original df and compute per-segment statistics."""
    profile_df = df.copy()
    profile_df["segment_id"] = labels
    profile_df["segment_name"] = [_map_label(l) for l in labels]

    numeric_cols = ["age", "income", "recency_days", "frequency", "monetary",
                    "clicks", "conversions", "email_open_rate", "nps_score",
                    "web_visits_30d", "app_sessions_30d"]
    available = [c for c in numeric_cols if c in profile_df.columns]

    profiles = profile_df.groupby("segment_name")[available].agg(["mean", "median", "std"]).round(2)
    profiles.columns = ["_".join(c) for c in profiles.columns]
    profiles["count"] = profile_df.groupby("segment_name").size()
    profiles["pct_of_total"] = (profiles["count"] / len(profile_df) * 100).round(2)

    return profiles


def run_segmentation(sample_size: int = 200_000) -> dict:
    """End-to-end: load features, fit pipeline, save artifacts."""
    path = DATA_PROC / "sample_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run data_pipeline.prepare_all() first. Missing: {path}")

    X = pd.read_parquet(path).values
    if sample_size and sample_size < len(X):
        idx = np.random.default_rng(42).choice(len(X), size=sample_size, replace=False)
        X = X[idx]
    print(f"Fitting segmentation on {len(X):,} rows ...")

    pipe = CustomerSegmentationPipeline()
    results = pipe.fit_transform(X)
    pipe.save()

    # Save embedding + labels
    emb_df = pd.DataFrame(results["embedding"], columns=["umap_x", "umap_y"])
    emb_df["hdbscan_label"] = results["hdbscan_labels"]
    emb_df["gmm_label"] = results["gmm_labels"]
    emb_df["segment_name"] = [_map_label(l) for l in results["gmm_labels"]]
    emb_df.to_parquet(DATA_PROC / "segment_embeddings.parquet", index=False)

    print("\nSegmentation complete.")
    print(f"  Metrics: {results['metrics']}")
    return results


if __name__ == "__main__":
    run_segmentation()
