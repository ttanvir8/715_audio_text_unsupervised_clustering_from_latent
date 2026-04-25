from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[easy_pca_baseline] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare easy VAE latent K-Means against a PCA + K-Means baseline "
            "using Silhouette Score and Calinski-Harabasz Index."
        )
    )
    parser.add_argument(
        "--latent-root",
        default="0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001",
        help="Root containing validation/full latent exports from 0_2_easy_inference.",
    )
    parser.add_argument(
        "--output-root",
        default="0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001",
        help="Root where comparison outputs will be written.",
    )
    parser.add_argument(
        "--lyrics-path",
        default="0_data_pre_processing/processed_dataset/lyrics_e5_large/lyrics_embeddings.parquet",
        help="Parquet file containing the original lyrics embeddings for the PCA baseline.",
    )
    parser.add_argument(
        "--lyrics-embedding-column",
        default="lyrics_e5_large_embedding",
        help="Original lyrics embedding column used by the PCA baseline.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which latent scope to compare.",
    )
    parser.add_argument(
        "--embedding",
        default="mu",
        choices=("mu", "logvar"),
        help="VAE latent array to compare against the PCA baseline.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
        help="PCA dimensions for the baseline before K-Means.",
    )
    parser.add_argument(
        "--language-k",
        type=int,
        default=0,
        help="Cluster count. 0 means number of observed language labels.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed.")
    parser.add_argument("--n-init", type=int, default=50, help="KMeans initializations.")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path)
    else:
        df.to_csv(output_path, index=False)


def load_latent_scope(scope_dir: Path, embedding: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    values = np.load(scope_dir / f"{embedding}.npy").astype(np.float32, copy=False)
    row_indices = np.load(scope_dir / "row_indices.npy").astype(np.int64, copy=False)
    labels = pq.read_table(scope_dir / "labels.parquet").to_pandas()

    if values.shape[0] != len(row_indices) or values.shape[0] != len(labels):
        raise ValueError(
            f"Latent rows, row indices, and labels must align in {scope_dir}: "
            f"{values.shape[0]}, {len(row_indices)}, {len(labels)}"
        )
    if "row_index" in labels.columns and not np.array_equal(labels["row_index"].to_numpy(), row_indices):
        raise ValueError(f"labels.parquet row_index does not match row_indices.npy in {scope_dir}")
    return values, row_indices, labels


def load_lyrics_embeddings(
    lyrics_path: Path,
    embedding_column: str,
    row_indices: np.ndarray,
) -> np.ndarray:
    table = pq.read_table(lyrics_path, columns=[embedding_column])
    selected = table.take(pa.array(row_indices, type=pa.int64()))
    values = np.asarray(selected[embedding_column].to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix from {embedding_column}, got {values.shape}")
    return values


def standardized(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values).astype(np.float32, copy=False)


def pca_features(values: np.ndarray, components: int, seed: int) -> tuple[np.ndarray, PCA]:
    max_components = min(values.shape[0], values.shape[1])
    component_count = min(int(components), max_components)
    if component_count < 1:
        raise ValueError("pca-components must be at least 1.")
    pca = PCA(n_components=component_count, random_state=seed)
    features = pca.fit_transform(values).astype(np.float32, copy=False)
    return features, pca


def fit_kmeans(values: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    if k < 2:
        raise ValueError("k must be at least 2.")
    if k > values.shape[0]:
        raise ValueError(f"k={k} exceeds sample count={values.shape[0]}.")
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    return model.fit_predict(values).astype(np.int64, copy=False)


def safe_silhouette(values: np.ndarray, clusters: np.ndarray) -> float:
    unique_count = len(np.unique(clusters))
    if unique_count < 2 or unique_count >= values.shape[0]:
        return float("nan")
    return float(silhouette_score(values, clusters))


def safe_calinski_harabasz(values: np.ndarray, clusters: np.ndarray) -> float:
    unique_count = len(np.unique(clusters))
    if unique_count < 2 or unique_count >= values.shape[0]:
        return float("nan")
    return float(calinski_harabasz_score(values, clusters))


def metric_row(
    scope: str,
    method: str,
    feature_source: str,
    values: np.ndarray,
    clusters: np.ndarray,
    k: int,
    explained_variance_ratio: float | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "method": method,
        "feature_source": feature_source,
        "k": int(k),
        "sample_count": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "cluster_count": int(len(np.unique(clusters))),
        "silhouette": safe_silhouette(values, clusters),
        "calinski_harabasz": safe_calinski_harabasz(values, clusters),
        "pca_explained_variance_ratio_sum": explained_variance_ratio,
    }


def run_scope(args: argparse.Namespace, scope: str) -> None:
    latent_scope_dir = project_path(args.latent_root) / scope
    output_dir = project_path(args.output_root) / scope
    lyrics_path = project_path(args.lyrics_path)

    vae_values, row_indices, labels = load_latent_scope(latent_scope_dir, args.embedding)
    language_names = sorted(labels["music_lang"].astype(str).unique().tolist())
    k = int(args.language_k) if args.language_k else len(language_names)

    log(
        f"Comparing {scope}: rows={vae_values.shape[0]}, "
        f"vae_dim={vae_values.shape[1]}, languages={language_names}, k={k}"
    )

    vae_features = standardized(vae_values)
    vae_clusters = fit_kmeans(vae_features, k, args.seed, args.n_init)

    lyrics_values = load_lyrics_embeddings(
        lyrics_path=lyrics_path,
        embedding_column=args.lyrics_embedding_column,
        row_indices=row_indices,
    )
    lyrics_features = standardized(lyrics_values)
    baseline_features, pca = pca_features(lyrics_features, args.pca_components, args.seed)
    baseline_clusters = fit_kmeans(baseline_features, k, args.seed, args.n_init)

    comparison = pd.DataFrame(
        [
            metric_row(
                scope=scope,
                method=f"vae_{args.embedding}_kmeans",
                feature_source=f"standardized_{args.embedding}",
                values=vae_features,
                clusters=vae_clusters,
                k=k,
            ),
            metric_row(
                scope=scope,
                method="pca_lyrics_kmeans",
                feature_source=f"standardized_{args.lyrics_embedding_column}_pca",
                values=baseline_features,
                clusters=baseline_clusters,
                k=k,
                explained_variance_ratio=float(np.sum(pca.explained_variance_ratio_)),
            ),
        ]
    )

    assignments = labels[["row_index", "music_id", "music_name", "music_lang"]].copy()
    assignments[f"vae_{args.embedding}_kmeans_k{k}"] = vae_clusters
    assignments[f"pca_lyrics_kmeans_k{k}"] = baseline_clusters

    write_dataframe(comparison, output_dir / "pca_kmeans_baseline_comparison.csv")
    write_dataframe(assignments, output_dir / "pca_kmeans_baseline_assignments.parquet")
    save_json(
        {
            "latent_scope_dir": str(latent_scope_dir),
            "lyrics_path": str(lyrics_path),
            "output_dir": str(output_dir),
            "scope": scope,
            "vae_embedding": args.embedding,
            "lyrics_embedding_column": args.lyrics_embedding_column,
            "pca_components_requested": int(args.pca_components),
            "pca_components_used": int(baseline_features.shape[1]),
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "language_names": language_names,
            "k": int(k),
            "seed": int(args.seed),
            "n_init": int(args.n_init),
            "outputs": [
                "pca_kmeans_baseline_comparison.csv",
                "pca_kmeans_baseline_assignments.parquet",
                "pca_kmeans_baseline_metadata.json",
            ],
        },
        output_dir / "pca_kmeans_baseline_metadata.json",
    )
    log(f"Saved {scope} PCA baseline comparison: {output_dir}")


def main() -> None:
    args = parse_args()
    for scope in scopes_from_arg(args.scope):
        run_scope(args, scope)
    log(f"PCA baseline comparison complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
