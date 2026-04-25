from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_ROOT = PROJECT_ROOT / "01_3_mid_clustering" / "extra_helper_functions"
if str(HELPER_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPER_ROOT))

from mid_clustering_utils import (  # noqa: E402
    fit_kmeans,
    load_latent_scope,
    observed_labels,
    parse_int_csv_list,
    project_path,
    purity_score,
    safe_davies_bouldin,
    safe_silhouette,
    save_json,
    standardized,
    write_dataframe,
)


def log(message: str) -> None:
    print(f"[mid_cluster_algorithms] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare K-Means, Agglomerative, and DBSCAN on medium audio+lyrics VAE latents."
        )
    )
    parser.add_argument(
        "--latent-root",
        default="01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001",
        help="Root containing validation/full latent exports from 01_2_inference.",
    )
    parser.add_argument(
        "--output-root",
        default="01_3_mid_clustering/clustering_outputs/audio_lyrics_vae_run_001",
        help="Root where algorithm-comparison outputs will be written.",
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
        help="Latent array to compare across clustering algorithms.",
    )
    parser.add_argument(
        "--label-column",
        default="music_lang",
        help="Target label column used for evaluation.",
    )
    parser.add_argument(
        "--label-id-column",
        default="music_lang_id",
        help="Integer id column aligned with --label-column. Falls back to factorization when missing.",
    )
    parser.add_argument(
        "--target-name",
        default="language",
        help="Display name for the evaluation target.",
    )
    parser.add_argument(
        "--cluster-k",
        type=int,
        default=0,
        help="Cluster count for K-Means and Agglomerative. 0 means number of observed labels.",
    )
    parser.add_argument(
        "--agglomerative-linkage",
        default="ward",
        choices=("ward", "average", "complete", "single"),
        help="Linkage used for Agglomerative clustering.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        default="5,10",
        help="Comma-separated min_samples values explored for DBSCAN.",
    )
    parser.add_argument(
        "--dbscan-percentiles",
        default="50,60,70,80,85,90,95",
        help="Percentiles of k-nearest-neighbour distances used to derive DBSCAN eps candidates.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed for K-Means.")
    parser.add_argument("--n-init", type=int, default=50, help="KMeans initializations.")
    return parser.parse_args()


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def assignment_base_columns(labels: pd.DataFrame, label_column: str, label_id_column: str | None) -> list[str]:
    desired = [
        "row_index",
        "music_id",
        "music_name",
        "art_id",
        "art_name",
        "music_lang",
        "music_lang_id",
        "main_genre",
        "main_genre_id",
    ]
    if label_column not in desired:
        desired.append(label_column)
    if label_id_column and label_id_column not in desired and label_id_column in labels.columns:
        desired.append(label_id_column)
    return [column for column in desired if column in labels.columns]


def metric_row(
    *,
    scope: str,
    algorithm: str,
    values: np.ndarray,
    clusters: np.ndarray,
    target_ids: np.ndarray,
    target_name: str,
    label_column: str,
    feature_name: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unique_labels = np.unique(clusters)
    non_noise_unique = unique_labels[unique_labels != -1]
    noise_count = int(np.sum(clusters == -1))
    noise_fraction = float(noise_count / len(clusters)) if len(clusters) else float("nan")
    has_enough_clusters = len(non_noise_unique) >= 2

    row: dict[str, Any] = {
        "scope": scope,
        "feature_name": feature_name,
        "algorithm": algorithm,
        "target": target_name,
        "label_column": label_column,
        "sample_count": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "cluster_count": int(len(unique_labels)),
        "cluster_count_excluding_noise": int(len(non_noise_unique)),
        "noise_count": noise_count,
        "noise_fraction": noise_fraction,
        "silhouette": safe_silhouette(values, clusters) if has_enough_clusters else float("nan"),
        "davies_bouldin": safe_davies_bouldin(values, clusters) if has_enough_clusters else float("nan"),
        "nmi": float(normalized_mutual_info_score(target_ids, clusters)),
        "ari": float(adjusted_rand_score(target_ids, clusters)),
        "purity": purity_score(target_ids, clusters),
    }
    if params:
        row.update(params)
    return row


def dbscan_eps_candidates(
    values: np.ndarray,
    min_samples_values: list[int],
    percentile_values: list[int],
) -> list[tuple[float, int, int]]:
    max_neighbors = max(min_samples_values)
    neighbors = NearestNeighbors(n_neighbors=max_neighbors, metric="euclidean")
    neighbors.fit(values)
    distances, _ = neighbors.kneighbors(values)

    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[float, int, int]] = []
    for min_samples in min_samples_values:
        kth_distances = distances[:, min_samples - 1]
        for percentile in percentile_values:
            eps = float(np.percentile(kth_distances, percentile))
            key = (min_samples, int(round(eps * 1_000_000)))
            if eps <= 0.0 or key in seen:
                continue
            seen.add(key)
            candidates.append((eps, min_samples, percentile))
    return candidates


def dbscan_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    silhouette = row["silhouette"]
    davies_bouldin = row["davies_bouldin"]
    return (
        float("-inf") if np.isnan(silhouette) else float(silhouette),
        float("-inf") if np.isnan(row["nmi"]) else float(row["nmi"]),
        float("-inf") if np.isnan(row["ari"]) else float(row["ari"]),
        float("-inf") if np.isnan(row["purity"]) else float(row["purity"]),
        float("inf") if np.isnan(davies_bouldin) else -float(davies_bouldin),
    )


def best_dbscan(
    *,
    values: np.ndarray,
    scope: str,
    target_ids: np.ndarray,
    target_name: str,
    label_column: str,
    embedding: str,
    percentile_values: list[int],
    min_samples_values: list[int],
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    candidates = dbscan_eps_candidates(values, min_samples_values, percentile_values)
    if not candidates:
        raise ValueError("No DBSCAN candidates were generated.")

    search_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_clusters: np.ndarray | None = None

    for eps, min_samples, percentile in candidates:
        clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1).fit_predict(values)
        row = metric_row(
            scope=scope,
            algorithm="dbscan",
            values=values,
            clusters=clusters,
            target_ids=target_ids,
            target_name=target_name,
            label_column=label_column,
            feature_name=embedding,
            params={
                "dbscan_eps": float(eps),
                "dbscan_min_samples": int(min_samples),
                "dbscan_percentile": int(percentile),
            },
        )
        search_rows.append(row)
        if best_row is None or dbscan_sort_key(row) > dbscan_sort_key(best_row):
            best_row = row
            best_clusters = clusters.astype(np.int64, copy=False)

    assert best_row is not None
    assert best_clusters is not None
    return best_clusters, best_row, pd.DataFrame(search_rows)


def run_scope(args: argparse.Namespace, scope: str) -> None:
    latent_scope_dir = project_path(args.latent_root) / scope
    output_dir = project_path(args.output_root) / scope

    values, _, labels = load_latent_scope(latent_scope_dir, args.embedding)
    values_std = standardized(values)
    target_ids, target_names = observed_labels(labels, args.label_column, args.label_id_column)
    cluster_k = int(args.cluster_k) if args.cluster_k else int(len(target_names))
    percentile_values = parse_int_csv_list(args.dbscan_percentiles)
    min_samples_values = parse_int_csv_list(args.dbscan_min_samples)

    log(
        f"Comparing algorithms for {scope}: rows={values.shape[0]}, dim={values.shape[1]}, "
        f"target={args.label_column}, labels={len(target_names)}, k={cluster_k}"
    )

    kmeans_clusters = fit_kmeans(values_std, cluster_k, args.seed, args.n_init)
    agglomerative_clusters = AgglomerativeClustering(
        n_clusters=cluster_k,
        linkage=args.agglomerative_linkage,
        metric="euclidean",
    ).fit_predict(values_std).astype(np.int64, copy=False)
    dbscan_clusters, dbscan_row, dbscan_search = best_dbscan(
        values=values_std,
        scope=scope,
        target_ids=target_ids,
        target_name=args.target_name,
        label_column=args.label_column,
        embedding=args.embedding,
        percentile_values=percentile_values,
        min_samples_values=min_samples_values,
    )

    comparison = pd.DataFrame(
        [
            metric_row(
                scope=scope,
                algorithm="kmeans",
                values=values_std,
                clusters=kmeans_clusters,
                target_ids=target_ids,
                target_name=args.target_name,
                label_column=args.label_column,
                feature_name=args.embedding,
                params={"k": int(cluster_k)},
            ),
            metric_row(
                scope=scope,
                algorithm=f"agglomerative_{args.agglomerative_linkage}",
                values=values_std,
                clusters=agglomerative_clusters,
                target_ids=target_ids,
                target_name=args.target_name,
                label_column=args.label_column,
                feature_name=args.embedding,
                params={"k": int(cluster_k)},
            ),
            {
                **dbscan_row,
                "k": np.nan,
            },
        ]
    )

    assignments = labels[assignment_base_columns(labels, args.label_column, args.label_id_column)].copy()
    assignments[f"kmeans__{args.embedding}__{args.target_name}_k{cluster_k}"] = kmeans_clusters
    assignments[
        f"agglomerative_{args.agglomerative_linkage}__{args.embedding}__{args.target_name}_k{cluster_k}"
    ] = agglomerative_clusters
    assignments[f"dbscan_best__{args.embedding}__{args.target_name}"] = dbscan_clusters

    write_dataframe(comparison, output_dir / "clustering_algorithm_comparison.csv")
    write_dataframe(assignments, output_dir / "clustering_algorithm_assignments.parquet")
    write_dataframe(dbscan_search, output_dir / "dbscan_parameter_sweep.csv")
    save_json(
        {
            "latent_scope_dir": str(latent_scope_dir),
            "output_dir": str(output_dir),
            "scope": scope,
            "embedding": args.embedding,
            "label_column": args.label_column,
            "label_id_column": args.label_id_column,
            "target_name": args.target_name,
            "target_labels": list(target_names),
            "cluster_k": int(cluster_k),
            "agglomerative_linkage": args.agglomerative_linkage,
            "dbscan_min_samples": min_samples_values,
            "dbscan_percentiles": percentile_values,
            "best_dbscan": {
                "dbscan_eps": float(dbscan_row["dbscan_eps"]),
                "dbscan_min_samples": int(dbscan_row["dbscan_min_samples"]),
                "dbscan_percentile": int(dbscan_row["dbscan_percentile"]),
                "cluster_count": int(dbscan_row["cluster_count"]),
                "cluster_count_excluding_noise": int(dbscan_row["cluster_count_excluding_noise"]),
                "noise_fraction": float(dbscan_row["noise_fraction"]),
            },
            "seed": int(args.seed),
            "n_init": int(args.n_init),
            "value_shape": list(values.shape),
            "standardized": True,
        },
        output_dir / "clustering_algorithm_metadata.json",
    )
    log(
        f"Saved algorithm comparison for {scope} to {output_dir}: "
        f"K-Means, Agglomerative, and tuned DBSCAN"
    )


def main() -> None:
    args = parse_args()
    for scope in scopes_from_arg(args.scope):
        run_scope(args, scope)


if __name__ == "__main__":
    main()
