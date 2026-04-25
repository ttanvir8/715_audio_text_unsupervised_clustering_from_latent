from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_ROOT = PROJECT_ROOT / "3_clustering" / "extra_helper_functions"
if str(HELPER_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPER_ROOT))

from clustering_utils import (  # noqa: E402
    assert_cuml_cuda_available,
    autoencoder_embedding,
    cluster_label_summary,
    cuml_device_summary,
    fit_predict,
    load_latent_scope,
    load_spectral_features,
    load_tabular_vectors,
    parse_csv_list,
    parse_int_csv_list,
    pca_embedding,
    save_json,
    standardized,
    supervised_metrics,
    safe_silhouette,
    write_dataframe,
)


def log(message: str) -> None:
    print(f"[cluster_latents] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-modal clustering on exported VAE encoder latents."
    )
    parser.add_argument(
        "--latent-root",
        required=True,
        help="Root containing validation/full latent exports from 2_inference.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root where clustering assignments and metrics will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which latent scope to cluster.",
    )
    parser.add_argument(
        "--embeddings",
        default="mu,metadata_assisted",
        help="Comma-separated embedding names: mu,metadata_assisted.",
    )
    parser.add_argument(
        "--algorithms",
        default="",
        help=(
            "Optional generic algorithms to run on --embeddings: cuml_kmeans,cuda_kmeans,kmeans,"
            "cuml_agglomerative,agglomerative,cuml_dbscan,dbscan,gmm."
        ),
    )
    parser.add_argument(
        "--comparison-methods",
        default=(
            "pca_kmeans,autoencoder_kmeans,spectral_feature_kmeans,"
            "vae_mu_kmeans,vae_metadata_assisted_kmeans"
        ),
        help=(
            "Comma-separated AGENTS.md comparison methods: pca_kmeans,"
            "autoencoder_kmeans,spectral_feature_kmeans,vae_mu_kmeans,"
            "vae_metadata_assisted_kmeans."
        ),
    )
    parser.add_argument(
        "--comparison-kmeans-algorithm",
        default="cuml_kmeans",
        choices=("cuml_kmeans", "cuda_kmeans", "kmeans"),
        help="K-Means backend used by comparison methods.",
    )
    parser.add_argument(
        "--model-input-path",
        default="0_data_pre_processing/processed_dataset/model_input/model_input_dataset.parquet",
        help="Processed model input parquet used for PCA/autoencoder baseline features.",
    )
    parser.add_argument(
        "--mel-tensor-dir",
        default="0_data_pre_processing/processed_dataset/mel_tensors",
        help="Mel tensor directory used for direct spectral feature clustering.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
        help="PCA dimensions for pca_kmeans.",
    )
    parser.add_argument(
        "--autoencoder-latent-dim",
        type=int,
        default=64,
        help="Latent dimensions for autoencoder_kmeans.",
    )
    parser.add_argument(
        "--autoencoder-epochs",
        type=int,
        default=40,
        help="Training epochs for the lightweight feature autoencoder baseline.",
    )
    parser.add_argument(
        "--autoencoder-batch-size",
        type=int,
        default=256,
        help="Batch size for the lightweight feature autoencoder baseline.",
    )
    parser.add_argument(
        "--autoencoder-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the lightweight feature autoencoder baseline.",
    )
    parser.add_argument(
        "--autoencoder-device",
        default="auto",
        help="Device for autoencoder_kmeans representation training: auto,cpu,cuda,cuda:0.",
    )
    parser.add_argument(
        "--spectral-time-bins",
        type=int,
        default=32,
        help="Number of pooled time bins included in compact spectral features.",
    )
    parser.add_argument(
        "--cuda-kmeans-device",
        default="cuda",
        help="Torch device used by cuda_kmeans.",
    )
    parser.add_argument(
        "--cuda-kmeans-n-init",
        type=int,
        default=20,
        help="Number of CUDA KMeans initializations.",
    )
    parser.add_argument(
        "--cuda-kmeans-max-iter",
        type=int,
        default=100,
        help="Maximum CUDA KMeans iterations per initialization.",
    )
    parser.add_argument(
        "--cuda-kmeans-tol",
        type=float,
        default=1e-4,
        help="CUDA KMeans convergence tolerance.",
    )
    parser.add_argument(
        "--cuda-kmeans-batch-size",
        type=int,
        default=0,
        help="Optional distance-computation batch size for CUDA KMeans; 0 computes all distances at once.",
    )
    parser.add_argument("--genre-k", type=int, default=51, help="Cluster count for genre metrics.")
    parser.add_argument("--language-k", type=int, default=4, help="Cluster count for language metrics.")
    parser.add_argument(
        "--agglomerative-linkage",
        default="single",
        choices=("single", "complete", "average", "ward"),
        help="Linkage used by agglomerative clustering. cuML currently supports single linkage.",
    )
    parser.add_argument("--dbscan-eps", type=float, default=0.5, help="DBSCAN epsilon radius.")
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=5,
        help="DBSCAN minimum samples for a core point.",
    )
    parser.add_argument(
        "--sweep-k",
        default="16,32,51,64",
        help="Comma-separated k values for unsupervised silhouette sweep.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5000,
        help="Sample size for silhouette on large scopes; use 0 for exact full-data silhouette.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed for clustering.")
    return parser.parse_args()


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def target_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "target": "genre",
            "label_column": "main_genre_id",
            "display_column": "main_genre",
            "k": int(args.genre_k),
        },
        {
            "target": "language",
            "label_column": "music_lang_id",
            "display_column": "music_lang",
            "k": int(args.language_k),
        },
    ]


def assignment_name(algorithm: str, embedding_name: str, k: int) -> str:
    return f"{algorithm}__{embedding_name}__k{k}"


def density_assignment_name(
    algorithm: str,
    embedding_name: str,
    eps: float,
    min_samples: int,
) -> str:
    eps_token = f"{eps:g}".replace(".", "p").replace("-", "m")
    return f"{algorithm}__{embedding_name}__eps{eps_token}__min{min_samples}"


def is_density_algorithm(algorithm: str) -> bool:
    return algorithm in {"dbscan", "cuml_dbscan"}


def cluster_count(clusters: np.ndarray, exclude_noise: bool = False) -> int:
    unique = set(np.unique(clusters).tolist())
    if exclude_noise:
        unique.discard(-1)
    return len(unique)


def noise_count(clusters: np.ndarray) -> int:
    return int(np.count_nonzero(clusters == -1))


COMPARISON_METHODS = {
    "pca_kmeans",
    "autoencoder_kmeans",
    "spectral_feature_kmeans",
    "vae_mu_kmeans",
    "vae_metadata_assisted_kmeans",
}

TABULAR_BASELINE_COLUMNS = [
    "lyrics_e5_large_embedding",
    "main_genre_embedding",
    "music_lang_embedding",
    "condition_embedding",
]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def needs_cuml(algorithms: list[str], comparison_methods: list[str], kmeans_backend: str) -> bool:
    if any(algorithm.startswith("cuml_") for algorithm in algorithms):
        return True
    return bool(comparison_methods) and kmeans_backend.startswith("cuml_")


def append_cluster_evaluation(
    *,
    scope: str,
    embedding_name: str,
    algorithm_name: str,
    assignment_column: str,
    clusters: np.ndarray,
    values: np.ndarray,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    metrics_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    summary_frames_genre: list[pd.DataFrame],
    summary_frames_language: list[pd.DataFrame],
    args: argparse.Namespace,
    k: int,
    kmeans_backend: str | None = None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int | None = None,
) -> None:
    assignments[assignment_column] = clusters.astype(np.int64)
    base_row = {
        "scope": scope,
        "embedding": embedding_name,
        "algorithm": algorithm_name,
        "k": int(k),
        "sample_count": int(values.shape[0]),
        "cluster_count": int(cluster_count(clusters)),
        "noise_count": int(noise_count(clusters)),
        "kmeans_backend": kmeans_backend,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
    }
    sweep_row = dict(base_row)
    sweep_row.update(
        {
            "silhouette": safe_silhouette(
                values,
                clusters,
                args.silhouette_sample_size,
                args.seed,
            ),
            "silhouette_sample_size": int(args.silhouette_sample_size),
        }
    )
    sweep_rows.append(sweep_row)

    for spec in target_specs(args):
        target_values = labels[spec["label_column"]].to_numpy()
        row = dict(base_row)
        row.update(
            {
                "target": spec["target"],
                "label_column": spec["label_column"],
            }
        )
        row.update(
            supervised_metrics(
                values,
                clusters,
                target_values,
                args.silhouette_sample_size,
                args.seed,
            )
        )
        row["silhouette_sample_size"] = int(args.silhouette_sample_size)
        metrics_rows.append(row)

        summary = cluster_label_summary(assignments, assignment_column, spec["display_column"])
        if spec["target"] == "genre":
            summary_frames_genre.append(summary)
        else:
            summary_frames_language.append(summary)


def get_spectral_values(
    cache: dict[str, np.ndarray],
    row_indices: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if "spectral" not in cache:
        log(
            "Loading compact spectral features from mel tensors: "
            f"rows={len(row_indices)}, time_bins={args.spectral_time_bins}"
        )
        cache["spectral"] = load_spectral_features(
            args.mel_tensor_dir,
            row_indices,
            time_bins=args.spectral_time_bins,
        )
        log(f"Loaded spectral features: shape={cache['spectral'].shape}")
    return cache["spectral"]


def get_raw_multimodal_values(
    cache: dict[str, np.ndarray],
    row_indices: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if "raw_multimodal" not in cache:
        spectral = get_spectral_values(cache, row_indices, args)
        log(
            "Loading tabular multimodal features: "
            f"columns={','.join(TABULAR_BASELINE_COLUMNS)}"
        )
        tabular = load_tabular_vectors(
            args.model_input_path,
            row_indices,
            TABULAR_BASELINE_COLUMNS,
        )
        cache["raw_multimodal"] = np.concatenate([spectral, tabular], axis=1).astype(
            np.float32,
            copy=False,
        )
        log(f"Loaded raw multimodal baseline features: shape={cache['raw_multimodal'].shape}")
    return cache["raw_multimodal"]


def comparison_values(
    method: str,
    embeddings: dict[str, np.ndarray],
    row_indices: np.ndarray,
    cache: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[str, np.ndarray]:
    if method == "vae_mu_kmeans":
        return "vae_mu", standardized(embeddings["mu"]).astype(np.float32, copy=False)
    if method == "vae_metadata_assisted_kmeans":
        return "vae_metadata_assisted", standardized(embeddings["metadata_assisted"]).astype(
            np.float32,
            copy=False,
        )
    if method == "spectral_feature_kmeans":
        return "spectral_features", standardized(get_spectral_values(cache, row_indices, args)).astype(
            np.float32,
            copy=False,
        )
    if method == "pca_kmeans":
        if "pca_multimodal" not in cache:
            raw = get_raw_multimodal_values(cache, row_indices, args)
            log(f"Building PCA baseline embedding: components={args.pca_components}")
            cache["pca_multimodal"] = pca_embedding(raw, args.pca_components, args.seed)
            log(f"Built PCA baseline embedding: shape={cache['pca_multimodal'].shape}")
        return "pca_multimodal", cache["pca_multimodal"]
    if method == "autoencoder_kmeans":
        if "autoencoder_multimodal" not in cache:
            raw = get_raw_multimodal_values(cache, row_indices, args)
            log(
                "Training autoencoder baseline embedding: "
                f"latent_dim={args.autoencoder_latent_dim}, epochs={args.autoencoder_epochs}, "
                f"batch_size={args.autoencoder_batch_size}, device={args.autoencoder_device}"
            )
            cache["autoencoder_multimodal"] = standardized(
                autoencoder_embedding(
                    raw,
                    latent_dim=args.autoencoder_latent_dim,
                    epochs=args.autoencoder_epochs,
                    batch_size=args.autoencoder_batch_size,
                    learning_rate=args.autoencoder_learning_rate,
                    seed=args.seed,
                    device_name=args.autoencoder_device,
                )
            ).astype(np.float32, copy=False)
            log(
                "Built autoencoder baseline embedding: "
                f"shape={cache['autoencoder_multimodal'].shape}"
            )
        return "autoencoder_multimodal", cache["autoencoder_multimodal"]
    raise ValueError(f"Unsupported comparison method: {method}")


def assignment_columns_for(assignments: pd.DataFrame, algorithm: str) -> list[str]:
    prefix = f"{algorithm}__"
    return [column for column in assignments.columns if column.startswith(prefix)]


def label_columns(assignments: pd.DataFrame) -> list[str]:
    return [column for column in assignments.columns if "__" not in column]


def algorithm_names_from_outputs(
    assignments: pd.DataFrame,
    metrics: pd.DataFrame,
    sweep: pd.DataFrame,
) -> list[str]:
    names: set[str] = set()
    for frame in (metrics, sweep):
        if "algorithm" in frame.columns:
            names.update(str(value) for value in frame["algorithm"].dropna().unique())
    for column in assignments.columns:
        if "__" in column:
            names.add(column.split("__", 1)[0])
    return sorted(names)


def summary_for_algorithm(summary: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    if summary.empty or "assignment" not in summary.columns:
        return pd.DataFrame()
    return summary[summary["assignment"].astype(str).str.startswith(f"{algorithm}__")].copy()


def add_metric_ranks(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    ranked = metrics.copy()
    group_columns = ["scope", "target", "k"]
    for metric in ["silhouette", "nmi", "ari", "purity"]:
        if metric in ranked.columns:
            ranked[f"{metric}_rank"] = ranked.groupby(group_columns)[metric].rank(
                ascending=False,
                method="min",
                na_option="bottom",
            )
    sort_columns = [column for column in ["target", "k", "nmi_rank", "purity_rank"] if column in ranked.columns]
    return ranked.sort_values(sort_columns, kind="stable")


def add_silhouette_ranks(sweep: pd.DataFrame) -> pd.DataFrame:
    if sweep.empty:
        return sweep.copy()
    ranked = sweep.copy()
    ranked["silhouette_rank"] = ranked.groupby(["scope", "k"])["silhouette"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    return ranked.sort_values(["k", "silhouette_rank"], kind="stable")


def write_organized_outputs(
    output_dir: Path,
    assignments: pd.DataFrame,
    metrics: pd.DataFrame,
    sweep: pd.DataFrame,
    genre_summary: pd.DataFrame,
    language_summary: pd.DataFrame,
) -> None:
    base_columns = label_columns(assignments)
    for algorithm in algorithm_names_from_outputs(assignments, metrics, sweep):
        algorithm_dir = output_dir / "by_algorithm" / algorithm
        assignment_columns = assignment_columns_for(assignments, algorithm)
        if assignment_columns:
            write_dataframe(
                assignments[base_columns + assignment_columns],
                algorithm_dir / "cluster_assignments.parquet",
            )
        if "algorithm" in metrics.columns:
            write_dataframe(
                metrics[metrics["algorithm"] == algorithm].copy(),
                algorithm_dir / "metrics.csv",
            )
        if "algorithm" in sweep.columns:
            write_dataframe(
                sweep[sweep["algorithm"] == algorithm].copy(),
                algorithm_dir / "silhouette_sweep.csv",
            )
        write_dataframe(
            summary_for_algorithm(genre_summary, algorithm),
            algorithm_dir / "cluster_summary_by_genre.csv",
        )
        write_dataframe(
            summary_for_algorithm(language_summary, algorithm),
            algorithm_dir / "cluster_summary_by_language.csv",
        )

    comparison_dir = output_dir / "comparison"
    comparison_algorithms = sorted(COMPARISON_METHODS)
    comparison_metrics = (
        metrics[metrics["algorithm"].isin(comparison_algorithms)].copy()
        if "algorithm" in metrics.columns
        else pd.DataFrame()
    )
    comparison_sweep = (
        sweep[sweep["algorithm"].isin(comparison_algorithms)].copy()
        if "algorithm" in sweep.columns
        else pd.DataFrame()
    )
    comparison_assignment_columns = [
        column
        for algorithm in comparison_algorithms
        for column in assignment_columns_for(assignments, algorithm)
    ]
    if comparison_assignment_columns:
        write_dataframe(
            assignments[base_columns + comparison_assignment_columns],
            comparison_dir / "cluster_assignments.parquet",
        )
    write_dataframe(comparison_metrics, comparison_dir / "metrics.csv")
    write_dataframe(comparison_sweep, comparison_dir / "silhouette_sweep.csv")
    write_dataframe(add_metric_ranks(comparison_metrics), comparison_dir / "ranked_metrics.csv")
    write_dataframe(
        add_silhouette_ranks(comparison_sweep),
        comparison_dir / "ranked_silhouette.csv",
    )


def maybe_cluster(
    algorithm: str,
    values: np.ndarray,
    k: int | None,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if k is not None and (k < 2 or k > values.shape[0]):
        log(f"Skipping {algorithm} k={k}; sample_count={values.shape[0]}")
        return None
    if algorithm in {"cuml_kmeans", "cuml_agglomerative", "cuml_dbscan"}:
        device_info = cuml_device_summary()
        display_name = {
            "cuml_kmeans": "cuML KMeans",
            "cuml_agglomerative": "cuML Agglomerative",
            "cuml_dbscan": "cuML DBSCAN",
        }[algorithm]
        params = (
            f"k={k}, n_init={args.cuda_kmeans_n_init}, max_iter={args.cuda_kmeans_max_iter}"
            if algorithm == "cuml_kmeans"
            else f"k={k}, linkage={args.agglomerative_linkage}"
            if algorithm == "cuml_agglomerative"
            else f"eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}"
        )
        log(
            f"Starting {display_name} on cuda:{device_info['device_id']} "
            f"({device_info['device_name']}, cc={device_info['compute_capability']}, "
            f"memory={device_info['memory_total_mb']}MB): "
            f"samples={values.shape[0]}, features={values.shape[1]}, {params}"
        )
    elif algorithm == "cuda_kmeans":
        log(
            f"Starting Torch CUDA KMeans on {args.cuda_kmeans_device}: "
            f"samples={values.shape[0]}, features={values.shape[1]}, k={k}, "
            f"n_init={args.cuda_kmeans_n_init}, max_iter={args.cuda_kmeans_max_iter}"
        )
    elif algorithm == "agglomerative":
        log(
            f"Starting agglomerative on CPU: samples={values.shape[0]}, "
            f"features={values.shape[1]}, k={k}, linkage={args.agglomerative_linkage}"
        )
    elif algorithm == "dbscan":
        log(
            f"Starting dbscan on CPU: samples={values.shape[0]}, features={values.shape[1]}, "
            f"eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}"
        )
    else:
        log(
            f"Starting {algorithm} on CPU: "
            f"samples={values.shape[0]}, features={values.shape[1]}, k={k}"
        )

    clusters = fit_predict(
        algorithm=algorithm,
        values=values,
        k=k,
        seed=args.seed,
        cuda_device=args.cuda_kmeans_device,
        cuda_n_init=args.cuda_kmeans_n_init,
        cuda_max_iter=args.cuda_kmeans_max_iter,
        cuda_tol=args.cuda_kmeans_tol,
        cuda_batch_size=args.cuda_kmeans_batch_size or None,
        agglomerative_linkage=args.agglomerative_linkage,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )
    log(
        f"Finished {algorithm}{'' if k is None else f' k={k}'}: "
        f"clusters={cluster_count(clusters)}, noise={noise_count(clusters)}, dtype={clusters.dtype}"
    )
    return clusters


def run_scope(
    scope: str,
    latent_root: Path,
    output_root: Path,
    embeddings_to_run: list[str],
    algorithms: list[str],
    comparison_methods: list[str],
    sweep_k: list[int],
    args: argparse.Namespace,
) -> None:
    scope_dir = latent_root / scope
    output_dir = output_root / scope
    log(f"Loading {scope} latents: {scope_dir}")
    embeddings, labels = load_latent_scope(scope_dir)

    assignments = labels.copy()
    metrics_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    summary_frames_genre = []
    summary_frames_language = []
    feature_cache: dict[str, np.ndarray] = {}
    row_indices = labels["row_index"].to_numpy(dtype=np.int64)

    for method in comparison_methods:
        embedding_name, values = comparison_values(
            method=method,
            embeddings=embeddings,
            row_indices=row_indices,
            cache=feature_cache,
            args=args,
        )
        log(f"Running comparison method {method}: embedding={embedding_name}, shape={values.shape}")
        needed_k = sorted({args.genre_k, args.language_k, *sweep_k})
        for k in needed_k:
            clusters = maybe_cluster(args.comparison_kmeans_algorithm, values, k, args)
            if clusters is None:
                continue
            append_cluster_evaluation(
                scope=scope,
                embedding_name=embedding_name,
                algorithm_name=method,
                assignment_column=assignment_name(method, embedding_name, k),
                clusters=clusters,
                values=values,
                labels=labels,
                assignments=assignments,
                metrics_rows=metrics_rows,
                sweep_rows=sweep_rows,
                summary_frames_genre=summary_frames_genre,
                summary_frames_language=summary_frames_language,
                args=args,
                k=k,
                kmeans_backend=args.comparison_kmeans_algorithm,
            )

    for embedding_name in embeddings_to_run:
        if not algorithms:
            break
        if embedding_name not in embeddings:
            raise KeyError(f"Embedding '{embedding_name}' not found. Available: {sorted(embeddings)}")
        values = standardized(embeddings[embedding_name])
        log(f"Clustering {scope}/{embedding_name}: shape={values.shape}")

        for algorithm in algorithms:
            if is_density_algorithm(algorithm):
                clusters = maybe_cluster(algorithm, values, None, args)
                if clusters is None:
                    continue
                detected_k = cluster_count(clusters, exclude_noise=True)
                column = density_assignment_name(
                    algorithm,
                    embedding_name,
                    args.dbscan_eps,
                    args.dbscan_min_samples,
                )
                assignments[column] = clusters.astype(np.int64)

                sweep_rows.append(
                    {
                        "scope": scope,
                        "embedding": embedding_name,
                        "algorithm": algorithm,
                        "k": int(detected_k),
                        "silhouette": safe_silhouette(
                            values,
                            clusters,
                            args.silhouette_sample_size,
                            args.seed,
                        ),
                        "silhouette_sample_size": int(args.silhouette_sample_size),
                        "sample_count": int(values.shape[0]),
                        "cluster_count": int(cluster_count(clusters)),
                        "noise_count": int(noise_count(clusters)),
                        "dbscan_eps": float(args.dbscan_eps),
                        "dbscan_min_samples": int(args.dbscan_min_samples),
                    }
                )

                for spec in target_specs(args):
                    target_values = labels[spec["label_column"]].to_numpy()
                    row = {
                        "scope": scope,
                        "embedding": embedding_name,
                        "algorithm": algorithm,
                        "k": int(detected_k),
                        "target": spec["target"],
                        "label_column": spec["label_column"],
                        "sample_count": int(values.shape[0]),
                        "cluster_count": int(cluster_count(clusters)),
                        "noise_count": int(noise_count(clusters)),
                        "dbscan_eps": float(args.dbscan_eps),
                        "dbscan_min_samples": int(args.dbscan_min_samples),
                    }
                    row.update(
                        supervised_metrics(
                            values,
                            clusters,
                            target_values,
                            args.silhouette_sample_size,
                            args.seed,
                        )
                    )
                    row["silhouette_sample_size"] = int(args.silhouette_sample_size)
                    metrics_rows.append(row)

                    summary = cluster_label_summary(assignments, column, spec["display_column"])
                    if spec["target"] == "genre":
                        summary_frames_genre.append(summary)
                    else:
                        summary_frames_language.append(summary)
                continue

            cached_predictions: dict[int, np.ndarray] = {}
            needed_k = sorted({args.genre_k, args.language_k, *sweep_k})
            for k in needed_k:
                clusters = maybe_cluster(algorithm, values, k, args)
                if clusters is None:
                    continue
                cached_predictions[k] = clusters
                column = assignment_name(algorithm, embedding_name, k)
                assignments[column] = clusters.astype(np.int64)

                sweep_rows.append(
                    {
                        "scope": scope,
                        "embedding": embedding_name,
                        "algorithm": algorithm,
                        "k": int(k),
                        "silhouette": safe_silhouette(
                            values,
                            clusters,
                            args.silhouette_sample_size,
                            args.seed,
                        ),
                        "silhouette_sample_size": int(args.silhouette_sample_size),
                        "sample_count": int(values.shape[0]),
                        "cluster_count": int(cluster_count(clusters)),
                        "noise_count": int(noise_count(clusters)),
                    }
                )

            for spec in target_specs(args):
                k = int(spec["k"])
                clusters = cached_predictions.get(k)
                if clusters is None:
                    continue
                target_values = labels[spec["label_column"]].to_numpy()
                row = {
                    "scope": scope,
                    "embedding": embedding_name,
                    "algorithm": algorithm,
                    "k": k,
                    "target": spec["target"],
                    "label_column": spec["label_column"],
                    "sample_count": int(values.shape[0]),
                    "cluster_count": int(cluster_count(clusters)),
                    "noise_count": int(noise_count(clusters)),
                }
                row.update(
                    supervised_metrics(
                        values,
                        clusters,
                        target_values,
                        args.silhouette_sample_size,
                        args.seed,
                    )
                )
                row["silhouette_sample_size"] = int(args.silhouette_sample_size)
                metrics_rows.append(row)

                column = assignment_name(algorithm, embedding_name, k)
                summary = cluster_label_summary(assignments, column, spec["display_column"])
                if spec["target"] == "genre":
                    summary_frames_genre.append(summary)
                else:
                    summary_frames_language.append(summary)

    metrics = pd.DataFrame(metrics_rows)
    sweep = pd.DataFrame(sweep_rows)

    write_dataframe(assignments, output_dir / "cluster_assignments.parquet")
    write_dataframe(metrics, output_dir / "metrics.csv")
    write_dataframe(sweep, output_dir / "silhouette_sweep.csv")

    genre_summary = (
        pd.concat(summary_frames_genre, ignore_index=True)
        if summary_frames_genre
        else pd.DataFrame()
    )
    language_summary = (
        pd.concat(summary_frames_language, ignore_index=True)
        if summary_frames_language
        else pd.DataFrame()
    )
    write_dataframe(genre_summary, output_dir / "cluster_summary_by_genre.csv")
    write_dataframe(language_summary, output_dir / "cluster_summary_by_language.csv")
    write_organized_outputs(
        output_dir=output_dir,
        assignments=assignments,
        metrics=metrics,
        sweep=sweep,
        genre_summary=genre_summary,
        language_summary=language_summary,
    )

    save_json(
        {
            "scope": scope,
            "latent_root": str(latent_root),
            "output_dir": str(output_dir),
            "embeddings": embeddings_to_run,
            "algorithms": algorithms,
            "comparison_methods": comparison_methods,
            "comparison_kmeans_algorithm": args.comparison_kmeans_algorithm,
            "baseline_inputs": {
                "model_input_path": str(args.model_input_path),
                "mel_tensor_dir": str(args.mel_tensor_dir),
                "tabular_columns": TABULAR_BASELINE_COLUMNS,
                "spectral_time_bins": int(args.spectral_time_bins),
            },
            "pca_kmeans": {
                "components": int(args.pca_components),
            },
            "autoencoder_kmeans": {
                "latent_dim": int(args.autoencoder_latent_dim),
                "epochs": int(args.autoencoder_epochs),
                "batch_size": int(args.autoencoder_batch_size),
                "learning_rate": float(args.autoencoder_learning_rate),
                "device": args.autoencoder_device,
            },
            "cuda_kmeans": {
                "device": args.cuda_kmeans_device,
                "n_init": int(args.cuda_kmeans_n_init),
                "max_iter": int(args.cuda_kmeans_max_iter),
                "tol": float(args.cuda_kmeans_tol),
                "batch_size": int(args.cuda_kmeans_batch_size),
            },
            "cuml_kmeans": {
                "n_init": int(args.cuda_kmeans_n_init),
                "max_iter": int(args.cuda_kmeans_max_iter),
                "tol": float(args.cuda_kmeans_tol),
            },
            "agglomerative": {
                "linkage": args.agglomerative_linkage,
            },
            "dbscan": {
                "eps": float(args.dbscan_eps),
                "min_samples": int(args.dbscan_min_samples),
            },
            "genre_k": int(args.genre_k),
            "language_k": int(args.language_k),
            "sweep_k": sweep_k,
            "silhouette_sample_size": int(args.silhouette_sample_size),
            "seed": int(args.seed),
            "notes": {
                "metadata_assisted": "Primary assisted multi-modal clustering view.",
                "mu": "Comparison view; current encoder was trained with metadata inputs.",
            },
        },
        output_dir / "run_metadata.json",
    )
    log(
        f"Saved {scope} clustering outputs: assignments={len(assignments)}, "
        f"metrics_rows={len(metrics_rows)}, output_dir={output_dir}"
    )


def main() -> None:
    args = parse_args()
    latent_root = project_path(args.latent_root)
    output_root = project_path(args.output_root)
    args.model_input_path = project_path(args.model_input_path)
    args.mel_tensor_dir = project_path(args.mel_tensor_dir)
    embeddings = parse_csv_list(args.embeddings)
    algorithms = parse_csv_list(args.algorithms)
    comparison_methods = parse_csv_list(args.comparison_methods)
    sweep_k = parse_int_csv_list(args.sweep_k)

    bad_embeddings = sorted(set(embeddings) - {"mu", "metadata_assisted"})
    if bad_embeddings:
        raise ValueError(f"Unsupported embeddings: {bad_embeddings}")
    supported_algorithms = {
        "cuml_kmeans",
        "cuda_kmeans",
        "kmeans",
        "cuml_agglomerative",
        "agglomerative",
        "cuml_dbscan",
        "dbscan",
        "gmm",
    }
    bad_algorithms = sorted(set(algorithms) - supported_algorithms)
    if bad_algorithms:
        raise ValueError(f"Unsupported algorithms: {bad_algorithms}")
    bad_methods = sorted(set(comparison_methods) - COMPARISON_METHODS)
    if bad_methods:
        raise ValueError(f"Unsupported comparison methods: {bad_methods}")

    if needs_cuml(algorithms, comparison_methods, args.comparison_kmeans_algorithm):
        cuda_info = assert_cuml_cuda_available()
        log(
            "cuML CUDA ready: "
            f"devices={cuda_info['device_count']}, "
            f"runtime={cuda_info['cuda_runtime']}, "
            f"driver={cuda_info['cuda_driver']}, "
            f"cuml={cuda_info['cuml_version']}"
        )

    for scope in scopes_from_arg(args.scope):
        run_scope(
            scope=scope,
            latent_root=latent_root,
            output_root=output_root,
            embeddings_to_run=embeddings,
            algorithms=algorithms,
            comparison_methods=comparison_methods,
            sweep_k=sweep_k,
            args=args,
        )

    log(f"Clustering complete: {output_root}")


if __name__ == "__main__":
    main()
