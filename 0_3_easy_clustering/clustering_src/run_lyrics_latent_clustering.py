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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[easy_cluster_latents] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster easy lyrics VAE encoder latents and evaluate against language labels."
    )
    parser.add_argument(
        "--latent-root",
        default="0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001",
        help="Root containing validation/full latent exports from 0_2_easy_inference.",
    )
    parser.add_argument(
        "--output-root",
        default="0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001",
        help="Root where clustering assignments and metrics will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which latent scope to cluster.",
    )
    parser.add_argument(
        "--embedding",
        default="mu",
        choices=("mu", "logvar"),
        help="Latent array to cluster.",
    )
    parser.add_argument(
        "--language-k",
        type=int,
        default=0,
        help="Cluster count for the main language run. 0 means number of language labels.",
    )
    parser.add_argument(
        "--sweep-k",
        default="2,3,4,5,6,8",
        help="Comma-separated k values for the silhouette and language-metric sweep.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed for clustering.")
    parser.add_argument(
        "--n-init",
        type=int,
        default=50,
        help="KMeans initializations.",
    )
    parser.add_argument(
        "--make-plot",
        action="store_true",
        help="Save a PCA scatter plot colored by cluster and marked by language.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    return [scope]


def parse_int_csv_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def load_latent_scope(scope_dir: str | Path, embedding: str) -> tuple[np.ndarray, pd.DataFrame]:
    path = Path(scope_dir)
    values = np.load(path / f"{embedding}.npy")
    labels = pq.read_table(path / "labels.parquet").to_pandas()
    if values.shape[0] != len(labels):
        raise ValueError(
            f"{embedding} rows ({values.shape[0]}) do not match labels ({len(labels)}) in {path}"
        )
    return values.astype(np.float32, copy=False), labels


def standardized(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values).astype(np.float32, copy=False)


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


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    contingency = pd.crosstab(pd.Series(y_pred, name="cluster"), pd.Series(y_true, name="target"))
    if contingency.empty:
        return float("nan")
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


def language_metrics(values: np.ndarray, clusters: np.ndarray, languages: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": safe_silhouette(values, clusters),
        "nmi": float(normalized_mutual_info_score(languages, clusters)),
        "ari": float(adjusted_rand_score(languages, clusters)),
        "purity": purity_score(languages, clusters),
    }


def cluster_language_summary(assignments: pd.DataFrame, assignment_column: str) -> pd.DataFrame:
    rows = []
    for cluster_id, group in assignments.groupby(assignment_column, sort=True):
        counts = group["music_lang"].value_counts(dropna=False)
        top_label = counts.index[0] if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        rows.append(
            {
                "assignment": assignment_column,
                "cluster": int(cluster_id),
                "size": int(len(group)),
                "top_language": top_label,
                "top_language_count": top_count,
                "top_language_fraction": float(top_count / len(group)) if len(group) else float("nan"),
                "unique_languages": int(group["music_lang"].nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)


def make_pca_plot(
    values: np.ndarray,
    assignments: pd.DataFrame,
    assignment_column: str,
    output_path: str | Path,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    component_count = min(2, values.shape[0], values.shape[1])
    if component_count < 2:
        return
    points = PCA(n_components=2, random_state=seed).fit_transform(values)
    languages = sorted(assignments["music_lang"].astype(str).unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    plt.figure(figsize=(8, 6))
    for language_index, language in enumerate(languages):
        mask = assignments["music_lang"].astype(str).to_numpy() == language
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            c=assignments.loc[mask, assignment_column].to_numpy(),
            cmap="tab10",
            marker=markers[language_index % len(markers)],
            s=28,
            alpha=0.82,
            label=language,
            edgecolors="none",
        )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"{assignment_column} on lyrics VAE latents")
    plt.legend(title="language", loc="best")
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def run_scope(args: argparse.Namespace, scope: str) -> None:
    latent_scope_dir = project_path(args.latent_root) / scope
    output_dir = project_path(args.output_root) / scope
    values, labels = load_latent_scope(latent_scope_dir, args.embedding)
    values_std = standardized(values)

    labels = labels.copy()
    language_codes, language_names = pd.factorize(labels["music_lang"], sort=True)
    labels["music_lang_id_easy"] = language_codes.astype(np.int64)
    languages = labels["music_lang_id_easy"].to_numpy()
    language_k = int(args.language_k) if args.language_k else int(len(language_names))

    log(
        f"Clustering {scope}: rows={values.shape[0]}, dim={values.shape[1]}, "
        f"languages={list(language_names)}, k={language_k}"
    )
    main_assignment = f"kmeans__{args.embedding}__k{language_k}"
    main_clusters = fit_kmeans(values_std, language_k, args.seed, args.n_init)
    assignments = labels[
        ["row_index", "music_id", "music_name", "music_lang", "music_lang_id_easy"]
    ].copy()
    assignments[main_assignment] = main_clusters

    metrics_rows: list[dict[str, Any]] = []
    main_metrics = {
        "scope": scope,
        "embedding": args.embedding,
        "algorithm": "kmeans",
        "assignment": main_assignment,
        "k": language_k,
        "sample_count": int(values.shape[0]),
        "cluster_count": int(len(np.unique(main_clusters))),
        "target": "language",
    }
    main_metrics.update(language_metrics(values_std, main_clusters, languages))
    metrics_rows.append(main_metrics)

    sweep_rows: list[dict[str, Any]] = []
    for k in parse_int_csv_list(args.sweep_k):
        if k < 2 or k > values_std.shape[0]:
            continue
        clusters = fit_kmeans(values_std, k, args.seed, args.n_init)
        row = {
            "scope": scope,
            "embedding": args.embedding,
            "algorithm": "kmeans",
            "k": int(k),
            "sample_count": int(values_std.shape[0]),
            "cluster_count": int(len(np.unique(clusters))),
            "target": "language",
        }
        row.update(language_metrics(values_std, clusters, languages))
        sweep_rows.append(row)

    summary = cluster_language_summary(assignments, main_assignment)
    write_dataframe(assignments, output_dir / "cluster_assignments.parquet")
    write_dataframe(pd.DataFrame(metrics_rows), output_dir / "metrics.csv")
    write_dataframe(pd.DataFrame(sweep_rows), output_dir / "silhouette_sweep.csv")
    write_dataframe(summary, output_dir / "cluster_summary_by_language.csv")
    save_json(
        {
            "latent_scope_dir": str(latent_scope_dir),
            "output_dir": str(output_dir),
            "scope": scope,
            "embedding": args.embedding,
            "language_names": list(language_names),
            "language_k": language_k,
            "seed": int(args.seed),
            "n_init": int(args.n_init),
            "sweep_k": parse_int_csv_list(args.sweep_k),
            "value_shape": list(values.shape),
            "standardized": True,
        },
        output_dir / "run_metadata.json",
    )
    if args.make_plot:
        make_pca_plot(
            values_std,
            assignments,
            main_assignment,
            output_dir / "latent_pca_cluster_scatter.png",
            args.seed,
        )
    log(f"Saved {scope} clustering outputs: {output_dir}")


def main() -> None:
    args = parse_args()
    for scope in scopes_from_arg(args.scope):
        run_scope(args, scope)
    log(f"Clustering complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()

