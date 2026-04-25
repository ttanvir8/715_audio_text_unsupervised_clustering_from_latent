from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[easy_latent_space] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize easy lyrics VAE latent space for validation and full scopes."
    )
    parser.add_argument(
        "--latent-root",
        default="0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001",
        help="Root containing validation/full latent exports.",
    )
    parser.add_argument(
        "--clustering-root",
        default="0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001",
        help="Root containing validation/full clustering outputs.",
    )
    parser.add_argument(
        "--output-root",
        default="0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001",
        help="Root where latent-space plots and point tables will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "val", "full", "both"),
        default="both",
        help="Which scope to visualize. val is an alias for validation.",
    )
    parser.add_argument(
        "--embeddings",
        default="mu,logvar",
        help="Comma-separated latent arrays to visualize with PCA.",
    )
    parser.add_argument(
        "--tsne-embedding",
        default="mu",
        choices=("mu", "logvar", "none"),
        help="Latent array to visualize with t-SNE. Use none to skip t-SNE.",
    )
    parser.add_argument(
        "--max-tsne-points",
        type=int,
        default=3000,
        help="Maximum rows for t-SNE per scope. Use 0 or a negative value for all rows.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Requested t-SNE perplexity.")
    parser.add_argument("--tsne-max-iter", type=int, default=1000, help="t-SNE optimization steps.")
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def scopes_from_arg(scope: str) -> list[str]:
    if scope == "both":
        return ["validation", "full"]
    if scope == "val":
        return ["validation"]
    return [scope]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path)


def assignment_column(assignments: pd.DataFrame) -> str:
    candidates = [column for column in assignments.columns if column.startswith("kmeans__")]
    if not candidates:
        raise ValueError("No K-Means assignment column found in cluster_assignments.parquet.")
    return candidates[0]


def load_metadata(latent_scope_dir: Path, clustering_scope_dir: Path) -> tuple[pd.DataFrame, str]:
    labels = pq.read_table(latent_scope_dir / "labels.parquet").to_pandas()
    assignments = pq.read_table(clustering_scope_dir / "cluster_assignments.parquet").to_pandas()
    assign_col = assignment_column(assignments)

    if len(labels) != len(assignments):
        raise ValueError(
            f"labels and assignments are not aligned: {len(labels)} vs {len(assignments)}"
        )
    if "row_index" in labels.columns and not labels["row_index"].equals(assignments["row_index"]):
        raise ValueError("labels.parquet and cluster_assignments.parquet row_index columns differ.")

    metadata = assignments[["row_index", "music_id", "music_name", "music_lang", assign_col]].copy()
    metadata = metadata.rename(columns={assign_col: "cluster"})
    return metadata, assign_col


def load_latent_values(latent_scope_dir: Path, embedding: str, expected_rows: int) -> np.ndarray:
    path = latent_scope_dir / f"{embedding}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing latent array: {path}")
    values = np.load(path).astype(np.float32, copy=False)
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"{embedding} rows ({values.shape[0]}) do not match metadata rows ({expected_rows})."
        )
    return values


def standardized(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values).astype(np.float32, copy=False)


def pca_2d(values: np.ndarray, seed: int) -> tuple[np.ndarray, PCA]:
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"Need at least two rows and two dims for PCA, got {values.shape}.")
    pca = PCA(n_components=2, random_state=seed)
    points = pca.fit_transform(values).astype(np.float32, copy=False)
    return points, pca


def balanced_sample_indices(metadata: pd.DataFrame, max_points: int, seed: int) -> np.ndarray:
    row_count = len(metadata)
    if max_points <= 0 or row_count <= max_points:
        return np.arange(row_count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    groups = [group.index.to_numpy(dtype=np.int64) for _, group in metadata.groupby("cluster", sort=True)]
    target_per_cluster = max(1, max_points // max(1, len(groups)))

    for indices in groups:
        take_count = min(len(indices), target_per_cluster)
        chosen = rng.choice(indices, size=take_count, replace=False)
        selected.update(int(index) for index in chosen)

    remaining = max_points - len(selected)
    if remaining > 0:
        available = np.array(
            [index for index in range(row_count) if index not in selected],
            dtype=np.int64,
        )
        if len(available):
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            selected.update(int(index) for index in extra)

    return np.asarray(sorted(selected)[:max_points], dtype=np.int64)


def effective_perplexity(sample_count: int, requested: float) -> float:
    if sample_count < 4:
        raise ValueError(f"Need at least 4 points for t-SNE, got {sample_count}.")
    upper = max(1.0, (sample_count - 1) / 3.0)
    return float(min(requested, upper))


def tsne_2d(values: np.ndarray, seed: int, perplexity: float, max_iter: int) -> np.ndarray:
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity(values.shape[0], perplexity),
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=max_iter,
    )
    return reducer.fit_transform(values).astype(np.float32, copy=False)


def category_order(values: pd.Series) -> list[str]:
    return sorted(values.astype(str).unique().tolist())


def scatter_by_category(
    points: np.ndarray,
    metadata: pd.DataFrame,
    category_column: str,
    title: str,
    output_path: str | Path,
) -> None:
    categories = metadata[category_column].astype(str)
    names = category_order(categories)
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(index % cmap.N) for index, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in names:
        mask = categories == name
        ax.scatter(
            points[mask.to_numpy(), 0],
            points[mask.to_numpy(), 1],
            s=18,
            alpha=0.78,
            color=colors[name],
            label=name,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.legend(title=category_column, loc="best", markerscale=1.35, fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_point_table(
    metadata: pd.DataFrame,
    points: np.ndarray,
    x_name: str,
    y_name: str,
    output_path: str | Path,
) -> None:
    table = metadata.copy()
    table[x_name] = points[:, 0]
    table[y_name] = points[:, 1]
    write_parquet(table, output_path)


def plot_pca_embedding(
    values: np.ndarray,
    metadata: pd.DataFrame,
    embedding: str,
    scope: str,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    points, pca = pca_2d(standardized(values), seed)
    point_path = output_dir / f"pca_{embedding}_points.parquet"
    cluster_path = output_dir / f"pca_{embedding}_colored_by_cluster.png"
    language_path = output_dir / f"pca_{embedding}_colored_by_language.png"

    save_point_table(metadata, points, "pca_x", "pca_y", point_path)
    scatter_by_category(
        points,
        metadata,
        "cluster",
        f"{scope} {embedding} PCA colored by cluster",
        cluster_path,
    )
    scatter_by_category(
        points,
        metadata,
        "music_lang",
        f"{scope} {embedding} PCA colored by language",
        language_path,
    )
    return {
        "embedding": embedding,
        "method": "pca",
        "rows": int(values.shape[0]),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "artifacts": [str(point_path), str(cluster_path), str(language_path)],
    }


def plot_tsne_embedding(
    values: np.ndarray,
    metadata: pd.DataFrame,
    embedding: str,
    scope: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_indices = balanced_sample_indices(metadata, int(args.max_tsne_points), int(args.seed))
    sampled_metadata = metadata.iloc[sample_indices].reset_index(drop=True)
    sampled_values = standardized(values)[sample_indices]
    points = tsne_2d(
        sampled_values,
        seed=int(args.seed),
        perplexity=float(args.perplexity),
        max_iter=int(args.tsne_max_iter),
    )

    point_path = output_dir / f"tsne_{embedding}_points.parquet"
    cluster_path = output_dir / f"tsne_{embedding}_colored_by_cluster.png"
    language_path = output_dir / f"tsne_{embedding}_colored_by_language.png"

    save_point_table(sampled_metadata, points, "tsne_x", "tsne_y", point_path)
    scatter_by_category(
        points,
        sampled_metadata,
        "cluster",
        f"{scope} {embedding} t-SNE colored by cluster",
        cluster_path,
    )
    scatter_by_category(
        points,
        sampled_metadata,
        "music_lang",
        f"{scope} {embedding} t-SNE colored by language",
        language_path,
    )
    return {
        "embedding": embedding,
        "method": "tsne",
        "rows": int(values.shape[0]),
        "plotted_rows": int(len(sampled_metadata)),
        "max_tsne_points": int(args.max_tsne_points),
        "artifacts": [str(point_path), str(cluster_path), str(language_path)],
    }


def run_scope(args: argparse.Namespace, scope: str) -> dict[str, Any]:
    latent_scope_dir = project_path(args.latent_root) / scope
    clustering_scope_dir = project_path(args.clustering_root) / scope
    output_dir = project_path(args.output_root) / scope / "letent_space"
    metadata, assign_col = load_metadata(latent_scope_dir, clustering_scope_dir)
    artifacts: list[dict[str, Any]] = []

    log(f"Visualizing latent space for {scope}: rows={len(metadata)}, assignment={assign_col}")
    for embedding in parse_csv(args.embeddings):
        values = load_latent_values(latent_scope_dir, embedding, len(metadata))
        artifacts.append(
            plot_pca_embedding(
                values=values,
                metadata=metadata,
                embedding=embedding,
                scope=scope,
                output_dir=output_dir,
                seed=int(args.seed),
            )
        )

    if args.tsne_embedding != "none":
        values = load_latent_values(latent_scope_dir, args.tsne_embedding, len(metadata))
        artifacts.append(
            plot_tsne_embedding(
                values=values,
                metadata=metadata,
                embedding=args.tsne_embedding,
                scope=scope,
                output_dir=output_dir,
                args=args,
            )
        )

    manifest = {
        "scope": scope,
        "latent_scope_dir": str(latent_scope_dir),
        "clustering_scope_dir": str(clustering_scope_dir),
        "output_dir": str(output_dir),
        "assignment_column": assign_col,
        "row_count": int(len(metadata)),
        "seed": int(args.seed),
        "artifacts": artifacts,
    }
    save_json(manifest, output_dir / "latent_space_manifest.json")
    log(f"Saved {scope} latent-space visualizations: {output_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    manifests = [run_scope(args, scope) for scope in scopes_from_arg(args.scope)]
    save_json(
        {
            "scopes": [manifest["scope"] for manifest in manifests],
            "output_subdir": "letent_space",
            "manifests": manifests,
        },
        project_path(args.output_root) / "letent_space_manifest.json",
    )
    log(f"Latent-space visualization complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
