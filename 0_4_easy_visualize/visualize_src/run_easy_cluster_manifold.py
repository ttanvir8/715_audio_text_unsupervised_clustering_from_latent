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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[easy_cluster_manifold] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize easy lyrics VAE clusters with t-SNE or UMAP."
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
        help="Root where t-SNE/UMAP plots and point tables will be written.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="both",
        help="Which scope to visualize.",
    )
    parser.add_argument(
        "--embedding",
        choices=("mu", "logvar"),
        default="mu",
        help="Latent array to visualize.",
    )
    parser.add_argument(
        "--method",
        choices=("tsne", "umap"),
        default="tsne",
        help="2D manifold method.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=3000,
        help="Maximum plotted rows per scope. Use 0 or a negative value to plot all rows.",
    )
    parser.add_argument("--seed", type=int, default=751, help="Random seed.")
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity. It is clipped to a valid value for small samples.",
    )
    parser.add_argument("--tsne-max-iter", type=int, default=1000, help="t-SNE optimization steps.")
    parser.add_argument("--umap-neighbors", type=int, default=30, help="UMAP neighbor count.")
    parser.add_argument("--umap-min-dist", type=float, default=0.10, help="UMAP minimum distance.")
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


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path)


def assignment_column(assignments: pd.DataFrame) -> str:
    candidates = [column for column in assignments.columns if column.startswith("kmeans__")]
    if not candidates:
        raise ValueError("No K-Means assignment column found in cluster_assignments.parquet.")
    return candidates[0]


def load_scope_inputs(
    latent_scope_dir: Path,
    clustering_scope_dir: Path,
    embedding: str,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    values = np.load(latent_scope_dir / f"{embedding}.npy").astype(np.float32, copy=False)
    labels = pq.read_table(latent_scope_dir / "labels.parquet").to_pandas()
    assignments = pq.read_table(clustering_scope_dir / "cluster_assignments.parquet").to_pandas()
    assign_col = assignment_column(assignments)

    if values.shape[0] != len(labels) or values.shape[0] != len(assignments):
        raise ValueError(
            "Scope inputs are not row-aligned: "
            f"values={values.shape[0]}, labels={len(labels)}, assignments={len(assignments)}"
        )
    if "row_index" in labels.columns and not labels["row_index"].equals(assignments["row_index"]):
        raise ValueError("labels.parquet and cluster_assignments.parquet row_index columns differ.")

    metadata = assignments[["row_index", "music_id", "music_name", "music_lang", assign_col]].copy()
    metadata = metadata.rename(columns={assign_col: "cluster"})
    return values, metadata, assign_col


def balanced_sample_indices(metadata: pd.DataFrame, max_points: int, seed: int) -> np.ndarray:
    row_count = len(metadata)
    if max_points <= 0 or row_count <= max_points:
        return np.arange(row_count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    group_indices = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in metadata.groupby("cluster", sort=True)
    ]
    target_per_cluster = max(1, max_points // max(1, len(group_indices)))
    selected_set: set[int] = set()

    for indices in group_indices:
        take_count = min(len(indices), target_per_cluster)
        chosen = rng.choice(indices, size=take_count, replace=False)
        for index in chosen:
            selected_set.add(int(index))

    remaining = max_points - len(selected_set)
    if remaining > 0:
        all_indices = np.arange(row_count, dtype=np.int64)
        available = np.array([index for index in all_indices if int(index) not in selected_set])
        if len(available):
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            for index in extra:
                selected_set.add(int(index))

    selected.extend(sorted(selected_set))
    return np.asarray(selected[:max_points], dtype=np.int64)


def standardized(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values).astype(np.float32, copy=False)


def effective_perplexity(sample_count: int, requested: float) -> float:
    if sample_count < 4:
        raise ValueError(f"Need at least 4 points for t-SNE, got {sample_count}.")
    upper = max(1.0, (sample_count - 1) / 3.0)
    return float(min(requested, upper))


def reduce_tsne(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    perplexity = effective_perplexity(values.shape[0], float(args.perplexity))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(args.seed),
        max_iter=int(args.tsne_max_iter),
    )
    return reducer.fit_transform(values).astype(np.float32, copy=False)


def reduce_umap(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "UMAP visualization needs the optional 'umap-learn' package. "
            "Use --method tsne with the current environment."
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(args.umap_neighbors),
        min_dist=float(args.umap_min_dist),
        random_state=int(args.seed),
    )
    return reducer.fit_transform(values).astype(np.float32, copy=False)


def reduce_2d(values: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.method == "tsne":
        return reduce_tsne(values, args)
    if args.method == "umap":
        return reduce_umap(values, args)
    raise ValueError(f"Unsupported method: {args.method}")


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
    ax.legend(title=category_column, loc="best", markerscale=1.4, fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_scope(args: argparse.Namespace, scope: str) -> dict[str, Any]:
    latent_scope_dir = project_path(args.latent_root) / scope
    clustering_scope_dir = project_path(args.clustering_root) / scope
    output_dir = project_path(args.output_root) / scope
    manifold_dir = output_dir / "manifold"

    values, metadata, assign_col = load_scope_inputs(
        latent_scope_dir=latent_scope_dir,
        clustering_scope_dir=clustering_scope_dir,
        embedding=args.embedding,
    )
    sample_indices = balanced_sample_indices(metadata, int(args.max_points), int(args.seed))
    values_std = standardized(values)
    sampled_values = values_std[sample_indices]
    sampled_metadata = metadata.iloc[sample_indices].reset_index(drop=True)

    log(
        f"Visualizing {scope}: method={args.method}, embedding={args.embedding}, "
        f"rows={len(metadata)}, plotted={len(sampled_metadata)}, assignment={assign_col}"
    )
    points = reduce_2d(sampled_values, args)
    point_table = sampled_metadata.copy()
    point_table[f"{args.method}_x"] = points[:, 0]
    point_table[f"{args.method}_y"] = points[:, 1]
    point_table["source_scope"] = scope
    point_table["source_embedding"] = args.embedding

    points_path = manifold_dir / f"{args.method}_{args.embedding}_points.parquet"
    cluster_path = manifold_dir / f"{args.method}_{args.embedding}_colored_by_cluster.png"
    language_path = manifold_dir / f"{args.method}_{args.embedding}_colored_by_language.png"

    write_parquet(point_table, points_path)
    scatter_by_category(
        points=points,
        metadata=point_table,
        category_column="cluster",
        title=f"{scope} {args.embedding} {args.method.upper()} colored by cluster",
        output_path=cluster_path,
    )
    scatter_by_category(
        points=points,
        metadata=point_table,
        category_column="music_lang",
        title=f"{scope} {args.embedding} {args.method.upper()} colored by language",
        output_path=language_path,
    )

    manifest = {
        "scope": scope,
        "method": args.method,
        "embedding": args.embedding,
        "latent_scope_dir": str(latent_scope_dir),
        "clustering_scope_dir": str(clustering_scope_dir),
        "output_dir": str(output_dir),
        "assignment_column": assign_col,
        "row_count": int(len(metadata)),
        "plotted_rows": int(len(point_table)),
        "max_points": int(args.max_points),
        "seed": int(args.seed),
        "artifacts": [
            str(points_path),
            str(cluster_path),
            str(language_path),
        ],
    }
    save_json(manifest, output_dir / "visualization_manifest.json")
    log(f"Saved {scope} manifold visualizations: {output_dir}")
    return manifest


def main() -> None:
    args = parse_args()
    manifests = [run_scope(args, scope) for scope in scopes_from_arg(args.scope)]
    save_json(
        {
            "scopes": [manifest["scope"] for manifest in manifests],
            "method": args.method,
            "embedding": args.embedding,
            "manifests": manifests,
        },
        project_path(args.output_root) / "visualization_manifest.json",
    )
    log(f"Visualization complete: {project_path(args.output_root)}")


if __name__ == "__main__":
    main()
