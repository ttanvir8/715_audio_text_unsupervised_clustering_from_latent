from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from cuml.manifold import UMAP
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "1_training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

# These specific imports depend on 1_training existing in sys.path
from training_scripts.model import build_model
from training_scripts.normalization import MelNormalizer
from dataloader.dataset import MusicVAEDataset, music_collate_fn
from training_scripts.training_utils import prepare_batch, select_device

def log(message: str) -> None:
    print(f"[multi_visualize] {message}", flush=True)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-name",
        default="mm_cbeta_vae_experiment_002",
        help="Name of the training experiment to load.",
    )
    parser.add_argument(
        "--latent-root",
        default="2_inference/latents_by_best_checkpoints",
        help="Root for inference outputs.",
    )
    parser.add_argument(
        "--clustering-root",
        default="3_clustering/clustering_outputs",
        help="Root for clustering outputs.",
    )
    parser.add_argument(
        "--output-root",
        default="4_Visulizations/cluster_visulization_outputs",
        help="Root to save all visualizations.",
    )
    parser.add_argument(
        "--scope",
        choices=("validation", "full", "both"),
        default="full",
        help="Which scope to visualize.",
    )
    return parser.parse_args()


# --------------- LATENT PLOTS ---------------

def tsne_or_umap_2d(values: np.ndarray, method="umap") -> np.ndarray:
    if values.shape[0] < 2:
        return PCA(n_components=2).fit_transform(StandardScaler().fit_transform(values))
    values_std = StandardScaler().fit_transform(values)
    if method == "umap":
        reducer = UMAP(n_components=2, n_neighbors=15, random_state=751)
        return reducer.fit_transform(values_std)
    else:
        # Fallback to PCA if requested
        return PCA(n_components=2, random_state=751).fit_transform(values_std)

def _label_color_mapping(values: pd.Series) -> tuple[list[Any], dict[Any, Any]]:
    unique_values = sorted(values.dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap("tab10" if len(unique_values) <= 10 else "tab20")
    colors = {value: cmap(index % cmap.N) for index, value in enumerate(unique_values)}
    return unique_values, colors

def scatter_by_category(
    points: np.ndarray,
    categories: pd.Series,
    title: str,
    output_path: str | Path,
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    category_names, colors = _label_color_mapping(categories)

    fig, ax = plt.subplots(figsize=(10, 8))
    category_values = categories.astype(str)
    for category in category_names:
        mask = category_values == category
        ax.scatter(
            points[mask.to_numpy(), 0],
            points[mask.to_numpy(), 1],
            s=12,
            alpha=0.6,
            color=colors[category],
            label=category,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Label", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7, markerscale=2.0)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

# --------------- METRICS ---------------

def plot_multi_metrics(metrics: pd.DataFrame, output_path: str | Path) -> None:
    metric_cols = ["silhouette", "nmi", "ari", "purity"]
    # Group by algorithm and calculate max values across k values
    best_by_algo = metrics.groupby('algorithm')[metric_cols].max()
    
    if best_by_algo.empty:
        return
        
    algorithms = best_by_algo.index.tolist()
    x = np.arange(len(metric_cols))
    width = 0.8 / len(algorithms)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        vals = best_by_algo.loc[algo].values
        ax.bar(x + i*width - 0.4 + width/2, vals, width, label=algo)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_title("Maximum Clustering Metrics By Algorithm")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

# --------------- DISTRIBUTIONS ---------------

def plot_distribution(
    cluster_summary: pd.DataFrame,
    category_name: str,
    output_path: str | Path,
) -> None:
    if cluster_summary.empty:
        return
    
    cluster_summary = cluster_summary.copy()
    cluster_summary['cluster'] = cluster_summary['cluster'].astype(str)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    clusters = cluster_summary['cluster'].unique()
    
    fractions = cluster_summary.set_index('cluster')['top_label_fraction'].to_dict()
    labels = cluster_summary.set_index('cluster')['top_label'].to_dict()
    
    x = np.arange(len(clusters))
    y = [fractions[c] for c in clusters]
    l = [labels[c] for c in clusters]
    
    bars = ax.bar(x, y, color="#4c78a8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel(f"Fraction of dominant {category_name}")
    ax.set_title(f"Dominant {category_name} per Cluster")
    
    for bar, txt in zip(bars, l):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, txt,
                ha='center', va='bottom', rotation=90, fontsize=6)
                
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# --------------- RECONSTRUCTION ---------------

def plot_reconstructions(
    model,
    loader,
    device,
    normalizer,
    output_dir: Path
):
    model.eval()
    batch = next(iter(loader))
    prepared = prepare_batch(batch, device, normalizer)
    
    with torch.no_grad():
        outputs = model(
            prepared["mel_norm"],
            prepared["lyrics"],
            prepared["metadata"],
            prepared["condition"]
        )
        
    mel_recon_norm = outputs["mel_recon_norm"]
    mel_orig_norm = prepared["mel_norm"]
    
    mel_recon = normalizer.denormalize(mel_recon_norm).cpu().numpy()
    mel_orig = normalizer.denormalize(mel_orig_norm).cpu().numpy()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = min(3, mel_orig.shape[0])
    
    for i in range(n_samples):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        orig_img = librosa.display.specshow(
            librosa.power_to_db(mel_orig[i, 0], ref=np.max),
            y_axis='mel', x_axis='time', ax=axes[0]
        )
        axes[0].set_title(f"Original Mel Spectrogram - Track {batch['music_name'][i][:20]}")
        fig.colorbar(orig_img, ax=axes[0], format='%+2.0f dB')
        
        recon_img = librosa.display.specshow(
            librosa.power_to_db(mel_recon[i, 0], ref=np.max),
            y_axis='mel', x_axis='time', ax=axes[1]
        )
        axes[1].set_title(f"Reconstructed Mel Spectrogram")
        fig.colorbar(recon_img, ax=axes[1], format='%+2.0f dB')
        
        fig.tight_layout()
        fig.savefig(output_dir / f"reconstruction_{i}.png", dpi=200)
        plt.close(fig)


# --------------- MAIN SCRIPT ---------------

def main():
    args = parse_args()
    
    output_root = project_path(args.output_root) / args.experiment_name
    output_root.mkdir(parents=True, exist_ok=True)
    
    latent_dir_base = project_path(args.latent_root) / args.experiment_name
    cluster_dir_base = project_path(args.clustering_root) / args.experiment_name
    
    for scope in scopes_from_arg(args.scope):
        log(f"--- Processing scope: {scope} ---")
        out_scope = output_root / scope
        lat_scope = latent_dir_base / scope
        clu_scope = cluster_dir_base / scope
        
        if not lat_scope.exists():
            log(f"Latent dir {lat_scope} missing, skipping.")
            continue
        
        # Load Labels
        labels = pq.read_table(lat_scope / "labels.parquet").to_pandas()
        
        # 1. Latent Plots (UMAP / PCA)
        log("Generating Latent Space UMAP Plots...")
        latent_out = out_scope / "latent_space"
        for embed_name in ["mu", "metadata_assisted"]:
            npy_file = lat_scope / f"{embed_name}.npy"
            if npy_file.exists():
                embed_vals = np.load(npy_file)
                # Ensure we only use up to 5000 samples for UMAP to avoid massive memory footprints
                if embed_vals.shape[0] > 5000:
                    idx = np.random.choice(embed_vals.shape[0], 5000, replace=False)
                    sub_embed = embed_vals[idx]
                    sub_labels = labels.iloc[idx].reset_index(drop=True)
                else:
                    sub_embed = embed_vals
                    sub_labels = labels.reset_index(drop=True)
                
                reduced = tsne_or_umap_2d(sub_embed, method="umap")
                scatter_by_category(reduced, sub_labels["music_lang"], f"{embed_name} UMAP by Language", latent_out / f"{embed_name}_umap_lang.png")
                scatter_by_category(reduced, sub_labels["main_genre"], f"{embed_name} UMAP by Genre", latent_out / f"{embed_name}_umap_genre.png")
        
        # 2. Cluster Distributions
        log("Generating Cluster Distribution Plots...")
        dist_out = out_scope / "distributions"
        algo = "vae_metadata_assisted_kmeans"
        genre_summary_file = clu_scope / "by_algorithm" / algo / "cluster_summary_by_genre.csv"
        lang_summary_file = clu_scope / "by_algorithm" / algo / "cluster_summary_by_language.csv"
        
        if genre_summary_file.exists():
            gs = pd.read_csv(genre_summary_file)
            # Pick a specific assignment K to plot, usually there are a few. Let's pick k=51 or k=32
            assignments = gs['assignment'].unique()
            for assign in assignments:
                sub_gs = gs[gs['assignment'] == assign]
                plot_distribution(sub_gs, "Genre", dist_out / f"{algo}_{assign}_genre_dist.png")
                
        if lang_summary_file.exists():
            ls = pd.read_csv(lang_summary_file)
            assignments = ls['assignment'].unique()
            for assign in assignments:
                sub_ls = ls[ls['assignment'] == assign]
                plot_distribution(sub_ls, "Language", dist_out / f"{algo}_{assign}_lang_dist.png")

        # 3. Metric Comparisons
        log("Generating Multi-Metric Chart...")
        met_out = out_scope / "metrics"
        comp_metrics_file = clu_scope / "comparison" / "metrics.csv"
        if comp_metrics_file.exists():
            metrics_df = pd.read_csv(comp_metrics_file)
            # Split language and genre metrics
            g_met = metrics_df[metrics_df['target'] == 'genre']
            l_met = metrics_df[metrics_df['target'] == 'language']
            plot_multi_metrics(g_met, met_out / "comparison_metrics_genre.png")
            plot_multi_metrics(l_met, met_out / "comparison_metrics_language.png")
            
        # 4. Reconstruction Examples (we will do this via the model)
        log("Generating Reconstructions via Checkpoint...")
        rec_out = out_scope / "reconstructions"
        exp_dir = project_path("1_training/experiments") / args.experiment_name
        ckpt_file = exp_dir / "best_checkpoint" / "model.pt"
        conf_file = exp_dir / "resolved_config.yaml"
        norm_file = exp_dir / "mel_normalization.json"
        
        import yaml
        
        if ckpt_file.exists() and conf_file.exists():
            with open(conf_file, 'r') as f:
                config = yaml.safe_load(f)
            
            dconf = config['data']
            dataset = MusicVAEDataset(
                model_input_path=project_path(dconf['model_input_path']),
                mel_tensor_dir=project_path(dconf['mel_tensor_dir']),
                mel_cache_chunks=2
            )
            # Take a small batch
            loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=music_collate_fn, shuffle=False)
            
            device = select_device("cuda")
            model = build_model(config['vae']['model']).to(device)
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            normalizer = MelNormalizer.from_file(norm_file)
            
            plot_reconstructions(model, loader, device, normalizer, rec_out)
            dataset.close()
        else:
            log("Missing checkpoint or config for reconstructions.")
            
    log("All visualizations completed successfully.")

if __name__ == "__main__":
    main()
