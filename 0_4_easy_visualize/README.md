# Easy Lyrics VAE Visualize

This stage visualizes the easy lyrics VAE clusters with t-SNE or UMAP.

The default method is t-SNE because it is available through `scikit-learn` in the project AI environment. UMAP is supported with `--method umap` if `umap-learn` is installed later.

## Run

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 0_4_easy_visualize/visualize_src/run_easy_cluster_manifold.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --clustering-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
  --output-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --scope both \
  --method tsne \
  --embedding mu
```

## Outputs

Per scope, the script writes:

```text
manifold/
|-- tsne_mu_colored_by_cluster.png
|-- tsne_mu_colored_by_language.png
`-- tsne_mu_points.parquet
visualization_manifest.json
```

For large scopes, `--max-points` controls the number of plotted rows. The default is `3000`, sampled in a cluster-balanced way so the smaller clusters remain visible.

## Latent Space

Create latent-space plots for both validation and full scopes:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 0_4_easy_visualize/visualize_src/run_easy_latent_space.py \
  --latent-root 0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001 \
  --clustering-root 0_3_easy_clustering/clustering_outputs/lyrics_vae_run_001 \
  --output-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --scope both
```

Per scope, outputs are written under the requested directory name:

```text
letent_space/
|-- pca_mu_colored_by_cluster.png
|-- pca_mu_colored_by_language.png
|-- pca_mu_points.parquet
|-- pca_logvar_colored_by_cluster.png
|-- pca_logvar_colored_by_language.png
|-- pca_logvar_points.parquet
|-- tsne_mu_colored_by_cluster.png
|-- tsne_mu_colored_by_language.png
|-- tsne_mu_points.parquet
`-- latent_space_manifest.json
```

## Zoomed Music Names

Create a zoomed latent-space plot with `music_name` labels drawn on the dots:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 0_4_easy_visualize/visualize_src/run_easy_latent_zoom_music_names.py \
  --points-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --output-root 0_4_easy_visualize/visualization_outputs/lyrics_vae_run_001 \
  --scope both \
  --method tsne \
  --embedding mu
```

The default zoom target is the smallest K-Means cluster for each scope. Outputs are written to:

```text
letent_space/zoomed_music_names/
|-- tsne_mu_cluster_<id>_smallest_music_names.png
|-- tsne_mu_cluster_<id>_smallest_labeled_points.parquet
`-- tsne_mu_cluster_<id>_smallest_manifest.json
```

Useful options:

```bash
--focus-cluster 0
--focus-language es
--music-name-contains "song title text"
--row-index 1234
--max-labels 80
```
