# Multi-Modal Clustering

This stage runs the clustering comparisons required by `AGENTS.md` and writes
metrics, assignments, summaries, and comparison tables for visualization.

## Run

Run from the project root:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 3_clustering/clustering_src/run_multimodal_clustering.py \
  --latent-root 2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002 \
  --output-root 3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002 \
  --scope both \
  --comparison-methods pca_kmeans,autoencoder_kmeans,spectral_feature_kmeans,vae_mu_kmeans,vae_metadata_assisted_kmeans \
  --comparison-kmeans-algorithm cuml_kmeans
```

`cuml_kmeans` uses RAPIDS cuML on CUDA. The script logs the CUDA device before
clustering, for example `cuda:0 (NVIDIA GeForce RTX 5060 Laptop GPU)`.

## Compared Methods

The command above runs these AGENTS.md comparison methods:

- `pca_kmeans`: PCA over raw multimodal features, then K-Means.
- `autoencoder_kmeans`: lightweight feature autoencoder embedding, then K-Means.
- `spectral_feature_kmeans`: direct compact mel/spectral features, then K-Means.
- `vae_mu_kmeans`: VAE `mu` latent embedding, then K-Means.
- `vae_metadata_assisted_kmeans`: VAE metadata-assisted latent embedding, then K-Means.

## Inputs

Default inputs used by the command:

```text
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/
|-- validation/
|   |-- mu.npy
|   |-- metadata_assisted.npy
|   `-- labels.parquet
`-- full/
    |-- mu.npy
    |-- metadata_assisted.npy
    `-- labels.parquet

0_data_pre_processing/processed_dataset/model_input/model_input_dataset.parquet
0_data_pre_processing/processed_dataset/mel_tensors/
```

## Outputs

Outputs are written to:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/
|-- validation/
`-- full/
```

Each scope contains combined outputs:

```text
cluster_assignments.parquet
metrics.csv
silhouette_sweep.csv
cluster_summary_by_genre.csv
cluster_summary_by_language.csv
run_metadata.json
```

Each scope also contains per-method outputs:

```text
by_algorithm/
|-- autoencoder_kmeans/
|-- pca_kmeans/
|-- spectral_feature_kmeans/
|-- vae_metadata_assisted_kmeans/
`-- vae_mu_kmeans/
```

Each method folder contains:

```text
cluster_assignments.parquet
metrics.csv
silhouette_sweep.csv
cluster_summary_by_genre.csv
cluster_summary_by_language.csv
```

The side-by-side comparison outputs are in:

```text
comparison/
|-- cluster_assignments.parquet
|-- metrics.csv
|-- ranked_metrics.csv
|-- ranked_silhouette.csv
`-- silhouette_sweep.csv
```

Use `comparison/ranked_metrics.csv` and `comparison/ranked_silhouette.csv` for
the final comparison plots.

## Metrics

The metrics files include:

- Silhouette Score
- Normalized Mutual Information, NMI
- Adjusted Rand Index, ARI
- Cluster Purity
- Cluster count
- Noise count, mainly useful for DBSCAN runs

Rows are reported for both targets:

- `genre`: compared against `main_genre_id`
- `language`: compared against `music_lang_id`

## Optional Generic Clustering

The runner can also cluster the VAE embeddings with generic algorithms:

```bash
python 3_clustering/clustering_src/run_multimodal_clustering.py \
  --latent-root 2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002 \
  --output-root 3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002_generic \
  --scope validation \
  --comparison-methods "" \
  --embeddings mu,metadata_assisted \
  --algorithms cuml_kmeans,cuml_agglomerative,cuml_dbscan \
  --agglomerative-linkage single \
  --dbscan-eps 0.5 \
  --dbscan-min-samples 5
```

cuML Agglomerative currently supports `single` linkage.
