# Clustering Visualizations

This stage creates the visualization artifacts required by the clustering items in
`AGENTS.md`.

## Directory Structure

```text
4_Visulizations/
|-- visualize_src/
`-- cluster_visulization_outputs/
```

Planned script location:

```text
4_Visulizations/visualize_src/run_cluster_visualizations.py
```

Planned output root:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/
|-- validation/
`-- full/
```

## Default Inputs

Clustering outputs:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/
|-- cluster_assignments.parquet
|-- metrics.csv
|-- silhouette_sweep.csv
|-- comparison/
|   |-- metrics.csv
|   |-- ranked_metrics.csv
|   `-- ranked_silhouette.csv
`-- by_algorithm/{method}/
    |-- cluster_assignments.parquet
    |-- metrics.csv
    |-- silhouette_sweep.csv
    |-- cluster_summary_by_genre.csv
    `-- cluster_summary_by_language.csv
```

Latent inputs:

```text
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/
|-- mu.npy
|-- metadata_assisted.npy
|-- labels.parquet
`-- row_indices.npy
```

Methods to visualize:

```text
pca_kmeans
autoencoder_kmeans
spectral_feature_kmeans
vae_mu_kmeans
vae_metadata_assisted_kmeans
```

Scopes:

```text
validation
full
```

## Plot Plan

### 1. Comparison Metric Bar Charts

Purpose:
Compare VAE-based clustering against PCA + K-Means, Autoencoder + K-Means, and direct spectral feature clustering.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/ranked_metrics.csv
```

Process:

1. Filter rows by `target`.
2. Plot each method on the x-axis.
3. Plot metric value on the y-axis.
4. Create separate charts for `silhouette`, `nmi`, `ari`, and `purity`.
5. Use `k=51` for genre charts and `k=4` for language charts.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/
```

Filenames:

```text
comparison_genre_silhouette_bar.png
comparison_genre_nmi_bar.png
comparison_genre_ari_bar.png
comparison_genre_purity_bar.png
comparison_language_silhouette_bar.png
comparison_language_nmi_bar.png
comparison_language_ari_bar.png
comparison_language_purity_bar.png
```

### 2. Ranked Comparison Tables

Purpose:
Make the quantitative comparison readable without opening CSV files.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/ranked_metrics.csv
```

Process:

1. Select columns: `algorithm`, `embedding`, `target`, `k`, `silhouette`, `nmi`, `ari`, `purity`.
2. Sort genre by `nmi` and `purity`.
3. Sort language by `nmi` and `purity`.
4. Render compact PNG tables.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/
```

Filenames:

```text
ranked_genre_metrics_table.png
ranked_language_metrics_table.png
```

### 3. Silhouette Sweep Line Plots

Purpose:
Show how cluster compactness changes across `k`.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/ranked_silhouette.csv
```

Process:

1. Plot `k` on the x-axis.
2. Plot `silhouette` on the y-axis.
3. Use one line per method.
4. Include all k values in `silhouette_sweep.csv`.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/
```

Filename:

```text
comparison_silhouette_sweep_line.png
```

### 4. Latent Space PCA Scatter Plots

Purpose:
Satisfy the `AGENTS.md` latent space plot requirement.

Input:

```text
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/mu.npy
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/metadata_assisted.npy
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/labels.parquet
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/cluster_assignments.parquet
```

Process:

1. Compute 2D PCA for `mu.npy`.
2. Compute 2D PCA for `metadata_assisted.npy`.
3. Join PCA coordinates with labels and cluster assignments.
4. Plot points colored by cluster assignment, genre, and language.
5. For cluster coloring, use the corresponding assignment column for each method.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/latent_space/
```

Filenames:

```text
latent_pca_mu_colored_by_genre.png
latent_pca_mu_colored_by_language.png
latent_pca_metadata_assisted_colored_by_genre.png
latent_pca_metadata_assisted_colored_by_language.png
latent_pca_vae_mu_kmeans_k51_colored_by_cluster.png
latent_pca_vae_metadata_assisted_kmeans_k51_colored_by_cluster.png
latent_pca_vae_mu_kmeans_k4_colored_by_cluster.png
latent_pca_vae_metadata_assisted_kmeans_k4_colored_by_cluster.png
```

### 5. Method Embedding PCA Scatter Plots

Purpose:
Visualize each comparison method in its own embedding space.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/comparison/cluster_assignments.parquet
```

Additional feature inputs:

```text
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/mu.npy
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/metadata_assisted.npy
0_data_pre_processing/processed_dataset/model_input/model_input_dataset.parquet
0_data_pre_processing/processed_dataset/mel_tensors/
```

Process:

1. Recreate each method embedding:
   - `pca_kmeans`: PCA multimodal embedding.
   - `autoencoder_kmeans`: autoencoder multimodal embedding.
   - `spectral_feature_kmeans`: compact spectral feature embedding.
   - `vae_mu_kmeans`: VAE `mu`.
   - `vae_metadata_assisted_kmeans`: VAE metadata-assisted embedding.
2. Compute 2D PCA for each method embedding.
3. Plot colored by method cluster assignment for `k=51` and `k=4`.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/method_embeddings/
```

Filenames:

```text
pca_kmeans_embedding_pca_k51_colored_by_cluster.png
pca_kmeans_embedding_pca_k4_colored_by_cluster.png
autoencoder_kmeans_embedding_pca_k51_colored_by_cluster.png
autoencoder_kmeans_embedding_pca_k4_colored_by_cluster.png
spectral_feature_kmeans_embedding_pca_k51_colored_by_cluster.png
spectral_feature_kmeans_embedding_pca_k4_colored_by_cluster.png
vae_mu_kmeans_embedding_pca_k51_colored_by_cluster.png
vae_mu_kmeans_embedding_pca_k4_colored_by_cluster.png
vae_metadata_assisted_kmeans_embedding_pca_k51_colored_by_cluster.png
vae_metadata_assisted_kmeans_embedding_pca_k4_colored_by_cluster.png
```

### 6. Cluster Distribution Over Genres

Purpose:
Satisfy the `AGENTS.md` cluster distribution over genres requirement.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/cluster_assignments.parquet
```

Process:

1. Select the genre assignment column for `k=51`.
2. Build a cluster-by-genre crosstab.
3. Normalize each row by cluster size.
4. Plot a heatmap.
5. Save the long-form distribution table for reuse.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/
```

Filenames:

```text
cluster_genre_distribution_k51_heatmap.png
cluster_genre_distribution_k51.csv
```

### 7. Cluster Distribution Over Languages

Purpose:
Satisfy the `AGENTS.md` cluster distribution over languages requirement.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/cluster_assignments.parquet
```

Process:

1. Select the language assignment column for `k=4`.
2. Build a cluster-by-language crosstab.
3. Normalize each row by cluster size.
4. Plot a heatmap.
5. Save the long-form distribution table for reuse.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/
```

Filenames:

```text
cluster_language_distribution_k4_heatmap.png
cluster_language_distribution_k4.csv
```

### 8. Cluster Size Distribution

Purpose:
Show whether methods create balanced clusters or a few dominant clusters.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/cluster_assignments.parquet
```

Process:

1. Count rows per cluster for `k=51`.
2. Count rows per cluster for `k=4`.
3. Plot bar charts.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/
```

Filenames:

```text
cluster_size_k51_bar.png
cluster_size_k4_bar.png
```

### 9. Cluster Representative Examples

Purpose:
Prepare examples for qualitative inspection and later reconstruction.

Input:

```text
3_clustering/clustering_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/cluster_assignments.parquet
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/mu.npy
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/metadata_assisted.npy
```

Process:

1. For each cluster, compute the cluster centroid in the relevant embedding.
2. Pick the closest item to the centroid as the representative.
3. Save `music_id`, `row_index`, `main_genre`, `music_lang`, `cluster`, and distance to centroid.
4. Select representatives for `k=51` and `k=4`.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/
```

Filenames:

```text
cluster_representatives_k51.csv
cluster_representatives_k4.csv
```

### 10. Reconstruction Example Grids

Purpose:
Satisfy the `AGENTS.md` reconstruction examples from VAE latent space requirement.

Input:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/cluster_representatives_k51.csv
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/by_algorithm/{method}/cluster_representatives_k4.csv
1_training/experiments/mm_cbeta_vae_experiment_002/best_checkpoint/model.pt
1_training/experiments/mm_cbeta_vae_experiment_002/mel_normalization.json
0_data_pre_processing/processed_dataset/mel_tensors/
2_inference/latents_by_best_checkpoints/mm_cbeta_vae_experiment_002/{scope}/mu.npy
```

Process:

1. Load representative rows.
2. Load original mel spectrograms.
3. Decode VAE `mu` with its condition embedding.
4. Denormalize reconstructed mel spectrograms.
5. Plot original vs reconstructed mel spectrogram pairs.

Output details:

```text
4_Visulizations/cluster_visulization_outputs/mm_cbeta_vae_experiment_002/{scope}/reconstruction_examples/
```

Filenames:

```text
reconstruction_examples_genre_k51_grid.png
reconstruction_examples_language_k4_grid.png
reconstruction_examples_manifest.csv
```

## Recommended Implementation Order

1. Generate comparison metric plots.
2. Generate silhouette sweep plots.
3. Generate genre/language heatmaps.
4. Generate latent PCA scatter plots.
5. Generate cluster representatives.
6. Generate reconstruction example grids.

