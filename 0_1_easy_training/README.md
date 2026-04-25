# Easy Lyrics VAE Training

This directory is the lyrics-only version of the VAE training pipeline. It trains a basic non-conditional VAE over precomputed lyrics embeddings and exports validation latent vectors for downstream clustering.

## Structure

```text
0_1_easy_training/
|-- configs/
|   |-- data_base.yaml
|   |-- experiments/run_001.yaml
|   `-- vae/lyrics_vae_default.yaml
|-- dataloader/
|-- training_scripts/
|-- sanity_check/
`-- experiments/
```

## Dataset

The default config reads:

```text
0_data_pre_processing/processed_dataset/lyrics_e5_large/lyrics_embeddings.parquet
```

Only the configured lyrics embedding column is used as model input. No audio, genre embedding, language embedding, metadata embedding, or conditional vector is passed into the VAE.

The default easy run now uses the full lyrics parquet (`9661` rows in the current processed dataset metadata) and then splits each language independently with the same `train_ratio` and `val_ratio`.

If you want to intentionally shrink the run later, set `max_rows_per_language` in `configs/data_base.yaml` to a positive integer.

## Environment

Use the project AI environment:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate
uv pip list | grep -E "torch|pyarrow|pandas|numpy|scikit-learn|PyYAML|tqdm"
```

## Sanity Check

From the repository root:

```bash
python 0_1_easy_training/sanity_check/check_lyrics_vae_pipeline.py \
  --experiment-config 0_1_easy_training/configs/experiments/run_001.yaml \
  --batch-size 4 \
  --device cpu
```

Expected shapes:

```text
lyrics_input: (4, 1024)
lyrics_recon: (4, 1024)
mu: (4, 64)
logvar: (4, 64)
```

## Smoke Run

```bash
python 0_1_easy_training/training_scripts/train_vae.py \
  --experiment-config 0_1_easy_training/configs/experiments/run_001.yaml \
  --epochs 1 \
  --batch-size 8 \
  --max-train-batches 2 \
  --max-val-batches 1
```

## Full Easy Run

```bash
python 0_1_easy_training/training_scripts/train_vae.py \
  --experiment-config 0_1_easy_training/configs/experiments/run_001.yaml
```

Outputs are written under:

```text
0_1_easy_training/experiments/lyrics_vae_run_001/
|-- resolved_config.yaml
|-- run_manifest.yaml
|-- split_indices.json
|-- metrics.csv
|-- checkpoints_of_each_epochs/
|-- best_checkpoint/
`-- latent_exports/
    |-- val_mu.npy
    |-- val_row_indices.npy
    |-- val_labels.parquet
    `-- latent_export_metadata.json
```
