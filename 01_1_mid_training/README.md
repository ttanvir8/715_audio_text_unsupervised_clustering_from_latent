# Medium Audio+Lyrics VAE Training

This directory is the medium-task training pipeline for a non-conditional VAE over audio and lyrics. It keeps the convolutional spectrogram encoder from the multimodal training direction, combines it with the lyrics embedding encoder, concatenates those two representations, and exports validation latent vectors for downstream clustering.

The model does not consume genre embeddings, language embeddings, metadata vectors, or condition embeddings.

## Structure

```text
01_1_mid_training/
|-- configs/
|   |-- data_base.yaml
|   |-- experiments/run_001.yaml
|   `-- vae/audio_lyrics_vae_default.yaml
|-- dataloader/
|-- training_scripts/
|-- sanity_check/
`-- experiments/
```

## Dataset

The default config reads the row-aligned medium inputs from:

```text
0_data_pre_processing/processed_dataset/model_input/model_input_dataset.parquet
0_data_pre_processing/processed_dataset/mel_tensors/
0_data_pre_processing/processed_dataset/genre_language/genre_language_embeddings.parquet
```

Inputs used by the VAE:

```text
lyrics_e5_large_embedding: [B, 1024]
melspectrogram: [B, 1, 128, 1292]
```

The train/validation split preserves the same ratio within each language. By default it uses all available rows; set `max_rows_per_language` in `configs/data_base.yaml` for a small debug run.

## Environment

Use the project AI environment:

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate
uv pip list | grep -E "torch|pyarrow|pandas|numpy|scikit-learn|PyYAML|tqdm"
```

## Sanity Check

From the repository root:

```bash
python 01_1_mid_training/sanity_check/check_audio_lyrics_vae_pipeline.py \
  --experiment-config 01_1_mid_training/configs/experiments/run_001.yaml \
  --batch-size 2 \
  --device cpu
```

Expected output includes:

```text
Audio+lyrics VAE sanity check passed.
lyrics_input: (2, 1024)
mel_input: (2, 1, 128, 1292)
mel_recon_norm: (2, 1, 128, 1292)
lyrics_recon: (2, 1024)
mu: (2, 64)
logvar: (2, 64)
```

## Smoke Run

```bash
python 01_1_mid_training/training_scripts/train_vae.py \
  --experiment-config 01_1_mid_training/configs/experiments/run_001.yaml \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --max-stat-batches 1
```

## Full Run

```bash
python 01_1_mid_training/training_scripts/train_vae.py \
  --experiment-config 01_1_mid_training/configs/experiments/run_001.yaml
```

Outputs are written under:

```text
01_1_mid_training/experiments/audio_lyrics_vae_run_001/
|-- resolved_config.yaml
|-- run_manifest.yaml
|-- split_indices.json
|-- mel_normalization.json
|-- metrics.csv
|-- checkpoints_of_each_epochs/
|-- best_checkpoint/
`-- latent_exports/
    |-- val_mu.npy
    |-- val_row_indices.npy
    |-- val_labels.parquet
    `-- latent_export_metadata.json
```

Use `latent_exports/val_mu.npy` as the fair audio+lyrics latent representation for the medium clustering task.
