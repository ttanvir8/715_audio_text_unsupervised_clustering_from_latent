# Medium Audio+Lyrics VAE Inference

This stage exports encoder latents from the best `01_1_mid_training` checkpoint.

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 01_2_inference/inference_src/export_best_checkpoint_latents.py \
  --experiment-dir 01_1_mid_training/experiments/audio_lyrics_vae_run_001 \
  --scope both \
  --device auto \
  --overwrite
```

Outputs:

```text
01_2_inference/latents_by_best_checkpoints/audio_lyrics_vae_run_001/
|-- validation/
|   |-- mu.npy
|   |-- logvar.npy
|   |-- row_indices.npy
|   |-- labels.parquet
|   `-- latent_export_metadata.json
`-- full/
    |-- mu.npy
    |-- logvar.npy
    |-- row_indices.npy
    |-- labels.parquet
    `-- latent_export_metadata.json
```

`full` means the selected medium train+validation rows from the original training split in
`01_1_mid_training/experiments/<run_name>/split_indices.json`.
