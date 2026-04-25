# Easy Lyrics VAE Inference

This stage exports encoder latents from the best `0_1_easy_training` checkpoint.

```bash
source /home/tanvir/fun/tts/voxtral/.venv/bin/activate

python 0_2_easy_inference/inference_src/export_best_checkpoint_latents.py \
  --experiment-dir 0_1_easy_training/experiments/lyrics_vae_run_001 \
  --scope both \
  --device auto \
  --overwrite
```

Outputs:

```text
0_2_easy_inference/latents_by_best_checkpoints/lyrics_vae_run_001/
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

`full` means the selected easy train+validation rows from training. With the current `0_1_easy_training/configs/data_base.yaml` setup, that is the full lyrics parquet used by the easy pipeline.
