from __future__ import annotations

import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from dataloader.dataset import (  # noqa: E402,F401
    DEFAULT_MEL_TENSOR_DIR,
    DEFAULT_MODEL_INPUT_PATH,
    MODEL_INPUT_COLUMNS,
    MusicVAEDataset,
    create_dataloader,
    music_collate_fn,
)


if __name__ == "__main__":
    loader = create_dataloader(batch_size=2)
    batch = next(iter(loader))
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: {value[:2]}")
