from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm


class MelNormalizer:
    def __init__(self, mean: float, std: float, eps: float = 1e-6) -> None:
        self.mean = float(mean)
        self.std = float(std)
        self.eps = float(eps)

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        mel_log = torch.log1p(mel.float())
        return (mel_log - self.mean) / (self.std + self.eps)

    def denormalize(self, mel_norm: torch.Tensor) -> torch.Tensor:
        mel_log = mel_norm * (self.std + self.eps) + self.mean
        return torch.expm1(mel_log).clamp_min(0.0)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mel_transform": "log1p",
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "MelNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(data["mean"]), float(data["std"]), float(data.get("eps", 1e-6)))

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@torch.no_grad()
def compute_mel_normalizer(
    loader,
    max_batches: int | None = None,
    desc: str = "mel stats",
) -> MelNormalizer:
    total = 0
    sum_values = 0.0
    sum_squares = 0.0
    iterable = islice(loader, max_batches) if max_batches is not None else loader
    progress_total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for batch in tqdm(iterable, desc=desc, total=progress_total, leave=False):
        mel_log = torch.log1p(batch["melspectrogram"].float())
        total += mel_log.numel()
        sum_values += float(mel_log.sum().item())
        sum_squares += float((mel_log * mel_log).sum().item())

    if total == 0:
        raise ValueError("Cannot compute mel normalization stats from an empty loader.")

    mean = sum_values / total
    variance = max(sum_squares / total - mean * mean, 1e-12)
    std = variance**0.5
    return MelNormalizer(mean=mean, std=std)

