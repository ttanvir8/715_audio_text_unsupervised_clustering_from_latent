from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    mel_weight: float = 1.0
    lyrics_weight: float = 0.4
    beta_max: float = 1.0
    beta_warmup_fraction: float = 0.25
    free_bits: float = 0.05
    active_kl_threshold: float = 0.01


def beta_for_step(global_step: int, total_steps: int, config: LossConfig) -> float:
    warmup_steps = max(1, int(total_steps * config.beta_warmup_fraction))
    return min(config.beta_max, config.beta_max * float(global_step) / float(warmup_steps))


def compute_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    beta: float,
    config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    mel_smooth_l1 = F.smooth_l1_loss(outputs["mel_recon_norm"], targets["mel_norm"])
    mel_mse = F.mse_loss(outputs["mel_recon_norm"], targets["mel_norm"])
    mel_recon_loss = 0.8 * mel_smooth_l1 + 0.2 * mel_mse

    lyrics_recon_unit = F.normalize(outputs["lyrics_recon"], dim=-1)
    lyrics_target_unit = F.normalize(targets["lyrics"], dim=-1)
    lyrics_cosine_loss = 1.0 - F.cosine_similarity(
        lyrics_recon_unit,
        lyrics_target_unit,
        dim=-1,
    ).mean()
    lyrics_mse_loss = F.mse_loss(outputs["lyrics_recon"], targets["lyrics"])
    lyrics_recon_loss = lyrics_cosine_loss + 0.05 * lyrics_mse_loss

    recon_loss = config.mel_weight * mel_recon_loss + config.lyrics_weight * lyrics_recon_loss

    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    raw_kl_loss = kl_per_dim.sum(dim=-1).mean()
    kl_loss = torch.clamp(kl_per_dim, min=config.free_bits).sum(dim=-1).mean()
    mean_kl_per_dim = kl_per_dim.mean(dim=0)
    active_kl_dims = (mean_kl_per_dim > config.active_kl_threshold).sum()
    mu_std_per_dim = mu.std(dim=0, unbiased=False)

    loss = recon_loss + beta * kl_loss
    metrics = {
        "loss": float(loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "mel_recon_loss": float(mel_recon_loss.detach().cpu()),
        "lyrics_recon_loss": float(lyrics_recon_loss.detach().cpu()),
        "lyrics_cosine_loss": float(lyrics_cosine_loss.detach().cpu()),
        "lyrics_mse_loss": float(lyrics_mse_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "raw_kl_loss": float(raw_kl_loss.detach().cpu()),
        "active_kl_dims": float(active_kl_dims.detach().cpu()),
        "mu_abs_mean": float(mu.abs().mean().detach().cpu()),
        "mu_std_mean": float(mu_std_per_dim.mean().detach().cpu()),
        "mu_std_min": float(mu_std_per_dim.min().detach().cpu()),
        "mu_std_max": float(mu_std_per_dim.max().detach().cpu()),
        "logvar_mean": float(logvar.mean().detach().cpu()),
        "beta": float(beta),
    }
    return loss, metrics


def build_loss_config(config_dict: dict) -> LossConfig:
    return LossConfig(
        mel_weight=float(config_dict.get("mel_weight", 1.0)),
        lyrics_weight=float(config_dict.get("lyrics_weight", 0.4)),
        beta_max=float(config_dict.get("beta_max", 1.0)),
        beta_warmup_fraction=float(config_dict.get("beta_warmup_fraction", 0.25)),
        free_bits=float(config_dict.get("free_bits", 0.05)),
        active_kl_threshold=float(config_dict.get("active_kl_threshold", 0.01)),
    )

