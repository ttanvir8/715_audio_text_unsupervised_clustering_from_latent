from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class LyricsVAEConfig:
    input_dim: int = 1024
    z_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.10


class BasicLyricsVAE(nn.Module):
    """A simple non-conditional VAE over precomputed lyrics embeddings."""

    def __init__(self, config: LyricsVAEConfig) -> None:
        super().__init__()
        self.config = config

        encoder_layers: list[nn.Module] = []
        previous_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            previous_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        latent_dim = previous_dim
        self.mu = nn.Linear(latent_dim, config.z_dim)
        self.logvar = nn.Linear(latent_dim, config.z_dim)

        decoder_layers: list[nn.Module] = []
        previous_dim = config.z_dim
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, lyrics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(lyrics)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, lyrics: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(lyrics)
        z = self.reparameterize(mu, logvar)
        return {
            "lyrics_recon": self.decode(z),
            "mu": mu,
            "logvar": logvar,
        }


def build_model(config_dict: dict) -> BasicLyricsVAE:
    config = LyricsVAEConfig(
        input_dim=int(config_dict.get("input_dim", 1024)),
        z_dim=int(config_dict.get("z_dim", 64)),
        hidden_dims=[int(value) for value in config_dict.get("hidden_dims", [512, 256])],
        dropout=float(config_dict.get("dropout", 0.10)),
    )
    return BasicLyricsVAE(config)

