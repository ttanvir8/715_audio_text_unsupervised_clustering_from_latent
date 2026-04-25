from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    lyrics_input_dim: int = 1024
    z_dim: int = 64
    audio_dim: int = 256
    lyrics_dim: int = 256
    fused_dim: int = 256
    dropout: float = 0.10


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=stride),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class AudioEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvNormAct(1, 32, stride=(2, 4)),
            ResidualDownBlock(32, 64, stride=(2, 2)),
            ResidualDownBlock(64, 128, stride=(2, 2)),
            ResidualDownBlock(128, 192, stride=(2, 3)),
            ResidualDownBlock(192, 256, stride=(2, 3)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, mel_norm: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(mel_norm))


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size
        self.conv = ConvNormAct(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return self.conv(x)


class MelDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256 * 4 * 9),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            UpsampleConvBlock(256, 192, (8, 27)),
            UpsampleConvBlock(192, 128, (16, 81)),
            UpsampleConvBlock(128, 64, (32, 162)),
            UpsampleConvBlock(64, 32, (64, 324)),
            UpsampleConvBlock(32, 16, (128, 1312)),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], 256, 4, 9)
        x = self.blocks(x)
        start = (x.shape[-1] - 1292) // 2
        return x[..., start : start + 1292]


class AudioLyricsConcatVAE(nn.Module):
    """Non-conditional VAE over concatenated audio and lyrics representations."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.audio_encoder = AudioEncoder(config.audio_dim)
        self.lyrics_encoder = nn.Sequential(
            nn.LayerNorm(config.lyrics_input_dim),
            nn.Linear(config.lyrics_input_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.lyrics_dim),
            nn.LayerNorm(config.lyrics_dim),
        )

        fusion_input_dim = config.audio_dim + config.lyrics_dim
        self.fusion_base = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.fused_dim),
        )
        self.fusion_gate = nn.Linear(fusion_input_dim, config.fused_dim)
        self.fusion_norm = nn.LayerNorm(config.fused_dim)

        self.latent_encoder = nn.Sequential(
            nn.Linear(config.fused_dim, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.mu = nn.Linear(128, config.z_dim)
        self.logvar = nn.Linear(128, config.z_dim)

        self.mel_decoder = MelDecoder(config.z_dim)
        self.lyrics_decoder = nn.Sequential(
            nn.Linear(config.z_dim, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, config.lyrics_input_dim),
        )

    def encode(self, mel_norm: torch.Tensor, lyrics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_embedding = self.audio_encoder(mel_norm)
        lyrics_embedding = self.lyrics_encoder(lyrics)
        fused_input = torch.cat([audio_embedding, lyrics_embedding], dim=-1)
        fused_base = self.fusion_base(fused_input)
        fused_gate = torch.sigmoid(self.fusion_gate(fused_input))
        fused = self.fusion_norm(fused_base * fused_gate)
        hidden = self.latent_encoder(fused)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mel_recon_norm": self.mel_decoder(z),
            "lyrics_recon": self.lyrics_decoder(z),
        }

    def forward(self, mel_norm: torch.Tensor, lyrics: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(mel_norm, lyrics)
        z = self.reparameterize(mu, logvar)
        outputs = self.decode(z)
        outputs.update({"mu": mu, "logvar": logvar})
        return outputs


def build_model(config_dict: dict) -> AudioLyricsConcatVAE:
    config = ModelConfig(
        lyrics_input_dim=int(config_dict.get("lyrics_input_dim", 1024)),
        z_dim=int(config_dict.get("z_dim", 64)),
        audio_dim=int(config_dict.get("audio_dim", 256)),
        lyrics_dim=int(config_dict.get("lyrics_dim", 256)),
        fused_dim=int(config_dict.get("fused_dim", 256)),
        dropout=float(config_dict.get("dropout", 0.10)),
    )
    return AudioLyricsConcatVAE(config)

