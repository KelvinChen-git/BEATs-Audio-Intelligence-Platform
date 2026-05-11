from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MelTransformerESC50(nn.Module):
    def __init__(
        self,
        num_classes: int = 50,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 400,
        hop_length: int = 160,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.input_proj = nn.Linear(n_mels, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.melspec(waveform)
        mel = self.amplitude_to_db(mel)
        mel = mel.transpose(1, 2)
        x = self.input_proj(mel)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)
