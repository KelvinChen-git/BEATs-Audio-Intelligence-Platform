from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def _resolve_beats_root(project_root: Path) -> Path:
    beats_root = project_root / "third_party" / "unilm" / "beats"
    required_files = [beats_root / "BEATs.py", beats_root / "backbone.py"]
    if all(path.exists() for path in required_files):
        return beats_root
    raise FileNotFoundError(
        "Official BEATs repo not found. Expected files under "
        f"{beats_root}. Run scripts/setup_beats.ps1 to clone microsoft/unilm first."
    )


def _add_beats_paths(project_root: Path) -> Path:
    beats_root = _resolve_beats_root(project_root)
    candidates = [beats_root, beats_root.parent]
    for path in candidates:
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)
    return beats_root


def _strip_state_dict_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _load_checkpoint_payload(checkpoint_path: Path) -> tuple[dict, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint object in {checkpoint_path}: expected a dict.")

    cfg = checkpoint.get("cfg")
    if cfg is None:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}: missing 'cfg'.")

    model_state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(model_state, dict):
        raise ValueError(
            f"Unexpected checkpoint format in {checkpoint_path}: expected 'model' or 'state_dict'."
        )
    return cfg, _strip_state_dict_prefixes(model_state)


class OfficialBEATsESC50(nn.Module):
    def __init__(
        self,
        project_root: str | Path,
        checkpoint_path: str | Path,
        num_classes: int = 50,
        freeze_backbone: bool = True,
        time_mask_width: int = 0,
        freq_mask_width: int = 0,
        num_time_masks: int = 0,
        num_freq_masks: int = 0,
        unfreeze_top_layers: int = 0,
    ) -> None:
        super().__init__()
        if min(time_mask_width, freq_mask_width, num_time_masks, num_freq_masks, unfreeze_top_layers) < 0:
            raise ValueError("SpecAugment and unfreeze settings must be non-negative.")

        project_root = Path(project_root)
        checkpoint_path = Path(checkpoint_path)
        _add_beats_paths(project_root)

        try:
            from BEATs import BEATs, BEATsConfig
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ImportError(
                "Could not import official BEATs. Run scripts/setup_beats.ps1 first and make sure "
                "third_party/unilm/beats exists."
            ) from exc

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Put a BEATs checkpoint under checkpoints/."
            )

        cfg_dict, model_state = _load_checkpoint_payload(checkpoint_path)
        cfg = BEATsConfig(cfg_dict)
        if getattr(cfg, "finetuned_model", False):
            cfg.finetuned_model = False
        self.backbone = BEATs(cfg)
        incompatible = self.backbone.load_state_dict(model_state, strict=False)
        allowed_unexpected = {"predictor.weight", "predictor.bias"}
        unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
        missing = set(incompatible.missing_keys)
        if unexpected or missing:
            raise ValueError(
                "Checkpoint could not be loaded into the official BEATs backbone. "
                f"Missing keys: {sorted(missing)}. Unexpected keys: {sorted(unexpected)}."
            )

        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

        if freeze_backbone or unfreeze_top_layers > 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if unfreeze_top_layers > 0:
            self._unfreeze_top_encoder_layers(unfreeze_top_layers)

        embed_dim = getattr(cfg, "encoder_embed_dim", None)
        if embed_dim is None:
            raise ValueError("Could not infer encoder_embed_dim from BEATs config.")
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _unfreeze_top_encoder_layers(self, layer_count: int) -> None:
        layers = getattr(getattr(self.backbone, "encoder", None), "layers", None)
        if layers is None:
            raise ValueError("Could not find BEATs encoder layers for partial unfreezing.")
        if layer_count > len(layers):
            raise ValueError(f"--unfreeze-top-layers must be <= {len(layers)}, got {layer_count}.")
        for layer in layers[-layer_count:]:
            for param in layer.parameters():
                param.requires_grad = True

    def _apply_specaugment(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return fbank
        if self.num_time_masks == 0 and self.num_freq_masks == 0:
            return fbank

        augmented = fbank.clone()
        batch_size, time_steps, freq_bins = augmented.shape
        for item in range(batch_size):
            for _ in range(self.num_time_masks):
                width = self._sample_mask_width(self.time_mask_width, time_steps, augmented.device)
                if width > 0:
                    start = torch.randint(0, time_steps - width + 1, (1,), device=augmented.device).item()
                    augmented[item, start : start + width, :] = 0
            for _ in range(self.num_freq_masks):
                width = self._sample_mask_width(self.freq_mask_width, freq_bins, augmented.device)
                if width > 0:
                    start = torch.randint(0, freq_bins - width + 1, (1,), device=augmented.device).item()
                    augmented[item, :, start : start + width] = 0
        return augmented

    @staticmethod
    def _sample_mask_width(max_width: int, axis_size: int, device: torch.device) -> int:
        max_width = min(max_width, axis_size)
        if max_width <= 0:
            return 0
        return int(torch.randint(1, max_width + 1, (1,), device=device).item())

    def _extract_features(self, waveform: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        autocast_off = (
            torch.amp.autocast(device_type=waveform.device.type, enabled=False)
            if waveform.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_off:
            fbank = self.backbone.preprocess(waveform.float())
        fbank = self._apply_specaugment(fbank)

        if padding_mask is not None:
            padding_mask = self.backbone.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.backbone.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.backbone.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.backbone.forward_padding_mask(features, padding_mask)

        if self.backbone.post_extract_proj is not None:
            features = self.backbone.post_extract_proj(features)

        features = self.backbone.dropout_input(features)
        features, _ = self.backbone.encoder(features, padding_mask=padding_mask)
        return features

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        elif waveform.dim() != 2:
            raise ValueError(
                f"Expected waveform with shape [batch, time] or [batch, 1, time], got {tuple(waveform.shape)}."
            )

        waveform = waveform.float()
        padding_mask = torch.zeros(waveform.shape, dtype=torch.bool, device=waveform.device)
        features = self._extract_features(waveform, padding_mask=padding_mask)

        if features.dim() == 3:
            pooled = features.mean(dim=1)
        elif features.dim() == 2:
            pooled = features
        else:
            raise ValueError(f"Unexpected BEATs feature shape: {tuple(features.shape)}")
        return self.classifier(pooled)
