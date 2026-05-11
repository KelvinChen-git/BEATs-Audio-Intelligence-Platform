from __future__ import annotations

import json
import re
import wave
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

LABEL_PATTERN = re.compile(r"/m/07rwj(\d{2})$")


def label_to_id(label: str) -> int:
    match = LABEL_PATTERN.search(label)
    if not match:
        raise ValueError(f"Unsupported label format: {label}")
    return int(match.group(1))


def _decode_pcm_wav(path: Path) -> Tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw_audio = wav_file.readframes(num_frames)

    if sample_width == 1:
        samples = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_audio, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(
            f"Unsupported WAV sample width: {sample_width} bytes. Install torchcodec for decoder support."
        )

    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).T
    else:
        samples = samples.reshape(1, -1)
    return torch.from_numpy(samples.copy()), sample_rate


def load_waveform(path: str | Path) -> Tuple[torch.Tensor, int]:
    path = Path(path)
    try:
        return torchaudio.load(str(path))
    except (ImportError, RuntimeError) as exc:
        if path.suffix.lower() == ".wav":
            try:
                return _decode_pcm_wav(path)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Could not load WAV file {path}. torchaudio failed first, and the built-in PCM WAV "
                    "fallback also failed. Install torchcodec with `py -m pip install torchcodec` or "
                    "reinstall a compatible torchaudio/torchcodec pair."
                ) from fallback_exc
        raise RuntimeError(
            f"Could not load audio file {path}. torchaudio requires TorchCodec in this environment. "
            "Install it with `py -m pip install torchcodec` or use PCM .wav files supported by the fallback."
        ) from exc


class ESC50ManifestDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        esc_root: str | Path,
        target_sample_rate: int = 16000,
        duration_seconds: float = 5.0,
    ) -> None:
        self.json_path = Path(json_path)
        self.esc_root = Path(esc_root)
        self.audio_root = self.esc_root / "audio"
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = int(target_sample_rate * duration_seconds)

        with open(self.json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "data" in payload:
            self.items: List[Dict[str, str]] = payload["data"]
        elif isinstance(payload, list):
            self.items = payload
        else:
            raise ValueError(f"Unexpected manifest format in {self.json_path}")

        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_audio_path(self, wav_from_json: str) -> Path:
        filename = Path(wav_from_json).name
        resolved = self.audio_root / filename
        if not resolved.exists():
            raise FileNotFoundError(
                f"Audio file not found: {resolved}\n"
                f"JSON item: {wav_from_json}\n"
                f"Put the full ESC-50 audio files under data/ESC-50-master/audio/."
            )
        return resolved

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sample_rate:
            return waveform
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        return self._resamplers[sr](waveform)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[-1]
        if num_samples > self.target_num_samples:
            waveform = waveform[..., : self.target_num_samples]
        elif num_samples < self.target_num_samples:
            pad = self.target_num_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[index]
        wav_path = self._resolve_audio_path(item["wav"])
        label_id = label_to_id(item["labels"])

        waveform, sr = load_waveform(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._resample(waveform, sr)
        waveform = self._fix_length(waveform)
        return waveform.squeeze(0), label_id, wav_path.name


class ESC50FileDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[str, int]],
        esc_root: str | Path,
        target_sample_rate: int = 16000,
        duration_seconds: float = 5.0,
    ) -> None:
        self.items = list(items)
        self.esc_root = Path(esc_root)
        self.audio_root = self.esc_root / "audio"
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = int(target_sample_rate * duration_seconds)
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.target_sample_rate:
            return waveform
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        return self._resamplers[sr](waveform)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[-1]
        if num_samples > self.target_num_samples:
            waveform = waveform[..., : self.target_num_samples]
        elif num_samples < self.target_num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_num_samples - num_samples))
        return waveform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        filename, label_id = self.items[index]
        wav_path = self.audio_root / filename
        if not wav_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {wav_path}\n"
                "Put the full ESC-50 audio files under data/ESC-50-master/audio/."
            )

        waveform, sr = load_waveform(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._resample(waveform, sr)
        waveform = self._fix_length(waveform)
        return waveform.squeeze(0), label_id, wav_path.name


def load_esc50_metadata_items(meta_csv: str | Path) -> list[dict[str, int | str]]:
    meta_csv = Path(meta_csv)
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing ESC-50 metadata CSV: {meta_csv}")

    items: list[dict[str, int | str]] = []
    with meta_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"filename", "fold", "target"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected ESC-50 metadata columns in {meta_csv}")
        for row in reader:
            items.append(
                {
                    "filename": row["filename"],
                    "fold": int(row["fold"]),
                    "target": int(row["target"]),
                }
            )
    return items


def split_esc50_standard_cv(
    meta_items: Sequence[dict[str, int | str]],
    fold: int,
    seed: int,
    val_ratio: float = 0.2,
) -> dict[str, list[Tuple[str, int]]]:
    test_items = [
        (str(item["filename"]), int(item["target"])) for item in meta_items if int(item["fold"]) == fold
    ]
    train_pool = [
        (str(item["filename"]), int(item["target"])) for item in meta_items if int(item["fold"]) != fold
    ]
    if not test_items or not train_pool:
        raise ValueError(f"Could not build ESC-50 standard CV split for fold {fold}.")

    generator = torch.Generator().manual_seed(seed + fold)
    by_label: dict[int, list[Tuple[str, int]]] = {}
    for item in train_pool:
        by_label.setdefault(item[1], []).append(item)

    train_items: list[Tuple[str, int]] = []
    val_items: list[Tuple[str, int]] = []
    for label_items in by_label.values():
        order = torch.randperm(len(label_items), generator=generator).tolist()
        val_count = max(1, int(round(len(label_items) * val_ratio)))
        val_indexes = set(order[:val_count])
        for index, item in enumerate(label_items):
            if index in val_indexes:
                val_items.append(item)
            else:
                train_items.append(item)

    return {"train": train_items, "val": val_items, "test": test_items}
