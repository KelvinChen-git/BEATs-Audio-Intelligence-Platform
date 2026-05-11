from pathlib import Path
import wave

import pytest
import torch

from src import dataset
from src.dataset import label_to_id, split_esc50_standard_cv


def test_label_to_id():
    assert label_to_id('/m/07rwj00') == 0
    assert label_to_id('/m/07rwj18') == 18
    assert label_to_id('/m/07rwj49') == 49


def test_manifest_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'datafiles' / 'esc_eval_data_1.json').exists()
    assert (root / 'datafiles' / 'run1_test.json').exists()


def test_load_waveform_falls_back_to_builtin_wav(monkeypatch, tmp_path):
    wav_path = tmp_path / "tiny.wav"
    samples = torch.tensor([0, 32767, -32768], dtype=torch.int16).numpy()
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    def raise_missing_torchcodec(_path):
        raise ImportError("TorchCodec is required for load_with_torchcodec")

    monkeypatch.setattr(dataset.torchaudio, "load", raise_missing_torchcodec)

    waveform, sample_rate = dataset.load_waveform(wav_path)

    assert sample_rate == 16000
    assert waveform.shape == (1, 3)
    assert waveform.dtype == torch.float32


def test_standard_cv_split_keeps_test_fold_separate():
    items = [
        {"filename": f"{fold}-{label}.wav", "fold": fold, "target": label}
        for fold in range(1, 6)
        for label in range(2)
        for _ in range(5)
    ]

    split = split_esc50_standard_cv(items, fold=2, seed=42, val_ratio=0.2)

    assert all(filename.startswith("2-") for filename, _ in split["test"])
    assert all(not filename.startswith("2-") for filename, _ in split["train"])
    assert all(not filename.startswith("2-") for filename, _ in split["val"])
    assert {label for _, label in split["val"]} == {0, 1}
