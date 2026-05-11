from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.beats_adapter import OfficialBEATsESC50
from src.train import validate_esc_audio_root


def test_validate_esc_audio_root_rejects_placeholder_only(tmp_path: Path):
    esc_root = tmp_path / "ESC-50-master"
    audio_root = esc_root / "audio"
    audio_root.mkdir(parents=True)
    (audio_root / "PLACE_FULL_ESC50_AUDIO_FILES_HERE.txt").write_text("placeholder", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing ESC-50 audio"):
        validate_esc_audio_root(esc_root)


def test_official_beats_requires_repo_before_loading_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "dummy.pt"
    checkpoint_path.write_bytes(b"not-used")

    with pytest.raises(FileNotFoundError, match="Run scripts/setup_beats.ps1"):
        OfficialBEATsESC50(project_root=tmp_path, checkpoint_path=checkpoint_path)


def test_specaugment_only_changes_training_features():
    model = OfficialBEATsESC50.__new__(OfficialBEATsESC50)
    nn.Module.__init__(model)
    model.time_mask_width = 2
    model.freq_mask_width = 2
    model.num_time_masks = 1
    model.num_freq_masks = 1

    fbank = torch.ones(1, 6, 5)

    model.eval()
    assert torch.equal(model._apply_specaugment(fbank), fbank)

    torch.manual_seed(0)
    model.train()
    augmented = model._apply_specaugment(fbank)

    assert augmented.shape == fbank.shape
    assert torch.count_nonzero(augmented == 0) > 0
