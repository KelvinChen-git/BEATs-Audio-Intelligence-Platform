import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import split_esc50_trainval_final
from src.train import evaluate, make_tta_views


class TinyWaveformDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        label = index % 2
        waveform = torch.full((8,), float(label))
        return waveform, label, f"{index}.wav"


class SumClassifier(nn.Module):
    def forward(self, waveforms):
        score = waveforms.sum(dim=-1)
        return torch.stack([-score, score], dim=-1)


def test_final_trainval_split_uses_1600_train_400_test_without_overlap():
    items = [
        {"filename": f"{fold}-{item_index}-{label}.wav", "fold": fold, "target": label}
        for fold in range(1, 6)
        for label in range(50)
        for item_index in range(8)
    ]

    split = split_esc50_trainval_final(items, fold=1)

    train_names = {filename for filename, _ in split["train"]}
    test_names = {filename for filename, _ in split["test"]}
    assert len(split["train"]) == 1600
    assert len(split["val"]) == 0
    assert len(split["test"]) == 400
    assert not train_names & test_names
    assert {label for _, label in split["test"]} == set(range(50))


def test_tta_disabled_matches_single_view_evaluation():
    loader = DataLoader(TinyWaveformDataset(), batch_size=2)
    model = SumClassifier()

    no_tta = evaluate(model, loader, device="cpu")
    one_view_tta = evaluate(model, loader, device="cpu", tta=True, tta_num_views=1)

    assert no_tta["acc"] == one_view_tta["acc"]


def test_tta_views_preserve_shape_and_count():
    waveforms = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    views = make_tta_views(waveforms, num_views=5, shift_samples=2)

    assert len(views) == 5
    assert all(view.shape == waveforms.shape for view in views)
    assert torch.equal(views[2], waveforms)
