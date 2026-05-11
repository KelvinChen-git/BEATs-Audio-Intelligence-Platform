# GPU Strong Fine-Tuning Guide

This guide is for running `official_beats_strong_ft` on a CUDA-enabled machine. Do not claim the BEATs ESC-50 reported 98.1% result unless your local experiment actually reaches it.

## 1. Check For An NVIDIA GPU

Run:

```powershell
nvidia-smi
```

If this command is not found, or it reports no GPU, this machine is not ready for CUDA training.

## 2. Check PyTorch CUDA

Run:

```powershell
py scripts\check_env.py
```

Look for:

```text
CUDA available: True
train.py selected device: cuda
```

If `torch version` or `torchaudio version` ends with `+cpu`, the installed PyTorch build is CPU-only. A CPU-only install will not use the GPU even if `nvidia-smi` works.

## 3. When To Reinstall CUDA PyTorch

Reinstall CUDA-enabled PyTorch when:

- `nvidia-smi` shows a usable NVIDIA GPU.
- `py scripts\check_env.py` reports `CUDA available: False`.
- `torch version` shows `+cpu`.

Use the official PyTorch install selector for the CUDA version supported by the target GPU driver. Keep `torch` and `torchaudio` on matching versions.

## 4. CPU Warning

Do not run strong fine-tuning on CPU unless you intentionally want a very long run. In this workspace, one strong fine-tuning epoch took about 14.6 minutes on CPU, so a 20-epoch fold-1 run is roughly 5 hours and a 5-fold run can take about a full day.

## 5. Fold-1 Priority Commands

Top-8, 20 epochs, standard ESC-50 CV:

```powershell
py -m src.train --model official_beats_strong_ft --protocol esc50_standard_cv --fold 1 --epochs 20 --batch-size 2 --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt --unfreeze-top-layers 8 --backbone-lr 1e-5 --head-lr 5e-4 --weight-decay 0.01 --label-smoothing 0.1 --grad-clip 1.0 --scheduler cosine --warmup-ratio 0.05 --amp --output-dir .\outputs\strong_ft_priority\top8_esc50_standard_cv
```

Top-12, 20 epochs, standard ESC-50 CV:

```powershell
py -m src.train --model official_beats_strong_ft --protocol esc50_standard_cv --fold 1 --epochs 20 --batch-size 2 --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt --unfreeze-top-layers 12 --backbone-lr 1e-5 --head-lr 5e-4 --weight-decay 0.01 --label-smoothing 0.1 --grad-clip 1.0 --scheduler cosine --warmup-ratio 0.05 --amp --output-dir .\outputs\strong_ft_priority\top12_esc50_standard_cv
```

Full fine-tuning, 20 epochs, standard ESC-50 CV:

```powershell
py -m src.train --model official_beats_strong_ft --protocol esc50_standard_cv --fold 1 --epochs 20 --batch-size 2 --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt --full-finetune --backbone-lr 5e-6 --head-lr 5e-4 --weight-decay 0.01 --label-smoothing 0.1 --grad-clip 1.0 --scheduler cosine --warmup-ratio 0.05 --amp --output-dir .\outputs\strong_ft_priority\full_esc50_standard_cv
```

## 6. Helper Scripts

Run the priority fold-1 sequence:

```powershell
.\scripts\run_strong_ft_fold1_priority.ps1
```

After selecting the best fold-1 setting, run the 5-fold template:

```powershell
.\scripts\run_strong_ft_5fold_best.ps1
```

Summarize top-level 5-fold strong fine-tuning metrics:

```powershell
py scripts\summarize_strong_ft.py
```

## 7. Recommended Experiment Order

1. Confirm CUDA with `nvidia-smi` and `py scripts\check_env.py`.
2. Run top-8 fold 1 with `esc50_standard_cv`.
3. If top-8 improves over the previous improved model, run top-12 fold 1.
4. If top-12 is stable and improves further, try full fine-tuning.
5. Run 5-fold training only for the best fold-1 configuration.

## 8. Threshold For 5-Fold Training

Use 5-fold training when fold-1 test accuracy is clearly above the previous improved result of about 91.55%, preferably at or above 94%. If fold-1 is below 91.55%, tune the fold-1 setup first instead of spending time on 5-fold training.
