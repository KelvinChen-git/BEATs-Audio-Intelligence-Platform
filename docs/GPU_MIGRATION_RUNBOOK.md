# GPU Migration Runbook

This machine currently reports CPU-only PyTorch, so it is not suitable for long `official_beats_strong_ft` experiments. One CPU epoch has already been slow enough that 20-epoch strong fine-tuning should move to a CUDA GPU machine.

## Check The GPU

Run:

```powershell
nvidia-smi
```

If this command is missing or cannot see a GPU, fix the NVIDIA driver or use another machine before training.

Then check PyTorch:

```powershell
py scripts\check_env.py
```

If torch or torchaudio versions include `+cpu`, the installed PyTorch build cannot use CUDA even if the machine has an NVIDIA GPU. Install CUDA-enabled PyTorch only after checking the official PyTorch selector:

https://pytorch.org/get-started/locally/

The selector is the source of truth for the exact CUDA wheel command. This repository includes a helper with a default CUDA 12.6 wheel index, but you should verify it before use:

```powershell
scripts\setup_cuda_pytorch_windows.ps1
```

Do not run strong fine-tuning on CPU unless you explicitly accept a very long runtime.

## Preflight

After installing CUDA-enabled PyTorch, run:

```powershell
scripts\gpu_preflight_check.ps1
```

The preflight checks `nvidia-smi`, CUDA availability, train.py device selection, the BEATs checkpoint, ESC-50 audio, ESC-50 metadata, and official BEATs source files.

## Recommended Order

1. Run tests:

```powershell
py -m pytest -q
```

2. Run GPU preflight:

```powershell
scripts\gpu_preflight_check.ps1
```

3. Run fold-1 top-8 strong fine-tuning:

```powershell
scripts\run_strong_ft_gpu_top8_fold1.ps1
```

4. If top-8 fold-1 `test_acc >= 0.94`, try top-12:

```powershell
scripts\run_strong_ft_gpu_top12_fold1.ps1
```

5. If top-12 improves, try full fine-tuning:

```powershell
scripts\run_strong_ft_gpu_full_fold1.ps1
```

6. Only after choosing the best fold-1 setting, run 5-fold training.

## Direct Commands

Top-8 fold 1:

```powershell
py -m src.train --model official_beats_strong_ft --protocol esc50_standard_cv --fold 1 --epochs 20 --batch-size 2 --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt --unfreeze-top-layers 8 --backbone-lr 1e-5 --head-lr 5e-4 --weight-decay 0.01 --label-smoothing 0.1 --grad-clip 1.0 --scheduler cosine --warmup-ratio 0.05 --amp --output-dir .\outputs\strong_ft_priority\top8_esc50_standard_cv
```

Top-12 fold 1:

```powershell
py -m src.train --model official_beats_strong_ft --protocol esc50_standard_cv --fold 1 --epochs 20 --batch-size 2 --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt --unfreeze-top-layers 12 --backbone-lr 1e-5 --head-lr 5e-4 --weight-decay 0.01 --label-smoothing 0.1 --grad-clip 1.0 --scheduler cosine --warmup-ratio 0.05 --amp --output-dir .\outputs\strong_ft_priority\top12_esc50_standard_cv
```

Full fine-tuning fold 1:

```powershell
py -m src.train --model official_beats_strong_ft --protocol esc50_standard_cv --fold 1 --epochs 20 --batch-size 2 --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt --full-finetune --backbone-lr 5e-6 --head-lr 5e-4 --weight-decay 0.01 --label-smoothing 0.1 --grad-clip 1.0 --scheduler cosine --warmup-ratio 0.05 --amp --output-dir .\outputs\strong_ft_priority\full_esc50_standard_cv
```

All three commands use `--protocol esc50_standard_cv`, which evaluates fold `k` on ESC-50 fold `k` and builds validation data from the remaining training folds.

## Output Folders

Priority fold-1 runs write to:

```text
outputs\strong_ft_priority\top8_esc50_standard_cv
outputs\strong_ft_priority\top12_esc50_standard_cv
outputs\strong_ft_priority\full_esc50_standard_cv
```

Each folder should contain `metrics_official_beats_strong_ft_fold_1.json` and the best checkpoint for that run. Summarize priority results with:

```powershell
py scripts\summarize_priority_runs.py
```

Do not claim the reported BEATs ESC-50 98.1% result unless this project actually reaches it in the completed experiment results, preferably on 5-fold evaluation.
