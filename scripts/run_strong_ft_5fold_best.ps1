$ErrorActionPreference = "Stop"

$Checkpoint = ".\checkpoints\BEATs_iter3_plus_AS2M.pt"
$OutputDir = ".\outputs\strong_ft_5fold_best\top8_esc50_standard_cv"

# Default: top-8, 20 epochs, standard ESC-50 CV.
# To switch to top-12, change "--unfreeze-top-layers", "8" to "12".
# To switch to full fine-tuning, remove "--unfreeze-top-layers", "8" and add "--full-finetune";
# also change "--backbone-lr", "1e-5" to "5e-6".

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

py -m src.train `
    --model official_beats_strong_ft `
    --protocol esc50_standard_cv `
    --epochs 20 `
    --batch-size 2 `
    --checkpoint-path $Checkpoint `
    --unfreeze-top-layers 8 `
    --backbone-lr 1e-5 `
    --head-lr 5e-4 `
    --weight-decay 0.01 `
    --label-smoothing 0.1 `
    --grad-clip 1.0 `
    --scheduler cosine `
    --warmup-ratio 0.05 `
    --amp `
    --output-dir $OutputDir
