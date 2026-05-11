$ErrorActionPreference = "Stop"

$Checkpoint = ".\checkpoints\BEATs_iter3_plus_AS2M.pt"

py -m src.train `
    --model official_beats_strong_ft `
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
    --protocol json_split
