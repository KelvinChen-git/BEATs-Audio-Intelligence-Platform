$ErrorActionPreference = "Stop"

foreach ($Fold in 1..5) {
    py -m src.train `
        --model official_beats_strong_ft `
        --protocol esc50_standard_cv `
        --use-trainval-for-final `
        --fold $Fold `
        --epochs 17 `
        --batch-size 2 `
        --checkpoint-path .\checkpoints\BEATs_iter3_plus_AS2M.pt `
        --full-finetune `
        --backbone-lr 3e-6 `
        --head-lr 5e-4 `
        --weight-decay 0.01 `
        --label-smoothing 0.1 `
        --grad-clip 1.0 `
        --scheduler cosine `
        --warmup-ratio 0.05 `
        --amp `
        --output-dir ".\outputs\replication_standard_cv\full_lr3e6_fold$Fold"
}
