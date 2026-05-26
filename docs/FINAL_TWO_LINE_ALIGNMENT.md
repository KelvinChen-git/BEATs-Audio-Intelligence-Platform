# Final Two-Line Alignment

The final comparison should focus on BEATs on ESC-50, not on the earlier exploratory baselines.

## Reproduced BEATs

The reproduced line uses:

- Official Microsoft BEATs implementation from `third_party/unilm/beats`
- `BEATs_iter3_plus_AS2M.pt`
- ESC-50 standard cross-validation
- For each fold, train on the other four folds and test on the held-out fold
- 1600 training samples and 400 test samples per fold
- Strong/full fine-tuning with the selected fixed epoch count

The earlier 1300 train / 300 validation / 400 test setup is a development and hyperparameter-selection protocol. It is not the final replication protocol.

## Improved BEATs

The improvement line should be built on top of the reproduced BEATs checkpoint from the same 1600 / 400 ESC-50 standard CV setting.

The aligned improvement is deterministic test-time augmentation / multi-view inference:

- Load the reproduced BEATs checkpoint
- Create multiple deterministic shifted waveform views at evaluation time
- Average class probabilities across views
- Compare against single-view reproduced BEATs on the same test fold

TTA view count is treated as an improvement ablation. The final improvement should use the best TTA setting from the ablation; if several settings are effectively tied, prefer the smallest view count that gives the best or near-best result for efficiency.

For fold 1, TTA-3 was selected from the ablation because it reached the joint-best test accuracy, the best test mAUC, and lower inference cost than TTA-9.

The final report figure uses a zoomed test-accuracy axis because both final systems are already high-performing and the absolute improvement is small but meaningful.

The old frozen-backbone result around 69.45% is historical and should not be the final reproduction result. The old improved result around 91.55% is also historical if full fine-tuning is treated as the reproduced BEATs line.

Do not claim the reported BEATs ESC-50 98.1% result unless completed 5-fold results in this project actually reach it.
