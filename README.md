# BEATs Environmental Sound Classification on ESC-50

A reproducible study of pretrained audio representation learning for environmental sound classification. This project implements and evaluates BEATs on the ESC-50 dataset, compares different fine-tuning strategies, and studies the accuracy–complexity trade-off of a large pretrained audio Transformer.

## Overview

Environmental sound classification maps short audio recordings to semantic categories such as dog barking, rain, coughing, sirens, and engine noise.

This project evaluates three modeling directions:

1. A log-mel Transformer trained from scratch as a task-specific baseline.
2. A selectively fine-tuned BEATs model using training-time SpecAugment.
3. Full BEATs fine-tuning followed by deterministic three-view test-time augmentation.

The strongest five-fold configuration achieved **94.90% mean accuracy**, while the full fine-tuning baseline achieved **94.70% mean accuracy** and **0.9985 macro-AUC**.

The project focuses on reproducible model training, evaluation, inference refinement, and complexity analysis. It does not claim an exact reproduction of the 98.1% result reported in the original BEATs paper.

## Dataset

Experiments use the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50), which contains:

- 2,000 labeled environmental audio clips
- 50 balanced sound classes
- 40 recordings per class
- Five major semantic categories
- Five-second audio clips
- Official five-fold cross-validation splits

The official fold assignments are preserved throughout the project. In each evaluation round, four folds are used for training and the remaining fold is held out for testing.

Keeping the official folds also prevents clips derived from the same original recording from appearing in both training and testing partitions.

## Model

The main model is [BEATs](https://arxiv.org/abs/2212.09058), a Transformer-based audio representation model pretrained using acoustic tokenizers and masked discrete-label prediction.

This project uses the official `BEATs_iter3_plus_AS2M` checkpoint and replaces the original output layer with a 50-class classification head for ESC-50.

The experiments evaluate two fine-tuning strategies:

### Selective fine-tuning

- Loads the pretrained BEATs checkpoint
- Unfreezes the top two BEATs encoder layers
- Trains the ESC-50 classification head
- Applies SpecAugment only during training
- Evaluates all five official folds

This configuration reduces the number of trainable encoder layers while retaining pretrained audio representations.

### Full fine-tuning

- Loads the same pretrained checkpoint
- Fine-tunes the complete BEATs model
- Uses a 50-class classification head
- Preserves the official ESC-50 fold assignments
- Serves as the primary BEATs reproduction configuration

## Audio Preprocessing

The preprocessing pipeline standardizes each recording before it is passed to the model:

1. Load WAV files using platform-safe paths.
2. Convert multichannel recordings to mono.
3. Resample audio to 16 kHz.
4. Clip or zero-pad each recording to a fixed duration of five seconds.
5. Store file paths, labels, and official fold assignments in JSON manifests.
6. Convert the waveform into the feature representation expected by BEATs.

The same preprocessing rules are applied across training, validation, and testing.

## Deterministic TTA-3

The final inference refinement uses deterministic three-view test-time augmentation.

For every test recording:

1. Construct three waveform views using fixed temporal offsets.
2. Run the same trained BEATs model on all three views.
3. Convert model outputs into class probabilities.
4. Average the three probability vectors.
5. Select the class with the highest averaged probability.

TTA-3 does not retrain the model or alter the dataset splits. It improves prediction stability by aggregating several deterministic views of the same recording.

## Evaluation Protocol

The project reports:

- Classification accuracy
- Macro-AUC across the 50 classes
- Mean and standard deviation across the five official folds
- Model parameter count
- Model size
- Multiply–accumulate operations
- Inference latency
- Throughput
- Accuracy per MMAC

Accuracy measures the final hard-label classification result, while macro-AUC evaluates how well the model ranks positive examples across all classes. Therefore, the two metrics may respond differently to the same model modification.

## Results

### Five-fold results

| Configuration | Mean Accuracy | Macro-AUC |
|---|---:|---:|
| Log-mel Transformer trained from scratch | 69.45% | — |
| BEATs with SpecAugment and top-two-layer unfreezing | 91.55% ± 1.85% | 0.9988 ± 0.0004 |
| BEATs full fine-tuning | 94.70% ± 1.81% | 0.9985 ± 0.0016 |
| BEATs full fine-tuning with deterministic TTA-3 | **94.90% ± 1.35%** | approximately 0.9985 |

The configurations above belong to separate experiment tracks and should not be interpreted as a single controlled ablation sequence. In particular, the difference between the scratch Transformer and BEATs models reflects both the architecture and the use of large-scale audio pretraining.

### Fold 1 comparison

| Configuration | Accuracy | Macro-AUC |
|---|---:|---:|
| BEATs full fine-tuning | 94.75% | 0.9987 |
| BEATs with deterministic TTA-3 | **95.50%** | **0.9988** |

On Fold 1, TTA-3 improved accuracy by **0.75 percentage points** without changing the trained model or data split.

## Efficiency Analysis

The complete BEATs model provides high classification accuracy at a substantial computational cost.

| Metric | Measured Value |
|---|---:|
| Parameters | 97.48M |
| Model size | 380.80 MB |
| Computation | 24,751.57 MMACs |
| Inference latency | 13.32 ms |
| Throughput | 75.06 samples/s |
| Accuracy per MMAC | 0.0038 |

Latency was measured over 50 repeated CUDA inference runs in the project’s local profiling environment. These values describe the tested environment and should not be treated as hardware-independent production benchmarks.

The results show that BEATs offers strong accuracy but is considerably more expensive than smaller task-specific models. This makes it suitable when classification quality is the main priority, while deployment on constrained devices would require compression or a smaller architecture.

## Key Findings

- Audio pretraining provides a large advantage over training a Transformer from scratch on the relatively small ESC-50 dataset.
- Full fine-tuning produced the strongest single-model five-fold accuracy in the completed experiments.
- Selective fine-tuning with SpecAugment remained competitive while updating fewer encoder layers.
- Deterministic TTA-3 produced a modest accuracy improvement and reduced fold-to-fold variance.
- Macro-AUC remained close to 1.0 across BEATs configurations, indicating strong class-ranking performance.
- The accuracy improvement comes with high parameter, memory, and compute requirements.

## My Contribution

This work was completed as part of a broader environmental sound classification study. My contribution focused on the BEATs experiment track:

- Reproduced BEATs fine-tuning on ESC-50 using the official pretrained checkpoint
- Implemented audio loading, mono conversion, 16 kHz resampling, fixed-length processing, and JSON manifests
- Preserved and evaluated the official five-fold protocol
- Implemented and evaluated SpecAugment with selective encoder-layer unfreezing
- Implemented deterministic TTA-3 and probability-level aggregation
- Compared accuracy, macro-AUC, model size, computation, latency, and throughput
- Analyzed the trade-off between predictive performance and computational cost

## Reproduction Checklist

To reproduce the experiments:

1. Download ESC-50 from the [official repository](https://github.com/karolpiczak/ESC-50).
2. Download the official `BEATs_iter3_plus_AS2M` checkpoint from the [Microsoft BEATs repository](https://github.com/microsoft/unilm/tree/master/beats).
3. Generate JSON manifests containing audio paths, labels, and official fold IDs.
4. Apply mono conversion, 16 kHz resampling, and five-second clipping or zero-padding.
5. Select either selective fine-tuning or full fine-tuning.
6. Train one model for each held-out ESC-50 fold.
7. Evaluate accuracy and macro-AUC separately for every fold.
8. Aggregate the five-fold mean and standard deviation.
9. Optionally apply deterministic TTA-3 to the saved full fine-tuning checkpoints.
10. Record the software environment, checkpoint hash, random seed, and per-fold outputs when publishing new results.

Exact commands should correspond to the training and evaluation entry points included in the repository.


## References

- [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50)
- [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)
- [Official Microsoft BEATs implementation](https://github.com/microsoft/unilm/tree/master/beats)

## Acknowledgements

This project builds on the ESC-50 dataset and the official Microsoft BEATs implementation. The dataset and pretrained checkpoint remain subject to their respective original licenses and terms.
