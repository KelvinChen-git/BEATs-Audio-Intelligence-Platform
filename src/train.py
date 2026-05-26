from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.beats_adapter import OfficialBEATsESC50
from src.dataset import (
    ESC50FileDataset,
    ESC50ManifestDataset,
    load_esc50_metadata_items,
    split_esc50_standard_cv,
    split_esc50_trainval_final,
)
from src.model_baseline import MelTransformerESC50
from src.utils import ensure_dir, save_json, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_DIR_DEFAULT = PROJECT_ROOT / "datafiles"
ESC_ROOT_DEFAULT = PROJECT_ROOT / "data" / "ESC-50-master"
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "outputs"


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESC-50 training entry point")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "official_beats", "official_beats_improved", "official_beats_strong_ft"],
    )
    parser.add_argument("--protocol", type=str, default="json_split", choices=["json_split", "esc50_standard_cv"])
    parser.add_argument("--json-dir", type=str, default=str(JSON_DIR_DEFAULT))
    parser.add_argument("--esc-root", type=str, default=str(ESC_ROOT_DEFAULT))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0, help="0 means run all 5 folds; otherwise run only 1..5")
    parser.add_argument("--use-trainval-for-final", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--time-mask-width", type=int, default=40)
    parser.add_argument("--freq-mask-width", type=int, default=16)
    parser.add_argument("--num-time-masks", type=int, default=2)
    parser.add_argument("--num-freq-masks", type=int, default=2)
    parser.add_argument("--unfreeze-top-layers", type=int, default=0)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--tta", action="store_true", help="Use deterministic multi-view evaluation.")
    parser.add_argument("--tta-num-views", type=int, default=5)
    parser.add_argument("--tta-shift-samples", type=int, default=1600)
    parser.add_argument("--eval-only-checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default=get_default_device())
    args = parser.parse_args()
    apply_model_defaults(args, parser)
    validate_runtime_args(args)
    return args


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.use_trainval_for_final and args.protocol != "esc50_standard_cv":
        raise ValueError("--use-trainval-for-final is only valid with --protocol esc50_standard_cv.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")
    if args.tta_num_views < 1:
        raise ValueError("--tta-num-views must be >= 1.")
    if args.tta_shift_samples < 0:
        raise ValueError("--tta-shift-samples must be >= 0.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested with --device, but torch.cuda.is_available() is False.")


def apply_model_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.model != "official_beats_strong_ft":
        return

    if args.epochs == parser.get_default("epochs"):
        args.epochs = 20
    if args.batch_size == parser.get_default("batch_size"):
        args.batch_size = 2
    if args.weight_decay == parser.get_default("weight_decay"):
        args.weight_decay = 0.01
    if args.unfreeze_top_layers == parser.get_default("unfreeze_top_layers") and not args.full_finetune:
        args.unfreeze_top_layers = 8
    if args.backbone_lr is None:
        args.backbone_lr = 1e-5
    if args.head_lr is None:
        args.head_lr = 5e-4
    if args.grad_clip == parser.get_default("grad_clip"):
        args.grad_clip = 1.0
    if args.label_smoothing == parser.get_default("label_smoothing"):
        args.label_smoothing = 0.1
    if args.warmup_ratio == parser.get_default("warmup_ratio"):
        args.warmup_ratio = 0.05
    if args.scheduler == parser.get_default("scheduler"):
        args.scheduler = "cosine"


def find_train_manifest(json_dir: Path, fold: int) -> Path:
    normal = json_dir / f"esc_train_data_{fold}.json"
    typo_version = json_dir / f"esc_trainl_data_{fold}.json"
    if normal.exists():
        return normal
    if typo_version.exists():
        return typo_version
    raise FileNotFoundError(f"Train manifest for fold {fold} not found in {json_dir}")


def get_fold_files(json_dir: Path, fold: int) -> Dict[str, Path]:
    return {
        "train": find_train_manifest(json_dir, fold),
        "val": json_dir / f"esc_eval_data_{fold}.json",
        "test": json_dir / f"run{fold}_test.json",
    }


def validate_esc_audio_root(esc_root: Path) -> None:
    audio_root = esc_root / "audio"
    if not audio_root.exists():
        raise FileNotFoundError(
            f"Missing ESC-50 audio directory: {audio_root}. Put the full ESC-50 audio files under "
            "data/ESC-50-master/audio/."
        )
    if not any(audio_root.glob("*.wav")):
        raise FileNotFoundError(
            f"No .wav files found under {audio_root}. This workspace only has the starter placeholder, so full "
            "training is blocked by missing ESC-50 audio."
        )


def build_loader(
    json_path: Path,
    esc_root: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = ESC50ManifestDataset(json_path=json_path, esc_root=esc_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_file_loader(
    items: list[tuple[str, int]],
    esc_root: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = ESC50FileDataset(items=items, esc_root=esc_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_loaders(
    args: argparse.Namespace,
    fold: int,
    esc_root: Path,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    if args.protocol == "json_split":
        json_dir = Path(args.json_dir)
        fold_files = get_fold_files(json_dir, fold)
        for split_name, split_path in fold_files.items():
            if not split_path.exists():
                raise FileNotFoundError(f"Missing {split_name} manifest: {split_path}")
        return (
            build_loader(
                fold_files["train"], esc_root, args.batch_size, args.num_workers, shuffle=True, pin_memory=pin_memory
            ),
            build_loader(
                fold_files["val"], esc_root, args.batch_size, args.num_workers, shuffle=False, pin_memory=pin_memory
            ),
            build_loader(
                fold_files["test"], esc_root, args.batch_size, args.num_workers, shuffle=False, pin_memory=pin_memory
            ),
        )

    meta_items = load_esc50_metadata_items(esc_root / "meta" / "esc50.csv")
    if args.use_trainval_for_final:
        splits = split_esc50_trainval_final(meta_items=meta_items, fold=fold)
        return (
            build_file_loader(splits["train"], esc_root, args.batch_size, args.num_workers, True, pin_memory),
            None,
            build_file_loader(splits["test"], esc_root, args.batch_size, args.num_workers, False, pin_memory),
        )

    splits = split_esc50_standard_cv(meta_items=meta_items, fold=fold, seed=args.seed)
    return (
        build_file_loader(splits["train"], esc_root, args.batch_size, args.num_workers, True, pin_memory),
        build_file_loader(splits["val"], esc_root, args.batch_size, args.num_workers, False, pin_memory),
        build_file_loader(splits["test"], esc_root, args.batch_size, args.num_workers, False, pin_memory),
    )


def make_tta_views(waveforms: torch.Tensor, num_views: int, shift_samples: int) -> list[torch.Tensor]:
    if num_views <= 1:
        return [waveforms]
    if shift_samples <= 0:
        return [waveforms for _ in range(num_views)]

    if num_views == 2:
        offsets = [-shift_samples, shift_samples]
    else:
        offsets = torch.linspace(-shift_samples, shift_samples, steps=num_views).round().to(torch.int64).tolist()
    return [torch.roll(waveforms, shifts=int(offset), dims=-1) for offset in offsets]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    tta: bool = False,
    tta_num_views: int = 5,
    tta_shift_samples: int = 1600,
) -> Dict[str, float]:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[torch.Tensor] = []

    for waveforms, labels, _ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        if tta:
            view_probs = [torch.softmax(model(view), dim=-1) for view in make_tta_views(waveforms, tta_num_views, tta_shift_samples)]
            probs = torch.stack(view_probs, dim=0).mean(dim=0)
        else:
            logits = model(waveforms)
            probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu())

    probs_np = torch.cat(all_probs, dim=0).numpy()
    acc = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, probs_np, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {"acc": float(acc), "mAUC": float(auc)}


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "baseline":
        return MelTransformerESC50(num_classes=50)
    if args.model == "official_beats":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required when --model official_beats")
        return OfficialBEATsESC50(
            project_root=PROJECT_ROOT,
            checkpoint_path=args.checkpoint_path,
            num_classes=50,
            freeze_backbone=args.freeze_backbone,
        )
    if args.model == "official_beats_improved":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required when --model official_beats_improved")
        return OfficialBEATsESC50(
            project_root=PROJECT_ROOT,
            checkpoint_path=args.checkpoint_path,
            num_classes=50,
            freeze_backbone=True,
            time_mask_width=args.time_mask_width,
            freq_mask_width=args.freq_mask_width,
            num_time_masks=args.num_time_masks,
            num_freq_masks=args.num_freq_masks,
            unfreeze_top_layers=args.unfreeze_top_layers,
        )
    if args.model == "official_beats_strong_ft":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required when --model official_beats_strong_ft")
        if args.full_finetune and args.unfreeze_top_layers > 0:
            raise ValueError("Use either --full-finetune or --unfreeze-top-layers, not both.")
        return OfficialBEATsESC50(
            project_root=PROJECT_ROOT,
            checkpoint_path=args.checkpoint_path,
            num_classes=50,
            freeze_backbone=not args.full_finetune,
            time_mask_width=args.time_mask_width,
            freq_mask_width=args.freq_mask_width,
            num_time_masks=args.num_time_masks,
            num_freq_masks=args.num_freq_masks,
            unfreeze_top_layers=0 if args.full_finetune else args.unfreeze_top_layers,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.model == "official_beats_strong_ft":
        backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
        head_params = [param for param in model.classifier.parameters() if param.requires_grad]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": args.head_lr})
        if not param_groups:
            raise ValueError("No trainable parameters found for official_beats_strong_ft.")
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )


def build_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> LambdaLR | None:
    if args.scheduler == "none":
        return None
    if total_steps <= 0:
        raise ValueError("total_steps must be positive when using a scheduler.")

    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    if args.scheduler == "cosine":
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return trainable, frozen


def log_training_start(
    args: argparse.Namespace,
    fold: int,
    device: str,
    trainable_params: int,
    frozen_params: int,
) -> None:
    print(
        "\n".join(
            [
                "[Training config]",
                f"model={args.model}",
                f"protocol={args.protocol}",
                f"fold={fold}",
                f"epochs={args.epochs}",
                f"device={device}",
                f"checkpoint_path={args.checkpoint_path or '-'}",
                f"trainable_parameters={trainable_params:,}",
                f"frozen_parameters={frozen_params:,}",
                f"amp={'enabled' if args.amp and device.startswith('cuda') else 'disabled'}",
                f"grad_accum_steps={args.grad_accum_steps}",
                f"use_trainval_for_final={args.use_trainval_for_final}",
                f"tta={'enabled' if args.tta else 'disabled'}",
            ]
        )
    )


def load_eval_checkpoint(model: nn.Module, checkpoint_path: str, device: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected evaluation checkpoint object in {path}: expected a state dict.")
    model.load_state_dict(state)


def result_metadata(
    args: argparse.Namespace,
    fold: int,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    test_loader: DataLoader,
) -> dict[str, object]:
    return {
        "model": args.model,
        "fold": fold,
        "protocol": args.protocol,
        "use_trainval_for_final": bool(args.use_trainval_for_final),
        "train_size": len(train_loader.dataset),
        "val_size": 0 if val_loader is None else len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "epochs": args.epochs,
        "tta_enabled": bool(args.tta),
        "tta_num_views": args.tta_num_views if args.tta else 1,
        "output_dir": str(Path(args.output_dir)),
    }


def train_one_fold(args: argparse.Namespace, fold: int) -> Dict[str, float]:
    device = args.device
    esc_root = Path(args.esc_root)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    pin_memory = str(device).startswith("cuda")

    validate_esc_audio_root(esc_root)
    train_loader, val_loader, test_loader = build_loaders(args, fold, esc_root, pin_memory)

    model = build_model(args).to(device)
    trainable_params, frozen_params = count_parameters(model)
    log_training_start(args, fold, device, trainable_params, frozen_params)

    if args.eval_only_checkpoint:
        load_eval_checkpoint(model, args.eval_only_checkpoint, device)
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            tta=args.tta,
            tta_num_views=args.tta_num_views,
            tta_shift_samples=args.tta_shift_samples,
        )
        result = {
            **result_metadata(args, fold, train_loader, val_loader, test_loader),
            "best_epoch": "eval_only",
            "val_acc": float("nan"),
            "val_mAUC": float("nan"),
            "test_acc": test_metrics["acc"],
            "test_mAUC": test_metrics["mAUC"],
            "checkpoint": str(Path(args.eval_only_checkpoint)),
        }
        save_json(result, out_dir / f"metrics_{args.model}_fold_{fold}.json")
        return result

    optimizer = build_optimizer(args, model)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = build_scheduler(args, optimizer, total_steps=args.epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = args.amp and str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if args.amp and not use_amp:
        print("--amp requested but disabled because the selected device is not CUDA.")

    best_val_acc = -math.inf
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        progress = tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch}/{args.epochs}", leave=False)
        optimizer.zero_grad()
        for step, (waveforms, labels, _) in enumerate(progress, start=1):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(waveforms)
                loss = criterion(logits, labels)
                scaled_loss = loss / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected. This usually indicates numeric instability; "
                    "try disabling --amp or lowering fine-tuning learning rates."
                )
            scaler.scale(scaled_loss).backward()

            should_step = step % args.grad_accum_steps == 0 or step == len(train_loader)
            if should_step:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        if val_loader is None:
            print(f"[Fold {fold}] Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}")
        else:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"[Fold {fold}] Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} | val_mAUC={val_metrics['mAUC']:.4f}"
            )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if val_loader is None:
        best_epoch = args.epochs
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_model_path = out_dir / f"final_{args.model}_fold_{fold}.pt"
    elif best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    else:
        best_model_path = out_dir / f"best_{args.model}_fold_{fold}.pt"

    torch.save(best_state, best_model_path)
    model.load_state_dict(best_state)

    if val_loader is None:
        val_metrics = {"acc": float("nan"), "mAUC": float("nan")}
    else:
        val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        tta=args.tta,
        tta_num_views=args.tta_num_views,
        tta_shift_samples=args.tta_shift_samples,
    )
    result = {
        **result_metadata(args, fold, train_loader, val_loader, test_loader),
        "best_epoch": best_epoch,
        "val_acc": val_metrics["acc"],
        "val_mAUC": val_metrics["mAUC"],
        "test_acc": test_metrics["acc"],
        "test_mAUC": test_metrics["mAUC"],
        "checkpoint": str(best_model_path),
    }
    save_json(result, out_dir / f"metrics_{args.model}_fold_{fold}.json")
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    folds = [args.fold] if args.fold in {1, 2, 3, 4, 5} else [1, 2, 3, 4, 5]
    all_results: List[Dict[str, float]] = []

    for fold in folds:
        result = train_one_fold(args, fold)
        all_results.append(result)
        print(
            f"[Fold {fold}] Done | val_acc={result['val_acc']:.4f} | val_mAUC={result['val_mAUC']:.4f} | "
            f"test_acc={result['test_acc']:.4f} | test_mAUC={result['test_mAUC']:.4f}"
        )

    summary_path = Path(args.output_dir) / f"metrics_summary_{args.model}.csv"
    summary_paths = [summary_path]
    if set(folds) == {1, 2, 3, 4, 5}:
        summary_paths.append(Path(args.output_dir) / f"metrics_summary_{args.model}_5fold.csv")

    for path in summary_paths:
        write_summary(path, all_results)

    print(f"Saved summary to: {summary_path}")


def write_summary(summary_path: Path, all_results: List[Dict[str, float]]) -> None:
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "fold",
                "best_epoch",
                "protocol",
                "use_trainval_for_final",
                "train_size",
                "val_size",
                "test_size",
                "epochs",
                "tta_enabled",
                "tta_num_views",
                "val_acc",
                "val_mAUC",
                "test_acc",
                "test_mAUC",
                "checkpoint",
                "output_dir",
            ],
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
        if len(all_results) > 1:
            writer.writerow({
                "model": all_results[0]["model"],
                "fold": "avg",
                "best_epoch": "-",
                "val_acc": sum(r["val_acc"] for r in all_results) / len(all_results),
                "val_mAUC": sum(r["val_mAUC"] for r in all_results) / len(all_results),
                "test_acc": sum(r["test_acc"] for r in all_results) / len(all_results),
                "test_mAUC": sum(r["test_mAUC"] for r in all_results) / len(all_results),
                "checkpoint": "-",
                "protocol": all_results[0].get("protocol", ""),
            })


if __name__ == "__main__":
    main()
