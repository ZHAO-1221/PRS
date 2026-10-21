"""Train the selected motion-aware TCN on generated dog skeletons."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pet_recognition.datasets.dog_behavior import (
    DOG_BEHAVIOR_CLASSES,
    DogBehaviorSkeletonDataset,
    class_counts,
    read_behavior_manifest,
)
from pet_recognition.evaluation.classification_metrics import classification_metrics, save_classification_report
from pet_recognition.models.skeleton_tcn import DogBehaviorMotionAuxTCN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, targets, predictions = 0.0, [], []
    for batch in loader:
        x = batch["keypoints"].to(device)
        mask = batch["frame_mask"].to(device)
        motion = batch["global_motion"].to(device)
        target = batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if hasattr(model, "forward_with_aux"):
                logits, movement_logit = model.forward_with_aux(x, mask, motion)
                centre_path = torch.linalg.vector_norm(motion[:, :, 2:4], dim=2)
                centre_path = (centre_path * mask).sum(1)
                movement_target = (centre_path > 0.1).float()
                auxiliary_loss = nn.functional.binary_cross_entropy_with_logits(
                    movement_logit, movement_target
                )
                loss = loss_fn(logits, target) + model.auxiliary_loss_weight * auxiliary_loss
            else:
                logits = model(x, mask, motion)
                loss = loss_fn(logits, target)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += loss.detach().item() * target.shape[0]
        targets.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    report = classification_metrics(targets, predictions, DOG_BEHAVIOR_CLASSES)
    report["loss"] = total_loss / len(loader.dataset)
    return report


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("PET_DATA_ROOT"))
    parser.add_argument("--manifest", default=str(project / "outputs" / "dog_behavior_manifest.csv"))
    parser.add_argument("--skeleton-dir")
    parser.add_argument(
        "--model",
        choices=("motion_aux_tcn",),
        default="motion_aux_tcn",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-global-motion", action="store_true",
                        help="Ablation: replace the global trajectory stream with zeros")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Use only after all model choices have been frozen")
    args = parser.parse_args()
    if not args.data_root and not args.skeleton_dir:
        parser.error("set PET_DATA_ROOT/--data-root or pass --skeleton-dir")
    seed_everything(args.seed)
    skeleton_dir = Path(args.skeleton_dir) if args.skeleton_dir else Path(args.data_root) / "skeletons_generated"
    output_dir = Path(args.output_dir) if args.output_dir else project / "work_dirs" / f"dog_behavior_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(manifest=args.manifest, skeleton_dir=skeleton_dir, target_frames=args.target_frames,
                  include_global_motion=not args.no_global_motion, seed=args.seed)
    train_set = DogBehaviorSkeletonDataset(split="train", training=True,
                                           augment=not args.no_augmentation, **common)
    val_set = DogBehaviorSkeletonDataset(split="val", training=False, augment=False, **common)
    test_set = (DogBehaviorSkeletonDataset(split="test", training=False, augment=False, **common)
                if args.evaluate_test else None)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)
    model = DogBehaviorMotionAuxTCN(num_classes=len(DOG_BEHAVIOR_CLASSES)).to(args.device)

    counts = class_counts(read_behavior_manifest(args.manifest, "train"))
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_f1, stale, history = -1.0, 0, []
    checkpoint = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        train_report = run_epoch(model, train_loader, loss_fn, args.device, optimizer)
        val_report = run_epoch(model, val_loader, loss_fn, args.device)
        scheduler.step()
        row = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_report["loss"], "train_accuracy": train_report["accuracy"],
            "train_macro_f1": train_report["macro_f1"], "val_loss": val_report["loss"],
            "val_accuracy": val_report["accuracy"], "val_macro_f1": val_report["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row))
        if val_report["macro_f1"] > best_f1:
            best_f1, stale = val_report["macro_f1"], 0
            torch.save({"model": model.state_dict(), "classes": DOG_BEHAVIOR_CLASSES,
                        "args": vars(args), "epoch": epoch, "val": val_report}, checkpoint)
            save_classification_report(val_report, DOG_BEHAVIOR_CLASSES, output_dir, "val")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    if args.evaluate_test:
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(saved["model"])
        test_loader = make_loader(test_set, args.batch_size, False, args.num_workers)
        test_report = run_epoch(model, test_loader, loss_fn, args.device)
        save_classification_report(test_report, DOG_BEHAVIOR_CLASSES, output_dir, "test")
        print("test", json.dumps({k: test_report[k] for k in ("loss", "accuracy", "macro_f1")}))
    print(f"best validation macro-F1={best_f1:.4f}; checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
