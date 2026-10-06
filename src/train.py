"""
Fine-tune a pretrained ResNet18 on the PCOS ultrasound classification task.

Two split modes, so the paper can quantify the effect of near-duplicate leakage:
  --split_mode group   : near-duplicate clusters stay entirely inside one split (clean)
  --split_mode random  : naive random split (near-copies can straddle train/test)

Usage (run both, into separate output folders):
    python src/train.py --image_dir /content/data/train/images \
        --label_file /content/data/train/class_label.xlsx \
        --group_csv results/dup_t0.csv --split_mode group \
        --epochs 15 --out_dir checkpoints/group

    python src/train.py --image_dir /content/data/train/images \
        --label_file /content/data/train/class_label.xlsx \
        --group_csv results/dup_t0.csv --split_mode random \
        --epochs 15 --out_dir checkpoints/random
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from dataset import PCOSDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from splits import load_cluster_map, get_cluster_ids, make_splits, leakage_report


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _ in tqdm(loader, desc="train" if train else "eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present in this split
    return {"loss": avg_loss, "accuracy": acc, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Folder containing the image files")
    parser.add_argument("--label_file", required=True, help="CSV or XLSX mapping imagePath -> class label")
    parser.add_argument("--group_csv", required=True, help="dedup_check.py output (use the threshold-0 run)")
    parser.add_argument("--split_mode", choices=["group", "random"], default="group")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | split_mode: {args.split_mode}")

    # Two dataset objects over the same samples: augmentation ONLY on train,
    # plain eval transform on val/test.
    train_ds_aug = PCOSDataset(args.image_dir, args.label_file, transform=TRAIN_TRANSFORM)
    eval_ds = PCOSDataset(args.image_dir, args.label_file, transform=EVAL_TRANSFORM)
    paths = [str(p) for p, _ in eval_ds.samples]
    labels = np.array([l for _, l in eval_ds.samples])

    cluster_ids = get_cluster_ids(paths, load_cluster_map(args.group_csv))
    train_idx, val_idx, test_idx = make_splits(cluster_ids, mode=args.split_mode, seed=args.seed)

    leak = leakage_report(train_idx, val_idx, test_idx, cluster_ids)
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"  {name}: {int((labels[idx] == 0).sum())} normal / {int((labels[idx] == 1).sum())} abnormal")
    print(
        f"Leakage: {leak['test_with_dup_in_train']:.1%} of test images have a near-duplicate "
        f"in train ({leak['val_with_dup_in_train']:.1%} of val)"
    )

    train_loader = DataLoader(Subset(train_ds_aug, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(eval_ds, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(Subset(eval_ds, test_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = -1.0
    history = []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} val_auc={val_metrics['auc']:.3f}"
        )
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> new best model saved (val_auc={best_val_auc:.3f})")

    # Final test-set evaluation with the best checkpoint
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal test metrics ({args.split_mode} split): {test_metrics}")

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(
            {"split_mode": args.split_mode, "leakage": leak, "history": history, "test_metrics": test_metrics},
            f, indent=2,
        )

    # Exact held-out test image paths, so faithfulness.py evaluates the same images
    with open(out_dir / "test_split_paths.json", "w") as f:
        json.dump([paths[i] for i in test_idx], f, indent=2)


if __name__ == "__main__":
    main()
