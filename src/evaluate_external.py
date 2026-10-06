"""
Evaluate the trained model on a second, independent PCOS ultrasound dataset.

This is the core comparison of the paper: does accuracy hold up outside the
training distribution, and does Grad-CAM faithfulness hold up alongside it
(or does it stay "visually convincing" while the model is actually failing)?

Usage (folder-structured Kaggle dataset, e.g. infected/notinfected):
    python src/evaluate_external.py --image_dir data/external \
        --positive_class_dir infected --negative_class_dir notinfected \
        --checkpoint checkpoints/best_model.pt --out results/external_eval.csv

Usage (PCOSGen-style CSV/XLSX label file, e.g. the official PCOSGen-test set):
    python src/evaluate_external.py --image_dir data/pcosgen_test/images \
        --label_file data/pcosgen_test/class_label.csv \
        --checkpoint checkpoints/best_model.pt --out results/pcosgen_test_eval.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import PCOSDataset, FolderPCOSDataset, EVAL_TRANSFORM


def build_model(num_classes: int = 2):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def run_accuracy_eval(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="external eval"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc, "n_images": len(all_labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_file", default=None, help="PCOSGen-style CSV/XLSX label file")
    parser.add_argument("--positive_class_dir", default=None, help="Folder-structured dataset: positive class subfolder")
    parser.add_argument("--negative_class_dir", default=None, help="Folder-structured dataset: negative class subfolder")
    parser.add_argument(
        "--exclude_csv",
        default=None,
        help="cross_dataset_dedup.py output: drop images in this dataset that have a "
        "near-duplicate in another dataset (e.g. train) before evaluating.",
    )
    parser.add_argument(
        "--exclude_col",
        default=None,
        help="Column name in --exclude_csv holding this dataset's image paths "
        "(e.g. 'test_path'). Required if --exclude_csv is given.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--indist_faithfulness_csv",
        default="results/faithfulness_indist.csv",
        help="Output of faithfulness.py on the in-distribution test split, for comparison",
    )
    args = parser.parse_args()

    if not args.label_file and not (args.positive_class_dir and args.negative_class_dir):
        raise ValueError(
            "Provide either --label_file (PCOSGen-style) or both "
            "--positive_class_dir and --negative_class_dir (folder-structured datasets)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    exclude_paths = set()
    if args.exclude_csv:
        if not args.exclude_col:
            raise ValueError("--exclude_col is required when --exclude_csv is given.")
        exclude_df = pd.read_csv(args.exclude_csv)
        exclude_paths = set(exclude_df[args.exclude_col].astype(str).unique())
        print(f"Excluding {len(exclude_paths)} images flagged as cross-dataset duplicates.")

    if args.label_file:
        dataset = PCOSDataset(args.image_dir, args.label_file, transform=EVAL_TRANSFORM, exclude_paths=exclude_paths)
    else:
        dataset = FolderPCOSDataset(
            args.image_dir, args.positive_class_dir, args.negative_class_dir,
            transform=EVAL_TRANSFORM, exclude_paths=exclude_paths,
        )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    acc_metrics = run_accuracy_eval(model, loader, device)
    print(f"\nExternal-dataset accuracy metrics: {acc_metrics}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([acc_metrics]).to_csv(args.out, index=False)
    print(f"Written to {args.out}")

    # Run faithfulness.py on this external set too, so the two are directly comparable
    faithfulness_out = str(Path(args.out).with_name("faithfulness_external.csv"))
    print("\nRunning quantitative Grad-CAM faithfulness on the external set...")
    faithfulness_cmd = [
        sys.executable,
        str(Path(__file__).parent / "faithfulness.py"),
        "--image_dir", args.image_dir,
        "--checkpoint", args.checkpoint,
        "--out", faithfulness_out,
        "--split", "external",
    ]
    if args.exclude_csv:
        faithfulness_cmd += ["--exclude_csv", args.exclude_csv, "--exclude_col", args.exclude_col]
    if args.label_file:
        faithfulness_cmd += ["--label_file", args.label_file]
    else:
        faithfulness_cmd += [
            "--positive_class_dir", args.positive_class_dir,
            "--negative_class_dir", args.negative_class_dir,
        ]
    subprocess.run(faithfulness_cmd, check=True)

    # Side-by-side summary if the in-distribution faithfulness file exists
    if Path(args.indist_faithfulness_csv).exists():
        indist = pd.read_csv(args.indist_faithfulness_csv)
        external = pd.read_csv(faithfulness_out)
        summary = pd.DataFrame(
            {
                "metric": ["accuracy", "mean_deletion_auc", "mean_insertion_auc"],
                "in_distribution": [
                    None,  # fill in manually from train.py's test_metrics if desired
                    indist["deletion_auc"].mean(),
                    indist["insertion_auc"].mean(),
                ],
                "external": [
                    acc_metrics["accuracy"],
                    external["deletion_auc"].mean(),
                    external["insertion_auc"].mean(),
                ],
            }
        )
        summary_path = str(Path(args.out).with_name("summary_comparison.csv"))
        summary.to_csv(summary_path, index=False)
        print(f"\nSide-by-side in-distribution vs. external summary written to {summary_path}")
        print(summary)


if __name__ == "__main__":
    main()
