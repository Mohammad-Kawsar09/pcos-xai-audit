"""
Quantitative Grad-CAM faithfulness: deletion & insertion AUC.

Idea (standard metric, see Petsiuk et al. "RISE", and common practice in the
XAI-faithfulness literature):

  Deletion: start from the original image, progressively replace the
  MOST-important pixels (per Grad-CAM) with a baseline (blur/zero), and track
  how fast the model's predicted-class probability falls. A faithful
  explanation should cause a FAST drop -> LOW area under the curve.

  Insertion: start from a blank/blurred image, progressively ADD BACK the
  most-important pixels, and track how fast probability rises. A faithful
  explanation should cause a FAST rise -> HIGH area under the curve.

This script evaluates both on a given test split and reports the mean AUCs,
alongside a random-saliency baseline (same procedure, random importance map)
so the numbers are interpretable: what matters is how far Grad-CAM beats random.

Usage:
    python src/faithfulness.py --image_dir data/pcosgen_train/images \
        --label_file data/pcosgen_train/class_label.xlsx --split test \
        --checkpoint checkpoints/best_model.pt --out results/faithfulness_indist.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import auc as sk_auc
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from dataset import PCOSDataset, FolderPCOSDataset, EVAL_TRANSFORM


def build_model(num_classes: int = 2):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def _get_prob(model, img, pred_class):
    with torch.no_grad():
        return torch.softmax(model(img), dim=1)[0, pred_class].item()


def _curves(model, image_tensor, blur, order, pred_class, device, n_steps):
    """Deletion and insertion probability curves for one pixel ordering
    (most important pixel first). Returns (del_probs, ins_probs)."""
    _, _, h, w = image_tensor.shape
    n_pixels = h * w
    step_size = max(1, n_pixels // n_steps)

    def chunk(step):
        end = n_pixels if step == n_steps - 1 else (step + 1) * step_size  # last step takes the remainder
        return order[step * step_size : end]

    # Deletion: start from the original, replace important pixels with the blurred image
    del_probs = [_get_prob(model, image_tensor, pred_class)]
    mask = torch.ones((h, w), device=device)
    for step in range(n_steps):
        ys, xs = np.unravel_index(chunk(step), (h, w))
        mask[ys, xs] = 0
        del_probs.append(_get_prob(model, image_tensor * mask + blur * (1 - mask), pred_class))

    # Insertion: start from the blurred image, reveal important pixels
    ins_probs = [_get_prob(model, blur, pred_class)]
    mask = torch.zeros((h, w), device=device)
    for step in range(n_steps):
        ys, xs = np.unravel_index(chunk(step), (h, w))
        mask[ys, xs] = 1
        ins_probs.append(_get_prob(model, image_tensor * mask + blur * (1 - mask), pred_class))
    return del_probs, ins_probs


def _auc(probs):
    return sk_auc(np.linspace(0, 1, len(probs)), probs)


def _random_saliency_order(h, w, cell_grid, rng):
    """Pixel order from a RANDOM saliency map at the same coarse resolution as
    Grad-CAM (cell_grid x cell_grid, bilinearly upsampled). Comparing against
    per-pixel random noise would be unfair: Grad-CAM masks are spatially
    coherent blobs, which disrupt an image more than scattered pixels do."""
    coarse = torch.tensor(rng.random((1, 1, cell_grid, cell_grid)), dtype=torch.float32)
    up = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.argsort(-up.flatten())


def deletion_insertion_scores(model, cam, image_tensor, pred_class, device, n_steps: int = 20,
                              n_random: int = 3, seed: int = 0):
    """Faithfulness of Grad-CAM for one image, plus a random-saliency baseline.

    Returns a dict with deletion/insertion AUC for Grad-CAM and for the mean of
    `n_random` random saliency maps, plus the model's confidence on the original
    image (p_orig) and on the fully blurred image (p_blur), which show how much
    room the metric has to move.
    """
    rng = np.random.default_rng(seed)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    cam_map = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]  # HxW, 0-1
    h, w = cam_map.shape
    blur = F.avg_pool2d(image_tensor, kernel_size=11, stride=1, padding=5)

    # Grad-CAM (after ReLU) has many exact zeros; break ties with tiny noise so
    # the order in those regions is random rather than top-left biased.
    cam_order = np.argsort(-(cam_map.flatten() + rng.random(h * w) * 1e-6))
    d, i = _curves(model, image_tensor, blur, cam_order, pred_class, device, n_steps)

    grid = 7  # ResNet18 layer4 output resolution for a 224x224 input
    rand_d, rand_i = [], []
    for _ in range(n_random):
        rd, ri = _curves(model, image_tensor, blur, _random_saliency_order(h, w, grid, rng),
                         pred_class, device, n_steps)
        rand_d.append(_auc(rd))
        rand_i.append(_auc(ri))

    return {
        "deletion_auc": _auc(d),
        "insertion_auc": _auc(i),
        "deletion_auc_rand": float(np.mean(rand_d)),
        "insertion_auc_rand": float(np.mean(rand_i)),
        "p_orig": d[0],
        "p_blur": i[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Folder containing the image files")
    parser.add_argument(
        "--label_file", default=None, help="CSV or XLSX mapping imagePath -> class label (PCOSGen-style)"
    )
    parser.add_argument(
        "--positive_class_dir",
        default=None,
        help="For folder-structured datasets (e.g. Kaggle sets): subfolder name for the positive/abnormal class",
    )
    parser.add_argument(
        "--negative_class_dir",
        default=None,
        help="For folder-structured datasets: subfolder name for the negative/normal class",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", help="Label only, used in output CSV")
    parser.add_argument(
        "--test_split_paths",
        default=None,
        help="Optional JSON list from train.py (test_split_paths.json) to "
        "restrict evaluation to the exact held-out test images.",
    )
    parser.add_argument("--max_images", type=int, default=200, help="Cap for runtime; sample size")
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument(
        "--exclude_csv", default=None,
        help="cross_dataset_dedup.py output: drop images that have a near-duplicate elsewhere before evaluating.",
    )
    parser.add_argument(
        "--exclude_col", default=None,
        help="Column in --exclude_csv holding this dataset's image paths. Required if --exclude_csv is given.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    if args.label_file:
        dataset = PCOSDataset(args.image_dir, args.label_file, transform=EVAL_TRANSFORM)
    elif args.positive_class_dir and args.negative_class_dir:
        dataset = FolderPCOSDataset(
            args.image_dir, args.positive_class_dir, args.negative_class_dir, transform=EVAL_TRANSFORM
        )
    else:
        raise ValueError(
            "Provide either --label_file (PCOSGen-style CSV/XLSX) or both "
            "--positive_class_dir and --negative_class_dir (folder-structured datasets)."
        )

    if args.exclude_csv:
        if not args.exclude_col:
            raise ValueError("--exclude_col is required when --exclude_csv is given.")
        exclude_df = pd.read_csv(args.exclude_csv)
        exclude_paths = set(exclude_df[args.exclude_col].astype(str).unique())
        before = len(dataset)
        keep_indices = [i for i, (p, _) in enumerate(dataset.samples) if str(p) not in exclude_paths]
        dataset = Subset(dataset, keep_indices)
        print(f"Excluded {before - len(dataset)} cross-dataset duplicate images ({len(dataset)} remain).")

    if args.test_split_paths:
        with open(args.test_split_paths) as f:
            keep_paths = set(json.load(f))
        indices = [i for i, (p, _) in enumerate(dataset.samples) if str(p) in keep_paths]
        dataset = Subset(dataset, indices)
        print(f"Restricted to {len(dataset)} held-out test images.")

    if len(dataset) > args.max_images:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(dataset), size=args.max_images, replace=False)
        dataset = Subset(dataset, sample_idx.tolist())
        print(f"Sampling {args.max_images} images for faithfulness evaluation.")

    records = []
    for i in tqdm(range(len(dataset)), desc="Faithfulness"):
        image_tensor, label, path = dataset[i]
        with torch.no_grad():
            pred_class = model(image_tensor.unsqueeze(0).to(device)).argmax(dim=1).item()
        scores = deletion_insertion_scores(
            model, cam, image_tensor, pred_class, device, n_steps=args.n_steps, seed=i
        )
        records.append({"path": path, "true_label": label, "pred_class": pred_class, **scores})

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    tag = f"[{args.split}]"
    print(f"\n{tag} images: {len(df)} | predicted abnormal: {(df['pred_class'] == 1).mean():.1%}")
    print(f"{tag} mean confidence on original image: {df['p_orig'].mean():.3f} | on fully blurred image: {df['p_blur'].mean():.3f}")
    print(f"{tag} DELETION  AUC (lower = more faithful):  Grad-CAM {df['deletion_auc'].mean():.4f} | random {df['deletion_auc_rand'].mean():.4f}")
    print(f"{tag} INSERTION AUC (higher = more faithful): Grad-CAM {df['insertion_auc'].mean():.4f} | random {df['insertion_auc_rand'].mean():.4f}")
    try:
        from scipy.stats import wilcoxon
        pd_ = wilcoxon(df["deletion_auc"], df["deletion_auc_rand"]).pvalue
        pi_ = wilcoxon(df["insertion_auc"], df["insertion_auc_rand"]).pvalue
        print(f"{tag} paired Wilcoxon, Grad-CAM vs random: deletion p={pd_:.2g}, insertion p={pi_:.2g}")
    except Exception as e:  # noqa: BLE001
        print(f"{tag} (Wilcoxon test skipped: {e})")
    print(f"Per-image results written to {args.out}")


if __name__ == "__main__":
    main()
