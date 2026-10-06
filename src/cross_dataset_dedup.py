"""
Check for near-duplicate images ACROSS two datasets (e.g. PCOSGen-train vs.
PCOSGen-test, or PCOSGen-train vs. an external Kaggle set). This is different
from dedup_check.py, which only looks for duplicates WITHIN one folder.

If a "held-out" test set shares images with train, evaluation on it is
contaminated -- the model has effectively already seen those images (or
crops of them). This script flags any test image that has a near-duplicate
in train, so you can exclude it and report a clean number.

Usage:
    python src/cross_dataset_dedup.py \
        --dir_a /content/data/train/images --label_a train \
        --dir_b /content/data/test/images  --label_b test \
        --hash_diff_threshold 0 \
        --out results/train_vs_test_duplicates.csv
"""
import argparse
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm


def compute_hashes(data_dir: Path, label: str):
    records = []
    paths = [p for p in data_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    for p in tqdm(paths, desc=f"Hashing {label}"):
        try:
            records.append({"path": str(p), "source": label, "phash": imagehash.phash(Image.open(p).convert("L"))})
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {p}: {e}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--dir_b", required=True)
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--hash_diff_threshold", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    recs_a = compute_hashes(Path(args.dir_a), args.label_a)
    recs_b = compute_hashes(Path(args.dir_b), args.label_b)
    print(f"{args.label_a}: {len(recs_a)} images | {args.label_b}: {len(recs_b)} images")

    matches = []
    for rb in tqdm(recs_b, desc=f"Comparing {args.label_b} against {args.label_a}"):
        for ra in recs_a:
            if rb["phash"] - ra["phash"] <= args.hash_diff_threshold:
                matches.append(
                    {
                        f"{args.label_b}_path": rb["path"],
                        f"{args.label_a}_path": ra["path"],
                        "hash_distance": rb["phash"] - ra["phash"],
                    }
                )

    df = pd.DataFrame(matches)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_contaminated = df[f"{args.label_b}_path"].nunique() if len(df) else 0
    print(f"\n{args.label_b} images with a near-duplicate in {args.label_a}: {n_contaminated} / {len(recs_b)} "
          f"({n_contaminated / max(len(recs_b), 1):.1%})")
    print(f"Full pair list written to {args.out}")
    print(
        f"\nNext step: if this is non-trivial, exclude those {args.label_b} images before "
        f"reporting accuracy/faithfulness on {args.label_b} -- see evaluate_external.py's "
        f"upcoming --exclude_csv flag."
    )


if __name__ == "__main__":
    main()
