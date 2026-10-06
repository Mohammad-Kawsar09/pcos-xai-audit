"""
Near-duplicate image detection via perceptual hashing.

PCOSGen (and similar scraped ultrasound datasets) are known to contain
near-duplicate images -- e.g. the same frame saved twice, or consecutive
frames from the same clip. If duplicates land on both sides of a train/test
split, reported accuracy is inflated. This script flags them before you
train anything.

Usage:
    python src/dedup_check.py --data_dir data/pcosgen --out results/pcosgen_duplicates.csv [--hash_diff_threshold 0]
"""
import argparse
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm


def compute_hashes(data_dir: Path):
    records = []
    image_paths = [
        p
        for p in data_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]
    for img_path in tqdm(image_paths, desc="Hashing images"):
        try:
            img = Image.open(img_path).convert("L")
            h = imagehash.phash(img)
            records.append({"path": str(img_path), "phash": str(h)})
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"Skipping {img_path}: {e}")
    return pd.DataFrame(records)


def find_near_duplicate_clusters(df: pd.DataFrame, hash_diff_threshold: int = 0):
    """Union-find over pairwise Hamming distance <= threshold."""
    hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in tqdm(range(n), desc="Comparing pairs"):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= hash_diff_threshold:
                union(i, j)

    cluster_ids = [find(i) for i in range(n)]
    df = df.copy()
    df["cluster_id"] = cluster_ids
    cluster_sizes = df.groupby("cluster_id")["path"].transform("count")
    df["cluster_size"] = cluster_sizes
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--hash_diff_threshold",
        type=int,
        default=0,
        help="Max Hamming distance between phashes to count as near-duplicate. "
        "0 = identical perceptual hash (recommended: on ultrasound data, thresholds >= 2 "
        "chain unrelated scans into giant clusters).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = compute_hashes(data_dir)
    df = find_near_duplicate_clusters(df, args.hash_diff_threshold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["cluster_size", "cluster_id"], ascending=False).to_csv(
        args.out, index=False
    )

    n_dup_clusters = (df.groupby("cluster_id")["path"].transform("count") > 1).sum()
    print(f"\nTotal images: {len(df)}")
    print(f"Images that belong to a near-duplicate cluster: {n_dup_clusters}")
    print(f"Full report written to {args.out}")
    print(
        "\nNext step: decide how to handle duplicates before training -- "
        "e.g. keep only one image per cluster, or at minimum make sure no "
        "cluster is split across train and test."
    )


if __name__ == "__main__":
    main()
