import hashlib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def analyze_clusters(csv_path="results/dup_t0.csv"):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Run dedup_check first.")
        return

    df = pd.read_csv(csv_path)
    multi = df[df.cluster_size > 1]

    print("=== Cluster Summary ===")
    print(f"Clusters with >1 image: {multi.cluster_id.nunique()}")
    print(
        f"10 largest cluster sizes: {sorted(multi.groupby('cluster_id').size(), reverse=True)[:10]}"
    )
    print(f"Unique images if keeping 1 per cluster: {df.cluster_id.nunique()}")

    # Check byte-identical groups (MD5)
    df["md5"] = df.path.map(
        lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    )
    byte_dups = (df.groupby("md5").path.transform("count") > 1).sum()
    print(f"Images in byte-identical groups: {byte_dups}")

    # Pixel difference analysis on the largest cluster
    largest_cid = multi.groupby("cluster_id").size().idxmax()
    cluster_paths = df[df.cluster_id == largest_cid].path.tolist()

    def to_small(p):
        return np.asarray(
            Image.open(p).convert("L").resize((64, 64)), dtype=float
        )

    ref_img = to_small(cluster_paths[0])
    diffs = [np.abs(to_small(p) - ref_img).mean() for p in cluster_paths[1:]]
    print(
        f"Largest cluster size: {len(cluster_paths)} | "
        f"Pixel diff vs first (min/median/max): {min(diffs):.1f} / {np.median(diffs):.1f} / {max(diffs):.1f}"
    )

    # Plot sample images from largest cluster
    os.makedirs("results", exist_ok=True)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    samples = np.random.default_rng(0).choice(
        cluster_paths, size=min(8, len(cluster_paths)), replace=False
    )
    for ax, p in zip(axs.flat, samples):
        ax.imshow(Image.open(p))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/duplicate_samples.png", dpi=300)
    print("Plot saved to results/duplicate_samples.png")


def check_corrupt_files(directory="data/external/data/test"):
    """Check for empty / 0-byte corrupt files in external dataset."""
    if not os.path.exists(directory):
        return
    empty = []
    for root, _, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            if os.path.getsize(path) == 0:
                empty.append(path)
    print(f"Corrupt / 0-byte files in {directory}: {len(empty)}")


if __name__ == "__main__":
    analyze_clusters()
    check_corrupt_files()