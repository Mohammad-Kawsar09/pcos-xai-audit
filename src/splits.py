"""
Cluster-aware train/val/test splitting.

PCOSGen contains many near-duplicate images (the same scan saved in several
crops/scales). A plain random split puts copies of one scan on both sides of
the split, which inflates test accuracy. `mode="group"` keeps every cluster
entirely inside one split. `mode="random"` is the naive split, kept so the
paper can report how much the leakage inflates results.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_cluster_map(group_csv: str) -> dict:
    """path -> cluster_id, from dedup_check.py output."""
    df = pd.read_csv(group_csv)
    return dict(zip(df["path"].astype(str), df["cluster_id"]))


def get_cluster_ids(paths, cluster_map):
    """Cluster id per image; images missing from the map become singletons."""
    ids, missing = [], 0
    for i, p in enumerate(paths):
        if p in cluster_map:
            ids.append(int(cluster_map[p]))
        else:
            ids.append(-(i + 1))  # unique negative id = its own cluster
            missing += 1
    if missing:
        print(f"[splits] Warning: {missing} images not found in the cluster CSV (treated as unique).")
    return np.array(ids)


def make_splits(cluster_ids, mode="group", seed=42, val_frac=0.15, test_frac=0.15):
    n = len(cluster_ids)
    idx = np.arange(n)
    if mode == "group":
        groups = cluster_ids
    elif mode == "random":
        groups = np.arange(n)  # every image its own group = plain random split
    else:
        raise ValueError("mode must be 'group' or 'random'")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, groups=groups))
    rel_val = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr, va = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))
    return trainval_idx[tr], trainval_idx[va], test_idx


def leakage_report(train_idx, val_idx, test_idx, cluster_ids):
    """Fraction of val/test images that have a near-duplicate in the train split."""
    train_clusters = set(cluster_ids[train_idx].tolist())
    val_leak = float(np.mean([c in train_clusters for c in cluster_ids[val_idx]]))
    test_leak = float(np.mean([c in train_clusters for c in cluster_ids[test_idx]]))
    return {"val_with_dup_in_train": val_leak, "test_with_dup_in_train": test_leak}
