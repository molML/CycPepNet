import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from cycpepnet.utils.provenance import atomic_json

# Convert numpy array of binary fingerprints to RDKit bitvect objects
def _numpy_to_bitvects(fp_array: np.ndarray) -> list:

    n_samples, n_bits = fp_array.shape
    bitvects          = []

    for i in range(n_samples):

        bv      = DataStructs.ExplicitBitVect(n_bits)
        on_bits = np.where(fp_array[i] == 1)[0]

        for bit in on_bits:
            bv.SetBit(int(bit))

        bitvects.append(bv)

    return bitvects

# Compute generalized Tanimoto distance for count fingerprints
def _count_tanimoto_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:

    numerator   = np.sum(np.minimum(fp1, fp2))
    denominator = np.sum(np.maximum(fp1, fp2))

    if denominator == 0:
        return 1.0

    similarity = numerator / denominator

    return 1.0 - similarity

# Compute simple-matching-coefficient distance for MAP6C fingerprints
def _map6c_smc_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:

    intersection = np.sum(fp1 == fp2)
    union        = len(fp1)

    if union == 0:
        return 1.0

    similarity = intersection / union

    return 1.0 - similarity

# Cluster Morgan binary fingerprints using Tanimoto distance
def butina_cluster(fingerprints: np.ndarray, cutoff: float = 0.5, precomputed_dists: np.ndarray = None) -> list[list[int]]:

    if precomputed_dists is not None:
        n     = len(fingerprints)
        dists = precomputed_dists

    else:
        bitvects = _numpy_to_bitvects(fingerprints)
        n        = len(bitvects)
        dists    = []

        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
            dists.extend([1.0 - s for s in sims])

    clusters_raw = Butina.ClusterData(dists, nPts=n, distThresh=cutoff, isDistData=True)
    clusters     = [list(c) for c in clusters_raw]

    clusters.sort(key=lambda c: len(c), reverse=True)

    return clusters

# Cluster Morgan count fingerprints using generalized Tanimoto distance
def butina_cluster_count(fingerprints: np.ndarray, cutoff: float = 0.5, precomputed_dists: np.ndarray = None) -> list[list[int]]:

    if precomputed_dists is not None:
        n     = len(fingerprints)
        dists = precomputed_dists

    else:
        n     = len(fingerprints)
        dists = []

        for i in range(1, n):
            for j in range(i):
                dist = _count_tanimoto_distance(fingerprints[i], fingerprints[j])
                dists.append(dist)

    clusters_raw = Butina.ClusterData(dists, nPts=n, distThresh=cutoff, isDistData=True)
    clusters    = [list(c) for c in clusters_raw]

    clusters.sort(key=lambda c: len(c), reverse=True)

    return clusters

# Cluster fingerprints using simple-matching-coefficient distance (MAP6C / MAP4C)
def _butina_cluster_smc(fingerprints: np.ndarray, cutoff: float, precomputed_dists: np.ndarray = None) -> list[list[int]]:

    if precomputed_dists is not None:
        n     = len(fingerprints)
        dists = precomputed_dists

    else:
        n     = len(fingerprints)
        dists = []

        for i in range(1, n):
            for j in range(i):
                dists.append(_map6c_smc_distance(fingerprints[i], fingerprints[j]))

    clusters_raw = Butina.ClusterData(dists, nPts=n, distThresh=cutoff, isDistData=True)
    clusters     = [list(c) for c in clusters_raw]
    clusters.sort(key=lambda c: len(c), reverse=True)

    return clusters

# Butina cluster using simple-matching-coefficient distance for MAP6C fingerprints
def butina_cluster_map6c(fingerprints: np.ndarray, cutoff: float = 0.55, precomputed_dists: np.ndarray = None) -> list[list[int]]:
    return _butina_cluster_smc(fingerprints, cutoff, precomputed_dists)

# Assign clusters to train/val/test splits using greedy allocation
def assign_clusters_to_splits(clusters: list[list[int]], n_samples: int, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42,) -> dict[str, np.ndarray]:

    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    n_splits_needed = sum(1 for r in [train_ratio, val_ratio, test_ratio] if r > 0)

    if len(clusters) < n_splits_needed:
        raise ValueError(
            f"Need at least {n_splits_needed} clusters for this split, got {len(clusters)}.")

    targets = {
        "train": n_samples * train_ratio,
        "val"  : n_samples * val_ratio,
        "test" : n_samples * test_ratio,
    }

    rng                 = np.random.default_rng(seed)
    clusters_with_noise = [(len(c), rng.random(), c) for c in clusters]

    clusters_with_noise.sort(key=lambda x: (-x[0], x[1]))
    sorted_clusters = [c for _, _, c in clusters_with_noise]

    reserved_test = sorted_clusters.pop() if test_ratio > 0 else []
    reserved_val  = sorted_clusters.pop() if val_ratio  > 0 else []

    splits = {"train": [], "val": list(reserved_val), "test": list(reserved_test)}

    counts = {
        "train": 0,
        "val"  : len(reserved_val),
        "test" : len(reserved_test),
        }

    for cluster in sorted_clusters:

        deficits   = {k: targets[k] - counts[k] for k in splits}
        best_split = max(deficits, key=deficits.get)

        splits[best_split].extend(cluster)
        counts[best_split] += len(cluster)

    return {k: np.sort(np.array(v, dtype=int)) for k, v in splits.items()}

# Shared implementation for all Butina split variants
def _butina_split(
    fingerprints: np.ndarray,
    smiles      : pd.Series,
    cluster_fn,
    cutoff      : float,
    train_ratio : float,
    val_ratio   : float,
    test_ratio  : float,
    seed        : int,
) -> tuple[dict[str, np.ndarray], list[list[int]], dict]:

    n_total       = len(smiles)
    unique_smiles = smiles.unique()
    n_unique      = len(unique_smiles)

    smiles_to_rows: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles):
        smiles_to_rows.setdefault(smi, []).append(idx)

    unique_fps_indices = [smiles_to_rows[smi][0] for smi in unique_smiles]
    unique_fps         = fingerprints[unique_fps_indices]
    clusters           = cluster_fn(unique_fps, cutoff=cutoff)
    unique_splits      = assign_clusters_to_splits(clusters, n_unique, train_ratio, val_ratio, test_ratio, seed)
    full_splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for split_name, unique_indices in unique_splits.items():
        for uid in unique_indices:
            full_splits[split_name].extend(smiles_to_rows[unique_smiles[uid]])

    split_dict = {k: np.sort(np.array(v, dtype=int)) for k, v in full_splits.items()}
    dedup_info = {"n_total": n_total, "n_unique_smiles": n_unique, "n_clusters": len(clusters)}

    return split_dict, clusters, dedup_info


# Butina split using Morgan binary fingerprints
def butina_split(fingerprints: np.ndarray, smiles: pd.Series, cutoff: float = 0.5,
                 train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                 seed: int = 42) -> tuple[dict[str, np.ndarray], list[list[int]], dict]:
    return _butina_split(fingerprints, smiles, butina_cluster, cutoff, train_ratio, val_ratio, test_ratio, seed)

# Random split by row index
def random_split(n_samples: int, train_ratio: float = 0.8, val_ratio: float = 0.1,
                 test_ratio: float = 0.1, seed: int = 42) -> dict[str, np.ndarray]:

    ratio_sum = train_ratio + val_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(n_samples)
    n_val  = round(n_samples * val_ratio)
    n_test = round(n_samples * test_ratio)

    val_idx   = np.sort(idx[:n_val])
    test_idx  = np.sort(idx[n_val:n_val + n_test])
    train_idx = np.sort(idx[n_val + n_test:])

    return {"train": train_idx, "val": val_idx, "test": test_idx}

# List every structural problem of a split: overlaps, duplicates, out-of-range or missing indices
def validate_split(

    split: dict[str, np.ndarray],
    n_samples: int,
    require_full_coverage: bool = True,

) -> list[str]:

    problems: list[str] = []
    split_names = ("train", "val", "test")
    missing = [name for name in split_names if name not in split]

    if missing:
        problems.append(f"missing split parts: {missing}")
        return problems

    arrays = {name: np.asarray(split[name], dtype=int) for name in split_names}

    for name, values in arrays.items():
        if values.ndim != 1:
            problems.append(f"{name} indices are not one-dimensional")
            continue

        if len(np.unique(values)) != len(values):
            problems.append(f"{name} contains duplicate indices")

        if values.size and (values.min() < 0 or values.max() >= n_samples):
            problems.append(f"{name} contains indices outside [0, {n_samples})")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = np.intersect1d(arrays[left], arrays[right])
        if overlap.size:
            problems.append(f"{left} and {right} overlap at {overlap.size} indices")

    if require_full_coverage:
        covered = np.unique(np.concatenate([arrays[name] for name in split_names]))
        if len(covered) != n_samples:
            problems.append(f"split covers {len(covered)}/{n_samples} rows")

    return problems


# Raise if a split file is structurally invalid for a dataset of n_samples rows
def assert_valid_split(
    split: dict[str, np.ndarray],
    n_samples: int,
    split_path: Path,

) -> None:

    problems = validate_split(split, n_samples)

    if problems:
        problem_text = "; ".join(problems)
        raise ValueError(f"Invalid split file {split_path}: {problem_text}")


# Save split indices and metadata to a JSON file
def save_split(split: dict[str, np.ndarray], output_path: Path, metadata: dict | None = None,) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": metadata or {},
        "train"   : split["train"].tolist(),
        "val"     : split["val"].tolist(),
        "test"    : split["test"].tolist(),
    }

    atomic_json(output_path, data)

    print(f"  Saved: {output_path}")
    return output_path

# Load split indices from a JSON file
def load_split(input_path: Path) -> tuple[dict[str, np.ndarray], dict]:

    with open(input_path) as f:
        data = json.load(f)

    for name in ("train", "val", "test"):
        if any(isinstance(x, bool) or not isinstance(x, int) for x in data[name]):
            raise ValueError(f"Non-integer indices in {input_path}: {name}")

    split = {
        "train": np.array(data["train"], dtype=int),
        "val"  : np.array(data["val"], dtype=int),
        "test" : np.array(data["test"], dtype=int),
    }

    return split, data.get("metadata", {})

# Build metadata dict for a split
def build_metadata(method: str, split: dict[str, np.ndarray], task_info: dict, seed: int = 42, cutoff: float | None = None, n_clusters: int | None = None,
                   n_unique_smiles: int | None = None, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,) -> dict:

    n_total = sum(len(split[k]) for k in ["train", "val", "test"])

    metadata = {
        "method"     : method,
        "seed"       : seed,
        "train_ratio": train_ratio,
        "val_ratio"  : val_ratio,
        "test_ratio" : test_ratio,
        "n_total"    : n_total,
        "n_train"    : len(split["train"]),
        "n_val"      : len(split["val"]),
        "n_test"     : len(split["test"]),
        "task"       : task_info,
        "created_at" : datetime.now().isoformat(timespec="seconds"),
    }

    if cutoff is not None:
        metadata["cutoff"] = cutoff

    if n_clusters is not None:
        metadata["n_clusters"] = n_clusters

    if n_unique_smiles is not None:
        metadata["n_unique_smiles"] = n_unique_smiles

    return metadata
