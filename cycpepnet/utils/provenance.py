import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np

# SHA-256 of a file, read in 1 MiB blocks
def file_sha256(path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

# SHA-256 of a JSON-serialisable value with a canonical key order
def json_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()

# Identity of a task table: the ordered list of its SMILES
def table_identity(frame) -> str:
    return json_sha256(frame["SMILES"].tolist())

# Input, task and model identities stored with every framework run
def run_provenance(task, frame, dataset_path, split_path, kwargs) -> dict:
    target = frame[task.target_column].to_numpy(dtype="<f8").tobytes()

    return {
        "dataset_sha256"        : file_sha256(dataset_path),
        "split_sha256"          : file_sha256(split_path),
        "task_smiles_sha256"    : table_identity(frame),
        "target_sha256"         : hashlib.sha256(target).hexdigest(),
        "task_definition_sha256": json_sha256(asdict(task)),
        "model_kwargs"          : kwargs,
        "model_kwargs_sha256"   : json_sha256(kwargs),
        **runtime_provenance(),
    }

# Source-code identity and versions of the packages that affect training
@lru_cache(maxsize=1)
def runtime_provenance() -> dict:
    root     = Path(__file__).resolve().parents[1]
    sources  = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))}
    versions = {}

    for package in ("numpy", "pandas", "rdkit", "scikit-learn", "torch", "xgboost", "optuna"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass

    return {"source_sha256": json_sha256(sources), "package_versions": versions}

# Write JSON through a temporary file so an interrupted write never leaves a partial record
def atomic_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)

# Write a compressed NPZ of predictions through a temporary file
def atomic_predictions(path, predictions) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        try:
            np.savez_compressed(handle, **{k: np.asarray(v) for k, v in predictions.items()})
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)

# True only for a complete record whose configuration and predictions match the expected run
def resumable(path, expected) -> bool:
    path = Path(path)

    try:
        record = json.loads(path.read_text())

        if any(record.get(key) != value for key, value in expected.items()):
            return False

        predictions = path.with_suffix(".npz")

        if record.get("predictions_sha256") != file_sha256(predictions):
            return False

        with np.load(predictions, allow_pickle=False) as data:
            score = "y_score" if record["task_type"] == "classification" else "y_pred"
            return len(data["y_true"]) == len(data[score]) == record["n_valid_test"]

    except (OSError, ValueError, KeyError):
        return False
