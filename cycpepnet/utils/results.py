import json
from pathlib import Path

# Read every framework run record below a folder, skipping auxiliary JSON and rejecting duplicate runs
def read_run_records(root) -> list[dict]:
    root    = Path(root)
    records = []
    seen    = set()

    for path in sorted(root.rglob("*.json")):

        if any(part.startswith("_archive") for part in path.relative_to(root).parts):
            continue

        record = json.loads(path.read_text())

        if not isinstance(record, dict) or not {"task", "model_key", "seed"} <= record.keys():
            continue

        key = tuple(record.get(k) for k in ("task", "model_key", "rep", "split_method", "seed"))

        if key in seen:
            raise ValueError(f"Duplicate framework run: {key}; encountered at {path}")

        seen.add(key)
        records.append(record)

    return records
