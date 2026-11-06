import argparse
import csv
import json
import sys
from pathlib import Path
from cycpepnet.utils.provenance import file_sha256, json_sha256

MANIFEST = Path("release-manifest.json")

# Check the acquired original inputs against the framework snapshot, without processed data
def check_inputs(root=Path("data/original"), inventory=Path("data/sources.csv")) -> None:
    with Path(inventory).open(newline="") as handle:
        records = list(csv.DictReader(handle))

    if not records:
        raise ValueError("The input inventory is empty.")

    errors = []

    for record in records:
        relative = Path(record["relative_path"]).relative_to("data/original")

        if ".." in relative.parts:
            raise ValueError("Input inventory paths must stay below data/original.")

        path = Path(root) / relative

        if not path.is_file():
            errors.append(f"Missing input: {path}")
        elif path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]:
            errors.append(f"Different input version: {path}")

    if errors:
        raise ValueError("Input verification failed; keep the framework snapshot unchanged:\n" + "\n".join(errors))

    print(f"Verified {len(records)} original input files against the framework snapshot.")

# Released data and result files whose identities the manifest records
def artifact_files() -> list[Path]:
    files  = list(Path("data/processed").rglob("*.csv"))
    files += [p for p in Path("data/processed").rglob("*.json") if p.parent.name == "splits"]
    files += [
        p for p in Path("results").rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".npz", ".csv", ".tex")
        and not any(part.startswith("_archive") for part in p.parts)
    ]
    return sorted(set(files))

# Hash released files, grouping run records and predictions by folder
def artifact_groups() -> dict[str, dict]:
    groups = {}

    for path in artifact_files():
        is_run_file = path.suffix in (".json", ".npz") and path.parts[0] == "results" and len(path.parts) > 2
        key         = str(path.parent) if is_run_file else str(path)
        groups.setdefault(key, {})[str(path)] = file_sha256(path)

    return {key: {"files": len(values), "sha256": json_sha256(values)} for key, values in groups.items()}

# Validate every released split against the task table it indexes and return how many were checked
def check_splits() -> int:
    from cycpepnet.benchmark import BENCHMARK_TASKS, load_task_table
    from cycpepnet.utils.data_paths import get_processed_dataset_dir
    from cycpepnet.utils.provenance import table_identity
    from cycpepnet.utils.splitting import assert_valid_split, load_split

    for task in BENCHMARK_TASKS:
        frame, _ = load_task_table(task)

        for method in ("butina_morgan", "random"):
            path            = get_processed_dataset_dir(task.dataset) / "splits" / f"{task.name}_{method}.json"
            split, metadata = load_split(path)
            assert_valid_split(split, len(frame), path)

            identity = metadata.get("task_smiles_sha256")
            if identity is not None and identity != table_identity(frame):
                raise ValueError(f"Split row identity mismatch: {path}")

    return 2 * len(BENCHMARK_TASKS)

# Compare reproduced CSV/TeX outputs with the released ones (CSV within rtol=atol=1e-8)
def compare_outputs(root, expected) -> None:
    import pandas as pd

    errors = []

    for relative in expected:
        reference = Path("results") / relative
        actual    = Path(root) / relative
        try:
            if reference.suffix == ".csv":
                pd.testing.assert_frame_equal(pd.read_csv(reference), pd.read_csv(actual),
                                              check_dtype=False, rtol=1e-8, atol=1e-8)
            elif reference.read_text() != actual.read_text():
                raise ValueError("Text differs")
        except (OSError, AssertionError, ValueError) as exc:
            errors.append(f"{relative}: {str(exc).splitlines()[0]}")

    if errors:
        raise ValueError("Output comparison failed:\n" + "\n".join(errors))

    print(f"Matched {len(expected)} numerical/text outputs (rtol=atol=1e-8 for CSV).")

# Verify inputs, release artifacts and splits, or record a new manifest (maintainers only)
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify original inputs, released artifacts and reproduced outputs.")
    parser.add_argument("--compare",         type=Path, help="Compare a reproduced results directory with the release.")
    parser.add_argument("--write-manifest",  action="store_true", help="Maintainer only: accept the current artifact identities.")
    parser.add_argument("--reproduced-root", type=Path, help="For --write-manifest: completed reproduction output directory.")
    parser.add_argument("--inputs-only",     action="store_true", help="Verify original inputs without processed data or results.")
    parser.add_argument("--original-root",   type=Path, default=Path("data/original"), help="Input directory for --inputs-only.")
    args = parser.parse_args()

    if args.inputs_only:
        if args.compare or args.write_manifest or args.reproduced_root:
            parser.error("--inputs-only cannot be combined with result/manifest operations")
        check_inputs(args.original_root)
        return

    if args.original_root != Path("data/original"):
        parser.error("--original-root requires --inputs-only")

    if args.write_manifest:
        if args.reproduced_root is None:
            parser.error("--write-manifest requires --reproduced-root")
        expected = sorted(
            str(p.relative_to(args.reproduced_root))
            for p in args.reproduced_root.rglob("*")
            if p.suffix in (".csv", ".tex")
        )
        compare_outputs(args.reproduced_root, expected)
        check_splits()
        manifest = {"schema_version": 1, "groups": artifact_groups(), "reproduced_outputs": expected}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {MANIFEST}")
        return

    manifest = json.loads(MANIFEST.read_text())
    actual   = artifact_groups()
    expected = manifest["groups"]
    changed  = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))

    if changed:
        raise ValueError("Release artifacts changed or missing:\n" + "\n".join(changed))

    n_splits = check_splits()
    print(f"Verified {sum(v['files'] for v in actual.values())} artifact files and {n_splits} task splits.")

    if args.compare:
        compare_outputs(args.compare, manifest["reproduced_outputs"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
