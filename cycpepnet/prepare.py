import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from cycpepnet.utils.data_paths import DATASET_PROPERTIES, get_processed_dataset_dir

SOURCES      = ("cycpeptmpdb", "macrocycledb", "npmmpd", "peplife2", "pepmsnd", "peptherdia")
COMBINATIONS = ("cycpeptmpdb_npmmpd", "halflife_peptides")

# Copy the released curated tables and splits into a separate output root, never replacing different files
def _copy_release(parser: argparse.ArgumentParser, dataset: str, output_root: Path) -> None:
    datasets = DATASET_PROPERTIES if dataset == "all" else (dataset,)

    for name in datasets:
        source      = get_processed_dataset_dir(name, Path("data/processed"))
        destination = get_processed_dataset_dir(name, output_root)

        if source.resolve() == destination.resolve():
            parser.error("Use a separate output root for representation generation.")

        for original in [source / f"{name}.csv", *sorted((source / "splits").glob("*.json"))]:
            target = destination / original.relative_to(source)
            if target.exists() and target.read_bytes() != original.read_bytes():
                parser.error(f"Refusing to replace different data at {target}; choose a fresh output root.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)

# Curate datasets and/or generate representations into an explicit output root
def main() -> None:
    parser = argparse.ArgumentParser(description="Curate datasets and generate molecular representations.")
    parser.add_argument("--dataset",         choices=("all", *SOURCES), default="all")
    parser.add_argument("--stage",           choices=("curate", "representations", "all"), default="curate")
    parser.add_argument("--original-root",   type=Path, default=Path("data/original"))
    parser.add_argument("--output-root",     type=Path, default=Path("outputs/processed"))
    parser.add_argument("--from-release",    action="store_true", help="Copy released curated tables and splits before representation generation.")
    parser.add_argument("--representations", nargs="+", help="Optional representation file stems, e.g. morgan_fp; default: all.")
    args = parser.parse_args()

    if args.from_release and args.stage != "representations":
        parser.error("--from-release is only valid with --stage representations")

    if args.from_release:
        _copy_release(parser, args.dataset, args.output_root)

    # Child processes resolve source.paths.DATA_ROOT from this variable at import time
    env   = dict(os.environ, CYCPEPNET_DATA_ROOT=str(args.output_root.resolve()))
    names = SOURCES if args.dataset == "all" else (args.dataset,)

    if args.stage in ("curate", "all"):
        modules = [f"cycpepnet.process.process_{name}" for name in names]
        if args.dataset == "all":
            modules += [f"cycpepnet.combine.combine_{name}" for name in COMBINATIONS]
        for module in modules:
            subprocess.run([sys.executable, "-m", module, "--stage", "curate",
                            "--original-root", str(args.original_root.resolve())], env=env, check=True)

    if args.stage in ("representations", "all"):
        extra = ["--representations", *args.representations] if args.representations else []
        subprocess.run([sys.executable, "-m", "cycpepnet.process._export", "--dataset", args.dataset, *extra],
                       env=env, check=True)


if __name__ == "__main__":
    main()
