import argparse
import hashlib
import json
from pathlib import Path
from cycpepnet import utils
from cycpepnet.utils.provenance import atomic_json, file_sha256

# Representation writers in source.utils and the file each one produces
EXPORTS = [
    ("save_morgan_fingerprints",           "morgan_fp.npy"),
    ("save_morgan_count_fingerprints",     "morgan_count_fp.npy"),
    ("save_morgan_r2_binary_fingerprints", "morgan_r2_fp.npy"),
    ("save_morgan_r2_count_fingerprints",  "morgan_r2_count_fp.npy"),
    ("save_map6c_fingerprints",            "map6c_fp.npy"),
    ("save_map4c_fingerprints",            "map4c_fp.npy"),
    ("save_mxfp_fingerprints",             "mxfp_fp.npy"),
    ("save_hdf_fingerprints",              "hdf_fp.npy"),
    ("save_pyg_graphs",                    "graphs.pt"),
    ("save_fragsmiles",                    "fragsmiles.txt"),
]

# Write the molecular representations of a curated table, refusing caches built from other rows or code
def export_representations(frame, output_dir, representations=None) -> None:
    output_dir     = Path(output_dir)
    digest         = hashlib.sha256(frame["SMILES"].to_json(orient="values").encode()).hexdigest()
    manifest       = output_dir / "representations.json"
    implementation = file_sha256(Path(__file__).parents[1] / "utils" / "representations.py")

    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["smiles_sha256"] != digest or previous.get("implementation_sha256") != implementation:
            raise ValueError(f"Stale representations in {output_dir}; use a fresh output root.")

    elif any((output_dir / filename).exists() for _, filename in EXPORTS):
        raise ValueError(f"Unverified legacy caches in {output_dir}; generate representations in a fresh output root.")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, {"smiles_sha256": digest, "rows": len(frame), "implementation_sha256": implementation})

    for function, filename in EXPORTS:
        if representations and Path(filename).stem not in representations:
            continue
        getattr(utils, function)(frame, output_dir / filename, smiles_col="SMILES")

# Command-line entry point shared by the process and combine scripts
def processor_cli(main) -> None:
    parser = argparse.ArgumentParser(description="Curate one dataset and optionally generate representations.")
    parser.add_argument("--stage",         choices=["curate", "all"], default="all")
    parser.add_argument("--original-root", type=Path,                 default=Path("data/original"))
    args = parser.parse_args()

    main(stage=args.stage, original_root=args.original_root)
