import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from cycpepnet.paths import DATA_ROOT
from cycpepnet.utils.provenance import table_identity
from cycpepnet.utils import (
    PHYSICOCHEMICAL_COLUMNS,
    build_metadata,
    butina_split,
    get_dataset_property,
    get_processed_dataset_dir,
    load_morgan_binary_fingerprints,
    load_morgan_count_fingerprints,
    print_completion_message,
    print_section_header,
    print_subsection_header,
    random_split,
    save_split,
)

# Framework task definition
@dataclass
class BenchmarkTask:
    name                     : str
    dataset                  : str
    target_column            : str
    filters                  : dict[str, str] | None  = None
    exclude_filters          : dict[str, list] | None = None
    exclude_zeros            : bool                   = False
    log_transform            : bool                   = False
    task_type                : str                    = "regression"
    classification_threshold : float | None           = None
    classify_above           : bool                   = True
    class_labels             : tuple[str, str]        = ("Negative", "Positive")
    supplementary            : bool                   = False

# Framework tasks registry

BENCHMARK_TASKS: list[BenchmarkTask] = [

    # ── Regression tasks ───────────────────────────

    # MacrocycleDB (OPRM1 KI)
    BenchmarkTask(
        name          = "macrocycledb_oprm1_ki_reg",
        dataset       = "macrocycledb",
        target_column = "Activity_Value",
        filters       = {"Activity_Type": "KI", "Target_Name": "OPRM_HUMAN"},
        log_transform = True,
    ),

    # MacrocycleDB (MC4R KI)
    BenchmarkTask(
        name          = "macrocycledb_mc4r_ki_reg",
        dataset       = "macrocycledb",
        target_column = "Activity_Value",
        filters       = {"Activity_Type": "KI", "Target_Name": "MC4R_HUMAN"},
        log_transform = True,
    ),

    # MacrocycleDB (SSTR2 IC50)
    BenchmarkTask(
        name          = "macrocycledb_sstr2_ic50_reg",
        dataset       = "macrocycledb",
        target_column = "Activity_Value",
        filters       = {"Activity_Type": "IC50", "Target_Name": "SSR2_HUMAN"},
        log_transform = True,
    ),

    # CycPeptMPDB + NPMMPD (PAMPA)
    BenchmarkTask(
        name          = "cycpeptmpdb_npmmpd_reg",
        dataset       = "cycpeptmpdb_npmmpd",
        target_column = "Permeability",
        log_transform = False, # Already in log scale
        supplementary = True,
    ),

    # NPMMPD (PAMPA, Log Peff)
    BenchmarkTask(
        name          = "npmmpd_reg",
        dataset       = "npmmpd",
        target_column = "Value",
        filters       = {"Assay": "PAMPA", "Endpoint": "Log Peff"},
        log_transform = False, # Already in log scale
    ),

    # CycPeptMPDB (PAMPA)
    BenchmarkTask(
        name          = "cycpeptmpdb_reg",
        dataset       = "cycpeptmpdb",
        target_column = "Permeability",
        filters       = {"Assay": "PAMPA"},
        log_transform = False, # Already in log scale
    ),

    # CycPeptMPDB - Single source (PAMPA, Townsend 2020)
    BenchmarkTask(
        name          = "cycpeptmpdb_townsend_reg",
        dataset       = "cycpeptmpdb",
        target_column = "Permeability",
        filters       = {"Assay": "PAMPA", "Source": "2020_Townsend"},
        log_transform = False,
        supplementary = True,
    ),

    # NPMMPD - Single source (PAMPA, Log Peff, Rzepiela 2022)
    BenchmarkTask(
        name          = "npmmpd_rzepiela_reg",
        dataset       = "npmmpd",
        target_column = "Value",
        filters       = {"Assay": "PAMPA", "Endpoint": "Log Peff", "Citation": "(Rzepiela et al., 2022)"},
        log_transform = False,
        supplementary = True,
    ),

    # Half-Life (PEPlife2 + PepMSND + PepTherDia)
    BenchmarkTask(
        name          = "peplife2_pepmsnd_peptherdia_reg",
        dataset       = "peplife2_pepmsnd_peptherdia",
        target_column = "Half_Life_Hours",
        log_transform = True,
    ),

    # Half-Life (PEPlife2 only)
    BenchmarkTask(
        name          = "half_life_peplife2_reg",
        dataset       = "peplife2_pepmsnd_peptherdia",
        target_column = "Half_Life_Hours",
        filters       = {"Source_Dataset": "peplife2"},
        log_transform = True,
        supplementary = True,
    ),

    # ── Classification tasks ───────────────────────

    # MacrocycleDB (OPRM1 KI)
    BenchmarkTask(
        name                     = "macrocycledb_oprm1_ki_cls",
        dataset                  = "macrocycledb",
        target_column            = "Activity_Value",
        filters                  = {"Activity_Type": "KI", "Target_Name": "OPRM_HUMAN"},
        log_transform            = True,
        task_type                = "classification",
        classification_threshold = np.log10(100.0),
        classify_above           = False,
        class_labels             = ("Non-potent", "Potent"),
    ),

    # MacrocycleDB (MC4R KI)
    BenchmarkTask(
        name                     = "macrocycledb_mc4r_ki_cls",
        dataset                  = "macrocycledb",
        target_column            = "Activity_Value",
        filters                  = {"Activity_Type": "KI", "Target_Name": "MC4R_HUMAN"},
        log_transform            = True,
        task_type                = "classification",
        classification_threshold = np.log10(100.0),
        classify_above           = False,
        class_labels             = ("Non-potent", "Potent"),
    ),

    # MacrocycleDB (SSTR2 IC50)
    BenchmarkTask(
        name                     = "macrocycledb_sstr2_ic50_cls",
        dataset                  = "macrocycledb",
        target_column            = "Activity_Value",
        filters                  = {"Activity_Type": "IC50", "Target_Name": "SSR2_HUMAN"},
        log_transform            = True,
        task_type                = "classification",
        classification_threshold = np.log10(100.0),
        classify_above           = False,
        class_labels             = ("Non-potent", "Potent"),
    ),

    # CycPeptMPDB + NPMMPD (PAMPA)
    BenchmarkTask(
        name                     = "cycpeptmpdb_npmmpd_cls",
        dataset                  = "cycpeptmpdb_npmmpd",
        target_column            = "Permeability",
        task_type                = "classification",
        classification_threshold = -6.0,
        classify_above           = True,
        class_labels             = ("Non-permeable", "Permeable"),
        supplementary            = True,
    ),

    # NPMMPD (PAMPA, Log Peff)
    BenchmarkTask(
        name                     = "npmmpd_cls",
        dataset                  = "npmmpd",
        target_column            = "Value",
        filters                  = {"Assay": "PAMPA", "Endpoint": "Log Peff"},
        task_type                = "classification",
        classification_threshold = -6.0,
        classify_above           = True,
        class_labels             = ("Non-permeable", "Permeable"),
    ),

    # CycPeptMPDB (PAMPA)
    BenchmarkTask(
        name                     = "cycpeptmpdb_cls",
        dataset                  = "cycpeptmpdb",
        target_column            = "Permeability",
        filters                  = {"Assay": "PAMPA"},
        task_type                = "classification",
        classification_threshold = -6.0,
        classify_above           = True,
        class_labels             = ("Non-permeable", "Permeable"),
    ),

    # CycPeptMPDB - Single source (PAMPA, Townsend 2020)
    BenchmarkTask(
        name                     = "cycpeptmpdb_townsend_cls",
        dataset                  = "cycpeptmpdb",
        target_column            = "Permeability",
        filters                  = {"Assay": "PAMPA", "Source": "2020_Townsend"},
        task_type                = "classification",
        classification_threshold = -6.0,
        classify_above           = True,
        class_labels             = ("Non-permeable", "Permeable"),
        supplementary            = True,
    ),

    # NPMMPD - Single source (PAMPA, Log Peff, Rzepiela 2022)
    BenchmarkTask(
        name                     = "npmmpd_rzepiela_cls",
        dataset                  = "npmmpd",
        target_column            = "Value",
        filters                  = {"Assay": "PAMPA", "Endpoint": "Log Peff", "Citation": "(Rzepiela et al., 2022)"},
        task_type                = "classification",
        classification_threshold = -6.0,
        classify_above           = True,
        class_labels             = ("Non-permeable", "Permeable"),
        supplementary            = True,
    ),

    # Half-life combined
    BenchmarkTask(
        name                     = "peplife2_pepmsnd_peptherdia_cls",
        dataset                  = "peplife2_pepmsnd_peptherdia",
        target_column            = "Half_Life_Hours",
        log_transform            = True,
        task_type                = "classification",
        classification_threshold = np.log10(8),
        classify_above           = True,
        class_labels             = ("Short-lived", "Long-lived"),
    ),

    # Half-life (PEPlife2 only)
    BenchmarkTask(
        name                     = "half_life_peplife2_cls",
        dataset                  = "peplife2_pepmsnd_peptherdia",
        target_column            = "Half_Life_Hours",
        filters                  = {"Source_Dataset": "peplife2"},
        log_transform            = True,
        task_type                = "classification",
        classification_threshold = np.log10(8),
        classify_above           = True,
        class_labels             = ("Short-lived", "Long-lived"),
        supplementary            = True,
    ),
]

# Load the filtered, deduplicated task table and the source-CSV row of each kept molecule
def load_task_table(
    task     : BenchmarkTask,
    data_root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, np.ndarray]:

    # Load the full dataset CSV and apply task-specific filters, exclusions, and transformations
    dataset_dir = get_processed_dataset_dir(task.dataset, data_root=data_root)
    df          = pd.read_csv(dataset_dir / f"{task.dataset}.csv", low_memory=False)

    # Apply task filters and drop missing targets
    if task.filters:
        for col, val in task.filters.items():
            df = df[df[col] == val]

    # Apply exclusion filters (e.g. outlier DOIs)
    if task.exclude_filters:
        for col, vals in task.exclude_filters.items():
            df = df[~df[col].isin(vals)]

    # Exclude zero values
    if task.exclude_zeros:
        df = df[df[task.target_column] != 0]

    df = df.dropna(subset=[task.target_column])

    # Deduplicate by SMILES
    CV_THRESHOLD = 1.0
    kept_indices = []
    for _, grp in df.groupby("SMILES", sort=False):
        vals = grp[task.target_column].dropna()
        if len(vals) == 0:
            continue
        if len(grp) == 1:
            kept_indices.append(grp.index[0])
            continue
        mean_v = vals.mean()
        cv     = vals.std() / abs(mean_v) if mean_v != 0 else float("inf")
        if cv > CV_THRESHOLD:
            continue
        df.at[grp.index[0], task.target_column] = mean_v
        kept_indices.append(grp.index[0])
    df = df.loc[kept_indices]

    # Log10-transform target: drop non-positive values first, then apply log10
    if task.log_transform:
        df                     = df[df[task.target_column] > 0].copy()
        df[task.target_column] = np.log10(df[task.target_column])

    # Binarize target for classification tasks
    if task.task_type == "classification":

        thr = task.classification_threshold

        if task.classify_above:
            df[task.target_column] = (df[task.target_column] >= thr).astype(np.float32)

        else:
            df[task.target_column] = (df[task.target_column] <= thr).astype(np.float32)

    orig_idx = df.index.values

    return df.reset_index(drop=True), orig_idx

# Load the task table with its Morgan binary/count fingerprints and the Morgan + physicochemical matrix
def load_task_data(
    task     : BenchmarkTask,
    data_root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:

    df, orig_idx = load_task_table(task, data_root)

    dataset_dir = get_processed_dataset_dir(task.dataset, data_root=data_root)
    n_full      = len(pd.read_csv(dataset_dir / f"{task.dataset}.csv", usecols=["SMILES"]))
    fps_morgan  = load_morgan_binary_fingerprints(dataset_dir / "morgan_fp.npy")
    count_path  = dataset_dir / "morgan_count_fp.npy"
    fps_count   = load_morgan_count_fingerprints(count_path) if count_path.exists() else None

    # Fingerprint files must be row-aligned with the unfiltered CSV
    for name, values in (("Morgan", fps_morgan), ("Count", fps_count)):
        if values is not None and len(values) != n_full:
            raise ValueError(f"[{task.dataset}] {name} FP length ({len(values)}) != CSV rows ({n_full})")

    fps_morgan = fps_morgan[orig_idx]
    fps_count  = fps_count[orig_idx] if fps_count is not None else None
    physchem   = df[PHYSICOCHEMICAL_COLUMNS].values.astype(np.float32)
    combined   = np.concatenate([fps_morgan.astype(np.float32), physchem], axis=1)

    return df, fps_morgan, fps_count, orig_idx, combined

# Group framework tasks by their data identity
def _task_groups() -> list[list[BenchmarkTask]]:

    groups: dict[tuple, list[BenchmarkTask]] = defaultdict(list)

    for task in BENCHMARK_TASKS:
        key = (task.dataset, tuple(sorted(task.filters.items())) if task.filters else (), task.log_transform)
        groups[key].append(task)

    return list(groups.values())

# Generate BUTINA splits for all framework tasks using Morgan binary fingerprints
def generate_all_splits(

    data_root    : Path  = DATA_ROOT,
    cutoff_morgan: float = 0.45,
    train_ratio  : float = 0.8,
    val_ratio    : float = 0.1,
    test_ratio   : float = 0.1,
    seed         : int   = 42,

) -> None:

    print_section_header("GENERATING FRAMEWORK SPLITS")

    groups    = _task_groups()
    n_unique  = len(groups)
    n_total   = len(BENCHMARK_TASKS)
    print(f"\n  Found {n_total} framework tasks ({n_unique} unique data splits)\n")

    generated = 0
    skipped   = 0

    for group in groups:

        task    = group[0]
        aliases = [t.name for t in group]

        print_subsection_header(f"Group: {aliases}")

        try:
            df, fps_morgan, _, _, _ = load_task_data(task, data_root)

        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            skipped += len(group)
            continue

        n_rows   = len(df)
        n_smiles = df["SMILES"].nunique()

        print(f"  Samples: {n_rows} ({n_smiles} unique SMILES)")
        print(f"  Target: {task.target_column}")

        if task.filters:
            print(f"  Filters: {task.filters}")

        splits_dir = get_processed_dataset_dir(task.dataset, data_root=data_root, create=True) / "splits"

        task_info = {
            "property"     : get_dataset_property(task.dataset),
            "dataset"      : task.dataset,
            "target_column": task.target_column,
            "filters"      : task.filters,
            "log_transform": task.log_transform,
            "tasks"        : aliases,
        }

        try:
            split_morgan, _, dedup_morgan = butina_split(fps_morgan, df["SMILES"], cutoff_morgan, train_ratio, val_ratio, test_ratio, seed)

            metadata_morgan = build_metadata(
                method          = "butina_morgan", # Morgan BINARY will be used for splitting
                split           = split_morgan,
                task_info       = task_info,
                seed            = seed,
                cutoff          = cutoff_morgan,
                n_clusters      = dedup_morgan["n_clusters"],
                n_unique_smiles = dedup_morgan["n_unique_smiles"],
                train_ratio     = train_ratio,
                val_ratio       = val_ratio,
                test_ratio      = test_ratio,
            )

            metadata_morgan["task_smiles_sha256"] = table_identity(df)

            for t in group:
                save_split(split_morgan, splits_dir / f"{t.name}_butina_morgan.json", metadata_morgan)

            print(
                f"  Morgan: train={len(split_morgan['train'])}, "
                f"val={len(split_morgan['val'])}, "
                f"test={len(split_morgan['test'])} "
                f"({dedup_morgan['n_clusters']} clusters)"
            )

        except ValueError as e:
            print(f"  Morgan split failed: {e}")
            skipped += len(group)
            continue

        generated += len(group)

    print()
    print_section_header("SUMMARY")
    print(f"\n  Tasks generated: {generated}")
    print(f"  Tasks skipped:   {skipped}")
    print(f"  Total tasks:     {n_total}")
    print()

    if skipped:
        raise RuntimeError(f"Split generation incomplete: {skipped} tasks failed.")

    print_completion_message()

# Generate RANDOM splits for all framework tasks
def generate_random_splits(

    data_root  : Path  = DATA_ROOT,
    train_ratio: float = 0.8,
    val_ratio  : float = 0.1,
    test_ratio : float = 0.1,
    seed       : int   = 42,

) -> None:

    print_section_header("GENERATING RANDOM SPLITS")

    groups   = _task_groups()
    n_unique = len(groups)
    n_total  = len(BENCHMARK_TASKS)

    print(f"\n  Found {n_total} framework tasks ({n_unique} unique data splits)\n")

    generated = 0
    skipped   = 0

    for group in groups:

        task    = group[0]
        aliases = [t.name for t in group]

        print_subsection_header(f"Group: {aliases}")

        try:
            df, _ = load_task_table(task, data_root)

        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            skipped += len(group)
            continue

        n_samples = len(df)
        n_smiles  = df["SMILES"].nunique()

        print(f"  Samples: {n_samples} ({n_smiles} unique SMILES)")
        print(f"  Target: {task.target_column}")

        if task.filters:
            print(f"  Filters: {task.filters}")

        splits_dir = get_processed_dataset_dir(task.dataset, data_root=data_root, create=True) / "splits"

        task_info = {
            "property"     : get_dataset_property(task.dataset),
            "dataset"      : task.dataset,
            "target_column": task.target_column,
            "filters"      : task.filters,
            "log_transform": task.log_transform,
            "tasks"        : aliases,
        }

        split_rand = random_split(n_samples, train_ratio, val_ratio, test_ratio, seed)

        metadata_rand = build_metadata(
            method      = "random",
            split       = split_rand,
            task_info   = task_info,
            seed        = seed,
            train_ratio = train_ratio,
            val_ratio   = val_ratio,
            test_ratio  = test_ratio,
        )

        metadata_rand["task_smiles_sha256"] = table_identity(df)

        for t in group:
            save_split(split_rand, splits_dir / f"{t.name}_random.json", metadata_rand)

        print(
            f"  Random: train={len(split_rand['train'])}, "
            f"val={len(split_rand['val'])}, "
            f"test={len(split_rand['test'])}"
        )

        generated += len(group)

    print()
    print_section_header("SUMMARY")
    print(f"\n  Tasks generated: {generated}")
    print(f"  Tasks skipped:   {skipped}")
    print(f"  Total tasks:     {n_total}")
    print()

    if skipped:
        raise RuntimeError(f"Split generation incomplete: {skipped} tasks failed.")

    print_completion_message()

# Generate new split files explicitly, refusing to replace existing ones unless --overwrite is given
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate new splits explicitly; analysis reuses existing splits.")
    parser.add_argument("--data-root", type=Path, default=Path("outputs/processed"))
    parser.add_argument("--method",    choices=["all", "butina_morgan", "random"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing split files.")
    args = parser.parse_args()

    methods  = ["butina_morgan", "random"] if args.method == "all" else [args.method]
    existing = [
        get_processed_dataset_dir(t.dataset, args.data_root) / "splits" / f"{t.name}_{m}.json"
        for t in BENCHMARK_TASKS for m in methods
    ]

    if not args.overwrite and any(p.exists() for p in existing):
        parser.error("Split files already exist. Reuse them, choose a fresh data root, or pass --overwrite explicitly.")

    if "butina_morgan" in methods:
        generate_all_splits(data_root=args.data_root)

    if "random" in methods:
        generate_random_splits(data_root=args.data_root)


if __name__ == "__main__":
    main()
