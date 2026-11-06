import sys
import csv
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycpepnet.paths import PLOTS_DIR, RESULTS_ROOT
from cycpepnet.benchmark import BENCHMARK_TASKS, DATA_ROOT, load_task_table
from cycpepnet.utils import (
    create_histogram,
    create_horizontal_bar_chart,
    create_pie_chart,
    get_processed_dataset_dir,
    print_completion_message,
    print_distribution_stats,
    print_numeric_stats,
    print_section_header,
    print_subsection_header,
    print_visualization_header,
    setup_plot_style,
    tee_stdout_to_log,
)

# Constants
PLOTS_ROOT   = PLOTS_DIR / "datasets" / "task"
SPLIT_COLORS = {"train": "#316B68", "val": "#E67E42", "test": "#9B59B6"}


# Strip the trailing _reg/_cls task-type suffix so plot folders stay clean
def task_plot_dirname(task_name: str) -> str:
    if task_name.endswith("_reg") or task_name.endswith("_cls"):
        return task_name.rsplit("_", 1)[0]
    return task_name


# Per-task extra stats config:
TASK_EXTRA_STATS: dict[str, dict] = {

    "macrocycledb_oprm1_ki_reg": {
        "distributions": [
            ("Molecule_Type",     "MOLECULE TYPE",     None, "pie"),
            ("Molecule_Category", "MOLECULE CATEGORY", None, "pie"),
        ],
        "numerics": [
            ("Peptide_Length", "PEPTIDE LENGTH (residues)", "Peptide Length (residues)", 1, True),
            ("Ring_Size",      "RING SIZE (atoms)",         "Ring Size (atoms)",         1, False),
        ],
    },

    "macrocycledb_mc4r_ki_reg": {
        "distributions": [
            ("Molecule_Type",     "MOLECULE TYPE",     None, "pie"),
            ("Molecule_Category", "MOLECULE CATEGORY", None, "pie"),
        ],
        "numerics": [
            ("Peptide_Length", "PEPTIDE LENGTH (residues)", "Peptide Length (residues)", 1, True),
            ("Ring_Size",      "RING SIZE (atoms)",         "Ring Size (atoms)",         1, False),
        ],
    },

    "macrocycledb_sstr2_ic50_reg": {
        "distributions": [
            ("Molecule_Type",     "MOLECULE TYPE",     None, "pie"),
            ("Molecule_Category", "MOLECULE CATEGORY", None, "pie"),
        ],
        "numerics": [
            ("Peptide_Length", "PEPTIDE LENGTH (residues)", "Peptide Length (residues)", 1, True),
            ("Ring_Size",      "RING SIZE (atoms)",         "Ring Size (atoms)",         1, False),
        ],
    },
    "cycpeptmpdb_reg": {
        "distributions": [
            ("Molecule_Type",  "MOLECULE TYPE",   None, "pie"),
            ("Source",         "SOURCE (TOP 15)", 15,   "bar"),
            ("Molecule_Shape", "MOLECULE SHAPE",  None, "pie"),
        ],
        "numerics": [
            ("Monomer_Length", "MONOMER LENGTH (residues)", "Monomer Length (residues)", 1, True),
            ("Ring_Size",      "RING SIZE (atoms)",         "Ring Size (atoms)",         1, False),
        ],
    },

    "cycpeptmpdb_townsend_reg": {
        "distributions": [
            ("Molecule_Type",  "MOLECULE TYPE",  None, "pie"),
            ("Molecule_Shape", "MOLECULE SHAPE", None, "pie"),
        ],
        "numerics": [
            ("Monomer_Length", "MONOMER LENGTH (residues)", "Monomer Length (residues)", 1, True),
            ("Ring_Size",      "RING SIZE (atoms)",         "Ring Size (atoms)",         1, False),
        ],
    },

    "npmmpd_rzepiela_reg": {
        "distributions": [
            ("Molecule_Type", "MOLECULE TYPE",   None, "pie"),
            ("Source Type",   "SOURCE TYPE",     None, "pie"),
        ],
        "numerics": [
            ("Macrocycle_Ring_Size", "RING SIZE (atoms)", "Ring Size (atoms)", 1, True),
        ],
    },

    "npmmpd_reg": {
        "distributions": [
            ("Molecule_Type", "MOLECULE TYPE",     None, "pie"),
            ("Citation",      "CITATION (TOP 15)", 15,   "bar"),
            ("Source Type",   "SOURCE TYPE",       None, "pie"),
        ],
        "numerics": [
            ("Macrocycle_Ring_Size", "RING SIZE (atoms)", "Ring Size (atoms)", 1, True),
        ],
    },

    "peplife2_pepmsnd_peptherdia_reg": {
        "distributions": [
            ("Molecule_Type",  "MOLECULE TYPE",  None, "pie"),
            ("Source_Dataset", "SOURCE DATASET", None, "pie"),
        ],
        "numerics": [
            ("Length",    "PEPTIDE LENGTH (residues)", "Peptide Length (residues)", 1, True),
            ("Ring_Size", "RING SIZE (atoms)",         "Ring Size (atoms)",         1, False),
        ],
    },
}


# Print extra distributions and numerics for a task
def _print_extras(df: pd.DataFrame, task_name: str) -> dict[str, tuple]:

    cfg = TASK_EXTRA_STATS.get(task_name)
    if cfg is None:
        return {}

    dist_data: dict[str, tuple] = {}
    for col, title, top_n, plot in cfg["distributions"]:
        if col not in df.columns:
            continue
        stats = print_distribution_stats(df, col, title, top_n=top_n)
        if plot:
            dist_data[col] = (stats, plot, top_n)

    num_data: dict[str, tuple] = {}
    for col, title, xlabel, decimals, plot in cfg["numerics"]:
        if col not in df.columns:
            continue
        print_numeric_stats(df, col, title, decimals=decimals)
        if plot:
            num_data[col] = (df[col].dropna(), xlabel)

    return {"dist": dist_data, "num": num_data}


# Generate extra plots for a task
def _plot_extras(plot_data: dict, plots_dir: Path) -> None:

    dist_data = plot_data.get("dist", {})
    num_data  = plot_data.get("num", {})

    for col, (stats, plot_type, top_n) in dist_data.items():

        if len(stats) <= 1:
            continue

        safe_col = col.lower().replace(" ", "_")
        out_path = plots_dir / f"{safe_col}_distribution.png"

        if plot_type == "pie":
            create_pie_chart(stats, col, "Count", "Percentage", out_path)
        elif plot_type == "bar":
            figsize = (20, max(6, min(top_n or len(stats), 15) * 0.6 + 2))
            create_horizontal_bar_chart(stats, col, "Count", "Percentage", out_path,
                                        figsize=figsize, top_n=top_n)

    for col, (series, xlabel) in num_data.items():

        safe_col   = col.lower().replace(" ", "_")
        out_path   = plots_dir / f"{safe_col}_distribution.png"
        is_integer = (series.dropna() % 1 == 0).all()
        bins       = range(int(series.min()), int(series.max()) + 2) if is_integer else 20

        create_histogram(
            series, xlabel, out_path, bins=bins,
            vertical_lines=[
                (series.mean(),   f"Mean ({series.mean():.1f})",     "#569CD6"),
                (series.median(), f"Median ({series.median():.1f})", "#C586C0"),
            ],
        )


# Analyze a single framework task
def analyze_task(task) -> dict:

    print_section_header(f"ANALYZING TASK: {task.name.upper()}")

    df, _ = load_task_table(task, data_root=DATA_ROOT)

    n_total  = len(df)
    n_unique = df["SMILES"].nunique()
    target   = task.target_column

    print(f"\n  Task:          {task.name}")
    print(f"  Dataset:       {task.dataset}")
    print(f"  Target column: {target}")
    print(f"  Log-transform: {task.log_transform}")

    if task.filters:
        print(f"  Filters:       {task.filters}")

    print(f"\n  Samples:       {n_total:,}")
    print(f"  Unique SMILES: {n_unique:,}\n")

    units = "log₁₀" if task.log_transform else "raw"
    print_numeric_stats(df, target, f"TARGET STATISTICS ({target}) [{units}]", decimals=3)

    plot_data = _print_extras(df, task.name)

    splits_dir = get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT) / "splits"
    split_file = splits_dir / f"{task.name}_butina_morgan.json"

    split = None

    if split_file.exists():

        with open(split_file) as f:
            split_data = json.load(f)

        train_idx = split_data["train"]
        val_idx   = split_data["val"]
        test_idx  = split_data["test"]

        print_subsection_header("BUTINA MORGAN SPLIT")
        print(f"\n  Train: {len(train_idx):,}  ({len(train_idx) / n_total * 100:.1f}%)")
        print(f"  Val:   {len(val_idx):,}  ({len(val_idx) / n_total * 100:.1f}%)")
        print(f"  Test:  {len(test_idx):,}  ({len(test_idx) / n_total * 100:.1f}%)\n")

        for split_name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            split_vals = df.iloc[idx][target].dropna()
            print(f"  {split_name}: mean={split_vals.mean():.3f}  std={split_vals.std():.3f}  "
                  f"min={split_vals.min():.3f}  max={split_vals.max():.3f}")

        print()
        split = True

    values = df[target].dropna()
    xlabel = f"log₁₀({target})" if task.log_transform else target

    if task.task_type == "classification":
        print_completion_message()
        return {
            "name" : task.name,
            "n"    : n_total,
            "mean" : float(values.mean()),
            "std"  : float(values.std()),
            "min"  : float(values.min()),
            "max"  : float(values.max()),
            "log"  : task.log_transform,
            "units": f"log₁₀({target})" if task.log_transform else target,
        }

    print_visualization_header()
    setup_plot_style()

    plots_dir = PLOTS_ROOT / task_plot_dirname(task.name)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_extras(plot_data, plots_dir)

    create_histogram(
        values, xlabel,
        plots_dir / "target_distribution.png",
        bins=40,
        vertical_lines=[
            (values.mean(),   f"Mean ({values.mean():.3f})",     "#569CD6"),
            (values.median(), f"Median ({values.median():.3f})", "#C586C0"),
        ],
    )

    if split is not None:
        train_vals = df.iloc[train_idx][target].dropna()
        val_vals   = df.iloc[val_idx][target].dropna()
        test_vals  = df.iloc[test_idx][target].dropna()

        fig, ax    = plt.subplots(figsize=(14, 8))
        all_values = pd.concat([train_vals, val_vals, test_vals])
        bins_range = np.linspace(all_values.min(), all_values.max(), 41)

        ax.hist(train_vals, bins=bins_range, alpha=0.6, color=SPLIT_COLORS["train"], label=f"Train (n={len(train_vals):,})", edgecolor='white')
        ax.hist(val_vals,   bins=bins_range, alpha=0.6, color=SPLIT_COLORS["val"],   label=f"Val (n={len(val_vals):,})",     edgecolor='white')
        ax.hist(test_vals,  bins=bins_range, alpha=0.6, color=SPLIT_COLORS["test"],  label=f"Test (n={len(test_vals):,})",   edgecolor='white')

        ax.set_xlabel(xlabel,      fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        split_path = plots_dir / "split_target_distribution.png"
        plt.savefig(split_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {split_path}")

    print_completion_message()

    return {
        "name" : task.name,
        "n"    : n_total,
        "mean" : float(values.mean()),
        "std"  : float(values.std()),
        "min"  : float(values.min()),
        "max"  : float(values.max()),
        "log"  : task.log_transform,
        "units": f"log₁₀({target})" if task.log_transform else target,
    }

# Order determines row order in splits_summary.csv.
_SPLIT_SUMMARY_ROWS: list[tuple[str, str, str]] = [
    ("MacrocycleDB - OPRM1 (Human)",      "Bioactivity", "macrocycledb_oprm1_ki_reg"),
    ("MacrocycleDB - MC4R (Human)",       "Bioactivity", "macrocycledb_mc4r_ki_reg"),
    ("MacrocycleDB - SSTR2 (Human)",      "Bioactivity", "macrocycledb_sstr2_ic50_reg"),
    ("CycPeptMPDB + NPMMPD",              "Permeability","cycpeptmpdb_npmmpd_reg"),
    ("NPMMPD",                            "Permeability","npmmpd_reg"),
    ("PEPlife2 + PEPMSND + PepTherDia",   "Half-life",   "peplife2_pepmsnd_peptherdia_reg"),
    ("CycPeptMPDB",                       "Permeability","cycpeptmpdb_reg"),
    ("CycPeptMPDB - Townsend 2020",       "Permeability","cycpeptmpdb_townsend_reg"),
    ("NPMMPD - Rzepiela 2022",            "Permeability","npmmpd_rzepiela_reg"),
]

# Generate a CSV summarizing the train/val/test split sizes for each task and split type
def generate_splits_summary(out_path: Path = RESULTS_ROOT / "splits_summary.csv") -> None:

    rows = []

    for display_name, prop, task_name in _SPLIT_SUMMARY_ROWS:

        if task_name not in TASK_MAP:

            rows.append({"Task": display_name, "Property": prop,
                         "Butina_Train": "", "Butina_Val": "", "Butina_Test": "",
                         "Random_Train": "", "Random_Val": "", "Random_Test": ""})

            continue

        task       = TASK_MAP[task_name]
        splits_dir = get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT) / "splits"

        butina_file = splits_dir / f"{task_name}_butina_morgan.json"
        random_file = splits_dir / f"{task_name}_random.json"

        def _read_sizes(path: Path) -> tuple[str, str, str]:
            if not path.exists():
                return "", "", ""
            with open(path) as f:
                d = json.load(f)
            return str(len(d["train"])), str(len(d["val"])), str(len(d["test"]))

        b_tr, b_va, b_te = _read_sizes(butina_file)
        r_tr, r_va, r_te = _read_sizes(random_file)

        rows.append({
            "Task"         : display_name,
            "Property"     : prop,
            "Butina_Train" : b_tr,
            "Butina_Val"   : b_va,
            "Butina_Test"  : b_te,
            "Random_Train" : r_tr,
            "Random_Val"   : r_va,
            "Random_Test"  : r_te,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["Task", "Property", "Butina_Train", "Butina_Val", "Butina_Test",
              "Random_Train", "Random_Val", "Random_Test"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved splits summary → {out_path}")

    for r in rows:
        b = f"{r['Butina_Train']}/{r['Butina_Val']}/{r['Butina_Test']}"
        ra = f"{r['Random_Train']}/{r['Random_Val']}/{r['Random_Test']}"
        print(f"    {r['Task']:<40} Butina={b:<20} Random={ra}")

# Task registry
TASK_MAP = {task.name: task for task in BENCHMARK_TASKS}

if __name__ == "__main__":

    with tee_stdout_to_log("analyze_tasks.log"):
        parser = argparse.ArgumentParser(description="Analyze framework tasks")

        parser.add_argument("tasks", nargs="*", default=list(TASK_MAP.keys()),
                            help="Task names to analyze (default: all)")
        parser.add_argument("--splits-summary", action="store_true",
                            help="Only regenerate results/splits_summary.csv and exit")

        args = parser.parse_args()

        if args.splits_summary:
            generate_splits_summary()
            sys.exit(0)

        tasks_data = []

        for name in args.tasks:

            if name not in TASK_MAP:
                print(f"Unknown task: {name}. Available: {', '.join(TASK_MAP.keys())}")
                continue

            tasks_data.append(analyze_task(TASK_MAP[name]))

        if len(tasks_data) > 1:
            generate_splits_summary()
