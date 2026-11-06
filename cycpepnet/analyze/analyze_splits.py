import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from cycpepnet.paths import DATA_ROOT, RESULTS_ROOT
from cycpepnet.benchmark import BENCHMARK_TASKS, load_task_table
from cycpepnet.utils import (
    get_processed_dataset_dir,
    print_completion_message,
    print_section_header,
    tee_stdout_to_log,
)
from cycpepnet.analyze._common import (
    CANONICAL_BUTINA_SPLIT,
    TASK_ORDER_CLS,
    TASK_ORDER_REG,
)

# Paths
OUTPUT_CSV   = RESULTS_ROOT / "splits_diagnostics.csv"

# Display names for tasks
TASK_DISPLAY: dict[str, str] = {
    "macrocycledb_oprm1_ki_reg":       "Binding\n(OPRM1 Ki)",
    "macrocycledb_mc4r_ki_reg":        "Binding\n(MC4R Ki)",
    "macrocycledb_sstr2_ic50_reg":     "Binding\n(SSTR2 IC50)",
    "npmmpd_reg":                      "Permeability\n(NPMMPD)",
    "cycpeptmpdb_reg":                 "Permeability\n(CycPeptMPDB)",
    "peplife2_pepmsnd_peptherdia_reg": "Half-life",
    "macrocycledb_oprm1_ki_cls":       "Binding\n(OPRM1 Ki)",
    "macrocycledb_mc4r_ki_cls":        "Binding\n(MC4R Ki)",
    "macrocycledb_sstr2_ic50_cls":     "Binding\n(SSTR2 IC50)",
    "npmmpd_cls":                      "Permeability\n(NPMMPD)",
    "cycpeptmpdb_cls":                 "Permeability\n(CycPeptMPDB)",
    "peplife2_pepmsnd_peptherdia_cls": "Half-life",
}

# Display labels for split methods
SPLIT_LABEL: dict[str, str] = {
    "butina_morgan": "Butina (Morgan)",
    "random":        "Random",
}

TASK_MAP = {task.name: task for task in BENCHMARK_TASKS}

# Target statistics (mean and std) for a given array of values
def _target_stats(values: np.ndarray) -> dict[str, float]:

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan")}

    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values)),
    }

# Compute diagnostics for one task + split method
def diagnose_split(task, split_method: str = CANONICAL_BUTINA_SPLIT) -> dict | None:

    splits_dir = get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT) / "splits"
    split_file = splits_dir / f"{task.name}_{split_method}.json"

    if not split_file.exists():
        return None

    with open(split_file) as f:
        split_data = json.load(f)

    metadata  = split_data.get("metadata", {})
    train_idx = np.array(split_data["train"])
    val_idx   = np.array(split_data["val"])
    test_idx  = np.array(split_data["test"])

    df, _ = load_task_table(task, data_root=DATA_ROOT)
    target         = task.target_column

    train_y = df.iloc[train_idx][target].dropna().values
    val_y   = df.iloc[val_idx][target].dropna().values
    test_y  = df.iloc[test_idx][target].dropna().values

    train_stats = _target_stats(train_y)
    val_stats   = _target_stats(val_y)
    test_stats  = _target_stats(test_y)

    rec: dict = {
        "task":         task.name,
        "task_type":    task.task_type,
        "split_method": split_method,
        "n_total":      len(df),
        "n_train":      len(train_idx),
        "n_val":        len(val_idx),
        "n_test":       len(test_idx),
        "train_mean":   train_stats["mean"],
        "train_std":    train_stats["std"],
        "val_mean":     val_stats["mean"],
        "val_std":      val_stats["std"],
        "test_mean":    test_stats["mean"],
        "test_std":     test_stats["std"],
    }

    n_clusters = metadata.get("n_clusters")
    if n_clusters is not None:
        rec["n_clusters"] = n_clusters

    if len(train_y) >= 5 and len(test_y) >= 5:
        ks_stat, ks_p = sp_stats.ks_2samp(train_y, test_y)
        rec["ks_stat_train_test"] = float(ks_stat)
        rec["ks_p_train_test"]    = float(ks_p)
        if task.task_type == "regression":
            rec["wasserstein_train_test"] = float(sp_stats.wasserstein_distance(train_y, test_y))

    if len(train_y) >= 5 and len(val_y) >= 5:
        ks_stat_v, ks_p_v = sp_stats.ks_2samp(train_y, val_y)
        rec["ks_stat_train_val"] = float(ks_stat_v)
        rec["ks_p_train_val"]    = float(ks_p_v)

    return rec

# Print a compact diagnostics summary for the given records
def _print_diagnostics(records: list[dict]) -> None:

    for rec in records:

        display = TASK_DISPLAY.get(rec["task"], rec["task"]).replace("\n", " ")
        split   = SPLIT_LABEL.get(rec["split_method"], rec["split_method"])

        print(f"\n  {display} ({split})")
        print(f"    Size:   n_train={rec['n_train']}  n_val={rec['n_val']}  n_test={rec['n_test']}")
        print(f"    Train:  mean={rec['train_mean']:.3f}  std={rec['train_std']:.3f}")
        print(f"    Val:    mean={rec['val_mean']:.3f}  std={rec['val_std']:.3f}")
        print(f"    Test:   mean={rec['test_mean']:.3f}  std={rec['test_std']:.3f}")

        if "n_clusters" in rec:
            print(f"    Clusters: {rec['n_clusters']}")

        if "ks_stat_train_test" in rec:
            ks   = rec["ks_stat_train_test"]
            ks_p = rec["ks_p_train_test"]
            wd   = rec.get("wasserstein_train_test", float("nan"))
            sig  = "significant" if ks_p < 0.05 else "non-significant"
            print(f"    KS(train→test): D={ks:.3f}  p={ks_p:.4f} [{sig}]  Wasserstein={wd:.3f}")

# Main analysis function
if __name__ == "__main__":

    with tee_stdout_to_log("analyze_splits.log"):

        all_records: list[dict] = []

        print_section_header("SPLIT DIAGNOSTICS - REGRESSION")
        reg_tasks = [TASK_MAP[name] for name in TASK_ORDER_REG if name in TASK_MAP]

        for task in reg_tasks:
            for split_method in ("butina_morgan", "random"):
                rec = diagnose_split(task, split_method)
                if rec:
                    all_records.append(rec)

        _print_diagnostics([r for r in all_records if r["task"] in TASK_ORDER_REG])

        cls_tasks = [TASK_MAP[name] for name in TASK_ORDER_CLS if name in TASK_MAP]

        if cls_tasks:
            print_section_header("SPLIT DIAGNOSTICS - CLASSIFICATION")

            for task in cls_tasks:
                for split_method in ("butina_morgan", "random"):
                    rec = diagnose_split(task, split_method)
                    if rec:
                        all_records.append(rec)

            _print_diagnostics([r for r in all_records if r["task"] in TASK_ORDER_CLS])

        if all_records:
            OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_records).to_csv(OUTPUT_CSV, index=False)
            print(f"\n  Saved: {OUTPUT_CSV}")

        print_completion_message()
