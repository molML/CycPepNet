import sys
import numpy as np
import pandas as pd
from cycpepnet.paths import DATA_ROOT, INPUT_RESULTS_ROOT, RESULTS_ROOT
from cycpepnet.utils.results import read_run_records
from cycpepnet.analyze._results_config import MAIN_ROW_MODELS
from cycpepnet.analyze._results_data import main_pool
from cycpepnet.benchmark import BENCHMARK_TASKS, load_task_table
from cycpepnet.analyze._common import (
    CANONICAL_BUTINA_SPLIT as CANONICAL_SPLIT,
    TASK_ORDER_CLS,
    TASK_ORDER_REG,
)
from cycpepnet.utils import (
    get_processed_dataset_dir,
    load_split,
    print_completion_message,
    print_section_header,
    tee_stdout_to_log,
)

LEARNABILITY_CSV         = RESULTS_ROOT / "learnability.csv"

_ACTIVE_TASKS = {task.name for task in BENCHMARK_TASKS}
_TASK_BY_NAME = {task.name: task for task in BENCHMARK_TASKS}

N_BOOTSTRAP  = 10_000
CI_ALPHA     = 0.05
Z_95         = 1.959964

# Metric used to pick the best model on validation
PRIMARY_METRIC = {"regression": "rmse", "classification": "balanced_acc"}

# Metrics for which a lower value is better
LOWER_IS_BETTER = {"rmse", "nrmse"}

# Metrics whose baseline is the measured training-mean predictor on the same test set
BASELINE_METRICS = {"rmse", "qf3", "balanced_acc"}

# Seed intervals share one bootstrap generator and are drawn in this order
TASK_TYPE_METRICS = {
    "regression":     ["nrmse", "qf3", "rmse", "r2"],
    "classification": ["auroc", "balanced_acc"],
}

# Role of each reported metric; a metric absent here is computed but not written
METRIC_ROLE = {
    "rmse":         "primary",
    "balanced_acc": "primary",
    "qf3":          "companion",
    "auroc":        "companion",
    "r2":           "descriptive",
}

# Map a task name to its property group
def _task_property(task_name: str) -> str:
    if task_name.startswith("macrocycledb"):
        return "binding"
    if task_name.startswith(("npmmpd", "cycpeptmpdb")):
        return "permeability"
    if task_name.startswith("peplife"):
        return "half-life"
    return "other"

# Load every non-archived per-seed result JSON into a single frame
def _load_seed_results() -> pd.DataFrame:

    records = read_run_records(INPUT_RESULTS_ROOT)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).rename(columns={"model_key": "model"})
    df = df[df["task"].isin(_ACTIVE_TASKS)].copy()
    df = df[df["model"].isin(set(MAIN_ROW_MODELS) | {"mean"})].copy()

    for source_col, target_col in (("qf3", "nrmse"), ("val_qf3", "val_nrmse")):
        if source_col in df.columns:
            values         = pd.to_numeric(df[source_col], errors="coerce")
            df[target_col] = np.sqrt(np.clip(1.0 - values, 0.0, None))

    return df

# Percentile bootstrap CI of the mean over seed-level values (training-stochasticity CI)
def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:

    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 2:
        return float("nan"), float("nan")

    means = rng.choice(vals, size=(N_BOOTSTRAP, len(vals)), replace=True).mean(axis=1)
    lo    = float(np.percentile(means, 100 * CI_ALPHA / 2))
    hi    = float(np.percentile(means, 100 * (1 - CI_ALPHA / 2)))

    return lo, hi

# Approximate test-set sampling 95% CI from the fixed test-set size
def test_sampling_ci(metric: str, mean: float, n_test: int | None) -> tuple[float, float]:

    if n_test is None or n_test < 2 or not np.isfinite(mean):
        return float("nan"), float("nan")

    if metric in ("rmse", "nrmse"):
        se = mean * np.sqrt(1.0 / (2.0 * n_test))
    elif metric in ("qf3", "r2"):
        se = (1.0 - mean) * np.sqrt(2.0 / n_test)
    else:
        return float("nan"), float("nan")

    return mean - Z_95 * se, mean + Z_95 * se


# Hanley-McNeil approximate test-set 95% CI for AUROC from per-class test counts
def _auroc_test_ci(auc: float, n_pos: int, n_neg: int) -> tuple[float, float]:

    if not np.isfinite(auc) or n_pos < 1 or n_neg < 1:
        return float("nan"), float("nan")

    q1  = auc / (2.0 - auc)
    q2  = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc ** 2)
        + (n_neg - 1) * (q2 - auc ** 2)
    ) / (n_pos * n_neg)
    se = np.sqrt(var) if var > 0 else 0.0

    return auc - Z_95 * se, auc + Z_95 * se


# Test-set 95% CI for balanced accuracy = mean(TPR, TNR), from per-class test counts
def _balanced_acc_test_ci(bal: float, tpr: float, tnr: float, n_pos: int, n_neg: int) -> tuple[float, float]:

    if not (np.isfinite(bal) and np.isfinite(tpr) and np.isfinite(tnr)) or n_pos < 1 or n_neg < 1:
        return float("nan"), float("nan")

    se = 0.5 * np.sqrt(tpr * (1.0 - tpr) / n_pos + tnr * (1.0 - tnr) / n_neg)

    return bal - Z_95 * se, bal + Z_95 * se


# Variance of the task target, used to convert an RMSE into a test-set reference
def _test_set_variance(task) -> float | None:

    try:
        df = load_task_table(task, data_root=DATA_ROOT)[0]
    except (FileNotFoundError, OSError, AssertionError, KeyError, ValueError):
        return None

    split_file = get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT) / "splits" / f"{task.name}_{CANONICAL_SPLIT}.json"
    if not split_file.exists():
        return None

    split, _ = load_split(split_file)
    test_idx = split["test"]

    y = df[task.target_column].to_numpy(dtype=float)
    if len(test_idx) < 2 or int(test_idx.max(initial=-1)) >= len(y):
        return None

    return float(np.var(y[test_idx], ddof=0))


# Positive/negative counts on the canonical test split (for classification CIs)
def _test_class_counts(task) -> tuple[int, int] | None:

    try:
        df = load_task_table(task, data_root=DATA_ROOT)[0]
    except (FileNotFoundError, OSError, AssertionError, KeyError, ValueError):
        return None

    split_file = get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT) / "splits" / f"{task.name}_{CANONICAL_SPLIT}.json"
    if not split_file.exists():
        return None

    split, _ = load_split(split_file)
    test_idx = split["test"]

    y = df[task.target_column].to_numpy(dtype=float)
    if len(test_idx) < 2 or int(test_idx.max(initial=-1)) >= len(y):
        return None

    y_test = y[test_idx]
    n_pos  = int(np.sum(y_test == 1))
    n_neg  = int(np.sum(y_test == 0))

    if n_pos < 1 or n_neg < 1:
        return None

    return n_pos, n_neg

# Build the per-task learnability table (best-achievable metric + both CIs)
def compute_learnability(
    df: pd.DataFrame,
    primary: dict[str, str] = PRIMARY_METRIC,
    roles: dict[str, str] = METRIC_ROLE,
) -> pd.DataFrame:

    rng  = np.random.default_rng(0)
    rows : list[dict] = []

    work = df[df["split_method"] == CANONICAL_SPLIT].copy()
    work = main_pool(work)

    order = [(t, "regression") for t in TASK_ORDER_REG] + [(t, "classification") for t in TASK_ORDER_CLS]

    for task, task_type in order:

        task_df = work[(work["task"] == task) & (work["model"] != "mean")]
        if task_df.empty:
            continue

        selection = primary[task_type]
        if selection not in task_df.columns:
            continue

        sel_metric = f"val_{selection}"
        if sel_metric not in task_df.columns or not task_df[sel_metric].notna().any():
            print(f"  WARNING: skipping {task}; no finite {sel_metric} values for validation selection.")
            continue

        model_sel = task_df.groupby("model")[sel_metric].mean()
        if model_sel.dropna().empty:
            continue
        best_model = model_sel.idxmin() if selection in LOWER_IS_BETTER else model_sel.idxmax()
        best_df    = task_df[task_df["model"] == best_model]
        n_test     = int(best_df["n_test"].iloc[0]) if "n_test" in best_df.columns else None

        base_df = work[(work["task"] == task) & (work["model"] == "mean")]

        var_test = (
            _test_set_variance(_TASK_BY_NAME[task])
            if task_type == "regression" and task in _TASK_BY_NAME else None
        )

        cls_counts = (
            _test_class_counts(_TASK_BY_NAME[task])
            if task_type == "classification" and task in _TASK_BY_NAME else None
        )

        for metric in TASK_TYPE_METRICS[task_type]:

            if metric == "r2":
                if var_test is None or var_test <= 0 or "rmse" not in best_df.columns:
                    continue
                rmse_vals = best_df["rmse"].to_numpy(dtype=float)
                rmse_vals = rmse_vals[np.isfinite(rmse_vals)]
                if len(rmse_vals) == 0:
                    continue
                vals = 1.0 - (rmse_vals ** 2) / var_test

            elif metric == "nrmse":
                if "qf3" not in best_df.columns:
                    continue
                qf3_vals = best_df["qf3"].to_numpy(dtype=float)
                qf3_vals = qf3_vals[np.isfinite(qf3_vals)]
                if len(qf3_vals) == 0:
                    continue
                vals = np.sqrt(np.clip(1.0 - qf3_vals, 0.0, None))

            elif metric not in best_df.columns:
                continue

            else:
                vals = best_df[metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue

            mean             = float(vals.mean())
            seed_lo, seed_hi = _bootstrap_ci(vals, rng)

            if task_type == "regression":
                test_lo, test_hi = test_sampling_ci(metric, mean, n_test)
            elif cls_counts is not None and metric == "auroc":
                test_lo, test_hi = _auroc_test_ci(mean, *cls_counts)
            elif cls_counts is not None and metric == "balanced_acc":
                tpr_m = float(best_df["tpr"].mean()) if "tpr" in best_df.columns else float("nan")
                tnr_m = float(best_df["tnr"].mean()) if "tnr" in best_df.columns else float("nan")
                test_lo, test_hi = _balanced_acc_test_ci(mean, tpr_m, tnr_m, *cls_counts)
            else:
                test_lo, test_hi = float("nan"), float("nan")

            if metric not in roles:
                continue

            metric_base = (
                float(base_df[metric].mean())
                if metric in BASELINE_METRICS and not base_df.empty and metric in base_df.columns
                else float("nan")
            )

            rows.append(dict(
                task        = task,
                property    = _task_property(task),
                task_type   = task_type,
                best_model  = best_model,
                selected_by = sel_metric,
                metric      = metric,
                metric_role = roles[metric],
                n_seeds    = int(len(vals)),
                n_test     = n_test if n_test is not None else "",
                mean       = round(mean, 4),
                seed_sd    = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
                seed_ci_lo = round(seed_lo, 4),
                seed_ci_hi = round(seed_hi, 4),
                test_ci_lo = round(test_lo, 4) if np.isfinite(test_lo) else "",
                test_ci_hi = round(test_hi, 4) if np.isfinite(test_hi) else "",
                test_var   = round(var_test, 4) if (metric == "r2" and var_test is not None) else "",
                baseline   = round(metric_base, 4) if np.isfinite(metric_base) else "",
            ))

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    role_rank = {"primary": 0, "historical": 0, "companion": 1, "descriptive": 2}
    table["_task"] = pd.Categorical(table["task"], categories=list(dict.fromkeys(table["task"])), ordered=True)
    table["_role"] = table["metric_role"].map(role_rank)
    return table.sort_values(["_task", "_role"], kind="stable").drop(columns=["_task", "_role"]).reset_index(drop=True)

# Main entry point for the uncertainty analysis script
def main() -> None:

    with tee_stdout_to_log("analyze_uncertainty.log"):

        df = _load_seed_results()
        if df.empty:
            print("No seed-level results found.")
            sys.exit(0)

        pd.set_option("display.width", 220)
        pd.set_option("display.max_rows", 200)

        print_section_header("TASK LEARNABILITY (best-achievable metric, with seed and test-sampling 95% CIs)")
        print("  RMSE/QF3²/R²/AUROC/balanced_acc are per-task learnability views, not cross-task model skill.")
        print("  best_model is selected by the VALIDATION metric (column selected_by) and reported on TEST.")
        print("  metric_role: primary (rmse, balanced_acc), companion (qf3, auroc), descriptive (r2).")
        print("  seed_ci = training stochasticity (split fixed across seeds). test_ci = small-test-set sampling:")
        print("  RMSE/QF3²/R² via Gaussian SE; AUROC via Hanley-McNeil; balanced_acc via per-class binomial.")
        print("  metric=rmse is in the units of each task's modelled response and is not comparable across tasks;")
        print("  baseline is the measured training-mean predictor on the same test set (rmse, qf3, balanced_acc).")
        print("  metric=r2 is the conventional test-set R² (1 - RMSE²/Var_test), recomputed from stored RMSE. It")
        print("  normalizes by TEST variance, which here exceeds train variance in every task, so it reads higher")
        print("  than QF3² implies; it is a learnability ceiling, NOT an absolute cross-task skill metric.")
        learnability = compute_learnability(df)
        print("\n" + learnability.to_string(index=False))
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        learnability.to_csv(LEARNABILITY_CSV, index=False)
        print(f"\n  Saved: {LEARNABILITY_CSV}  ({len(learnability)} rows)")
        print("  Learnability figure is rendered by analyze_results.py (main/fig_task_learnability.png).")

        print()
        print_completion_message()

if __name__ == "__main__":
    main()
