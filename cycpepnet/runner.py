import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import wandb
import torch
from cycpepnet.benchmark import BENCHMARK_TASKS, BenchmarkTask, DATA_ROOT, load_task_data
from cycpepnet.models.baseline import MeanPredictor
from cycpepnet.models.fps import RandomForestModel, XGBoostModel
from cycpepnet.models.graphs import GCNModel, MPNNModel
from cycpepnet.models.pretrained import ChemBERTa2Model, MolCLRModel, PretrainedGINModel, PeptideCLM2Model
from cycpepnet.models.strings import LSTMModel, TransformerModel, FragSMILESTokenizer
from cycpepnet.evaluation import RepType, evaluate_model
from cycpepnet.utils.provenance import atomic_json, atomic_predictions, file_sha256, resumable, run_provenance
from cycpepnet.utils import print_completion_message, print_section_header, print_subsection_header
from cycpepnet.utils import get_processed_dataset_dir
from cycpepnet.utils.splitting import assert_valid_split, load_split


# Configuration constants
RESULTS_ROOT              = Path("outputs/runs")
SEEDS                     = [0, 1, 2, 3, 4]
CANONICAL_SPLIT           = "butina_morgan"
SPLIT_METHODS             = ["butina_morgan", "random"]
NEURAL_KEYS: set[str]     = {"lstm", "lstm_fragsmiles", "transformer", "transformer_fragsmiles", "gcn", "mpnn", "chemberta2", "peptide_clm2", "pretrained_gin", "molclr"}

# Weights & Biases configuration
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "CycPepNet")


# Framework matrix
@dataclass
class BenchmarkEntry:
    key      : str
    model_cls: type
    rep      : RepType
    kwargs   : dict
    tasks    : list[str] | None = None

BENCHMARK_MATRIX: list[BenchmarkEntry] = [

    # Baselines
    BenchmarkEntry("mean",          MeanPredictor,     RepType.MORGAN_BINARY_PHYSCHEM, {}),
    BenchmarkEntry("rf_physchem",   RandomForestModel, RepType.PHYSCHEM_ONLY,          {"n_trials": 50}),
    BenchmarkEntry("xgb_physchem",  XGBoostModel,      RepType.PHYSCHEM_ONLY,          {"n_trials": 50}),

    # ECFP4b
    BenchmarkEntry("rf_binary_r2",     RandomForestModel, RepType.MORGAN_BINARY_R2_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("xgb_binary_r2",    XGBoostModel,      RepType.MORGAN_BINARY_R2_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("rf_binary_r2_fp",  RandomForestModel, RepType.MORGAN_BINARY_R2,          {"n_trials": 50}),
    BenchmarkEntry("xgb_binary_r2_fp", XGBoostModel,      RepType.MORGAN_BINARY_R2,          {"n_trials": 50}),

    # ECFP4c
    BenchmarkEntry("rf_count_r2",    RandomForestModel, RepType.MORGAN_COUNT_R2_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("xgb_count_r2",   XGBoostModel,      RepType.MORGAN_COUNT_R2_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("rf_count_r2_fp", RandomForestModel, RepType.MORGAN_COUNT_R2,          {"n_trials": 50}),
    BenchmarkEntry("xgb_count_r2_fp",XGBoostModel,      RepType.MORGAN_COUNT_R2,          {"n_trials": 50}),

    # ECFP6b
    BenchmarkEntry("rf_binary",    RandomForestModel, RepType.MORGAN_BINARY_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("xgb_binary",   XGBoostModel,      RepType.MORGAN_BINARY_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("rf_binary_fp", RandomForestModel, RepType.MORGAN_BINARY,          {"n_trials": 50}),
    BenchmarkEntry("xgb_binary_fp",XGBoostModel,      RepType.MORGAN_BINARY,          {"n_trials": 50}),

    # ECFP6c
    BenchmarkEntry("rf_count",     RandomForestModel, RepType.MORGAN_COUNT_PHYSCHEM,  {"n_trials": 50}),
    BenchmarkEntry("xgb_count",    XGBoostModel,      RepType.MORGAN_COUNT_PHYSCHEM,  {"n_trials": 50}),
    BenchmarkEntry("rf_count_fp",  RandomForestModel, RepType.MORGAN_COUNT,           {"n_trials": 50}),
    BenchmarkEntry("xgb_count_fp", XGBoostModel,      RepType.MORGAN_COUNT,           {"n_trials": 50}),

    # MAP4C
    BenchmarkEntry("rf_map4c",    RandomForestModel, RepType.MAP4C_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("xgb_map4c",   XGBoostModel,      RepType.MAP4C_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("rf_map4c_fp", RandomForestModel, RepType.MAP4C,          {"n_trials": 50}),
    BenchmarkEntry("xgb_map4c_fp",XGBoostModel,      RepType.MAP4C,          {"n_trials": 50}),

    # MAP6C
    BenchmarkEntry("rf_map6c",    RandomForestModel, RepType.MAP6C_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("xgb_map6c",   XGBoostModel,      RepType.MAP6C_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("rf_map6c_fp", RandomForestModel, RepType.MAP6C,          {"n_trials": 50}),
    BenchmarkEntry("xgb_map6c_fp",XGBoostModel,      RepType.MAP6C,          {"n_trials": 50}),

    # MXFP
    BenchmarkEntry("rf_mxfp",    RandomForestModel, RepType.MXFP_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("xgb_mxfp",   XGBoostModel,      RepType.MXFP_PHYSCHEM, {"n_trials": 50}),
    BenchmarkEntry("rf_mxfp_fp", RandomForestModel, RepType.MXFP,          {"n_trials": 50}),
    BenchmarkEntry("xgb_mxfp_fp",XGBoostModel,      RepType.MXFP,          {"n_trials": 50}),

    # HDF
    BenchmarkEntry("rf_hdf",     RandomForestModel, RepType.HDF_PHYSCHEM,  {"n_trials": 50}),
    BenchmarkEntry("xgb_hdf",    XGBoostModel,      RepType.HDF_PHYSCHEM,  {"n_trials": 50}),
    BenchmarkEntry("rf_hdf_fp",  RandomForestModel, RepType.HDF,           {"n_trials": 50}),
    BenchmarkEntry("xgb_hdf_fp", XGBoostModel,      RepType.HDF,           {"n_trials": 50}),

    # Graph neural networks
    BenchmarkEntry("gcn",  GCNModel,  RepType.GRAPHS, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),
    BenchmarkEntry("mpnn", MPNNModel, RepType.GRAPHS, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),

    # Sequence neural models
    BenchmarkEntry("lstm",        LSTMModel,        RepType.SMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),
    BenchmarkEntry("transformer", TransformerModel, RepType.SMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),

    # Pretrained chemical language models
    BenchmarkEntry("chemberta2",       ChemBERTa2Model,     RepType.SMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),
    BenchmarkEntry("peptide_clm2",     PeptideCLM2Model,    RepType.SMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),

    # Pretrained graph neural networks
    BenchmarkEntry("pretrained_gin",   PretrainedGINModel,  RepType.SMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),
    BenchmarkEntry("molclr",           MolCLRModel,         RepType.SMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400}),

    # fragSMILES sequence variants
    BenchmarkEntry("lstm_fragsmiles",        LSTMModel,        RepType.FRAGSMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400, "tokenizer_cls": FragSMILESTokenizer}),
    BenchmarkEntry("transformer_fragsmiles", TransformerModel, RepType.FRAGSMILES, {"n_trials": 50, "device": "cuda", "hpo_timeout": 14400, "tokenizer_cls": FragSMILESTokenizer}),
]

# Apply a shared timeout
def _with_hpo_timeout(entries: list[BenchmarkEntry], timeout_s: int) -> list[BenchmarkEntry]:

    overridden = []

    for entry in entries:

        kwargs = dict(entry.kwargs)

        if kwargs.get("n_trials", 0) > 0:
            kwargs["hpo_timeout"] = timeout_s

        overridden.append(BenchmarkEntry(entry.key, entry.model_cls, entry.rep, kwargs, tasks=entry.tasks))

    return overridden

# Preferred neural device for the current runtime
def _runtime_neural_device() -> str:

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

# Adapt neural entry devices to the current runtime without changing framework logic
def _normalize_entry_devices(entries: list[BenchmarkEntry]) -> tuple[list[BenchmarkEntry], str | None]:

    runtime_device = _runtime_neural_device()
    normalized     = []
    downgraded     = False

    for entry in entries:

        kwargs = dict(entry.kwargs)

        if entry.key in NEURAL_KEYS and kwargs.get("device") == "cuda" and runtime_device != "cuda":
            kwargs["device"] = runtime_device
            downgraded = True

        normalized.append(BenchmarkEntry(entry.key, entry.model_cls, entry.rep, kwargs, tasks=entry.tasks))

    return normalized, runtime_device if downgraded else None

# Split file path for a given task and split method (e.g. "butina_morgan")
def split_path(task: BenchmarkTask, data_root: Path, method: str = CANONICAL_SPLIT) -> Path:
    return get_processed_dataset_dir(task.dataset, data_root=data_root) / "splits" / f"{task.name}_{method}.json"

# Result file path for a given (task, model entry, seed, split_method) tuple
def result_path(results_root: Path, task: BenchmarkTask, entry: BenchmarkEntry, seed: int, split_method: str = CANONICAL_SPLIT) -> Path:

    suffix    = "" if split_method == CANONICAL_SPLIT else f"__{split_method}"
    task_type = "classification" if task.task_type == "classification" else "regression"
    subfolder = "supplementary" if task.supplementary else "main"

    return results_root / task_type / subfolder / task.name / f"{entry.key}__seed{seed:02d}{suffix}.json"

# Helper to save a single result JSON
def _save_result(path: Path, payload: dict) -> None:
    atomic_json(path, payload)

# Save per-molecule predictions (y_true + y_pred/y_score) next to the result JSON
def _save_predictions(json_path: Path, preds: dict | None) -> None:
    if preds:
        atomic_predictions(json_path.with_suffix(".npz"), preds)

# Run the full framework matrix and save results to disk
def run_benchmark(

    data_root:    Path                        = DATA_ROOT,
    results_root: Path                        = RESULTS_ROOT,
    seeds:        list[int]          | None   = None,
    tasks:        list[BenchmarkTask] | None  = None,
    entries:      list[BenchmarkEntry] | None = None,
    resume:       bool                        = True,
    verbose:      bool                        = True,
    split_method: str                         = CANONICAL_SPLIT,
    wandb_mode:   str                | None   = "disabled",

) -> None:

    if seeds   is None: seeds   = SEEDS
    if tasks   is None: tasks   = BENCHMARK_TASKS
    if entries is None: entries = BENCHMARK_MATRIX

    entries, fallback_device = _normalize_entry_devices(entries)

    print_section_header("CycPepNet FRAMEWORK RUNNER")

    n_total = sum(
        1
        for task in tasks
        for entry in entries
        if entry.tasks is None or task.name in entry.tasks
    ) * len(seeds)

    print(f"\n  Tasks:      {len(tasks)}")
    print(f"  Model rep:    {len(entries)}")
    print(f"  Seeds:        {seeds}")
    print(f"  Split method: {split_method}")
    print(f"  Total runs:   {n_total}")
    print(f"  Results root: {results_root}\n")

    if fallback_device is not None:
        print(f"  CUDA not available; neural models will run on {fallback_device}.\n")

    completed = skipped = errors = 0

    for task in tasks:

        print_subsection_header(f"Task: {task.name}")

        sp = split_path(task, data_root, method=split_method)

        if not sp.exists():

            print(f"  Split not found: {sp}")
            print("  → Run generate_all_splits() / generate_random_splits() first. Skipping task.")

            errors += sum(1 for entry in entries if entry.tasks is None or task.name in entry.tasks) * len(seeds)

            continue

        eligible_entries = [
            entry for entry in entries
            if entry.tasks is None or task.name in entry.tasks
        ]

        try:
            split, _  = load_split(sp)
            task_data = load_task_data(task, data_root)
            assert_valid_split(split, len(task_data[0]), sp)
        except Exception as exc:
            print(f"  Invalid or incompatible split: {exc}")
            print("  Skipping task to avoid training on stale or malformed split indices.")
            errors += len(eligible_entries) * len(seeds)
            continue

        for entry in eligible_entries:

            for seed in seeds:

                rp = result_path(results_root, task, entry, seed, split_method)

                serializable_kwargs = {k: (v.__name__ if isinstance(v, type) else v) for k, v in entry.kwargs.items()}

                provenance = run_provenance(
                    task,
                    task_data[0],
                    get_processed_dataset_dir(task.dataset, data_root=data_root) / f"{task.dataset}.csv",
                    sp,
                    serializable_kwargs,
                )

                expected = dict(
                    task         = task.name,
                    task_type    = task.task_type,
                    model_key    = entry.key,
                    rep          = entry.rep.value,
                    seed         = seed,
                    split_method = split_method,
                    **provenance,
                )

                if resume and rp.exists():

                    if not resumable(rp, expected):
                        raise RuntimeError(
                            f"Cannot safely resume {rp}: incomplete or incompatible record. "
                            "Use a fresh results root or --no-resume."
                        )

                    if verbose:
                        print(f"  [skip]  {entry.key}  seed={seed}")

                    skipped += 1
                    continue

                wandb_run = None

                try:

                    serializable_kwargs = {
                        k: (v.__name__ if isinstance(v, type) else v)
                        for k, v in entry.kwargs.items()
                    }
                    model          = entry.model_cls(seed=seed, task_type=task.task_type, **entry.kwargs)
                    model_metadata = {
                        k: v for k, v in model.benchmark_metadata().items()
                        if v is not None
                    }

                    wandb_run = wandb.init(
                        entity  = WANDB_ENTITY,
                        project = WANDB_PROJECT,
                        mode    = wandb_mode,
                        name    = f"{entry.key}__seed{seed:02d}",
                        group   = f"{task.name}__{entry.key}__{split_method}",
                        config  = {
                            "task"          : task.name,
                            "task_type"     : task.task_type,
                            "model"         : entry.key,
                            "rep"           : entry.rep.value,
                            "seed"          : seed,
                            "split_method"  : split_method,
                            "log_transform" : task.log_transform,
                            **serializable_kwargs,
                            **model_metadata,
                        },
                        reinit  = True,
                    )

                    wandb_run.define_metric("hpo/trial")
                    wandb_run.define_metric("hpo/*",   step_metric="hpo/trial")
                    wandb_run.define_metric("train_epoch")
                    wandb_run.define_metric("train/*", step_metric="train_epoch")
                    wandb_run.define_metric("val/*",   step_metric="train_epoch")

                    log_fn = wandb_run.log

                    if verbose:
                        print(f"\n  [run]   {entry.key}  seed={seed}")

                    metrics, preds = evaluate_model(
                        task         = task,
                        model        = model,
                        rep_type     = entry.rep,
                        split_file   = sp,
                        data_root    = data_root,
                        verbose      = verbose,
                        log_fn       = log_fn,
                        return_preds = True,
                        task_data    = task_data,
                    )

                    if task.task_type == "classification":
                        wandb_run.summary["final/test_auroc"]         = metrics["auroc"]
                        wandb_run.summary["final/test_tpr"]           = metrics["tpr"]
                        wandb_run.summary["final/test_tnr"]           = metrics["tnr"]
                        wandb_run.summary["final/test_precision"]     = metrics["precision"]
                        wandb_run.summary["final/test_balanced_acc"]  = metrics["balanced_acc"]
                        y_true  = preds["y_true"]
                        y_score = preds["y_score"]
                        y_probas = np.column_stack([1 - y_score, y_score])
                        wandb_run.log({
                            "plots/roc_curve": wandb.plot.roc_curve(
                                y_true, y_probas, labels=list(task.class_labels),
                            ),
                            "plots/confusion_matrix": wandb.plot.confusion_matrix(
                                y_true=y_true,
                                preds=(y_score >= 0.5).astype(int),
                                class_names=list(task.class_labels),
                            ),
                        })

                    else:
                        wandb_run.summary["final/test_rmse"] = metrics["rmse"]
                        wandb_run.summary["final/test_qf3"]  = metrics["qf3"]


                    payload = {
                        "task"          : task.name,
                        "task_type"     : task.task_type,
                        "model_key"     : entry.key,
                        "rep"           : entry.rep.value,
                        "seed"          : seed,
                        "split_method"  : split_method,
                        "log_transform" : task.log_transform,
                        "timestamp"     : datetime.now().isoformat(timespec="seconds"),
                        **model_metadata,
                        **provenance,
                        "hpo": getattr(model, "hpo_metadata", None),
                        **metrics,
                    }
                    _save_predictions(rp, preds)
                    payload["predictions_sha256"] = file_sha256(rp.with_suffix(".npz"))
                    _save_result(rp, payload)

                    completed += 1

                except Exception:
                    print(f"  [ERROR] {entry.key} seed={seed}")
                    traceback.print_exc()
                    errors += 1

                finally:
                    if wandb_run is not None:
                        wandb_run.finish()

    print()
    print_section_header("RUN SUMMARY")
    print(f"\n  Completed: {completed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}\n")

    aggregate_results(results_root, verbose=True)

    if errors:
        raise RuntimeError(f"Framework run incomplete: {errors} runs failed or lacked compatible input data.")

    print_completion_message()

# Aggregate individual result JSON files into a summary CSV and print to console
def aggregate_results(

    results_root:     Path = RESULTS_ROOT,
    verbose:          bool = True,
    include_inactive: bool = False,

) -> pd.DataFrame:

    result_files = [
        f
        for f in sorted(results_root.glob("**/*.json"))
        if not any(part.startswith("_archive") for part in f.relative_to(results_root).parts)
    ]

    if not result_files:
        print("  No result files found.")
        return pd.DataFrame()

    records = [json.loads(f.read_text()) for f in result_files]
    df_all  = pd.DataFrame(records)

    if not include_inactive and "model_key" in df_all.columns:
        active_model_keys = {entry.key for entry in BENCHMARK_MATRIX}
        inactive_mask     = ~df_all["model_key"].isin(active_model_keys)

        if inactive_mask.any() and verbose:
            inactive = sorted(df_all.loc[inactive_mask, "model_key"].dropna().unique())
            print(f"  Ignoring inactive result models during aggregation: {inactive}")

        df_all = df_all.loc[~inactive_mask].copy()

        if df_all.empty:
            print("  No active result files found.")
            return pd.DataFrame()

    if not include_inactive and "task" in df_all.columns:

        active_task_names = {task.name for task in BENCHMARK_TASKS}
        inactive_mask     = ~df_all["task"].isin(active_task_names)

        if inactive_mask.any() and verbose:
            inactive = sorted(df_all.loc[inactive_mask, "task"].dropna().unique())
            print(f"  Ignoring inactive result tasks during aggregation: {inactive}")

        df_all = df_all.loc[~inactive_mask].copy()

        if df_all.empty:
            print("  No active result files found.")
            return pd.DataFrame()

    rows = []

    for (task_name, model_key, rep, split_method), grp in df_all.groupby(["task", "model_key", "rep", "split_method"]):

        n      = len(grp)
        is_cls = grp["task_type"].iloc[0] == "classification" if "task_type" in grp.columns else ("auroc" in grp.columns and grp["auroc"].notna().any())

        row: dict = {
            "task"        : task_name,
            "model"       : model_key,
            "rep"         : rep,
            "split_method": split_method,
            "task_type"   : "classification" if is_cls else "regression",
            "n_seeds"     : n,
            "n_train"     : int(grp["n_train"].iloc[0]),
            "n_test"      : int(grp["n_test"].iloc[0]),
        }

        if is_cls:

            for col in ["auroc", "tpr", "tnr", "precision", "balanced_acc"]:
                if col in grp:
                    row[f"{col}_mean"] = grp[col].mean()
                    row[f"{col}_std"]  = grp[col].std(ddof=1) if n > 1 else 0.0

            for col in ["val_auroc", "val_tpr", "val_tnr", "val_precision", "val_balanced_acc"]:
                if col in grp:
                    row[f"{col}_mean"] = grp[col].mean()
                    row[f"{col}_std"]  = grp[col].std(ddof=1) if n > 1 else 0.0

        else:
            for col in ["val_rmse", "val_qf3", "rmse", "qf3"]:
                if col in grp:
                    row[f"{col}_mean"] = grp[col].mean()
                    row[f"{col}_std"]  = grp[col].std(ddof=1) if n > 1 else 0.0

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["split_method", "task", "model"], na_position="last")

    out = results_root / "summary.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    if verbose:
        _print_summary_table(summary)
        print(f"\n  Summary saved → {out}\n")

    _save_best_models(summary, results_root)
    _save_validation_selected_models(summary, results_root)

    return summary

# Extract best model per task (by test metric: auroc for classification, qf3 for regression)
def _save_best_models(summary: pd.DataFrame, results_root: Path) -> None:

    best_rows = []

    for (split_method, task_name), task_df in summary.groupby(["split_method", "task"]):

        is_cls = task_df["task_type"].iloc[0] == "classification"
        metric = "auroc_mean" if is_cls else "qf3_mean"
        valid  = task_df.dropna(subset=[metric])

        if valid.empty:
            print(f"  Skipping best-model selection for {task_name} [{split_method}]: no valid {metric} values.")
            continue

        best_idx = valid[metric].idxmax()
        best_rows.append(valid.loc[best_idx])

    if not best_rows:
        print("  Best models summary skipped: no valid rows found.")
        return

    best_models = pd.DataFrame(best_rows).sort_values(["split_method", "task"])

    out = results_root / "best_models_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    best_models.to_csv(out, index=False)

    print(f"  Best models summary saved → {out}")


# Best configuration per task chosen on validation, the counterpart of the test-ranked diagnostic above
def _save_validation_selected_models(summary: pd.DataFrame, results_root: Path) -> None:

    rows = []

    for _, group in summary.groupby(["split_method", "task"]):

        classification = group["task_type"].iloc[0] == "classification"
        metric         = "val_balanced_acc_mean" if classification else "val_rmse_mean"

        if metric not in group:
            continue

        valid = group.dropna(subset=[metric])

        if valid.empty:
            continue

        index              = valid[metric].idxmax() if classification else valid[metric].idxmin()
        row                = valid.loc[index].copy()
        row["selected_by"] = metric
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(results_root / "validation_selected_models.csv", index=False)

# Print summary table to console with aligned columns
def _print_summary_table(summary: pd.DataFrame) -> None:

    print()
    print_section_header("RESULTS SUMMARY")

    col_w = {"model": 18, "rep": 16, "metric": 20}

    for (split_method, task_name), task_df in summary.groupby(["split_method", "task"]):

        is_cls = task_df["task_type"].iloc[0] == "classification"

        print(f"\n  Task: {task_name}  [{split_method}]  [{'classification' if is_cls else 'regression'}]")

        if is_cls:

            hdr = (
                f"  {'Model':<{col_w['model']}} "
                f"{'Rep':<{col_w['rep']}} "
                f"{'Val AUROC (μ±σ)':<{col_w['metric']}} "
                f"{'Test AUROC (μ±σ)':<{col_w['metric']}} "
                f"{'Test BalAcc (μ±σ)':<{col_w['metric']}} "
                f"{'Test TPR (μ±σ)':<{col_w['metric']}} "
                f"{'Test TNR (μ±σ)':<{col_w['metric']}} "
                f"{'Test Prec (μ±σ)':<{col_w['metric']}} "
                f"{'N'}"
            )

            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

            for _, row in task_df.iterrows():
                def _fmt(m: str) -> str:
                    mn = row.get(f"{m}_mean", float("nan"))
                    sd = row.get(f"{m}_std",  0.0)
                    return f"{mn:.4f} ± {sd:.4f}"
                print(
                    f"  {row['model']:<{col_w['model']}} "
                    f"{row['rep']:<{col_w['rep']}} "
                    f"{_fmt('val_auroc'):<{col_w['metric']}} "
                    f"{_fmt('auroc'):<{col_w['metric']}} "
                    f"{_fmt('balanced_acc'):<{col_w['metric']}} "
                    f"{_fmt('tpr'):<{col_w['metric']}} "
                    f"{_fmt('tnr'):<{col_w['metric']}} "
                    f"{_fmt('precision'):<{col_w['metric']}} "
                    f"{row['n_seeds']:.0f}"
                )
        else:
            hdr = (
                f"  {'Model':<{col_w['model']}} "
                f"{'Rep':<{col_w['rep']}} "
                f"{'Val RMSE (μ±σ)':<{col_w['metric']}} "
                f"{'Val QF3² (μ±σ)':<{col_w['metric']}} "
                f"{'Test RMSE (μ±σ)':<{col_w['metric']}} "
                f"{'Test QF3² (μ±σ)':<{col_w['metric']}} "
                f"{'N'}"
            )
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for _, row in task_df.iterrows():
                def _fmt(m: str) -> str:
                    mn = row.get(f"{m}_mean", float("nan"))
                    sd = row.get(f"{m}_std",  0.0)
                    return f"{mn:.4f} ± {sd:.4f}"
                print(
                    f"  {row['model']:<{col_w['model']}} "
                    f"{row['rep']:<{col_w['rep']}} "
                    f"{_fmt('val_rmse'):<{col_w['metric']}} "
                    f"{_fmt('val_qf3'):<{col_w['metric']}} "
                    f"{_fmt('rmse'):<{col_w['metric']}} "
                    f"{_fmt('qf3'):<{col_w['metric']}} "
                    f"{row['n_seeds']:.0f}"
                )


_TASK_MAP  = {t.name: t for t in BENCHMARK_TASKS}
_ENTRY_MAP = {e.key: e for e in BENCHMARK_MATRIX}

# Build argument parser for CLI
def _build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(description="CycPepNet framework runner.")

    p.add_argument(
        "--tasks", nargs="+", choices=list(_TASK_MAP), default=None,
        help="Subset of tasks to run (default: all).",
    )

    p.add_argument(
        "--models", nargs="+", choices=list(_ENTRY_MAP), default=None,
        help="Subset of model keys to run (default: all).",
    )

    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Random seeds (default: 0 1 2 3 4).",
    )

    p.add_argument(
        "--no-resume", action="store_true",
        help="Re-run even if result file already exists.",
    )

    p.add_argument(
        "--aggregate-only", action="store_true",
        help="Only aggregate existing results; skip training.",
    )

    p.add_argument(
        "--include-inactive-results", action="store_true",
        help="Include result files from models that are no longer in the active framework matrix.",
    )

    p.add_argument(
        "--hpo-timeout", type=int, default=None,
        help="Max seconds for Optuna tuning per model fit (applies when n_trials > 0).",
    )

    p.add_argument(
        "--model-type", choices=["all", "neural", "non-neural"], default="all",
        help="Run only neural (GPU) models, only non-neural (CPU) models, or all (default).",
    )

    p.add_argument("--data-root",    default="data/processed", type=Path)

    p.add_argument("--results-root", default="outputs/runs",  type=Path)

    p.add_argument(
        "--split-method", choices=SPLIT_METHODS, default=None,
        help="Split strategy to use. If omitted, runs all split methods sequentially.",
    )

    p.add_argument(
        "--wandb", action="store_true",
        help="Opt in to online Weights & Biases logging; disabled by default.",
    )

    return p

if __name__ == "__main__":

    args = _build_parser().parse_args()

    if args.aggregate_only:
        aggregate_results(
            args.results_root,
            verbose=True,
            include_inactive=args.include_inactive_results,
        )
        sys.exit(0)

    selected_tasks    = [_TASK_MAP[t]  for t in args.tasks]  if args.tasks  else None
    selected_entries  = [_ENTRY_MAP[m] for m in args.models] if args.models else None

    base_pool = selected_entries if selected_entries is not None else BENCHMARK_MATRIX

    if args.model_type == "neural":
        base_pool = [e for e in base_pool if e.key in NEURAL_KEYS]

    elif args.model_type == "non-neural":
        base_pool = [e for e in base_pool if e.key not in NEURAL_KEYS]

    selected_entries = base_pool

    entries_to_run    = selected_entries

    if args.hpo_timeout is not None:
        base_entries   = selected_entries if selected_entries is not None else BENCHMARK_MATRIX
        entries_to_run = _with_hpo_timeout(base_entries, args.hpo_timeout)

    methods    = [args.split_method] if args.split_method else SPLIT_METHODS
    wandb_mode = None if args.wandb else "disabled"

    for method in methods:
        run_benchmark(
            data_root    = args.data_root,
            results_root = args.results_root,
            seeds        = args.seeds,
            tasks        = selected_tasks,
            entries      = entries_to_run,
            resume       = not args.no_resume,
            verbose      = True,
            split_method = method,
            wandb_mode   = wandb_mode,
        )
