from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from cycpepnet.analyze._common import (
    CANONICAL_BUTINA_SPLIT as CANONICAL_SPLIT,
    TASK_ORDER_CLS,
    TASK_ORDER_REG,
)
from cycpepnet.analyze._results_config import MAIN_ROW_MODELS
from cycpepnet.analyze._results_data import (
    main_pool,
    _load_seed_level_results,
    representation_of,
)
from cycpepnet.analyze._stats_core import _friedman_nemenyi
from cycpepnet.paths import RESULTS_ROOT
from cycpepnet.utils import print_completion_message, print_section_header, tee_stdout_to_log

OUTPUT_RANKING_CSV = RESULTS_ROOT / "method_ranking.csv"

# Primary metric per task type
PRIMARY_METRIC_BY_TASKTYPE: dict[str, str] = {
    "regression":     "rmse",
    "classification": "balanced_acc",
}

_LOWER_IS_BETTER_METRICS = {"rmse", "nrmse"}

METHOD_RANKING_COLUMNS = [
    "analysis",
    "task_type",
    "metric",
    "n_tasks",
    "n_methods",
    "friedman_stat",
    "friedman_p",
    "critical_difference",
    "n_pairwise",
    "n_pairwise_significant",
    "method",
    "mean_rank",
]

# Model family of each runner key
MODEL_FAMILY: dict[str, str] = {}
for _m in MAIN_ROW_MODELS:
    if _m.startswith("rf_"):
        MODEL_FAMILY[_m] = "RF"
    elif _m.startswith("xgb_"):
        MODEL_FAMILY[_m] = "XGBoost"
    elif _m in ("lstm", "lstm_fragsmiles"):
        MODEL_FAMILY[_m] = "LSTM"
    elif _m in ("transformer", "transformer_fragsmiles"):
        MODEL_FAMILY[_m] = "Transformer"
    elif _m == "gcn":
        MODEL_FAMILY[_m] = "GCN"
    elif _m == "mpnn":
        MODEL_FAMILY[_m] = "MPNN"
    elif _m == "chemberta2":
        MODEL_FAMILY[_m] = "ChemBERTa"
    elif _m == "peptide_clm2":
        MODEL_FAMILY[_m] = "PeptideCLM"
    elif _m == "pretrained_gin":
        MODEL_FAMILY[_m] = "GIN"
    elif _m == "molclr":
        MODEL_FAMILY[_m] = "MolCLR"

# Fingerprint-model suffixes mapped to fingerprint-family names
FP_FAMILY: dict[str, str] = {
    "binary_r2_fp": "ECFP4b", "binary_r2": "ECFP4b",
    "count_r2_fp":  "ECFP4c", "count_r2":  "ECFP4c",
    "binary_fp":    "ECFP6b", "binary":    "ECFP6b",
    "count_fp":     "ECFP6c", "count":     "ECFP6c",
    "map4c_fp":     "MAP4C",  "map4c":     "MAP4C",
    "map6c_fp":     "MAP6C",  "map6c":     "MAP6C",
    "mxfp_fp":      "MXFP",   "mxfp":      "MXFP",
    "hdf_fp":       "HDF",    "hdf":       "HDF",
}

# Map a fingerprint-model key to its fingerprint family
def _fp_family_of(model: str) -> str | None:
    for suffix, family in FP_FAMILY.items():
        if model.endswith(suffix):
            return family
    return None

# Add NRMSE columns derived from QF3²
def _add_nrmse(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    for source_col, target_col in (("qf3", "nrmse"), ("val_qf3", "val_nrmse")):

        if source_col not in out.columns:
            continue

        values          = pd.to_numeric(out[source_col], errors="coerce")
        out[target_col] = np.sqrt(np.clip(1.0 - values, 0.0, None))

    return out

# Restrict the seed-level frame to the canonical comparison pool for one task type
def _filter_main(df: pd.DataFrame, task_type: str, tasks: list[str]) -> pd.DataFrame:
    out = df[df["task_type"] == task_type].copy()
    out = out[out["model"] != "mean"]
    out = out[out["split_method"] == CANONICAL_SPLIT]
    out = out[out["task"].isin(tasks)]
    out = main_pool(out)
    return out

# Rank-based "which model/representation wins overall" via _friedman_nemenyi
#
# One configuration per method and task is selected on the validation metric and
# then ranked on the held-out test set, as described in the Methods.
def analyze_method_ranking(
    work        : pd.DataFrame,
    metric      : str,
    analysis    : str,
    method_col  : str,
    task_type   : str,
    val_metric  : str,
) -> list[dict]:

    work = work.dropna(subset=[method_col, metric, val_metric])
    if work.empty:
        return []

    configs = (
        work.groupby(["task", method_col, "model", "rep"], dropna=False)
        .agg(validation_mean=(val_metric, "mean"), test_mean=(metric, "mean"))
        .reset_index()
        .dropna(subset=["validation_mean", "test_mean"])
    )
    if configs.empty:
        return []

    lower_is_better   = metric in _LOWER_IS_BETTER_METRICS
    validation_groups = configs.groupby(["task", method_col])["validation_mean"]
    selected          = validation_groups.idxmin() if lower_is_better else validation_groups.idxmax()

    cell   = configs.loc[selected]
    matrix = cell.pivot(index="task", columns=method_col, values="test_mean")

    res = _friedman_nemenyi(matrix, higher_is_better=not lower_is_better)
    if res is None:
        return []

    n_pairwise     = len(res["pairwise"])
    n_pairwise_sig = sum(1 for pw in res["pairwise"] if pw["significant"] == "significant")

    common = {
        "analysis":               analysis,
        "task_type":              task_type,
        "metric":                 metric,
        "n_tasks":                res["n_blocks"],
        "n_methods":              res["n_methods"],
        "friedman_stat":          round(res["friedman_stat"], 4),
        "friedman_p":             res["friedman_p"],
        "critical_difference":    round(res["critical_difference"], 4),
        "n_pairwise":             n_pairwise,
        "n_pairwise_significant": n_pairwise_sig,
    }

    return [
        {**common, "method": method, "mean_rank": round(mean_rank, 4)}
        for method, mean_rank in sorted(res["mean_ranks"].items(), key=lambda kv: kv[1])
    ]

# Build the model-family, fingerprint-family and representation rankings for a task type
def _method_ranking_rows(df: pd.DataFrame, metric: str, tasks: list[str], task_type: str) -> list[dict]:

    rows: list[dict] = []

    val_metric = f"val_{metric}"
    if val_metric not in df.columns:
        return rows

    fam           = _filter_main(df, task_type, tasks)
    fam           = fam[~fam["rep"].str.contains(r"\+physchem", na=False)].copy()
    fam["family"] = fam["model"].map(MODEL_FAMILY)

    rows += analyze_method_ranking(fam, metric, "model_families_ranking", "family", task_type, val_metric)

    fp = _filter_main(df, task_type, tasks)
    fp = fp[fp["model"].str.startswith(("rf_", "xgb_"))]
    fp = fp[~fp["rep"].str.contains(r"\+physchem", na=False)].copy()

    fp["fp_family"] = fp["model"].map(_fp_family_of)

    rows += analyze_method_ranking(fp, metric, "fp_families_ranking", "fp_family", task_type, val_metric)

    rep = _filter_main(df, task_type, tasks).copy()
    rep["representation"] = [representation_of(m, r) for m, r in zip(rep["model"], rep["rep"])]
    rep = rep.dropna(subset=["representation"])

    rows += analyze_method_ranking(rep, metric, "representation_ranking", "representation", task_type, val_metric)

    return rows

# Print one Friedman block with its mean ranks and Nemenyi verdict
def _print_ranking_block(analysis_name: str, block: list[dict]) -> None:

    head = block[0]
    print(f"\n  {analysis_name}  (Friedman p={head['friedman_p']:.4g}, "
          f"CD={head['critical_difference']}, n_tasks={head['n_tasks']})")

    for r in block:
        print(f"    {r['method']:14s}  mean rank = {r['mean_rank']:.3f}")

    n_pw  = head["n_pairwise"]
    n_sig = head["n_pairwise_significant"]

    if n_sig == 0:
        print(f"    [post-hoc] 0/{n_pw} pairwise gaps exceed the critical difference "
              "→ ranking is DESCRIPTIVE only (no method significantly wins).")
    else:
        print(f"    [post-hoc] {n_sig}/{n_pw} pairwise gaps exceed the critical difference.")

# Rank model families, fingerprint families and representations across the six main tasks
def main() -> None:

    df = _add_nrmse(_load_seed_level_results())

    if df.empty:
        print("No seed-level results found.")
        sys.exit(0)

    ranking_rows: list[dict] = []

    for task_type, tasks in (("regression", list(TASK_ORDER_REG)),
                             ("classification", list(TASK_ORDER_CLS))):

        metric = PRIMARY_METRIC_BY_TASKTYPE[task_type]
        print_section_header(f"METHOD RANKING (Friedman + Nemenyi, {metric}) - {task_type}")

        type_ranking = _method_ranking_rows(df, metric, tasks, task_type)

        for analysis_name in sorted({r["analysis"] for r in type_ranking}):
            block = [r for r in type_ranking if r["analysis"] == analysis_name]
            if block:
                _print_ranking_block(analysis_name, block)

        ranking_rows.extend(type_ranking)

    if ranking_rows:
        OUTPUT_RANKING_CSV.parent.mkdir(parents=True, exist_ok=True)
        ranking_df = pd.DataFrame(ranking_rows).reindex(columns=METHOD_RANKING_COLUMNS)
        ranking_df.to_csv(OUTPUT_RANKING_CSV, index=False)
        print(f"\n  Saved: {OUTPUT_RANKING_CSV}  ({len(ranking_df)} rows)")

    print_completion_message()


if __name__ == "__main__":
    with tee_stdout_to_log("analyze_ranking.log"):
        main()
