from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT, TASK_ORDER_CLS, TASK_ORDER_REG
from cycpepnet.analyze._stats_core import _benjamini_hochberg, _holm_correction, _significance
from cycpepnet.analyze._result_helpers import METRIC_HIGHER_IS_BETTER, load_seed_results, task_property
from cycpepnet.analyze._figure_ablation import count_binary_pairs, ecfp_radius_pairs, map_radius_pairs, physchem_pairs
from cycpepnet.analyze.analyze_uncertainty import METRIC_ROLE, _load_seed_results, compute_learnability
from cycpepnet.analyze.statistical_tests import (
    NON_CONFIRMATORY,
    _delta_row,
    _family_pair_models,
    _group_contrast,
    _paired_deltas,
    _split_effect,
    _split_effect_relative,
    _write_latex_table,
)
from cycpepnet.utils import print_completion_message, print_section_header

# Reproduces the statistics released with RMSE/SD_train as the primary regression metric

DEFAULT_OUTPUT = Path("outputs") / "historical_nrmse"

HISTORICAL_PRIMARY = {"regression": "nrmse", "classification": "balanced_acc"}
HISTORICAL_REFERENCE = {"nrmse": 1.0, "balanced_acc": 0.5}

HISTORICAL_COLUMNS = [
    "analysis", "task_type", "metric", "comparison", "direction", "inference",
    "n_obs", "n_tasks", "n_clusters",
    "estimate", "std_error", "ci_lo", "ci_hi", "z_value", "p_value",
    "lmm_status", "lmm_p_value", "var_task", "var_cluster", "var_resid",
    "tc_estimate", "tc_std_error", "tc_ci_lo", "tc_ci_hi", "tc_p_value",
    "median_delta", "median_ci_lo", "median_ci_hi",
    "wilcoxon_task_p", "rank_biserial",
    "confirmatory", "p_holm", "p_fdr", "significance",
]


# Released no-skill test: seed t-test against the fixed reference and test-CI check, Holm over
# every task of both formulations
def _historical_noskill_rows(seed_df: pd.DataFrame, learn: pd.DataFrame) -> list[dict]:

    rows: list[dict] = []

    for task_type, tasks in (("regression", TASK_ORDER_REG), ("classification", TASK_ORDER_CLS)):

        metric    = HISTORICAL_PRIMARY[task_type]
        reference = HISTORICAL_REFERENCE[metric]
        higher    = METRIC_HIGHER_IS_BETTER[metric]

        for task in tasks:

            entry = learn[(learn["task"] == task) & (learn["metric"] == metric)]

            if entry.empty:
                continue

            entry  = entry.iloc[0]
            model  = str(entry["best_model"])

            chosen = seed_df.loc[
                (seed_df["task"] == task)
                & (seed_df["model"] == model)
                & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
            ].dropna(subset=[metric])

            values = chosen[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size < 2:
                continue

            t_stat, p_two = sp_stats.ttest_1samp(values, reference)
            better        = (values.mean() > reference) if higher else (values.mean() < reference)
            p_one         = p_two / 2.0 if better else 1.0 - p_two / 2.0

            test_lo = float(entry["test_ci_lo"])
            test_hi = float(entry["test_ci_hi"])
            test_excludes = bool(np.isfinite(test_lo) and np.isfinite(test_hi)
                                 and not (test_lo <= reference <= test_hi))

            rows.append({
                "task":                task,
                "task_type":           task_type,
                "property":            task_property(task),
                "metric":              metric,
                "configuration":       model,
                "n_seeds":             int(values.size),
                "n_test":              int(entry["n_test"]),
                "mean":                float(values.mean()),
                "reference":           float(reference),
                "direction_beats_ref": bool(better),
                "t_statistic":         float(t_stat),
                "p_one_sided_seed":    float(p_one),
                "test_ci_lo":          test_lo,
                "test_ci_hi":          test_hi,
                "test_ci_excludes_ref": test_excludes,
            })

    holm = _holm_correction([r["p_one_sided_seed"] for r in rows])
    for r, p in zip(rows, holm):
        r["p_holm_seed"]      = float(p)
        r["seed_significant"] = _significance(p)
        r["beats_reference"]  = bool(
            r["direction_beats_ref"]
            and r["seed_significant"] == "significant"
            and r["test_ci_excludes_ref"]
        )

    return rows


# Run the released contrast set on RMSE/SD_train and write it to output_dir
def main(output_dir: Path = DEFAULT_OUTPUT) -> None:

    seed_df = load_seed_results()
    rows: list[dict] = []
    delta_frames: list[pd.DataFrame] = []

    string_scratch    = ["lstm", "lstm_fragsmiles", "transformer", "transformer_fragsmiles"]
    string_pretrained = ["chemberta2", "peptide_clm2"]

    for task_type, tasks in (("regression", list(TASK_ORDER_REG)), ("classification", list(TASK_ORDER_CLS))):

        metric = HISTORICAL_PRIMARY[task_type]
        print_section_header(f"HISTORICAL CONTRASTS - {task_type} ({metric})")

        result = _split_effect(seed_df, tasks, task_type, metric)
        if result:
            row, deltas = result
            rows.append(row)
            delta_frames.append(deltas.assign(task_type=task_type, metric=metric, variant="raw"))

        if task_type == "regression":
            result = _split_effect_relative(seed_df, tasks, task_type, metric)
            if result:
                row, deltas = result
                rows.append(row)
                delta_frames.append(deltas.assign(task_type=task_type, metric=metric, variant="relative"))

        pair_specs = [
            ("physchem_ablation", "FP+PhysChem vs FP-only",   physchem_pairs()),
            ("radius_effect",     "ECFP6 vs ECFP4",           ecfp_radius_pairs()),
            ("radius_effect",     "MAP6C vs MAP4C",           map_radius_pairs()),
            ("count_effect",      "ECFP4c vs ECFP4b",         count_binary_pairs()),
            ("fragsmiles",        "fragSMILES vs SMILES (LSTM)",        [("lstm_fragsmiles", "lstm")]),
            ("fragsmiles",        "fragSMILES vs SMILES (Transformer)", [("transformer_fragsmiles", "transformer")]),
        ]
        for analysis, comparison, pairs in pair_specs:
            row = _delta_row(analysis, task_type, metric, comparison, "variant - baseline",
                             _paired_deltas(seed_df, tasks, task_type, metric, pairs))
            if row:
                rows.append(row)

        for family_a, family_b in [("RF", "XGBoost"), ("LSTM", "Transformer"),
                                   ("GCN", "MPNN"), ("ChemBERTa", "PeptideCLM"), ("GIN", "MolCLR")]:
            row = _delta_row("model_families", task_type, metric, f"{family_a} vs {family_b}",
                             f"{family_a} - {family_b}",
                             _paired_deltas(seed_df, tasks, task_type, metric, _family_pair_models(family_a, family_b)))
            if row:
                rows.append(row)

        for comparison, models_a, models_b in [
            ("String: scratch vs pretrained", string_scratch, string_pretrained),
            ("Graph: scratch vs pretrained", ["gcn", "mpnn"], ["pretrained_gin", "molclr"]),
        ]:
            row = _group_contrast(seed_df, tasks, task_type, metric, "scratch_vs_pretrained", comparison, models_a, models_b)
            if row:
                rows.append(row)

        if task_type == "regression":
            row = _group_contrast(seed_df, tasks, task_type, metric, "scratch_vs_pretrained_relative",
                                  "String: scratch vs pretrained (relative)", string_scratch, string_pretrained,
                                  rescale=True)
            if row:
                row["metric"] = f"{metric}_relative"
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame["confirmatory"] = ~frame["analysis"].isin(NON_CONFIRMATORY)
    frame["p_holm"]       = np.nan
    frame["p_fdr"]        = np.nan

    for _, block in frame.groupby(["analysis", "task_type"]):
        frame.loc[block.index, "p_holm"] = _holm_correction(block["p_value"].tolist())

    confirmatory = frame.index[frame["confirmatory"]]
    frame.loc[confirmatory, "p_fdr"] = _benjamini_hochberg(frame.loc[confirmatory, "p_value"].tolist())
    frame["significance"] = [
        _significance(p) if ok else "descriptive"
        for p, ok in zip(frame["p_fdr"], frame["confirmatory"])
    ]
    frame = frame.reindex(columns=HISTORICAL_COLUMNS)

    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "source_data").mkdir(parents=True, exist_ok=True)

    frame.to_csv(output_dir / "statistical_tests.csv", index=False)
    pd.concat(delta_frames, ignore_index=True).to_csv(output_dir / "source_data" / "split_effect_deltas_analysed.csv", index=False)
    _write_latex_table(frame, output_dir / "tables" / "statistical_tests.tex")

    learn = compute_learnability(_load_seed_results(), primary=HISTORICAL_PRIMARY,
                                 roles={**METRIC_ROLE, "nrmse": "historical"})
    pd.DataFrame(_historical_noskill_rows(seed_df, learn)).to_csv(output_dir / "noskill_tests.csv", index=False)
    print(f"  Saved: {output_dir}")

    print_completion_message()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce the superseded RMSE/SD_train inference into its own directory.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output_dir)
