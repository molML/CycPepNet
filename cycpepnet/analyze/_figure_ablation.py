from __future__ import annotations
import numpy as np
import pandas as pd
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT
from cycpepnet.analyze._result_helpers import MAIN_CLS_TASKS, MAIN_REG_TASKS, METRIC_HIGHER_IS_BETTER, TABLES_DIR, TASK_LABEL, TASK_RESPONSE_SCALE, save_source_table, save_table, task_property

# Runs eligible for the representation ablations: canonical Butina split, framework families only
def ablation_pool(seed_df: pd.DataFrame, tasks: list[str], task_type: str) -> pd.DataFrame:
    return seed_df.loc[
        (seed_df["task"].isin(tasks))
        & (seed_df["task_type"] == task_type)
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (~seed_df["family"].isin(["Baseline", "Other"]))
    ].copy()

# Each ablation as (comparison, [(variant model, baseline model)], comparison group)
def ablation_specs() -> list[tuple[str, list[tuple[str, str]], str]]:
    return [
        ("PhysChem", physchem_pairs(), "PhysChem"),
        ("ECFP4c - ECFP4b", count_binary_pairs(), "Fingerprint encoding"),
        ("ECFP6 - ECFP4", ecfp_radius_pairs(), "Fingerprint radius"),
        ("MAP6C - MAP4C", map_radius_pairs(), "Fingerprint radius"),
        ("fragSMILES - SMILES (LSTM)", [("lstm_fragsmiles", "lstm")], "String tokenization"),
        ("fragSMILES - SMILES (Transformer)", [("transformer_fragsmiles", "transformer")], "String tokenization"),
    ]

# Build ablation deltas
def build_ablation_deltas(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> pd.DataFrame:
    work = ablation_pool(seed_df, tasks, task_type)

    frames = [
        pair_model_delta(work, metric, comparison, pairs, group)
        for comparison, pairs, group in ablation_specs()
    ]
    out = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    out["property"] = out["task"].map(task_property)
    return out

PHYSCHEM_FINGERPRINTS = (
    ("binary_r2", "ECFP4 binary"),
    ("count_r2",  "ECFP4 count"),
    ("binary",    "ECFP6 binary"),
    ("count",     "ECFP6 count"),
    ("map4c",     "MAP4C"),
    ("map6c",     "MAP6C"),
    ("mxfp",      "MXFP"),
    ("hdf",       "HDF"),
)

ABLATION_ORDER = [
    "PhysChem",
    *(f"PhysChem {label}" for _, label in PHYSCHEM_FINGERPRINTS),
    "ECFP4c - ECFP4b",
    "ECFP6 - ECFP4",
    "MAP6C - MAP4C",
    "fragSMILES - SMILES (LSTM)",
    "fragSMILES - SMILES (Transformer)",
]

ABLATION_LABEL = {
    "PhysChem": "+ PhysChem",
    **{f"PhysChem {label}": f"+ PhysChem ({label})" for _, label in PHYSCHEM_FINGERPRINTS},
    "ECFP4c - ECFP4b": "ECFP4c - ECFP4b",
    "ECFP6 - ECFP4": "ECFP6 - ECFP4",
    "MAP6C - MAP4C": "MAP6C - MAP4C",
    "fragSMILES - SMILES (LSTM)": "LSTM: fragSMILES - SMILES",
    "fragSMILES - SMILES (Transformer)": "Transformer: fragSMILES - SMILES",
}

# Return PhysChem ablation pairs
def physchem_pairs() -> list[tuple[str, str]]:
    pairs = []
    suffixes = ["binary", "count", "binary_r2", "count_r2", "map4c", "map6c", "mxfp", "hdf"]
    for family_prefix in ["rf", "xgb"]:
        for suffix in suffixes:
            pairs.append((f"{family_prefix}_{suffix}", f"{family_prefix}_{suffix}_fp"))
    return pairs

# Return count-versus-binary fingerprint pairs
def count_binary_pairs() -> list[tuple[str, str]]:
    pairs = []
    for family_prefix in ["rf", "xgb"]:
        pairs.extend(
            [
                (f"{family_prefix}_count_r2", f"{family_prefix}_binary_r2"),
                (f"{family_prefix}_count_r2_fp", f"{family_prefix}_binary_r2_fp"),
            ]
        )
    return pairs

# Return ECFP radius comparison pairs
def ecfp_radius_pairs() -> list[tuple[str, str]]:
    pairs = []
    for family_prefix in ["rf", "xgb"]:
        pairs.extend(
            [
                (f"{family_prefix}_binary", f"{family_prefix}_binary_r2"),
                (f"{family_prefix}_binary_fp", f"{family_prefix}_binary_r2_fp"),
                (f"{family_prefix}_count", f"{family_prefix}_count_r2"),
                (f"{family_prefix}_count_fp", f"{family_prefix}_count_r2_fp"),
            ]
        )
    return pairs

# Return MAP radius comparison pairs
def map_radius_pairs() -> list[tuple[str, str]]:
    pairs = []
    for family_prefix in ["rf", "xgb"]:
        pairs.extend(
            [
                (f"{family_prefix}_map6c", f"{family_prefix}_map4c"),
                (f"{family_prefix}_map6c_fp", f"{family_prefix}_map4c_fp"),
            ]
        )
    return pairs

# Pair model results and calculate metric deltas
def pair_model_delta(
    work: pd.DataFrame,
    metric: str,
    comparison: str,
    pairs: list[tuple[str, str]],
    comparison_group: str,
) -> pd.DataFrame:
    rows = []
    for model_a, model_b in pairs:
        sub = work.loc[work["model"].isin([model_a, model_b]), ["task", "model", "seed", metric]].copy()
        wide = sub.pivot_table(index=["task", "seed"], columns="model", values=metric, aggfunc="mean")
        if model_a not in wide.columns or model_b not in wide.columns:
            continue
        wide = wide.dropna(subset=[model_a, model_b]).reset_index()
        wide["delta"] = wide[model_a] - wide[model_b]
        wide["value_a"] = wide[model_a]
        wide["value_b"] = wide[model_b]
        wide["comparison"] = comparison
        wide["comparison_group"] = comparison_group
        wide["model_a"] = model_a
        wide["model_b"] = model_b
        rows.append(wide)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# PhysChem deltas split by fingerprint
def physchem_fingerprint_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    physchem = deltas.loc[deltas["comparison"] == "PhysChem"]
    return pd.concat(
        [
            physchem.loc[physchem["model_b"].isin([f"{prefix}_{suffix}_fp" for prefix in ("rf", "xgb")])]
            .assign(comparison=f"PhysChem {label}")
            for suffix, label in PHYSCHEM_FINGERPRINTS
        ],
        ignore_index=True,
    )

# Metric pairs of the two ablation source-data files
ABLATION_EXPORTS = (
    ("representation_ablation_paired_deltas.csv",
     [("regression", MAIN_REG_TASKS, "rmse"), ("classification", MAIN_CLS_TASKS, "balanced_acc")]),
    ("representation_ablation_paired_deltas_companion_metrics.csv",
     [("regression", MAIN_REG_TASKS, "qf3"), ("classification", MAIN_CLS_TASKS, "auroc")]),
)

# Metrics on one dimensionless scale in every task
ABLATION_POOLABLE_METRICS = {"balanced_acc", "qf3", "auroc"}

# Every expected (ablation, pair, task, seed) observation that could not be paired, and why
def ablation_exclusions(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> pd.DataFrame:

    work = ablation_pool(seed_df, tasks, task_type)
    rows = []

    for comparison, pairs, _ in ablation_specs():
        for variant, baseline in pairs:
            for task in tasks:

                in_task  = work.loc[work["task"] == task]
                expected = sorted(in_task["seed"].dropna().astype(int).unique())
                present  = {
                    model: set(in_task.loc[(in_task["model"] == model) & in_task[metric].notna(), "seed"].astype(int))
                    for model in (variant, baseline)
                }

                for seed in expected:
                    missing = [role for role, model in (("variant", variant), ("baseline", baseline))
                               if seed not in present[model]]
                    if missing:
                        rows.append(dict(task_type=task_type, metric=metric, comparison=comparison, task=task,
                                         model_variant=variant, model_baseline=baseline, seed=seed,
                                         reason=f"no finite {metric} for the {' and '.join(missing)} run"))

    return pd.DataFrame(rows, columns=["task_type", "metric", "comparison", "task", "model_variant",
                                       "model_baseline", "seed", "reason"])

# Per task and ablation: both group summaries and the summaries of the paired differences
def summarise_ablation_by_task(deltas: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> pd.DataFrame:

    higher_is_better = METRIC_HIGHER_IS_BETTER[metric]
    rows = []

    for comparison in ABLATION_ORDER:
        for task in tasks:

            group = deltas.loc[(deltas["comparison"] == comparison) & (deltas["task"] == task)]
            if group.empty:
                continue

            variant  = group["value_a"].to_numpy(dtype=float)
            baseline = group["value_b"].to_numpy(dtype=float)
            delta    = group["delta"].to_numpy(dtype=float)
            favours  = delta > 0 if higher_is_better else delta < 0
            against  = delta < 0 if higher_is_better else delta > 0

            rows.append(dict(
                task_type = task_type,
                metric    = metric,
                comparison = comparison,
                comparison_label = ABLATION_LABEL[comparison],
                task      = task,
                task_label = TASK_LABEL[task],
                property  = task_property(task),
                response_scale = TASK_RESPONSE_SCALE[task] if metric == "rmse" else "dimensionless",
                delta_definition = "variant - baseline",
                favours_variant_when = "delta > 0" if higher_is_better else "delta < 0",
                n_configuration_pairs = int(group.groupby(["model_a", "model_b"]).ngroups),
                n_seeds   = int(group["seed"].nunique()),
                n_observations = int(len(group)),
                mean_variant  = float(variant.mean()),
                mean_baseline = float(baseline.mean()),
                difference_of_means = float(variant.mean() - baseline.mean()),
                mean_of_paired_differences = float(delta.mean()),
                median_variant  = float(np.median(variant)),
                median_baseline = float(np.median(baseline)),
                difference_of_medians = float(np.median(variant) - np.median(baseline)),
                median_of_paired_differences = float(np.median(delta)),
                q1_of_paired_differences = float(np.percentile(delta, 25)),
                q3_of_paired_differences = float(np.percentile(delta, 75)),
                iqr_of_paired_differences = float(np.percentile(delta, 75) - np.percentile(delta, 25)),
                n_favour_variant  = int(favours.sum()),
                n_favour_baseline = int(against.sum()),
                n_ties = int((delta == 0).sum()),
            ))

    return pd.DataFrame(rows)

# Per property: the median paired difference over every task, pair and seed of that property,
# and the median across properties of the matching IQR, as printed in Tables S1 and S3
def summarise_ablation_by_property(deltas: pd.DataFrame, task_type: str, metric: str) -> pd.DataFrame:

    if metric not in ABLATION_POOLABLE_METRICS:
        raise ValueError(f"{metric} cannot be pooled across the tasks of a property")

    rows = []

    for comparison in ABLATION_ORDER:

        group = deltas.loc[deltas["comparison"] == comparison]
        if group.empty:
            continue

        per_property = []
        for property_name in ("Binding", "Permeability", "Half-life"):
            values = group.loc[group["property"] == property_name, "delta"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            per_property.append(dict(
                task_type = task_type,
                metric    = metric,
                comparison = comparison,
                comparison_label = ABLATION_LABEL[comparison],
                delta_definition = "variant - baseline",
                property  = property_name,
                n_tasks   = int(group.loc[group["property"] == property_name, "task"].nunique()),
                n_observations = int(values.size),
                median_of_paired_differences = float(np.median(values)),
                iqr_of_paired_differences = float(np.percentile(values, 75) - np.percentile(values, 25)),
            ))

        iqr_median = float(np.median([entry["iqr_of_paired_differences"] for entry in per_property]))
        rows.extend({**entry, "iqr_median_across_properties": iqr_median} for entry in per_property)

    return pd.DataFrame(rows)

# LaTeX label of each ablation, as printed in the Supporting Information tables
ABLATION_LATEX_LABEL = {
    "PhysChem":                          r"Fingerprint $+$\,PhysChem vs.\ fingerprint",
    **{f"PhysChem {label}": rf"\quad {label}" for _, label in PHYSCHEM_FINGERPRINTS},
    "ECFP4c - ECFP4b":                  r"ECFP4 count vs.\ binary",
    "ECFP6 - ECFP4":                     r"ECFP6 vs.\ ECFP4",
    "MAP6C - MAP4C":                     r"MAP6C vs.\ MAP4C",
    "fragSMILES - SMILES (LSTM)":        r"fragSMILES vs.\ SMILES (LSTM)",
    "fragSMILES - SMILES (Transformer)": r"fragSMILES vs.\ SMILES (Transformer)",
}

# Block heading of a per-property ablation table, as printed in the Supporting Information
ABLATION_LATEX_BLOCK = {
    "balanced_acc": r"\multicolumn{5}{@{}l}{\textit{Classification --- balanced accuracy}}\\",
}

# Signed three-decimal number; a value that rounds to zero is printed +0.000, as in the SI
def signed_cell(value: float) -> str:
    text = f"{value:+.3f}"
    return "+0.000" if text == "-0.000" else text

# Regression ablations in wide form: per task, the median and IQR of the paired RMSE differences
def ablation_regression_wide(by_task: pd.DataFrame) -> pd.DataFrame:

    rmse = by_task.loc[by_task["metric"] == "rmse"]
    rows = []

    for comparison in ABLATION_ORDER:
        entry = dict(comparison=comparison, comparison_label=ABLATION_LABEL[comparison],
                     delta_definition="variant - baseline", favours_variant_when="delta < 0")
        for task in MAIN_REG_TASKS:
            cell = rmse.loc[(rmse["comparison"] == comparison) & (rmse["task"] == task)].iloc[0]
            entry[f"{task}__median_of_paired_differences"] = cell["median_of_paired_differences"]
            entry[f"{task}__iqr_of_paired_differences"]    = cell["iqr_of_paired_differences"]
            entry[f"{task}__n_observations"]               = int(cell["n_observations"])
        rows.append(entry)

    return pd.DataFrame(rows)

# LaTeX rows: regression ablations, six task columns of "median (IQR)" in RMSE units
def ablation_regression_latex(wide: pd.DataFrame) -> str:

    lines = []
    for row in wide.to_dict("records"):
        cells = [
            f"${signed_cell(row[f'{task}__median_of_paired_differences'])}$ "
            f"(${row[f'{task}__iqr_of_paired_differences']:.3f}$)"
            for task in MAIN_REG_TASKS
        ]
        lines.append(f"{ABLATION_LATEX_LABEL[row['comparison']]} & " + " & ".join(cells) + r" \\")

    return "\n".join(lines) + "\n"

# LaTeX rows: per-property ablation blocks for dimensionless metrics, in the SI layout
def ablation_property_latex(by_property: pd.DataFrame, metrics: list[str]) -> str:

    lines = []
    for metric in metrics:
        block = by_property.loc[by_property["metric"] == metric]
        lines += [r"\midrule", ABLATION_LATEX_BLOCK[metric], r"\midrule"]
        for comparison in ABLATION_ORDER:
            cells = block.loc[block["comparison"] == comparison].set_index("property")
            values = [f"${signed_cell(cells.loc[name, 'median_of_paired_differences'])}$"
                      for name in ("Binding", "Permeability", "Half-life")]
            iqr = cells["iqr_median_across_properties"].iloc[0]
            lines.append(f"{ABLATION_LATEX_LABEL[comparison]} & " + " & ".join(values) + f" & ${iqr:.3f}$ " + r"\\")

    return "\n".join(lines) + "\n"

# Write a LaTeX rows file next to the other tables
def save_latex_rows(text: str, name: str) -> None:

    out = TABLES_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"  Saved: {out}")

# Export representation ablation deltas, their exclusions and the descriptive summaries
def export_representation_ablation_deltas(seed_df: pd.DataFrame) -> None:

    by_task, by_property, exclusions = [], [], []

    for filename, panels in ABLATION_EXPORTS:

        all_deltas = []

        for task_type, tasks, metric in panels:
            deltas = build_ablation_deltas(seed_df, tasks, task_type, metric)
            all_deltas.append(deltas.assign(metric=metric))
            summary_deltas = pd.concat([deltas, physchem_fingerprint_deltas(deltas)], ignore_index=True)
            by_task.append(summarise_ablation_by_task(summary_deltas, tasks, task_type, metric))
            exclusions.append(ablation_exclusions(seed_df, tasks, task_type, metric))
            if metric in ABLATION_POOLABLE_METRICS:
                by_property.append(summarise_ablation_by_property(summary_deltas, task_type, metric))

        save_source_table(pd.concat(all_deltas, ignore_index=True), filename)

    by_task     = pd.concat(by_task, ignore_index=True)
    by_property = pd.concat(by_property, ignore_index=True)
    wide        = ablation_regression_wide(by_task)

    save_source_table(pd.concat(exclusions, ignore_index=True), "representation_ablation_exclusions.csv")
    save_table(by_task, "representation_ablation_by_task.csv")
    save_table(by_property, "representation_ablation_by_property.csv")
    save_table(wide, "representation_ablation_regression_by_task_wide.csv")

    save_latex_rows(ablation_regression_latex(wide), "representation_ablation_regression_by_task.tex")
    save_latex_rows(ablation_property_latex(by_property, ["balanced_acc"]), "representation_ablation_classification_by_property.tex")
