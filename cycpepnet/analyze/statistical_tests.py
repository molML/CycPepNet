from __future__ import annotations
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sp_stats
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT, TASK_ORDER_CLS, TASK_ORDER_REG
from cycpepnet.analyze._results_config import RESULTS_ROOT
from cycpepnet.analyze._results_data import main_pool
from cycpepnet.analyze._stats_core import _benjamini_hochberg, _holm_correction, _significance
from cycpepnet.analyze._result_helpers import METRIC_HIGHER_IS_BETTER, load_seed_results, task_property
from cycpepnet.analyze._figure_ablation import count_binary_pairs, ecfp_radius_pairs, map_radius_pairs, pair_model_delta, physchem_pairs
from cycpepnet.analyze._figure_split import split_effect_observations
from cycpepnet.utils import print_completion_message, print_section_header, tee_stdout_to_log

OUTPUT_CSV       = RESULTS_ROOT / "statistical_tests.csv"
OUTPUT_NOSKILL   = RESULTS_ROOT / "noskill_tests.csv"
OUTPUT_DELTAS    = RESULTS_ROOT / "source_data" / "split_effect_deltas_analysed.csv"
OUTPUT_TEX       = RESULTS_ROOT / "tables" / "statistical_tests.tex"
OUTPUT_BY_TASK   = RESULTS_ROOT / "tables" / "regression_contrasts_by_task.csv"
OUTPUT_RELATIVE  = RESULTS_ROOT / "tables" / "relative_to_training_mean_by_task.csv"
OUTPUT_RETIRED   = RESULTS_ROOT / "tables" / "statistical_tests_retired.csv"
OUTPUT_FAMILIES  = RESULTS_ROOT / "tables" / "multiplicity_families.csv"
RANDOM_SPLIT     = "random"
BOOTSTRAP_DRAWS  = 10000
BOOTSTRAP_SEED   = 20260812
ALPHA            = 0.05

# Primary metric per task type, aligned with the manuscript's metrics policy
PRIMARY_METRIC = {"regression": "rmse", "classification": "balanced_acc"}

# Metrics on one dimensionless scale in every task
POOLABLE_METRICS = {"qf3", "balanced_acc", "auroc"}

# Fixed no-skill value of a metric
NOSKILL_REFERENCE = {"balanced_acc": 0.5}

# Where each analysis draws its runs; the split contrast shares Figure 4's observations
POOL_FIGURE4 = "Figure 4 split pool: main analysis pool, no baselines, paired Random-Butina"
POOL_RELEASED = "released contrast pool (unchanged)"

CSV_COLUMNS = [
    "analysis", "task_type", "metric", "comparison", "direction", "inference",
    "n_obs", "n_tasks", "n_clusters",
    "estimate", "std_error", "ci_lo", "ci_hi", "z_value", "p_value",
    "lmm_status", "lmm_std_error", "lmm_ci_lo", "lmm_ci_hi", "lmm_p_value",
    "var_task", "var_cluster", "var_resid",
    "tc_estimate", "tc_std_error", "tc_ci_lo", "tc_ci_hi", "tc_p_value",
    "median_delta", "median_ci_lo", "median_ci_hi",
    "wilcoxon_task_p", "rank_biserial",
    "confirmatory", "contrast_id", "pool",
    "holm_family", "holm_family_size", "p_holm",
    "bh_family", "bh_family_size", "p_fdr", "significance",
]

# Families of hypotheses, as defined in the Methods (Multiplicity)
BH_FAMILY = "bh:confirmatory_primary"
BH_FAMILY_DEFINITION = (
    "Confirmatory set: every split, representation-ablation and model-family comparison on the "
    "primary metrics (Methods, Multiplicity). The pooled regression comparisons are retired, so "
    "the set now holds the classification comparisons only; revision of the released 30-comparison family."
)
HOLM_FAMILY_DEFINITION = (
    "Related comparisons of one analysis within one task type, the grouping statistical_tests.py "
    "has always used (Methods, Multiplicity: 'within each family of related comparisons', e.g. the "
    "five model-family contrasts)."
)
NOSKILL_FAMILY = "holm:noskill:classification"
NOSKILL_FAMILY_DEFINITION = (
    "One-sided seed tests of the six classification tasks against balanced accuracy = 0.5. "
    "Revision: the released family also held six regression tests against RMSE/SD_train = 1, now retired."
)

# Comparisons excluded from the Benjamini-Hochberg family
NON_CONFIRMATORY = {"scratch_vs_pretrained_relative"}


# Build a long frame of paired deltas with one row per (task, cluster, seed)
def _paired_deltas(
    seed_df: pd.DataFrame,
    tasks: list[str],
    task_type: str,
    metric: str,
    pairs: list[tuple[str, str]],
    split_method: str = CANONICAL_BUTINA_SPLIT,
) -> pd.DataFrame:

    work = seed_df.loc[
        seed_df["task"].isin(tasks)
        & (seed_df["task_type"] == task_type)
        & (seed_df["split_method"] == split_method)
    ].copy()

    deltas = pair_model_delta(work, metric, "pair", pairs, "pair")

    if deltas.empty:
        return deltas

    deltas["cluster"] = deltas["model_a"].astype(str) + "|" + deltas["model_b"].astype(str)
    return deltas[["task", "cluster", "seed", "delta"]].dropna(subset=["delta"])


# Percentile bootstrap confidence interval for the median of a set of values
def _median_bootstrap_ci(values: np.ndarray) -> tuple[float, float]:

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]

    if clean.size < 2:
        return float("nan"), float("nan")

    rng    = np.random.default_rng(BOOTSTRAP_SEED)
    draws  = rng.choice(clean, size=(BOOTSTRAP_DRAWS, clean.size), replace=True)
    medians = np.median(draws, axis=1)

    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


# One-sample t-test on per-task mean deltas
#
# This is the cluster-robust inference with tasks as the clusters. Its
# reference distribution has n_tasks - 1 degrees of freedom, so it answers
# "does this effect hold across tasks?" with the honest sample size.
def _task_clustered_test(deltas: pd.DataFrame) -> dict:

    per_task = deltas.groupby("task")["delta"].mean().to_numpy(dtype=float)
    per_task = per_task[np.isfinite(per_task)]
    n        = per_task.size

    if n < 2:
        return {"tc_estimate": float("nan"), "tc_std_error": float("nan"),
                "tc_ci_lo": float("nan"), "tc_ci_hi": float("nan"), "tc_p_value": float("nan")}

    mean   = float(per_task.mean())
    stderr = float(per_task.std(ddof=1) / np.sqrt(n))

    if stderr == 0.0:
        return {"tc_estimate": mean, "tc_std_error": 0.0,
                "tc_ci_lo": mean, "tc_ci_hi": mean, "tc_p_value": 0.0}

    crit = float(sp_stats.t.ppf(1.0 - ALPHA / 2.0, n - 1))
    _, p = sp_stats.ttest_1samp(per_task, 0.0)

    return {"tc_estimate": mean, "tc_std_error": stderr,
            "tc_ci_lo": mean - crit * stderr, "tc_ci_hi": mean + crit * stderr,
            "tc_p_value": float(p)}


# Fit delta ~ 1 with random intercepts for task and, where identifiable, cluster
#
# Seeds enter as replicates within (task, cluster). They vary training
# stochasticity only, because every task has a single fixed split, so the
# resulting interval covers configuration, task and seed variation but not
# split variation.
#
# When a contrast has only one cluster the crossed parameterization is not
# identifiable, so task alone is used as the grouping factor. Fits whose
# fixed-effect standard error collapses far below the between-task standard
# error have lost the task-level variance component; those are flagged as
# degenerate. This fit is a sensitivity check, not the primary inference.
def _fit_lmm(deltas: pd.DataFrame, clustered_se: float) -> dict:

    out = {
        "estimate":    float("nan"),
        "std_error":   float("nan"),
        "ci_lo":       float("nan"),
        "ci_hi":       float("nan"),
        "z_value":     float("nan"),
        "lmm_std_error": float("nan"),
        "lmm_ci_lo":   float("nan"),
        "lmm_ci_hi":   float("nan"),
        "lmm_p_value": float("nan"),
        "var_task":    float("nan"),
        "var_cluster": float("nan"),
        "var_resid":   float("nan"),
        "lmm_status":  "not_fitted",
    }

    if len(deltas) < 4 or deltas["delta"].nunique() < 2:
        return out

    data       = deltas.copy()
    n_clusters = data["cluster"].nunique()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if n_clusters >= 2:
                data["_all"] = 1
                model = smf.mixedlm(
                    "delta ~ 1", data, groups=data["_all"],
                    vc_formula={"task": "0 + C(task)", "cluster": "0 + C(cluster)"},
                )
            else:
                model = smf.mixedlm("delta ~ 1", data, groups=data["task"], re_formula="1")

            fit = model.fit(reml=True, method="lbfgs")

    except Exception as exc:
        out["lmm_status"] = f"failed:{type(exc).__name__}"
        return out

    estimate = float(fit.fe_params["Intercept"])
    stderr   = float(fit.bse_fe["Intercept"])

    if not np.isfinite(estimate) or not np.isfinite(stderr) or stderr <= 0:
        out["lmm_status"] = "non_finite"
        return out

    vcomp = list(np.atleast_1d(fit.vcomp)) if hasattr(fit, "vcomp") else []

    components = dict(zip(getattr(model.exog_vc, "names", []), vcomp))
    random_cov = np.asarray(fit.cov_re).ravel()

    out.update(
        estimate    = estimate,
        std_error   = stderr,
        ci_lo       = estimate - 1.959963985 * stderr,
        ci_hi       = estimate + 1.959963985 * stderr,
        z_value     = estimate / stderr,
        lmm_std_error = stderr,
        lmm_ci_lo   = estimate - 1.959963985 * stderr,
        lmm_ci_hi   = estimate + 1.959963985 * stderr,
        lmm_p_value = float(fit.pvalues["Intercept"]),
        var_task    = float(components.get("task", random_cov[0] if random_cov.size else float("nan"))),
        var_cluster = float(components.get("cluster", float("nan"))),
        var_resid   = float(fit.scale),
        lmm_status  = "ok",
    )

    # A fixed-effect standard error far below the between-task standard error
    # means the task variance component collapsed to zero and the replicates
    # were effectively treated as independent.
    if np.isfinite(clustered_se) and clustered_se > 0 and stderr < 0.5 * clustered_se:
        out["lmm_status"] = "degenerate"

    return out


# Matched-pairs rank-biserial correlation
def _rank_biserial(delta: np.ndarray) -> float:

    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d) & (d != 0.0)]

    if d.size == 0:
        return 0.0

    ranks = sp_stats.rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())

    return (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0.0


# Task-level paired Wilcoxon on per-task mean deltas, kept as a conservative cross-check
def _task_level_wilcoxon(deltas: pd.DataFrame) -> tuple[float, float, float, float, float]:

    per_task = deltas.groupby("task")["delta"].mean().to_numpy(dtype=float)
    per_task = per_task[np.isfinite(per_task)]

    if per_task.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    median = float(np.median(per_task))
    lo, hi = _median_bootstrap_ci(per_task)

    try:
        _, p = sp_stats.wilcoxon(per_task, alternative="two-sided")
    except ValueError:
        p = float("nan")

    return median, lo, hi, float(p), _rank_biserial(per_task)


# Per-task description of one contrast whose metric cannot be pooled across tasks
def _per_task_summary(analysis: str, task_type: str, metric: str, comparison: str, direction: str,
                      delta_kind: str, deltas: pd.DataFrame, tasks: list[str], pool: str) -> pd.DataFrame:

    rows = []

    for task in tasks:

        group = deltas.loc[deltas["task"] == task]
        if group.empty:
            continue

        values = group["delta"].to_numpy(dtype=float)
        rows.append(dict(
            analysis=analysis, task_type=task_type, metric=metric, comparison=comparison,
            direction=direction, delta_kind=delta_kind, pool=pool, task=task,
            n_obs=int(values.size), n_clusters=int(group["cluster"].nunique()),
            n_seeds=int(group["seed"].nunique()),
            mean_of_differences=float(values.mean()),
            median_of_differences=float(np.median(values)),
            q1_of_differences=float(np.percentile(values, 25)),
            q3_of_differences=float(np.percentile(values, 75)),
            n_positive=int((values > 0).sum()), n_negative=int((values < 0).sum()),
            n_zero=int((values == 0).sum()),
        ))

    return pd.DataFrame(rows)


# Assemble one result row from a set of paired deltas
#
# The reported estimate and p-value are task-clustered; the mixed model is a sensitivity check in the lmm_ columns.
def _delta_row(analysis: str, task_type: str, metric: str, comparison: str,
               direction: str, deltas: pd.DataFrame) -> dict | None:

    if deltas.empty:
        return None

    row = {"analysis": analysis, "task_type": task_type, "metric": metric,
           "comparison": comparison, "direction": direction,
           "n_obs": int(len(deltas)), "n_tasks": int(deltas["task"].nunique()),
           "n_clusters": int(deltas["cluster"].nunique())}

    clustered = _task_clustered_test(deltas)
    row.update(clustered)
    row.update(_fit_lmm(deltas, clustered["tc_std_error"]))

    row["inference"] = "task_clustered"
    row.update(
        estimate  = clustered["tc_estimate"],
        std_error = clustered["tc_std_error"],
        ci_lo     = clustered["tc_ci_lo"],
        ci_hi     = clustered["tc_ci_hi"],
        p_value   = clustered["tc_p_value"],
    )

    median, lo, hi, wp, rb = _task_level_wilcoxon(deltas)
    row.update(median_delta=median, median_ci_lo=lo, median_ci_hi=hi,
               wilcoxon_task_p=wp, rank_biserial=rb)

    return row


# Random vs Butina deltas, paired within task, configuration and seed
def _split_deltas(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> pd.DataFrame:

    work = seed_df.loc[
        seed_df["task"].isin(tasks)
        & (seed_df["task_type"] == task_type)
        & (seed_df["split_method"].isin([CANONICAL_BUTINA_SPLIT, RANDOM_SPLIT]))
    ].copy()

    wide = work.pivot_table(
        index=["task", "model", "rep", "seed"],
        columns="split_method",
        values=metric,
        aggfunc="mean",
    ).reset_index()

    if CANONICAL_BUTINA_SPLIT not in wide.columns or RANDOM_SPLIT not in wide.columns:
        return pd.DataFrame()

    wide = wide.dropna(subset=[CANONICAL_BUTINA_SPLIT, RANDOM_SPLIT])
    wide["delta"]   = wide[RANDOM_SPLIT] - wide[CANONICAL_BUTINA_SPLIT]
    wide["cluster"] = wide["model"].astype(str) + "|" + wide["rep"].astype(str)

    return wide[["task", "cluster", "seed", "delta"]]


# Random vs Butina split, paired within task, configuration and seed
def _split_effect(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> tuple[dict, pd.DataFrame] | None:

    deltas = _split_deltas(seed_df, tasks, task_type, metric)
    row    = _delta_row(
        "split_effect", task_type, metric, "Random vs Butina", "random - butina", deltas,
    )

    return (row, deltas) if row else None


# Random vs Butina split, rescaled by the trivial predictor of each split
def _split_relative_deltas(seed_df: pd.DataFrame, tasks: list[str], task_type: str,
                          metric: str, canonical: bool = True) -> tuple[pd.DataFrame, pd.Series]:

    work = seed_df.loc[
        seed_df["task"].isin(tasks)
        & (seed_df["task_type"] == task_type)
        & (seed_df["split_method"].isin([CANONICAL_BUTINA_SPLIT, RANDOM_SPLIT]))
    ].copy()

    reference = work.loc[work["model"] == "mean"].groupby(["task", "split_method"])[metric].mean()
    work      = work.loc[work["model"] != "mean"]

    if canonical:
        work = main_pool(work)

    if reference.empty or work.empty:
        return pd.DataFrame(), reference

    keys = list(zip(work["task"], work["split_method"]))

    if any(key not in reference.index for key in keys):
        return pd.DataFrame(), reference

    work[metric] = work[metric].to_numpy(dtype=float) / reference.loc[keys].to_numpy(dtype=float)

    wide = work.pivot_table(
        index=["task", "model", "rep", "seed"],
        columns="split_method",
        values=metric,
        aggfunc="mean",
    ).reset_index()

    if CANONICAL_BUTINA_SPLIT not in wide.columns or RANDOM_SPLIT not in wide.columns:
        return pd.DataFrame(), reference

    wide = wide.dropna(subset=[CANONICAL_BUTINA_SPLIT, RANDOM_SPLIT])
    wide["delta"]   = wide[RANDOM_SPLIT] - wide[CANONICAL_BUTINA_SPLIT]
    wide["cluster"] = wide["model"].astype(str) + "|" + wide["rep"].astype(str)

    return wide[["task", "cluster", "seed", "delta"]], reference


# Random vs Butina split, rescaled by the trivial predictor of each split (historical inference)
def _split_effect_relative(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> tuple[dict, pd.DataFrame] | None:

    deltas, _ = _split_relative_deltas(seed_df, tasks, task_type, metric, canonical=False)
    row    = _delta_row(
        "split_effect", task_type, f"{metric}_relative",
        "Random vs Butina (relative)", "random - butina", deltas,
    )

    return (row, deltas) if row else None


# Random vs Butina deltas on exactly the observations of Figure 4
def _split_deltas_figure4(seed_df: pd.DataFrame, task_type: str) -> pd.DataFrame:

    observations, _ = split_effect_observations(seed_df)
    observations    = observations.loc[observations["task_type"] == task_type].copy()

    observations["delta"]   = observations["random"] - observations["butina"]
    observations["cluster"] = observations["model"].astype(str) + "|" + observations["rep"].astype(str)

    return observations[["task", "cluster", "seed", "delta"]].reset_index(drop=True)


# Group-mean deltas (scratch vs pretrained): per task and seed, mean of group a minus mean of b
def _group_deltas(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str,
                  models_a: list[str], models_b: list[str], rescale: bool = False) -> pd.DataFrame:

    work = seed_df.loc[
        seed_df["task"].isin(tasks)
        & (seed_df["task_type"] == task_type)
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
    ].copy()

    if rescale:
        reference = work.loc[work["model"] == "mean"].groupby(["task", "split_method"])[metric].mean()
        work      = work.loc[work["model"] != "mean"]

        if reference.empty or work.empty:
            return pd.DataFrame()

        keys = list(zip(work["task"], work["split_method"]))

        if any(key not in reference.index for key in keys):
            return pd.DataFrame()

        work[metric] = work[metric].to_numpy(dtype=float) / reference.loc[keys].to_numpy(dtype=float)

    frames = []

    for label, models in (("a", models_a), ("b", models_b)):
        sub = work[work["model"].isin(models)]
        if sub.empty:
            return pd.DataFrame()
        frames.append(sub.groupby(["task", "seed"])[metric].mean().rename(label))

    joined = pd.concat(frames, axis=1).dropna().reset_index()
    joined["delta"]   = joined["a"] - joined["b"]
    joined["cluster"] = "group"

    return joined[["task", "cluster", "seed", "delta"]]


# Group-mean comparison (scratch vs pretrained)
def _group_contrast(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str,
                    analysis: str, comparison: str, models_a: list[str], models_b: list[str],
                    rescale: bool = False) -> dict | None:

    deltas = _group_deltas(seed_df, tasks, task_type, metric, models_a, models_b, rescale)

    return _delta_row(analysis, task_type, metric, comparison, "a - b", deltas)


# Model-family contrasts paired on a shared representation
def _family_pair_models(family_a: str, family_b: str) -> list[tuple[str, str]]:

    suffixes = ["binary", "count", "binary_r2", "count_r2", "map4c", "map6c", "mxfp", "hdf", "physchem"]
    prefix   = {"RF": "rf", "XGBoost": "xgb"}

    if family_a in prefix and family_b in prefix:
        return [(f"{prefix[family_a]}_{s}", f"{prefix[family_b]}_{s}") for s in suffixes] + [
            (f"{prefix[family_a]}_{s}_fp", f"{prefix[family_b]}_{s}_fp")
            for s in suffixes if s != "physchem"
        ]

    single = {
        "LSTM": ["lstm", "lstm_fragsmiles"], "Transformer": ["transformer", "transformer_fragsmiles"],
        "ChemBERTa": ["chemberta2"], "PeptideCLM": ["peptide_clm2"],
        "GCN": ["gcn"], "MPNN": ["mpnn"], "GIN": ["pretrained_gin"], "MolCLR": ["molclr"],
    }
    a, b = single.get(family_a, []), single.get(family_b, [])

    return [(x, y) for x, y in zip(a, b)]


# Test the best validation-selected classifier against balanced accuracy = 0.5
def _noskill_rows(seed_df: pd.DataFrame, learn: pd.DataFrame) -> list[dict]:

    rows: list[dict] = []

    for task_type, tasks in (("regression", TASK_ORDER_REG), ("classification", TASK_ORDER_CLS)):

        metric    = PRIMARY_METRIC[task_type]
        reference = NOSKILL_REFERENCE.get(metric)

        if reference is None:
            continue

        higher = METRIC_HIGHER_IS_BETTER[metric]

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

            test_lo = float(entry.get("test_ci_lo", float("nan")))
            test_hi = float(entry.get("test_ci_hi", float("nan")))
            test_excludes = bool(np.isfinite(test_lo) and np.isfinite(test_hi)
                                 and not (test_lo <= reference <= test_hi))

            rows.append({
                "task":                task,
                "task_type":           task_type,
                "property":            task_property(task),
                "metric":              metric,
                "configuration":       model,
                "selected_by":         str(entry["selected_by"]),
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
                "contrast_id":         f"noskill:{task}",
            })

    holm = _holm_correction([r["p_one_sided_seed"] for r in rows])

    for r, p in zip(rows, holm):
        r["holm_family"]      = NOSKILL_FAMILY
        r["holm_family_size"] = len(rows)
        r["p_holm_seed"]      = float(p)
        r["seed_significant"] = _significance(p)
        r["beats_reference"]  = bool(
            r["direction_beats_ref"]
            and r["seed_significant"] == "significant"
            and r["test_ci_excludes_ref"]
        )

    return rows


# Format a p-value for the manuscript table
def _fmt_p(p: float) -> str:

    if not np.isfinite(p):
        return "---"
    if p < 0.001:
        return "$<0.001$"
    if f"{p:.3f}" == "0.050":
        return f"${p:.4f}$"

    return f"${p:.3f}$"


# Write the contrast table as a LaTeX tabular body for the Supporting Information
def _write_latex_table(frame: pd.DataFrame, path: Path) -> None:

    label = {
        "Random vs Butina":                   "Random vs.\\ Butina split",
        "Random vs Butina (relative)":        "Random vs.\\ Butina split, relative to the training-mean predictor",
        "FP+PhysChem vs FP-only":             "Fingerprint $+$\\,PhysChem vs.\\ fingerprint",
        "ECFP6 vs ECFP4":                     "ECFP6 vs.\\ ECFP4",
        "MAP6C vs MAP4C":                     "MAP6C vs.\\ MAP4C",
        "ECFP4c vs ECFP4b":                   "ECFP4 count vs.\\ binary",
        "fragSMILES vs SMILES (LSTM)":        "fragSMILES vs.\\ SMILES (LSTM)",
        "fragSMILES vs SMILES (Transformer)": "fragSMILES vs.\\ SMILES (Transformer)",
        "String: scratch vs pretrained":      "String models: scratch vs.\\ pretrained",
        "Graph: scratch vs pretrained":       "Graph models: scratch vs.\\ pretrained",
        "RF vs XGBoost":                      "Random Forest vs.\\ XGBoost",
        "LSTM vs Transformer":                "LSTM vs.\\ Transformer",
        "GCN vs MPNN":                        "GCN vs.\\ MPNN",
        "ChemBERTa vs PeptideCLM":            "ChemBERTa-2 vs.\\ PeptideCLM-2",
        "GIN vs MolCLR":                      "Pretrained GIN vs.\\ MolCLR",
    }

    lines: list[str] = []

    for task_type, unit in (("regression", r"$\mathrm{RMSE}/\mathrm{SD}_{\mathrm{train}}$"),
                            ("classification", "balanced accuracy")):

        block = frame[(frame["task_type"] == task_type) & frame["confirmatory"]].sort_values("p_fdr")

        if block.empty:
            continue

        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{task_type.capitalize()} --- {unit}}}}}\\")
        lines.append(r"\midrule")

        for r in block.itertuples():
            marker = r"$^{*}$" if r.significance == "significant" else ""
            lines.append(
                f"{label.get(r.comparison, r.comparison)}{marker} & "
                f"${r.estimate:+.3f}$ & "
                f"$[{r.ci_lo:+.3f},\\,{r.ci_hi:+.3f}]$ & "
                f"{_fmt_p(r.p_value)} & {_fmt_p(r.p_fdr)} & "
                f"{int(r.n_tasks)}/{int(r.n_clusters)}/{int(r.n_obs)} \\\\"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# Stable identifier of a contrast, e.g. classification:model_families:rf_vs_xgboost
def _contrast_id(task_type: str, analysis: str, comparison: str) -> str:
    return f"{task_type}:{analysis}:{re.sub(r'[^a-z0-9]+', '_', comparison.lower()).strip('_')}"


# Per-task description of a contrast relative to the training-mean predictor
def _relative_summary(analysis: str, comparison: str, direction: str, formula: str, delta_kind: str,
                      deltas: pd.DataFrame, reference: pd.Series, tasks: list[str],
                      pool: str = POOL_RELEASED) -> pd.DataFrame:

    summary = _per_task_summary(analysis, "regression", "rmse_relative", comparison, direction,
                                delta_kind, deltas, tasks, pool)
    summary.insert(summary.columns.get_loc("task") + 1, "formula", formula)

    for split, column in ((RANDOM_SPLIT, "rmse_training_mean_random"), (CANONICAL_BUTINA_SPLIT, "rmse_training_mean_butina")):
        values = [reference.get((task, split), np.nan) for task in summary["task"]]
        if analysis == "scratch_vs_pretrained_relative" and split == RANDOM_SPLIT:
            values = [np.nan] * len(values)
        summary[column] = values

    return summary


# Run the active contrasts and write the machine-readable tables
def main() -> None:

    seed_df = load_seed_results()

    if seed_df.empty:
        print("  No seed-level result rows found.")
        print_completion_message()
        return

    rows: list[dict] = []
    delta_frames: list[pd.DataFrame] = []
    by_task: list[pd.DataFrame] = []
    relative: list[pd.DataFrame] = []
    retired: list[dict] = []

    string_scratch    = ["lstm", "transformer"]
    string_pretrained = ["chemberta2", "peptide_clm2"]
    historical = "results/statistical_tests.csv as released (reproduced by statistical_tests_nrmse_historical.py)"

    for task_type, tasks in (("regression", list(TASK_ORDER_REG)), ("classification", list(TASK_ORDER_CLS))):

        metric   = PRIMARY_METRIC[task_type]
        poolable = metric in POOLABLE_METRICS
        print_section_header(f"CONTRASTS - {task_type} ({metric})")

        # (analysis, comparison, direction, delta kind, pool, deltas), in the order of the table
        split = _split_deltas_figure4(seed_df, task_type)
        delta_frames.append(split.assign(task_type=task_type, metric=metric, variant="raw"))
        contrasts = [("split_effect", "Random vs Butina", "random - butina", "paired difference", POOL_FIGURE4, split)]

        pair_specs = [
            ("physchem_ablation", "FP+PhysChem vs FP-only",   physchem_pairs()),
            ("radius_effect",     "ECFP6 vs ECFP4",           ecfp_radius_pairs()),
            ("radius_effect",     "MAP6C vs MAP4C",           map_radius_pairs()),
            ("count_effect",      "ECFP4c vs ECFP4b",         count_binary_pairs()),
            ("fragsmiles",        "fragSMILES vs SMILES (LSTM)",        [("lstm_fragsmiles", "lstm")]),
            ("fragsmiles",        "fragSMILES vs SMILES (Transformer)", [("transformer_fragsmiles", "transformer")]),
        ]
        for analysis, comparison, pairs in pair_specs:
            contrasts.append((analysis, comparison, "variant - baseline", "paired difference", POOL_RELEASED,
                              _paired_deltas(seed_df, tasks, task_type, metric, pairs)))

        for family_a, family_b in [("RF", "XGBoost"), ("LSTM", "Transformer"),
                                   ("GCN", "MPNN"), ("ChemBERTa", "PeptideCLM"), ("GIN", "MolCLR")]:
            contrasts.append(("model_families", f"{family_a} vs {family_b}", f"{family_a} - {family_b}",
                              "paired difference", POOL_RELEASED,
                              _paired_deltas(seed_df, tasks, task_type, metric, _family_pair_models(family_a, family_b))))

        for comparison, models_a, models_b in [
            ("String: scratch vs pretrained", string_scratch, string_pretrained),
            ("Graph: scratch vs pretrained", ["gcn", "mpnn"], ["pretrained_gin", "molclr"]),
        ]:
            contrasts.append(("scratch_vs_pretrained", comparison, "a - b", "difference of group means per seed",
                              POOL_RELEASED, _group_deltas(seed_df, tasks, task_type, metric, models_a, models_b)))

        for analysis, comparison, direction, kind, pool, deltas in contrasts:

            if deltas.empty:
                continue

            if poolable:
                row = _delta_row(analysis, task_type, metric, comparison, direction, deltas)
                if row:
                    row.update(contrast_id=_contrast_id(task_type, analysis, comparison), pool=pool)
                    rows.append(row)
                continue

            by_task.append(_per_task_summary(analysis, task_type, metric, comparison, direction, kind, deltas, tasks, pool))
            retired.append(dict(
                source_table="statistical_tests.csv", contrast_id=_contrast_id(task_type, analysis, comparison),
                analysis=analysis, task_type=task_type, comparison=comparison, released_metric="nrmse",
                status="retired",
                reason="pooled across tasks; RMSE is in each task's response units and no pooled scale is adopted",
                replacement="tables/regression_contrasts_by_task.csv (descriptive, per task, no test)",
                historical=historical,
            ))

        if task_type == "regression":

            split_rel, reference = _split_relative_deltas(seed_df, tasks, task_type, metric)
            delta_frames.append(split_rel.assign(task_type=task_type, metric=metric, variant="relative"))
            relative.append(_relative_summary(
                "split_effect_relative", "Random vs Butina (relative)", "random - butina",
                "RMSE_config,random / RMSE_mean,random - RMSE_config,butina / RMSE_mean,butina",
                "paired difference of ratios, split-specific denominators", split_rel, reference, tasks,
                POOL_FIGURE4))

            scratch_rel = _group_deltas(seed_df, tasks, task_type, metric, string_scratch, string_pretrained, rescale=True)
            relative.append(_relative_summary(
                "scratch_vs_pretrained_relative", "String: scratch vs pretrained (relative)", "a - b",
                "(mean RMSE_scratch - mean RMSE_pretrained) / RMSE_mean,butina",
                "difference of group means per seed, one denominator", scratch_rel, reference, tasks))

            for analysis, comparison, confirmatory, reason in (
                ("split_effect", "Random vs Butina (relative)", True,
                 "introduced to offset SD_train moving between splits, which raw RMSE does not have; the estimand is a "
                 "change relative to two different baselines, not a test of Figure 4; pooled inference re-entered the same runs"),
                ("scratch_vs_pretrained_relative", "String: scratch vs pretrained (relative)", False,
                 "algebraically delta RMSE / training-mean RMSE pooled across tasks, a normalisation not adopted for pooled contrasts"),
            ):
                retired.append(dict(
                    source_table="statistical_tests.csv", contrast_id=_contrast_id(task_type, analysis, comparison),
                    analysis=analysis, task_type=task_type, comparison=comparison, released_metric="nrmse_relative",
                    status="descriptive only" if confirmatory else "descriptive only (was non-confirmatory)",
                    reason=reason,
                    replacement="tables/relative_to_training_mean_by_task.csv (per task, no test)",
                    historical=historical,
                ))

    for task in TASK_ORDER_REG:
        retired.append(dict(
            source_table="noskill_tests.csv", contrast_id=f"noskill:{task}", analysis="noskill",
            task_type="regression", comparison="selected model vs RMSE/SD_train = 1", released_metric="nrmse",
            status="retired",
            reason="the reference is a training-variance threshold (QF3 = 0), not the training-mean predictor; RMSE has no fixed no-skill value",
            replacement="descriptive model vs measured training-mean predictor on the same test set (Figure 2, learnability.csv baseline)",
            historical="results/noskill_tests.csv as released (reproduced by statistical_tests_nrmse_historical.py)",
        ))

    frame = pd.DataFrame(rows)
    frame["confirmatory"] = ~frame["analysis"].isin(NON_CONFIRMATORY)
    frame["holm_family"]  = "holm:" + frame["task_type"] + ":" + frame["analysis"]
    frame["p_holm"]       = np.nan

    for family, block in frame.groupby("holm_family"):
        frame.loc[block.index, "p_holm"]           = _holm_correction(block["p_value"].tolist())
        frame.loc[block.index, "holm_family_size"] = len(block)

    confirmatory = frame.index[frame["confirmatory"]]
    frame["bh_family"]      = np.where(frame["confirmatory"], BH_FAMILY, "")
    frame["bh_family_size"] = np.where(frame["confirmatory"], len(confirmatory), 0)
    frame["p_fdr"]          = np.nan
    frame.loc[confirmatory, "p_fdr"] = _benjamini_hochberg(frame.loc[confirmatory, "p_value"].tolist())
    frame["significance"] = [
        _significance(p) if ok else "descriptive"
        for p, ok in zip(frame["p_fdr"], frame["confirmatory"])
    ]
    frame["holm_family_size"] = frame["holm_family_size"].astype(int)
    frame = frame.reindex(columns=CSV_COLUMNS)

    for path in (OUTPUT_CSV, OUTPUT_BY_TASK, OUTPUT_RELATIVE, OUTPUT_RETIRED, OUTPUT_FAMILIES, OUTPUT_DELTAS):
        path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved: {OUTPUT_CSV}  ({len(frame)} rows)")
    pd.concat(by_task, ignore_index=True).to_csv(OUTPUT_BY_TASK, index=False)
    print(f"  Saved: {OUTPUT_BY_TASK}")
    pd.concat(relative, ignore_index=True).to_csv(OUTPUT_RELATIVE, index=False)
    print(f"  Saved: {OUTPUT_RELATIVE}")
    pd.DataFrame(retired).to_csv(OUTPUT_RETIRED, index=False)
    print(f"  Saved: {OUTPUT_RETIRED}  ({len(retired)} entries)")
    pd.concat(delta_frames, ignore_index=True).to_csv(OUTPUT_DELTAS, index=False)
    print(f"  Saved: {OUTPUT_DELTAS}")

    _write_latex_table(frame, OUTPUT_TEX)

    noskill = _noskill_rows(seed_df, pd.read_csv(RESULTS_ROOT / "learnability.csv"))
    pd.DataFrame(noskill).to_csv(OUTPUT_NOSKILL, index=False)
    print(f"  Saved: {OUTPUT_NOSKILL}  ({len(noskill)} rows)")

    families = [
        dict(contrast_id=r.contrast_id, source_table="statistical_tests.csv", family_id=r.holm_family,
             family_definition=HOLM_FAMILY_DEFINITION, procedure="Holm", n_hypotheses=int(r.holm_family_size),
             p_raw=r.p_value, p_adjusted=r.p_holm)
        for r in frame.itertuples()
    ] + [
        dict(contrast_id=r.contrast_id, source_table="statistical_tests.csv", family_id=r.bh_family,
             family_definition=BH_FAMILY_DEFINITION, procedure="Benjamini-Hochberg", n_hypotheses=int(r.bh_family_size),
             p_raw=r.p_value, p_adjusted=r.p_fdr)
        for r in frame.itertuples() if r.confirmatory
    ] + [
        dict(contrast_id=r["contrast_id"], source_table="noskill_tests.csv", family_id=r["holm_family"],
             family_definition=NOSKILL_FAMILY_DEFINITION, procedure="Holm", n_hypotheses=r["holm_family_size"],
             p_raw=r["p_one_sided_seed"], p_adjusted=r["p_holm_seed"])
        for r in noskill
    ]
    pd.DataFrame(families).to_csv(OUTPUT_FAMILIES, index=False)
    print(f"  Saved: {OUTPUT_FAMILIES}  ({len(families)} rows)")

    for task_type, block in frame.groupby("task_type"):
        print_section_header(f"SUMMARY - {task_type}")
        for r in block.itertuples():
            flag = "*" if r.significance == "significant" else " "
            print(f"  {flag} {r.comparison:38s} est={r.estimate:+.4f} "
                  f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}] p={r.p_value:.3g} p_FDR={r.p_fdr:.3f} "
                  f"[{r.inference}/{r.lmm_status}] (n={r.n_obs}, tasks={r.n_tasks}, clusters={r.n_clusters})")

    print_section_header("NO-SKILL TESTS (classification, validation-selected configuration)")
    for r in noskill:
        flag = "*" if r["beats_reference"] else " "
        print(f"  {flag} {r['task']:34s} mean={r['mean']:.4f} ref={r['reference']:.2f} "
              f"seed p_Holm={r['p_holm_seed']:.4f}  test CI [{r['test_ci_lo']:.3f},{r['test_ci_hi']:.3f}] "
              f"excl={r['test_ci_excludes_ref']}  n_test={r['n_test']}")

    print_completion_message()


if __name__ == "__main__":
    with tee_stdout_to_log("statistical_tests.log"):
        main()
