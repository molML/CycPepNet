from __future__ import annotations
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycpepnet.paths import DATA_ROOT
from cycpepnet.benchmark import load_task_table
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT
from cycpepnet.analyze._results_config import ANNOTATION_SIZE, AX_FACE, EXPECTED_N_SEEDS, FIG_WIDTH_SUPP, MODEL_GROUP_MAP, NEUTRAL_LABEL, PANEL_LABEL_X, RESULTS_ROOT, SPINE_COLOR
from cycpepnet.analyze._results_data import main_pool
from cycpepnet.analyze._results_style import _polish_axis, _save_plot
from cycpepnet.utils import get_processed_dataset_dir, load_split
from cycpepnet.analyze._result_helpers import MAIN_CLS_TASKS, OUT_ROOT, PROPERTY_COLOR, TASK_BY_NAME, TASK_LABEL, TASK_RESPONSE_SCALE, add_panel_label, add_panel_title, save_source_table, save_table, task_property
from cycpepnet.analyze._figure_methods import configuration_representation

# Seed-level sensitivity/specificity: marker styling and numerical tolerances for the consistency checks
SENS_SPEC_SEED_SIZE        = 68

SENS_SPEC_HALO_SIZE        = 118

SENS_SPEC_HALO_ALPHA       = 0.20

SENS_SPEC_TICKS            = [0.0, 0.25, 0.5, 0.75, 1.0]

SENS_SPEC_BA_TOL           = 1e-9

SENS_SPEC_ROUNDED_TOL      = 5e-5 + 1e-9

SENS_SPEC_COUNT_TOL        = 1e-6

# Positive and negative counts in a task's test partition
def task_test_class_counts(task_name: str, split_method: str) -> tuple[int, int, int]:

    task     = TASK_BY_NAME[task_name]
    y        = load_task_table(task, data_root=DATA_ROOT)[0][task.target_column].to_numpy(dtype=float)
    split, _ = load_split(get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT) / "splits" / f"{task_name}_{split_method}.json")
    y_test   = y[split["test"]]

    return int(len(y_test)), int(np.sum(y_test == 1)), int(np.sum(y_test == 0))

# Validation-selected configuration per classification task, as recorded in learnability.csv
def learnability_selected_configurations(seed_df: pd.DataFrame, tasks: list[str]) -> dict[str, str]:

    learn = pd.read_csv(RESULTS_ROOT / "learnability.csv")
    learn = learn.loc[(learn["task_type"] == "classification") & (learn["metric"] == "balanced_acc")]

    pool = main_pool(
        seed_df.loc[
            (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
            & (seed_df["task_type"] == "classification")
            & (seed_df["model"] != "mean")
        ]
    )

    selected = {}
    for task in tasks:

        stored = learn.loc[learn["task"] == task]
        if len(stored) != 1:
            raise RuntimeError(f"learnability.csv has {len(stored)} balanced-accuracy rows for {task}")
        if stored["selected_by"].iloc[0] != "val_balanced_acc":
            raise RuntimeError(f"{task} was not selected on val_balanced_acc in learnability.csv")

        rederived = pool.loc[pool["task"] == task].groupby("model")["val_balanced_acc"].mean().idxmax()
        if rederived != stored["best_model"].iloc[0]:
            raise RuntimeError(
                f"{task}: learnability.csv selects {stored['best_model'].iloc[0]}, "
                f"but validation re-selection gives {rederived}"
            )

        selected[task] = rederived

    return selected

# Label rule of a classification task on its modelled response, e.g. "log10(Ki / nM) <= 2.000"
def class_rule(task) -> str:
    stem  = task.name.removesuffix("_cls") + "_reg"
    scale = re.sub(r"\$|\\mathrm|[{}_^]", "", TASK_RESPONSE_SCALE.get(stem, task.target_column))
    return f"{scale} {'>=' if task.classify_above else '<='} {task.classification_threshold:.3f}"

# Per-seed test sensitivity and specificity of the configurations selected on validation in learnability.csv
def sensitivity_specificity_seed_table(seed_df: pd.DataFrame) -> pd.DataFrame:

    learn    = pd.read_csv(RESULTS_ROOT / "learnability.csv")
    selected = learnability_selected_configurations(seed_df, MAIN_CLS_TASKS)
    rows     = []

    for task, model in selected.items():

        runs = seed_df.loc[
            (seed_df["task"] == task)
            & (seed_df["model"] == model)
            & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        ].sort_values("seed")

        if len(runs) != EXPECTED_N_SEEDS or runs["seed"].nunique() != EXPECTED_N_SEEDS or runs["rep"].nunique() != 1:
            raise RuntimeError(
                f"{task}/{model}: expected {EXPECTED_N_SEEDS} distinct seeds of one representation, "
                f"found {len(runs)} rows, seeds {runs['seed'].tolist()}, reps {runs['rep'].unique().tolist()}"
            )

        n_test, n_pos, n_neg = task_test_class_counts(task, CANONICAL_BUTINA_SPLIT)

        for run in runs.itertuples():

            if int(run.n_test) != n_test or int(getattr(run, "n_valid_test", n_test)) != n_test:
                raise RuntimeError(f"{task}/{model}/seed {run.seed}: stored n_test does not match the Butina test split")

            sensitivity = float(run.tpr)
            specificity = float(run.tnr)

            if np.isfinite(sensitivity) != (n_pos > 0) or np.isfinite(specificity) != (n_neg > 0):
                raise RuntimeError(f"{task}/{model}/seed {run.seed}: definedness of TPR/TNR contradicts the test class counts")

            recomputed_ba = (sensitivity + specificity) / 2.0
            stored_ba     = float(run.balanced_acc)
            if np.isfinite(recomputed_ba) and abs(recomputed_ba - stored_ba) > SENS_SPEC_BA_TOL:
                raise RuntimeError(f"{task}/{model}/seed {run.seed}: (TPR + TNR) / 2 = {recomputed_ba} != stored {stored_ba}")

            tp = sensitivity * n_pos if n_pos > 0 else np.nan
            tn = specificity * n_neg if n_neg > 0 else np.nan
            for name, count in (("TP", tp), ("TN", tn)):
                if np.isfinite(count) and abs(count - round(count)) > SENS_SPEC_COUNT_TOL:
                    raise RuntimeError(f"{task}/{model}/seed {run.seed}: implied {name} = {count} is not an integer")

            rows.append(dict(
                task                    = task,
                task_label              = TASK_LABEL[task],
                property                = task_property(task),
                configuration           = model,
                model_family            = MODEL_GROUP_MAP.get(model, "Other"),
                representation          = configuration_representation(model, run.rep)[1],
                rep_field               = run.rep,
                split_method            = CANONICAL_BUTINA_SPLIT,
                selected_by             = "val_balanced_acc",
                seed                    = int(run.seed),
                n_test                  = n_test,
                n_positive              = n_pos,
                n_negative              = n_neg,
                tp                      = round(tp) if np.isfinite(tp) else pd.NA,
                fn                      = n_pos - round(tp) if np.isfinite(tp) else pd.NA,
                tn                      = round(tn) if np.isfinite(tn) else pd.NA,
                fp                      = n_neg - round(tn) if np.isfinite(tn) else pd.NA,
                sensitivity             = sensitivity,
                specificity             = specificity,
                positive_class          = TASK_BY_NAME[task].class_labels[1],
                positive_class_rule     = class_rule(TASK_BY_NAME[task]),
                decision_threshold      = "predicted probability >= 0.5",
                sensitivity_defined     = bool(np.isfinite(sensitivity)),
                specificity_defined     = bool(np.isfinite(specificity)),
                balanced_acc_stored     = stored_ba,
                balanced_acc_from_rates = recomputed_ba,
            ))

    table = pd.DataFrame(rows)

    for task, group in table.groupby("task"):
        plotted = learn.loc[(learn["task"] == task) & (learn["metric"] == "balanced_acc"), "mean"].iloc[0]
        if abs(group["balanced_acc_stored"].mean() - plotted) > SENS_SPEC_ROUNDED_TOL:
            raise RuntimeError(f"{task}: seed-mean balanced accuracy differs from learnability.csv ({plotted})")

    return table

# Supplementary figure: per-seed specificity against sensitivity for the configuration selected on validation
def plot_sensitivity_specificity(seed_df: pd.DataFrame) -> None:

    if not (RESULTS_ROOT / "learnability.csv").exists():
        print("  Skipped sfig_sensitivity_specificity: results/learnability.csv is missing.")
        return

    table = sensitivity_specificity_seed_table(seed_df)
    tasks = MAIN_CLS_TASKS

    summary = (
        table.groupby(["task", "task_label", "property", "configuration", "model_family", "representation"], sort=False)
        .agg(
            n_seeds               = ("seed", "size"),
            n_test                = ("n_test", "first"),
            n_positive            = ("n_positive", "first"),
            n_negative            = ("n_negative", "first"),
            sensitivity_n_defined = ("sensitivity_defined", "sum"),
            sensitivity_mean      = ("sensitivity", "mean"),
            sensitivity_sd        = ("sensitivity", "std"),
            specificity_n_defined = ("specificity_defined", "sum"),
            specificity_mean      = ("specificity", "mean"),
            specificity_sd        = ("specificity", "std"),
            balanced_acc_mean     = ("balanced_acc_stored", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_SUPP, 5.40), sharex=True, sharey=True)
    axes      = axes.ravel()

    for index, (ax, task, panel) in enumerate(zip(axes, tasks, "abcdef")):

        group = table.loc[table["task"] == task].sort_values("seed")
        color = PROPERTY_COLOR.get(task_property(task), NEUTRAL_LABEL)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(SENS_SPEC_TICKS)
        ax.set_yticks(SENS_SPEC_TICKS)
        ax.set_xticklabels([f"{tick:.2f}" for tick in SENS_SPEC_TICKS])
        ax.set_yticklabels([f"{tick:.2f}" for tick in SENS_SPEC_TICKS])
        ax.set_aspect("equal", adjustable="box")
        ax.plot([0.0, 1.0], [0.0, 1.0], color=SPINE_COLOR, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)

        defined = group.loc[group["sensitivity_defined"] & group["specificity_defined"]]

        if len(defined) < len(group):
            print(f"  WARNING: {TASK_LABEL[task]}: {len(group) - len(defined)}/{len(group)} seeds have undefined sensitivity or specificity and are not drawn.")

        ax.scatter(
            defined["specificity"], defined["sensitivity"],
            s=SENS_SPEC_HALO_SIZE, color=color, edgecolor="none", alpha=SENS_SPEC_HALO_ALPHA,
            clip_on=False, zorder=2,
        )
        ax.scatter(
            defined["specificity"], defined["sensitivity"],
            s=SENS_SPEC_SEED_SIZE, color=color, edgecolor=AX_FACE, linewidth=1.3, alpha=0.92,
            clip_on=False, zorder=3,
        )

        add_panel_title(ax, TASK_LABEL[task], weight="bold")
        add_panel_label(ax, panel, x=PANEL_LABEL_X if index % 3 == 0 else None)
        _polish_axis(ax, "none")
        ax.tick_params(axis="both", pad=5.0)
        ax.tick_params(labelbottom=True)

    axes[4].set_xlabel("Specificity = TN / (TN + FP)")
    axes[0].set_ylabel("Sensitivity = TP / (TP + FN)")
    axes[0].yaxis.set_label_coords(-0.20, -0.12)

    property_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor=AX_FACE, markersize=8, label=prop)
        for prop, color in PROPERTY_COLOR.items()
    ]
    encoding_handles = [
        plt.Line2D([0], [0], color=SPINE_COLOR, linestyle=(0, (4, 3)), linewidth=0.8, label="Sensitivity = specificity"),
    ]
    fig.legend(
        handles=property_handles + encoding_handles,
        loc="lower center",
        ncol=len(property_handles) + len(encoding_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=ANNOTATION_SIZE,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.16, wspace=0.14, hspace=0.26)

    save_source_table(table, "sensitivity_specificity_per_seed.csv")
    save_table(summary, "sensitivity_specificity_summary.csv")
    _save_plot(fig, OUT_ROOT / "supplementary" / "sfig_sensitivity_specificity.png")
