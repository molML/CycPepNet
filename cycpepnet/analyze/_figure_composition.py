from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT
from cycpepnet.analyze._results_config import ANNOTATION_SIZE, FIG_WIDTH_FULL, FIG_WIDTH_SUPP, PANEL_LABEL_X, TEXT_COLOR
from cycpepnet.analyze._results_data import main_pool
from cycpepnet.analyze._results_style import _polish_axis, _save_plot, _two_decimal_ticks
from cycpepnet.analyze._result_helpers import METRIC_HIGHER_IS_BETTER, OUT_ROOT, PROPERTY_COLOR, SPLIT_BOX_COLOR, SPLIT_BOX_LINEWIDTH, SPLIT_DIMENSIONLESS_METRICS, SPLIT_MEDIAN_LINEWIDTH, SPLIT_OBSERVATION_ALPHA, SPLIT_OBSERVATION_SIZE, SPLIT_UNIT_INTERVAL_METRICS, UNIT_INTERVAL_TICKS, add_panel_label, add_panel_title, save_source_table, save_table, task_property
from cycpepnet.analyze._figure_learnability import add_figure_legend, draw_training_mean, metric_tick_locator, property_axis_limits, training_mean_legend_handle, training_mean_values

# Calculate paired task comparison deltas
def paired_task_comparison_deltas(
    seed_df: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    task_type: str,
    metric: str,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    rows = []
    work = seed_df.loc[
        (seed_df["task_type"] == task_type)
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (~seed_df["family"].isin(["Baseline", "Other"]))
    ].copy()
    work = main_pool(work)

    for label, task_a, task_b in comparisons:
        sub = work.loc[work["task"].isin([task_a, task_b]), ["task", "model", "rep", "seed", metric]].copy()
        wide = sub.pivot_table(index=["model", "rep", "seed"], columns="task", values=metric, aggfunc="mean")
        if task_a not in wide.columns or task_b not in wide.columns:
            continue
        wide = wide.dropna(subset=[task_a, task_b]).reset_index()
        wide["delta"] = wide[task_a] - wide[task_b] if higher_is_better else wide[task_b] - wide[task_a]
        wide["comparison"] = label
        wide["task_a"] = task_a
        wide["task_b"] = task_b
        rows.append(wide)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

COMPOSITION_COMPARISONS_REG: list[tuple[str, str, str]] = [
    ("CycPeptMPDB full vs merged", "cycpeptmpdb_reg", "cycpeptmpdb_npmmpd_reg"),
    ("NPMMPD full vs merged", "npmmpd_reg", "cycpeptmpdb_npmmpd_reg"),
    ("Townsend vs CycPeptMPDB full", "cycpeptmpdb_townsend_reg", "cycpeptmpdb_reg"),
    ("Rzepiela vs NPMMPD full", "npmmpd_rzepiela_reg", "npmmpd_reg"),
    ("PEPlife2 vs half-life merge", "half_life_peplife2_reg", "peplife2_pepmsnd_peptherdia_reg"),
]

COMPOSITION_COMPARISONS_CLS: list[tuple[str, str, str]] = [
    ("CycPeptMPDB full vs merged", "cycpeptmpdb_cls", "cycpeptmpdb_npmmpd_cls"),
    ("NPMMPD full vs merged", "npmmpd_cls", "cycpeptmpdb_npmmpd_cls"),
    ("Townsend vs CycPeptMPDB full", "cycpeptmpdb_townsend_cls", "cycpeptmpdb_cls"),
    ("Rzepiela vs NPMMPD full", "npmmpd_rzepiela_cls", "npmmpd_cls"),
    ("PEPlife2 vs half-life merge", "half_life_peplife2_cls", "peplife2_pepmsnd_peptherdia_cls"),
]

# Export permeability composition deltas
def export_permeability_composition_deltas(seed_df: pd.DataFrame) -> None:
    all_deltas = []
    for task_type, comparisons, metric in [
        ("regression", COMPOSITION_COMPARISONS_REG, "rmse"),
        ("classification", COMPOSITION_COMPARISONS_CLS, "balanced_acc"),
    ]:
        higher_is_better = METRIC_HIGHER_IS_BETTER[metric]
        deltas = paired_task_comparison_deltas(seed_df, comparisons, task_type, metric, higher_is_better=higher_is_better)
        all_deltas.append(deltas.assign(metric=metric, higher_is_better=higher_is_better))
    save_source_table(pd.concat(all_deltas, ignore_index=True), "dataset_composition_paired_deltas.csv")

# Composition deltas scaled by each level's own training mean
def export_composition_relative_to_training_mean(seed_df: pd.DataFrame) -> None:

    baseline = training_mean_values(seed_df, "rmse")
    work     = main_pool(
        seed_df.loc[
            (seed_df["task_type"] == "regression")
            & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
            & (~seed_df["family"].isin(["Baseline", "Other"]))
        ]
    )

    rows = []
    for label, task_a, task_b in COMPOSITION_COMPARISONS_REG:

        missing = [t for t in (task_a, task_b) if (t, CANONICAL_BUTINA_SPLIT) not in baseline]
        if missing:
            print(f"  WARNING: no training-mean RMSE for {', '.join(missing)}; {label} is skipped.")
            continue

        denominator_a = baseline[(task_a, CANONICAL_BUTINA_SPLIT)]
        denominator_b = baseline[(task_b, CANONICAL_BUTINA_SPLIT)]

        wide = (
            work.loc[work["task"].isin([task_a, task_b]), ["task", "model", "rep", "seed", "rmse"]]
            .pivot_table(index=["model", "rep", "seed"], columns="task", values="rmse", aggfunc="mean")
            .dropna(subset=[task_a, task_b])
            .reset_index()
        )

        ratio_a    = wide[task_a] / denominator_a
        ratio_b    = wide[task_b] / denominator_b
        difference = ratio_b - ratio_a

        rows.append(dict(
            analysis               = "composition_effect_relative",
            task_type              = "regression",
            metric                 = "rmse_relative",
            comparison             = label,
            task_a                 = task_a,
            task_b                 = task_b,
            direction              = f"{task_b} - {task_a}",
            delta_kind             = "paired difference of ratios, level-specific denominators",
            formula                = "RMSE_config,b / RMSE_mean,b - RMSE_config,a / RMSE_mean,a",
            n_obs                  = int(len(wide)),
            n_configurations       = int(wide.groupby(["model", "rep"]).ngroups),
            n_seeds                = int(wide["seed"].nunique()),
            mean_of_differences    = float(difference.mean()),
            median_of_differences  = float(difference.median()),
            q1_of_differences      = float(difference.quantile(0.25)),
            q3_of_differences      = float(difference.quantile(0.75)),
            n_positive             = int((difference > 0).sum()),
            n_negative             = int((difference < 0).sum()),
            n_zero                 = int((difference == 0).sum()),
            median_ratio_a         = float(ratio_a.median()),
            median_ratio_b         = float(ratio_b.median()),
            rmse_training_mean_a   = denominator_a,
            rmse_training_mean_b   = denominator_b,
        ))

    save_table(pd.DataFrame(rows), "dataset_composition_relative_to_training_mean.csv")

COMPOSITION_LINEAGES = [
    ("CycPeptMPDB", "Permeability", [
        ("Townsend 2020", "cycpeptmpdb_townsend"),
        ("CycPeptMPDB", "cycpeptmpdb"),
        ("CycPeptMPDB\n+ NPMMPD", "cycpeptmpdb_npmmpd"),
    ]),
    ("NPMMPD", "Permeability", [
        ("Rzepiela 2022", "npmmpd_rzepiela"),
        ("NPMMPD", "npmmpd"),
        ("NPMMPD\n+ CycPeptMPDB", "cycpeptmpdb_npmmpd"),
    ]),
    ("Half-life", "Half-life", [
        ("PEPlife2", "half_life_peplife2"),
        ("PEPlife2\n+ PepMSND\n+ PepTherDia", "peplife2_pepmsnd_peptherdia"),
    ]),
]

COMPOSITION_JITTER_SEED = 20260924

COMPOSITION_JITTER      = 0.20

COMPOSITION_BOX_WIDTH   = 0.56

# Every configuration and seed a composition figure draws for one metric, on the canonical Butina split
def composition_pool(seed_df: pd.DataFrame, task_type: str, metric: str) -> pd.DataFrame:

    work = seed_df.loc[
        (seed_df["task_type"] == task_type)
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (~seed_df["family"].isin(["Baseline", "Other"]))
    ]

    return main_pool(work).dropna(subset=[metric])

# Single-source against pooled datasets: one box per dataset level over every configuration and seed
def draw_composition_figure(
    seed_df: pd.DataFrame,
    rows_spec: list[tuple[str, str, str]],
    stem: str,
    table_stem: str,
) -> None:

    rng      = np.random.default_rng(COMPOSITION_JITTER_SEED)
    n_cols   = len(COMPOSITION_LINEAGES)
    box_line = dict(color=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH)
    baseline = {metric: training_mean_values(seed_df, metric) for _, metric, _ in rows_spec}

    fig, axes = plt.subplots(len(rows_spec), n_cols, figsize=(FIG_WIDTH_SUPP, 4.35))
    summary: list[dict]           = []
    points:  list[pd.DataFrame]   = []

    for row_idx, (task_type, metric, xlabel) in enumerate(rows_spec):

        work   = composition_pool(seed_df, task_type, metric)
        suffix = "reg" if task_type == "regression" else "cls"

        panels = [
            (title, prop, [
                (label, f"{base}_{suffix}", work.loc[work["task"] == f"{base}_{suffix}", metric].to_numpy(dtype=float))
                for label, base in items
            ])
            for title, prop, items in COMPOSITION_LINEAGES
        ]

        extremes = pd.DataFrame([
            dict(task=task, lo=float(values.min()), hi=float(values.max()))
            for _, _, levels in panels for _, task, values in levels if len(values)
        ])
        limits = (
            None if metric in SPLIT_UNIT_INTERVAL_METRICS
            else property_axis_limits(extremes, list(extremes["task"]),
                                      floor_zero=(metric == "rmse"), columns=["lo", "hi"])
        )

        for col_idx, (title, prop, levels) in enumerate(panels):

            ax = axes[row_idx][col_idx]

            if any(len(values) == 0 for _, _, values in levels):
                ax.set_axis_off()
                continue

            color = PROPERTY_COLOR[prop]

            for idx, (label, task, values) in enumerate(levels):

                offsets = rng.uniform(-COMPOSITION_JITTER, COMPOSITION_JITTER, size=len(values))
                ax.scatter(values, idx + offsets, s=SPLIT_OBSERVATION_SIZE, color=color,
                           edgecolor="none", alpha=SPLIT_OBSERVATION_ALPHA, zorder=2)
                ax.boxplot(
                    values, positions=[idx], widths=COMPOSITION_BOX_WIDTH, orientation="horizontal",
                    showfliers=False, whis=(0, 100),
                    medianprops=dict(color=TEXT_COLOR, linewidth=SPLIT_MEDIAN_LINEWIDTH, solid_capstyle="butt"),
                    boxprops=dict(facecolor="none", edgecolor=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH),
                    whiskerprops=box_line, capprops=box_line,
                    patch_artist=True, zorder=4,
                )

                q1, median, q3 = np.percentile(values, [25, 50, 75])
                summary.append(dict(
                    lineage=title, level=label.replace("\n", " "), task=task, metric=metric,
                    n_observations=int(len(values)), minimum=float(values.min()), q1=float(q1),
                    median=float(median), q3=float(q3), maximum=float(values.max()),
                ))
                points.append(pd.DataFrame(dict(
                    lineage=title, level=label.replace("\n", " "), task=task, metric=metric,
                    value=values, y_plotted=idx + offsets,
                )))

            draw_training_mean(ax, metric, {
                idx: baseline[metric][(task, CANONICAL_BUTINA_SPLIT)]
                for idx, (_, task, _) in enumerate(levels)
                if (task, CANONICAL_BUTINA_SPLIT) in baseline.get(metric, {})
            }, horizontal=True)

            ax.set_xlim(*(0.0, 1.0) if limits is None else limits[task_property(levels[0][1])])
            if limits is None:
                ax.set_xticks(UNIT_INTERVAL_TICKS)
            elif metric in SPLIT_DIMENSIONLESS_METRICS:
                ax.xaxis.set_major_locator(metric_tick_locator(metric))
            _two_decimal_ticks(ax.xaxis)

            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([label for label, _, _ in levels])
            ax.set_ylim(-0.6, len(levels) - 0.4)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)

            if row_idx == 0:
                add_panel_title(ax, title, weight="bold")

            add_panel_label(ax, "abcdef"[row_idx * n_cols + col_idx], x=PANEL_LABEL_X if col_idx == 0 else None)
            _polish_axis(ax, "none")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PROPERTY_COLOR[prop],
                   markeredgecolor="none", alpha=SPLIT_OBSERVATION_ALPHA + 0.25, markersize=4.6, label=prop)
        for prop in dict.fromkeys(prop for _, prop, _ in COMPOSITION_LINEAGES)
    ]
    handles.append(training_mean_legend_handle())

    add_figure_legend(fig, handles, loc="lower center", ncol=len(handles), frameon=False,
                      bbox_to_anchor=(0.5, 0.015), fontsize=ANNOTATION_SIZE)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.16, wspace=0.72, hspace=0.42)

    save_table(pd.DataFrame(summary), f"{table_stem}_plotted_values.csv")
    save_source_table(pd.concat(points, ignore_index=True), f"{table_stem}_plotted_points.csv")
    _save_plot(fig, OUT_ROOT / "supplementary" / f"{stem}.png")

# Supplementary figure 7 compares single-source and pooled datasets
def plot_permeability_composition_lineage(seed_df: pd.DataFrame) -> None:
    draw_composition_figure(
        seed_df,
        [
            ("regression",     "rmse",         r"Test RMSE $\downarrow$"),
            ("classification", "balanced_acc", r"Test balanced accuracy $\uparrow$"),
        ],
        "sfig_dataset_composition",
        "dataset_composition",
    )

# Supplementary figure 8: the same comparison on Q2F3 and AUROC
def plot_permeability_composition_lineage_companion(seed_df: pd.DataFrame) -> None:
    draw_composition_figure(
        seed_df,
        [
            ("regression",     "qf3",   r"Test $Q^2_{\mathrm{F3}}$ $\uparrow$"),
            ("classification", "auroc", r"Test AUROC $\uparrow$"),
        ],
        "sfig_dataset_composition_companion",
        "dataset_composition_companion",
    )

    comparisons_reg = [
        ("CycPeptMPDB full vs merged", "cycpeptmpdb_reg", "cycpeptmpdb_npmmpd_reg"),
        ("NPMMPD full vs merged", "npmmpd_reg", "cycpeptmpdb_npmmpd_reg"),
        ("Townsend vs CycPeptMPDB full", "cycpeptmpdb_townsend_reg", "cycpeptmpdb_reg"),
        ("Rzepiela vs NPMMPD full", "npmmpd_rzepiela_reg", "npmmpd_reg"),
        ("PEPlife2 vs half-life merge", "half_life_peplife2_reg", "peplife2_pepmsnd_peptherdia_reg"),
    ]
    comparisons_cls = [
        ("CycPeptMPDB full vs merged", "cycpeptmpdb_cls", "cycpeptmpdb_npmmpd_cls"),
        ("NPMMPD full vs merged", "npmmpd_cls", "cycpeptmpdb_npmmpd_cls"),
        ("Townsend vs CycPeptMPDB full", "cycpeptmpdb_townsend_cls", "cycpeptmpdb_cls"),
        ("Rzepiela vs NPMMPD full", "npmmpd_rzepiela_cls", "npmmpd_cls"),
        ("PEPlife2 vs half-life merge", "half_life_peplife2_cls", "peplife2_pepmsnd_peptherdia_cls"),
    ]
    source_rows = []
    for task_type, comparisons, metric in [
        ("regression", comparisons_reg, "qf3"),
        ("classification", comparisons_cls, "auroc"),
    ]:
        higher_is_better = METRIC_HIGHER_IS_BETTER[metric]
        deltas = paired_task_comparison_deltas(
            seed_df,
            comparisons,
            task_type,
            metric,
            higher_is_better=higher_is_better,
        )
        source_rows.append(
            deltas.assign(metric=metric, higher_is_better=higher_is_better)
        )
    save_source_table(
        pd.concat(source_rows, ignore_index=True),
        "dataset_composition_paired_deltas_companion_metrics.csv",
    )

# Main-text permeability panels under the split figure
def plot_permeability_composition_panels(seed_df: pd.DataFrame) -> None:

    rng      = np.random.default_rng(COMPOSITION_JITTER_SEED)
    work     = composition_pool(seed_df, "regression", "rmse")
    baseline = training_mean_values(seed_df, "rmse")
    box_line = dict(color=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH)
    lineages = [lineage for lineage in COMPOSITION_LINEAGES if lineage[1] == "Permeability"]

    values = {
        base: work.loc[work["task"] == f"{base}_reg", "rmse"].to_numpy(dtype=float)
        for _, _, levels in lineages for _, base in levels
    }
    marks  = [baseline[(f"{base}_reg", CANONICAL_BUTINA_SPLIT)] for base in values]
    spread = np.concatenate(list(values.values()) + [np.array(marks)])
    limits = (np.floor(spread.min() * 10) / 10 - 0.02, np.ceil(spread.max() * 10) / 10 + 0.02)

    fig, axes = plt.subplots(1, len(lineages), figsize=(FIG_WIDTH_FULL, 2.45), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.86, bottom=0.3, wspace=0.55)

    for ax, (title, prop, levels), letter in zip(axes, lineages, "cd"):

        for idx, (_, base) in enumerate(levels):

            points = values[base]
            ax.scatter(points, idx + rng.uniform(-COMPOSITION_JITTER, COMPOSITION_JITTER, len(points)),
                       s=SPLIT_OBSERVATION_SIZE, color=PROPERTY_COLOR[prop], edgecolor="none",
                       alpha=SPLIT_OBSERVATION_ALPHA, zorder=2)
            ax.boxplot(
                points, positions=[idx], widths=COMPOSITION_BOX_WIDTH, orientation="horizontal",
                showfliers=False, whis=(0, 100),
                medianprops=dict(color=TEXT_COLOR, linewidth=SPLIT_MEDIAN_LINEWIDTH, solid_capstyle="butt"),
                boxprops=dict(facecolor="none", edgecolor=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH),
                whiskerprops=box_line, capprops=box_line,
                patch_artist=True, zorder=4,
            )

        draw_training_mean(ax, "rmse", {
            idx: baseline[(f"{base}_reg", CANONICAL_BUTINA_SPLIT)] for idx, (_, base) in enumerate(levels)
        }, horizontal=True)

        ax.set_xlim(*limits)
        _two_decimal_ticks(ax.xaxis)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([label for label, _ in levels])
        ax.set_ylim(-0.6, len(levels) - 0.4)
        ax.invert_yaxis()
        ax.set_xlabel(r"Test RMSE $\downarrow$")
        add_panel_title(ax, title, weight="bold")
        add_panel_label(ax, letter)
        _polish_axis(ax, "none")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PROPERTY_COLOR["Permeability"],
                   markeredgecolor="none", alpha=SPLIT_OBSERVATION_ALPHA + 0.25, markersize=4.6,
                   label="One configuration × seed"),
        training_mean_legend_handle()[0],
    ]
    add_figure_legend(fig, handles, loc="lower center", ncol=len(handles), frameon=False,
                      bbox_to_anchor=(0.5, 0.0), fontsize=ANNOTATION_SIZE)

    _save_plot(fig, OUT_ROOT / "main" / "fig_composition_permeability.png")
