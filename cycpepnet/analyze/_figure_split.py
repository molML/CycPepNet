from __future__ import annotations
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT
from cycpepnet.analyze._results_config import ANNOTATION_SIZE, ANNOTATION_SMALL_SIZE, AXIS_LABEL_SIZE, AX_FACE, FIG_WIDTH_FULL, NEUTRAL_LABEL, PANEL_LABEL_DY, PANEL_TITLE_DY, TEXT_COLOR
from cycpepnet.analyze._stats_core import _benjamini_hochberg
from cycpepnet.analyze._results_data import main_pool
from cycpepnet.analyze._results_style import _polish_axis, _save_plot, _two_decimal_ticks
from cycpepnet.analyze._result_helpers import MAIN_CLS_TASKS, MAIN_REG_TASKS, OUT_ROOT, PROPERTY_COLOR, RANDOM_SPLIT, SPLIT_BOX_COLOR, SPLIT_BOX_LINEWIDTH, SPLIT_MEDIAN_LINEWIDTH, SPLIT_METRIC_SCALE, SPLIT_OBSERVATION_ALPHA, SPLIT_OBSERVATION_SIZE, SPLIT_UNIT_INTERVAL_METRICS, TASK_LABEL, TASK_RESPONSE_SCALE, add_panel_label, add_panel_title, save_source_table, save_table, task_property
from cycpepnet.analyze._figure_learnability import add_figure_legend, draw_training_mean, metric_axis_limits, metric_tick_locator, training_mean_legend_handle, training_mean_values

# ---------------------------------------------------------------------------
# Figure 4: the effect of the split strategy, as full per-task distributions
# ---------------------------------------------------------------------------

SPLIT_BLOCKS = [
    ("regression",     "rmse",         r"Test RMSE $\downarrow$"),
    ("classification", "balanced_acc", r"Test balanced accuracy $\uparrow$"),
]

# Companion metrics of figure 4, drawn on the same paired observations
SPLIT_COMPANION_BLOCKS = [
    ("regression",     "qf3",   r"Test $Q^2_{\mathrm{F3}}$ $\uparrow$"),
    ("classification", "auroc", r"Test AUROC $\uparrow$"),
]

SPLIT_JITTER_SEED    = 20260917

SPLIT_JITTER_SPREAD  = 0.17

SPLIT_CATEGORY_LABEL_SIZE = ANNOTATION_SMALL_SIZE

# Paired Random / Butina observations behind figure 4, one row per configuration and seed
def split_effect_observations(seed_df: pd.DataFrame, blocks: list = SPLIT_BLOCKS) -> tuple[pd.DataFrame, pd.DataFrame]:

    paired: list[pd.DataFrame] = []
    excluded: list[dict] = []

    for task_type, metric, _ in blocks:

        tasks = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS

        candidate = seed_df.loc[
            (seed_df["task"].isin(tasks))
            & (seed_df["task_type"] == task_type)
            & (seed_df["split_method"].isin([CANONICAL_BUTINA_SPLIT, RANDOM_SPLIT]))
        ]

        for reason, mask in (
            ("baseline or non-benchmark family", candidate["family"].isin(["Baseline", "Other"])),
            ("outside the main pool: higher fingerprint radius or PhysChem augmentation",
             ~candidate["family"].isin(["Baseline", "Other"])
             & ~candidate.index.isin(main_pool(candidate).index)),
        ):
            for (model, rep), group in candidate.loc[mask].groupby(["model", "rep"], dropna=False):
                excluded.append(dict(task_type=task_type, model=model, rep=str(rep),
                                     n_rows=int(len(group)), reason=reason))

        work = main_pool(candidate.loc[~candidate["family"].isin(["Baseline", "Other"])])

        wide = work.pivot_table(
            index=["task", "model", "rep", "seed"],
            columns="split_method",
            values=metric,
            aggfunc="mean",
        )

        unpaired = wide.loc[wide[[CANONICAL_BUTINA_SPLIT, RANDOM_SPLIT]].isna().any(axis=1)]
        for row in unpaired.reset_index().itertuples():
            excluded.append(dict(
                task_type=task_type, model=row.model, rep=str(row.rep), n_rows=1,
                reason=f"{metric} missing or undefined under one split (task {row.task}, seed {row.seed})",
            ))
        if len(unpaired):
            print(f"  WARNING: {len(unpaired)} {task_type} observations are not paired across splits and are excluded.")

        wide = wide.dropna(subset=[CANONICAL_BUTINA_SPLIT, RANDOM_SPLIT]).reset_index()
        wide = wide.rename(columns={CANONICAL_BUTINA_SPLIT: "butina", RANDOM_SPLIT: "random"})
        wide["task_type"]      = task_type
        wide["metric"]         = metric
        wide["response_scale"] = (wide["task"].map(TASK_RESPONSE_SCALE) if metric == "rmse"
                                  else SPLIT_METRIC_SCALE[metric])
        paired.append(wide[["task_type", "metric", "task", "model", "rep", "seed",
                            "random", "butina", "response_scale"]])

    return pd.concat(paired, ignore_index=True), pd.DataFrame(excluded)

# Per task and split: median, quartiles and the observed extremes of those observations
def split_effect_summary(observations: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for (task_type, metric, task), group in observations.groupby(["task_type", "metric", "task"], sort=False):

        entry = dict(
            task_type = task_type,
            metric    = metric,
            task      = task,
            task_label = TASK_LABEL[task],
            response_scale = group["response_scale"].iloc[0],
            n_observations = int(len(group)),
            n_configurations = int(group.groupby(["model", "rep"]).ngroups),
            n_seeds = int(group["seed"].nunique()),
        )

        for split in ("random", "butina"):
            values = group[split].to_numpy(dtype=float)
            entry.update({
                f"{split}_min":    float(values.min()),
                f"{split}_q1":     float(np.percentile(values, 25)),
                f"{split}_median": float(np.median(values)),
                f"{split}_q3":     float(np.percentile(values, 75)),
                f"{split}_max":    float(values.max()),
            })

        entry["median_difference_butina_minus_random"] = entry["butina_median"] - entry["random_median"]
        rows.append(entry)

    return pd.DataFrame(rows)

# Legend entries shared by both split figures; the split itself is named on the x axis
def split_legend_handles() -> list:
    return [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor=AX_FACE, markersize=5.6, label=prop)
        for prop, color in PROPERTY_COLOR.items()
    ]

# Jittered horizontal position of every plotted observation, one row per observation and split
def split_effect_plotted_points(observations: pd.DataFrame, blocks: list = SPLIT_BLOCKS) -> pd.DataFrame:

    rng    = np.random.default_rng(SPLIT_JITTER_SEED)
    frames = []

    for task_type, metric, _ in blocks:
        tasks = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS
        for task in tasks:
            group = observations.loc[(observations["task_type"] == task_type) & (observations["task"] == task)]
            for position, split in enumerate(("random", "butina")):
                offsets = rng.uniform(-SPLIT_JITTER_SPREAD, SPLIT_JITTER_SPREAD, size=len(group))
                frames.append(group[["task_type", "metric", "task", "model", "rep", "seed"]].assign(
                    split=split, value=group[split].to_numpy(dtype=float),
                    category_position=position, x_plotted=position + offsets,
                ))

    return pd.concat(frames, ignore_index=True)

SPLIT_MARK_COLOR    = "#5F5A54"

SPLIT_BRACKET_DROP  = 0.045

# Significance label of an adjusted p value, or n.s. when it clears no threshold
def significance_label(p_adjusted: float) -> str:

    for threshold, stars in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p_adjusted < threshold:
            return stars

    return "n.s."

# Paired Wilcoxon of Butina against Random on the configuration means of each task,
# seeds averaged first, Benjamini-Hochberg adjusted across the panels of one figure
def split_effect_per_task_tests(observations: pd.DataFrame, blocks: list) -> pd.DataFrame:

    rows = []

    for task_type, metric, _ in blocks:

        tasks = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS

        for task in tasks:

            group  = observations.loc[(observations["task_type"] == task_type) & (observations["task"] == task)]
            paired = group.groupby(["model", "rep"])[["random", "butina"]].mean()

            statistic, p_value = wilcoxon(
                paired["butina"].to_numpy(dtype=float),
                paired["random"].to_numpy(dtype=float),
                alternative="two-sided",
            )

            rows.append(dict(
                task_type        = task_type,
                metric           = metric,
                task             = task,
                task_label       = TASK_LABEL[task],
                n_configurations = int(len(paired)),
                median_random    = float(paired["random"].median()),
                median_butina    = float(paired["butina"].median()),
                median_delta     = float((paired["butina"] - paired["random"]).median()),
                wilcoxon_statistic = float(statistic),
                p_value          = float(p_value),
            ))

    frame = pd.DataFrame(rows)
    frame["p_fdr"]       = _benjamini_hochberg(frame["p_value"].tolist())
    frame["label"]       = [significance_label(p) for p in frame["p_fdr"]]
    frame["family_size"] = len(frame)

    return frame

# The mark between the two boxes of one panel, drawn just under the top of the axis
def annotate_split_significance(ax: plt.Axes, label: str) -> None:

    if not label:
        return

    low, high = ax.get_ylim()
    span      = high - low
    y         = high - span * SPLIT_BRACKET_DROP
    drop      = span * 0.018

    ax.plot([0, 0, 1, 1], [y - drop, y, y, y - drop], color=SPLIT_MARK_COLOR,
            linewidth=0.7, clip_on=False, zorder=6)
    ax.text(0.5, y + span * 0.012, label, ha="center", va="bottom",
            fontsize=ANNOTATION_SMALL_SIZE, color=SPLIT_MARK_COLOR, clip_on=False, zorder=6)

# Six task columns of Random and Butina boxes over the individual observations, one row per block
def draw_split_distributions(observations: pd.DataFrame, points: pd.DataFrame, blocks: list,
                             tests: pd.DataFrame, top: float,
                             baselines: dict[str, dict] | None = None) -> plt.Figure:

    box_line = dict(color=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH)
    marks    = tests.set_index(["metric", "task"])["label"].to_dict()
    baseline = baselines or {}

    fig, axes = plt.subplots(2, len(MAIN_REG_TASKS), figsize=(FIG_WIDTH_FULL, 4.70))

    for row, (task_type, metric, axis_label) in enumerate(blocks):

        tasks = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS

        extremes = observations.loc[observations["task_type"] == task_type].assign(
            lo=lambda frame: frame[["random", "butina"]].min(axis=1),
            hi=lambda frame: frame[["random", "butina"]].max(axis=1),
        )

        limits = metric_axis_limits(extremes, tasks, metric, floor_zero=(metric == "rmse"), columns=["lo", "hi"])

        for column, task in enumerate(tasks):

            ax    = axes[row][column]
            group = observations.loc[(observations["task_type"] == task_type) & (observations["task"] == task)]
            drawn = points.loc[(points["task_type"] == task_type) & (points["task"] == task)]

            ax.scatter(drawn["x_plotted"], drawn["value"], s=SPLIT_OBSERVATION_SIZE,
                       color=PROPERTY_COLOR.get(task_property(task), NEUTRAL_LABEL),
                       edgecolor="none", alpha=SPLIT_OBSERVATION_ALPHA, zorder=2)

            ax.boxplot(
                [group["random"].to_numpy(dtype=float), group["butina"].to_numpy(dtype=float)],
                positions=[0, 1], widths=0.56, vert=True,
                showfliers=False,
                whis=(0, 100),
                medianprops=dict(color=TEXT_COLOR, linewidth=SPLIT_MEDIAN_LINEWIDTH, solid_capstyle="butt"),
                boxprops=dict(facecolor="none", edgecolor=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH),
                whiskerprops=box_line, capprops=box_line,
                patch_artist=True, zorder=4,
            )

            draw_training_mean(ax, metric, {
                position: baseline[metric][(task, split_method)]
                for position, split_method in enumerate([RANDOM_SPLIT, CANONICAL_BUTINA_SPLIT])
                if (task, split_method) in baseline.get(metric, {})
            })

            ax.set_xlim(-0.58, 1.58)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Random", "Butina"])

            if metric in SPLIT_UNIT_INTERVAL_METRICS:
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            else:
                ax.set_ylim(*limits[task_property(task)])
                ax.yaxis.set_major_locator(metric_tick_locator(metric))
            _two_decimal_ticks(ax.yaxis)

            _polish_axis(ax, "none")
            ax.tick_params(axis="x", labelsize=SPLIT_CATEGORY_LABEL_SIZE, length=0, pad=2.5)

            annotate_split_significance(ax, marks.get((metric, task), ""))

            if row == 0:
                add_panel_title(ax, TASK_LABEL[task], dy=PANEL_TITLE_DY + 1.0, weight="bold")
            if column == 0:
                ax.set_ylabel(axis_label, fontsize=AXIS_LABEL_SIZE - 0.6)
                add_panel_label(ax, "ab"[row], dy=PANEL_LABEL_DY + (14.0 if row == 0 else 4.0))

    add_figure_legend(fig, split_legend_handles() + [training_mean_legend_handle()],
                      loc="lower center", bbox_to_anchor=(0.5, 0.0),
                      ncol=4, frameon=False, fontsize=ANNOTATION_SIZE)
    fig.subplots_adjust(left=0.098, right=0.995, top=top, bottom=0.135, wspace=0.52, hspace=0.42)

    return fig

# Figure 4: Random against Butina for every task, as boxes over the individual observations
def plot_random_vs_butina(seed_df: pd.DataFrame) -> None:

    observations, excluded = split_effect_observations(seed_df)
    summary = split_effect_summary(observations)
    points  = split_effect_plotted_points(observations)

    save_source_table(observations, "split_effect_paired_observations.csv")
    save_source_table(excluded, "split_effect_exclusions.csv")
    save_source_table(points, "split_effect_plotted_points.csv")
    save_table(summary, "split_effect_distribution_statistics.csv")

    tests = split_effect_per_task_tests(observations, SPLIT_BLOCKS)
    save_table(tests, "split_effect_per_task_tests.csv")

    baselines = {metric: training_mean_values(seed_df, metric) for _, metric, _ in SPLIT_BLOCKS}

    fig = draw_split_distributions(observations, points, SPLIT_BLOCKS, tests, top=0.875,
                                   baselines=baselines)
    _save_plot(fig, OUT_ROOT / "main" / "fig_random_vs_butina.png", align_labels=False)

# Each (task, configuration, seed) of Figure 4 against the companion pool, per task
def split_companion_pool_comparison(primary: pd.DataFrame, companion: pd.DataFrame) -> pd.DataFrame:

    rows = []
    keys = ["model", "rep", "seed"]

    for task_type, tasks in (("regression", MAIN_REG_TASKS), ("classification", MAIN_CLS_TASKS)):
        for task in tasks:
            a = primary.loc[(primary["task_type"] == task_type) & (primary["task"] == task), keys]
            b = companion.loc[(companion["task_type"] == task_type) & (companion["task"] == task), keys]
            a_keys = set(map(tuple, a.astype(str).to_numpy()))
            b_keys = set(map(tuple, b.astype(str).to_numpy()))
            rows.append(dict(
                task_type=task_type, task=task, task_label=TASK_LABEL[task],
                n_figure4=len(a_keys), n_companion=len(b_keys),
                n_lost=len(a_keys - b_keys), n_added=len(b_keys - a_keys),
                lost=";".join("|".join(k) for k in sorted(a_keys - b_keys)),
                added=";".join("|".join(k) for k in sorted(b_keys - a_keys)),
            ))

    return pd.DataFrame(rows)

# Supplementary companion of figure 4: the same layout and observations on Q2F3 and AUROC
def plot_random_vs_butina_companion(seed_df: pd.DataFrame) -> None:

    primary, _ = split_effect_observations(seed_df)
    observations, excluded = split_effect_observations(seed_df, SPLIT_COMPANION_BLOCKS)
    summary = split_effect_summary(observations)
    points  = split_effect_plotted_points(observations, SPLIT_COMPANION_BLOCKS)

    save_source_table(observations, "split_effect_companion_paired_observations.csv")
    save_source_table(excluded, "split_effect_companion_exclusions.csv")
    save_source_table(points, "split_effect_companion_plotted_points.csv")
    save_table(summary, "split_effect_companion_distribution_statistics.csv")
    save_table(split_companion_pool_comparison(primary, observations), "split_effect_companion_pool_comparison.csv")

    tests = split_effect_per_task_tests(observations, SPLIT_COMPANION_BLOCKS)
    save_table(tests, "split_effect_companion_per_task_tests.csv")

    baselines = {metric: training_mean_values(seed_df, metric) for _, metric, _ in SPLIT_COMPANION_BLOCKS}

    fig = draw_split_distributions(observations, points, SPLIT_COMPANION_BLOCKS, tests, top=0.895,
                                   baselines=baselines)
    _save_plot(fig, OUT_ROOT / "supplementary" / "fig_random_vs_butina_companion.png", align_labels=False)
