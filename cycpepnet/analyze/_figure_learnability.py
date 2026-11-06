from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from cycpepnet.paths import DATA_ROOT
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT
from cycpepnet.analyze._results_config import ANNOTATION_SIZE, ANNOTATION_SMALL_SIZE, AXIS_LABEL_SIZE, AX_FACE, BASELINE_LINESTYLE, BASELINE_LINEWIDTH, FIG_WIDTH_FULL, NEUTRAL_LABEL, PANEL_LABEL_DY, PANEL_TITLE_DY, RESULTS_ROOT, TEXT_COLOR
from cycpepnet.analyze._results_data import main_pool
from cycpepnet.analyze.analyze_uncertainty import test_sampling_ci
from cycpepnet.analyze._results_style import _polish_axis, _save_plot, _two_decimal_ticks
from cycpepnet.utils import get_processed_dataset_dir, load_split
from cycpepnet.analyze._result_helpers import MAIN_CLS_TASKS, MAIN_REG_TASKS, OUT_ROOT, PROPERTY_COLOR, SPLIT_BOX_COLOR, SPLIT_BOX_LINEWIDTH, SPLIT_DIMENSIONLESS_METRICS, SPLIT_MEDIAN_LINEWIDTH, SPLIT_METRIC_SCALE, SPLIT_OBSERVATION_ALPHA, SPLIT_OBSERVATION_SIZE, SPLIT_UNIT_INTERVAL_METRICS, TASK_BY_NAME, TASK_LABEL, TASK_RESPONSE_SCALE, add_panel_label, add_panel_title, fit_xtick_labels, save_source_table, save_table, task_property

SELECTED_LABEL      = "Selected model"

TRAINING_MEAN_LABEL = "Training-mean predictor"

NO_SKILL_LABEL      = "No-skill model"

TRAINING_MEAN_COLOR = NEUTRAL_LABEL

# The training-mean predictor is deterministic, so its per-seed RMSE must not move at all
TRAINING_MEAN_RMSE_TOL = 1e-9

# Test score of the training-mean predictor on the canonical Butina test split
def training_mean_baseline(seed_df: pd.DataFrame, metric: str = "rmse") -> dict[str, dict]:

    work = seed_df.loc[
        (seed_df["model"] == "mean")
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (seed_df["task"].isin(MAIN_REG_TASKS))
    ]

    baselines: dict[str, dict] = {}

    for task, group in work.groupby("task"):

        values = pd.to_numeric(group[metric], errors="coerce").dropna()

        if values.empty:
            print(f"  WARNING: no training-mean {metric} stored for {task}; baseline omitted.")
            continue

        if float(values.max() - values.min()) > TRAINING_MEAN_RMSE_TOL:
            raise RuntimeError(
                f"{task}: training-mean {metric} varies across seeds "
                f"({values.min()} to {values.max()}), but MeanPredictor is deterministic"
            )

        value  = float(values.iloc[0])
        n_test = int(group["n_test"].iloc[0])
        lo, hi = test_sampling_ci(metric, value, n_test)

        baselines[task] = dict(
            value      = value,
            n_test     = n_test,
            n_runs     = int(len(values)),
            test_ci_lo = lo,
            test_ci_hi = hi,
        )

    return baselines

# Content hashes a run stores for the data and partition it was scored on
SPLIT_PROVENANCE_HASHES = ["split_sha256", "split_smiles_sha256", "target_sha256", "dataset_sha256"]

# Check that a task's selected model and its training-mean baseline were scored on one test set
def verify_shared_test_partition(seed_df: pd.DataFrame, task_name: str, model_key: str) -> dict:

    runs = seed_df.loc[
        (seed_df["task"] == task_name)
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (seed_df["model"].isin([model_key, "mean"]))
    ]

    selected = runs.loc[runs["model"] == model_key]
    baseline = runs.loc[runs["model"] == "mean"]

    if selected.empty or baseline.empty:
        raise RuntimeError(f"{task_name}: missing runs for {model_key} or the training-mean baseline")

    compared = []

    for column in SPLIT_PROVENANCE_HASHES:

        if column not in runs.columns:
            continue

        selected_values = set(selected[column].dropna())
        baseline_values = set(baseline[column].dropna())

        if not selected_values or not baseline_values:
            continue

        if selected_values != baseline_values:
            raise RuntimeError(
                f"{task_name}: {model_key} and the training-mean baseline disagree on {column}"
            )

        compared.append(column)

    task = TASK_BY_NAME[task_name]
    path = get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT) / "splits" / f"{task_name}_{CANONICAL_BUTINA_SPLIT}.json"

    if not path.exists():
        raise RuntimeError(f"{task_name}: committed split file {path} is missing")

    split, _   = load_split(path)
    n_test     = len(split["test"])
    recorded   = set(pd.to_numeric(runs["n_test"], errors="coerce").dropna().astype(int))

    if recorded != {n_test}:
        raise RuntimeError(
            f"{task_name}: split file holds {n_test} test molecules but runs recorded {sorted(recorded)}"
        )

    return dict(
        split_file      = str(path),
        n_test          = n_test,
        hashes_compared = ",".join(compared) if compared else "none stored",
        evidence        = "identical content hashes" if compared else "shared committed split file and matching n_test",
    )

# Metrics drawn by the learnability figure and its companion
LEARNABILITY_VIEWS = {
    "primary": dict(
        metrics              = {"regression": "rmse", "classification": "balanced_acc"},
        stem                 = "fig_task_learnability",
        folder               = "main",
        table_stem           = "task_learnability",
        regression_label     = r"Test RMSE $\downarrow$",
        classification_label = r"Test balanced accuracy $\uparrow$",
        floor_zero           = True,
    ),
    "companion": dict(
        metrics              = {"regression": "qf3", "classification": "auroc"},
        stem                 = "fig_task_learnability_companion",
        folder               = "supplementary",
        table_stem           = "task_learnability_companion",
        regression_label     = r"Test $Q^2_{\mathrm{F3}}$ $\uparrow$",
        classification_label = r"Test AUROC $\uparrow$",
        floor_zero           = False,
    ),
}

# The criterion results/learnability.csv records in its selected_by column
LEARNABILITY_SELECTION = {"regression": "val_rmse", "classification": "val_balanced_acc"}

METRIC_TEXT = {"rmse": "RMSE", "qf3": "Q2F3", "balanced_acc": "balanced_acc", "auroc": "auroc"}

# Every value drawn in the interval-style learnability figure or its companion, as one tidy row per plotted point
def learnability_figure_data(seed_df: pd.DataFrame, view: str = "primary") -> pd.DataFrame:

    learn   = pd.read_csv(RESULTS_ROOT / "learnability.csv")
    metrics = LEARNABILITY_VIEWS[view]["metrics"]

    for col in ["mean", "seed_ci_lo", "seed_ci_hi", "test_ci_lo", "test_ci_hi"]:
        if col in learn.columns:
            learn[col] = pd.to_numeric(learn[col], errors="coerce")

    pool = main_pool(
        seed_df.loc[
            (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
            & (seed_df["model"] != "mean")
        ]
    )

    baselines = training_mean_baseline(seed_df, metrics["regression"])
    rows: list[dict] = []

    specs = [
        ("regression",     MAIN_REG_TASKS, True),
        ("classification", MAIN_CLS_TASKS, False),
    ]

    for task_type, tasks, lower_is_better in specs:

        metric     = metrics[task_type]
        val_metric = LEARNABILITY_SELECTION[task_type]

        for task in tasks:

            stored = learn.loc[(learn["task"] == task) & (learn["metric"] == metric)]

            if len(stored) != 1:
                raise RuntimeError(f"learnability.csv has {len(stored)} {metric} rows for {task}")

            stored = stored.iloc[0]

            selection  = pool.loc[pool["task"] == task].groupby("model")[val_metric].mean()
            rederived  = selection.idxmin() if lower_is_better else selection.idxmax()

            if rederived != stored["best_model"]:
                raise RuntimeError(
                    f"{task}: learnability.csv selects {stored['best_model']} on "
                    f"{stored['selected_by']}, but selection on {val_metric} gives {rederived}"
                )

            if view != "primary" and stored["selected_by"] != val_metric:
                raise RuntimeError(
                    f"{task}: the {metric} row of learnability.csv was selected on "
                    f"{stored['selected_by']}, not on the primary {val_metric}"
                )

            runs     = pool.loc[(pool["task"] == task) & (pool["model"] == rederived)]
            evidence = (
                verify_shared_test_partition(seed_df, task, rederived)
                if task_type == "regression"
                else None
            )

            rows.append(dict(
                panel             = "a-f" if task_type == "regression" else "g",
                task              = task,
                task_label        = TASK_LABEL[task],
                property          = task_property(task),
                task_type         = task_type,
                category          = SELECTED_LABEL,
                configuration     = rederived,
                representation    = str(runs["rep"].iloc[0]),
                selected_by       = val_metric,
                metric            = metric,
                response_scale    = TASK_RESPONSE_SCALE.get(task, "dimensionless") if metric == "rmse" else "dimensionless",
                estimator         = f"mean of {int(stored['n_seeds'])} per-seed test {metric} values",
                n_seeds           = int(stored["n_seeds"]),
                n_test            = int(stored["n_test"]),
                value             = float(stored["mean"]),
                seed_ci_lo        = float(stored["seed_ci_lo"]),
                seed_ci_hi        = float(stored["seed_ci_hi"]),
                seed_ci_type      = "percentile bootstrap (10000 draws) of the mean over seeds",
                test_ci_lo        = float(stored["test_ci_lo"]),
                test_ci_hi        = float(stored["test_ci_hi"]),
                test_ci_type      = TEST_CI_METHOD[metric],
                shared_test_evidence = evidence["evidence"] if evidence else "single-category panel",
                hashes_compared      = evidence["hashes_compared"] if evidence else "",
            ))

            if task_type != "regression" or task not in baselines:
                continue

            base = baselines[task]

            rows.append(dict(
                panel             = "a-f",
                task              = task,
                task_label        = TASK_LABEL[task],
                property          = task_property(task),
                task_type         = task_type,
                category          = TRAINING_MEAN_LABEL,
                configuration     = "mean",
                representation    = "none (constant predictor)",
                selected_by       = "not selected (fixed baseline)",
                metric            = metric,
                response_scale    = TASK_RESPONSE_SCALE.get(task, "dimensionless") if metric == "rmse" else "dimensionless",
                estimator         = f"single deterministic test {METRIC_TEXT[metric]} (identical in all seeds)",
                n_seeds           = base["n_runs"],
                n_test            = base["n_test"],
                value             = base["value"],
                seed_ci_lo        = np.nan,
                seed_ci_hi        = np.nan,
                seed_ci_type      = "none (deterministic predictor, no seed variability)",
                test_ci_lo        = base["test_ci_lo"],
                test_ci_hi        = base["test_ci_hi"],
                test_ci_type      = TEST_CI_METHOD[metric],
                shared_test_evidence = evidence["evidence"],
                hashes_compared      = evidence["hashes_compared"],
            ))

    return pd.DataFrame(rows)

# Provenance of each analytic test-sampling interval, as implemented in analyze_uncertainty
TEST_CI_METHOD = {
    "rmse":         "analytic Gaussian SE, RMSE * sqrt(1 / (2 * n_test)), z = 1.96",
    "qf3":          "analytic Gaussian SE (delta method of the RMSE SE, SD_train fixed), (1 - Q2F3) * sqrt(2 / n_test), z = 1.96",
    "balanced_acc": "analytic per-class binomial SE on mean TPR and TNR, z = 1.96",
    "auroc":        "Hanley-McNeil SE from the test class counts, z = 1.96",
}

LEARNABILITY_SPAN_COLUMNS = ["value", "seed_ci_lo", "seed_ci_hi", "test_ci_lo", "test_ci_hi"]

# One y-range per endpoint property, so every task of a property is read on the same scale
def property_axis_limits(
    data: pd.DataFrame,
    tasks: list[str],
    floor_zero: bool,
    columns: list[str] = LEARNABILITY_SPAN_COLUMNS,
) -> dict[str, tuple[float, float]]:

    limits: dict[str, tuple[float, float]] = {}

    for prop in dict.fromkeys(task_property(task) for task in tasks):

        members = [task for task in tasks if task_property(task) == prop]
        span    = data.loc[data["task"].isin(members), columns].to_numpy(dtype=float).ravel()
        span    = span[np.isfinite(span)]

        if len(span) == 0:
            continue

        lo, hi      = float(span.min()), float(span.max())
        pad         = max((hi - lo) * 0.12, 0.02)
        limits[prop] = (max(0.0, lo - pad) if floor_zero else lo - pad, hi + pad)

    return limits

LEARNABILITY_MEAN_MARKER       = "D"

LEARNABILITY_MEAN_COLOR        = "#A5A19C"

LEARNABILITY_MEAN_SIZE         = 22

LEARNABILITY_SELECTED_MARKER   = "D"

LEARNABILITY_SELECTED_SIZE     = 22

LEARNABILITY_MARKER_EDGEWIDTH  = 0.7

LEARNABILITY_LEGEND_MARKERSIZE = 4.7

LEARNABILITY_LEGEND_TUPLE_PAD  = 0.30

LEARNABILITY_LEGEND_HANDLELENGTH = 3.2

LEARNABILITY_LEGEND_HANDLETEXTPAD = 1.0

LEARNABILITY_LEGEND_COLUMNSPACING = 2.0

LEARNABILITY_SELECTED_SOFTEN   = 1.0 - (SPLIT_OBSERVATION_ALPHA + 0.25)

# Blend a colour towards white
def soften(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)

# Training-mean score per task and split
def training_mean_values(seed_df: pd.DataFrame, metric: str) -> dict[tuple[str, str], float]:

    work   = seed_df.loc[seed_df["model"] == "mean"]
    values: dict[tuple[str, str], float] = {}

    for (task, split_method), group in work.groupby(["task", "split_method"]):

        numeric = pd.to_numeric(group[metric], errors="coerce").dropna()

        if numeric.empty:
            continue

        values[(task, split_method)] = float(numeric.mean())

    return values

# The grey diamond, drawn the same way in every figure
def draw_training_mean_marker(ax: plt.Axes, x: float, y: float, *, horizontal: bool = False) -> None:
    ax.scatter(
        *((y, x) if horizontal else (x, y)),
        s=LEARNABILITY_MEAN_SIZE, marker=LEARNABILITY_MEAN_MARKER,
        color=LEARNABILITY_MEAN_COLOR, edgecolor=AX_FACE,
        linewidth=LEARNABILITY_MARKER_EDGEWIDTH, zorder=6,
    )

# Metrics drawn with a line instead
TRAINING_MEAN_LINE_METRICS = {"balanced_acc", "auroc"}

TRAINING_MEAN_LINE = dict(color=TRAINING_MEAN_COLOR, linestyle=BASELINE_LINESTYLE, linewidth=BASELINE_LINEWIDTH)

# Training-mean predictor of one panel
def draw_training_mean(ax: plt.Axes, metric: str, values: dict[float, float], *, horizontal: bool = False,
                       line: bool = False) -> None:

    if metric not in TRAINING_MEAN_LINE_METRICS and not line:
        for position, value in values.items():
            draw_training_mean_marker(ax, position, value, horizontal=horizontal)
        return

    if not values:
        return

    scores = np.array(list(values.values()), dtype=float)

    if not np.allclose(scores, scores[0]):
        raise RuntimeError(f"training-mean {metric} differs between the categories of one panel: {sorted(scores)}")

    (ax.axvline if horizontal else ax.axhline)(scores[0], zorder=1, **TRAINING_MEAN_LINE)

# Legend handle: diamond and line
def training_mean_legend_handle() -> tuple[plt.Line2D, plt.Line2D]:
    return (
        plt.Line2D(
            [0], [0], marker=LEARNABILITY_MEAN_MARKER, color="none",
            markerfacecolor=LEARNABILITY_MEAN_COLOR, markeredgecolor=AX_FACE,
            markeredgewidth=LEARNABILITY_MARKER_EDGEWIDTH,
            markersize=LEARNABILITY_LEGEND_MARKERSIZE, label=NO_SKILL_LABEL,
        ),
        plt.Line2D([0], [0], label=NO_SKILL_LABEL, **TRAINING_MEAN_LINE),
    )

# Figure legend with tuple handles
def add_figure_legend(fig: plt.Figure, handles: list, **kwargs) -> None:

    labels = [handle[0].get_label() if isinstance(handle, tuple) else handle.get_label() for handle in handles]
    kwargs.setdefault("handlelength", LEARNABILITY_LEGEND_HANDLELENGTH)

    fig.legend(handles=handles, labels=labels,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=LEARNABILITY_LEGEND_TUPLE_PAD)},
               **kwargs)

# One y-range per endpoint property, or a single one for every task when the metric has no endpoint units
def metric_axis_limits(
    data: pd.DataFrame,
    tasks: list[str],
    metric: str,
    floor_zero: bool,
    columns: list[str],
) -> dict[str, tuple[float, float]]:

    if metric in SPLIT_DIMENSIONLESS_METRICS:
        shared = property_axis_limits(data.assign(task=tasks[0]), tasks[:1], floor_zero, columns)
        return {task_property(task): next(iter(shared.values())) for task in tasks}

    return property_axis_limits(data, tasks, floor_zero, columns)

# Major ticks of a metric axis
def metric_tick_locator(metric: str) -> MaxNLocator:

    if metric in SPLIT_DIMENSIONLESS_METRICS:
        return MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])

    return MaxNLocator(nbins=4)

# ---------------------------------------------------------------------------
# Figure 2 as full distributions: every configuration the validation selection
# searched, per task, with the selected configuration and the baseline kept.
# ---------------------------------------------------------------------------

POOL_JITTER_SEED   = 20260923

POOL_JITTER_SPREAD = 0.30

# The (task type, metric, axis label) of both rows of a learnability view, regression above classification
def learnability_blocks(view: str) -> list[tuple[str, str, str]]:

    spec = LEARNABILITY_VIEWS[view]

    return [
        ("regression",     spec["metrics"]["regression"],     spec["regression_label"]),
        ("classification", spec["metrics"]["classification"], spec["classification_label"]),
    ]

# Test score of every configuration and seed of the pool figure 2 selects from
def learnability_pool_observations(seed_df: pd.DataFrame, view: str = "primary") -> pd.DataFrame:

    frames = []
    blocks = learnability_blocks(view)

    for task_type, metric, _ in blocks:

        tasks = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS

        pool = main_pool(
            seed_df.loc[
                (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
                & (seed_df["model"] != "mean")
                & (seed_df["task"].isin(tasks))
                & (~seed_df["family"].isin(["Baseline", "Other"]))
            ]
        )

        frame = pool[["task", "model", "rep", "family", "seed"]].copy()
        frame["task_type"]      = task_type
        frame["metric"]         = metric
        frame["value"]          = pd.to_numeric(pool[metric], errors="coerce")
        frame["task_label"]     = frame["task"].map(TASK_LABEL)
        frame["property"]       = frame["task"].map(task_property)
        frame["response_scale"] = (frame["task"].map(TASK_RESPONSE_SCALE) if metric == "rmse"
                                   else SPLIT_METRIC_SCALE[metric])

        missing = int(frame["value"].isna().sum())
        if missing:
            print(f"  WARNING: {missing} {task_type} pool observations have no {metric} and are dropped.")

        frames.append(frame.dropna(subset=["value"]))

    return pd.concat(frames, ignore_index=True)

# Jittered horizontal position of every plotted pool observation
def learnability_pool_points(observations: pd.DataFrame) -> pd.DataFrame:

    rng    = np.random.default_rng(POOL_JITTER_SEED)
    frames = []

    for _, group in observations.groupby(["task_type", "task"], sort=False):
        offsets = rng.uniform(-POOL_JITTER_SPREAD, POOL_JITTER_SPREAD, size=len(group))
        frames.append(group.assign(category_position=0, x_plotted=offsets))

    return pd.concat(frames, ignore_index=True)

# Per task: quartiles and observed extremes of the pool, beside the two points figure 2 draws
def learnability_pool_summary(observations: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for (task_type, metric, task), group in observations.groupby(["task_type", "metric", "task"], sort=False):

        values = group["value"].to_numpy(dtype=float)
        marks  = selected.loc[(selected["task"] == task) & (selected["metric"] == metric)].set_index("category")

        rows.append(dict(
            task_type        = task_type,
            metric           = metric,
            task             = task,
            task_label       = TASK_LABEL[task],
            property         = task_property(task),
            response_scale   = group["response_scale"].iloc[0],
            n_observations   = int(len(group)),
            n_configurations = int(group.groupby(["model", "rep"]).ngroups),
            n_seeds          = int(group["seed"].nunique()),
            pool_min         = float(values.min()),
            pool_q1          = float(np.percentile(values, 25)),
            pool_median      = float(np.median(values)),
            pool_q3          = float(np.percentile(values, 75)),
            pool_max         = float(values.max()),
            selected_value   = float(marks.loc[SELECTED_LABEL, "value"]) if SELECTED_LABEL in marks.index else np.nan,
            selected_config  = marks.loc[SELECTED_LABEL, "configuration"] if SELECTED_LABEL in marks.index else "",
            training_mean    = float(marks.loc[TRAINING_MEAN_LABEL, "value"]) if TRAINING_MEAN_LABEL in marks.index else np.nan,
        ))

    return pd.DataFrame(rows)

# The reference points of a learnability view, for both formulations, on the pool figure's scale
def learnability_pool_marks(seed_df: pd.DataFrame, view: str = "primary") -> pd.DataFrame:

    blocks       = learnability_blocks(view)
    class_metric = LEARNABILITY_VIEWS[view]["metrics"]["classification"]

    marks = learnability_figure_data(seed_df, view)
    marks = marks.loc[marks["metric"].isin([metric for _, metric, _ in blocks])]

    baseline = seed_df.loc[
        (seed_df["model"] == "mean")
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (seed_df["task"].isin(MAIN_CLS_TASKS))
    ]

    rows = []
    for task, group in baseline.groupby("task"):
        values = pd.to_numeric(group[class_metric], errors="coerce").dropna()
        if values.empty:
            print(f"  WARNING: no training-mean {class_metric} stored for {task}; baseline omitted.")
            continue
        rows.append(dict(
            task           = task,
            task_label     = TASK_LABEL[task],
            property       = task_property(task),
            task_type      = "classification",
            category       = TRAINING_MEAN_LABEL,
            configuration  = "mean",
            representation = "none (constant predictor)",
            metric         = class_metric,
            value          = float(values.mean()),
        ))

    columns = ["task", "task_label", "property", "task_type", "category",
               "configuration", "representation", "metric", "value"]

    return pd.concat([marks[columns], pd.DataFrame(rows, columns=columns)], ignore_index=True)

# Figure 2, or its companion on Q2F3 and AUROC with view="companion": the whole selection pool per task,
# as boxes over every observation, beside the training-mean predictor
def plot_task_learnability(seed_df: pd.DataFrame, view: str = "primary") -> None:

    spec         = LEARNABILITY_VIEWS[view]
    observations = learnability_pool_observations(seed_df, view)
    points       = learnability_pool_points(observations)
    marks        = learnability_pool_marks(seed_df, view)
    summary      = learnability_pool_summary(observations, marks)

    save_source_table(points, f"{spec['table_stem']}_points.csv")
    save_source_table(marks, f"{spec['table_stem']}_marks.csv")
    save_table(summary, f"{spec['table_stem']}_statistics.csv")

    box_line = dict(color=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH)

    fig, axes = plt.subplots(2, len(MAIN_REG_TASKS), figsize=(FIG_WIDTH_FULL, 4.95))

    for row, (task_type, metric, axis_label) in enumerate(learnability_blocks(view)):

        tasks  = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS
        limits = metric_axis_limits(
            summary.loc[summary["task_type"] == task_type], tasks, metric, spec["floor_zero"],
            columns=["pool_min", "pool_max", "training_mean"],
        )

        for column, task in enumerate(tasks):

            ax    = axes[row][column]
            group = observations.loc[(observations["task_type"] == task_type) & (observations["task"] == task)]
            drawn = points.loc[(points["task_type"] == task_type) & (points["task"] == task)]
            color = PROPERTY_COLOR.get(task_property(task), NEUTRAL_LABEL)

            ax.scatter(drawn["x_plotted"], drawn["value"], s=SPLIT_OBSERVATION_SIZE, color=color,
                       edgecolor="none", alpha=SPLIT_OBSERVATION_ALPHA, zorder=2)

            ax.boxplot(
                [group["value"].to_numpy(dtype=float)],
                positions=[0], widths=0.72, vert=True,
                showfliers=False,
                whis=(0, 100),
                medianprops=dict(color=TEXT_COLOR, linewidth=SPLIT_MEDIAN_LINEWIDTH, solid_capstyle="butt"),
                boxprops=dict(facecolor="none", edgecolor=SPLIT_BOX_COLOR, linewidth=SPLIT_BOX_LINEWIDTH),
                whiskerprops=box_line, capprops=box_line,
                patch_artist=True, zorder=4,
            )

            task_marks = marks.loc[(marks["task"] == task) & (marks["metric"] == metric)].set_index("category")

            if TRAINING_MEAN_LABEL in task_marks.index:
                draw_training_mean(ax, metric, {0: float(task_marks.loc[TRAINING_MEAN_LABEL, "value"])}, line=True)

            if SELECTED_LABEL in task_marks.index:
                ax.scatter(0, float(task_marks.loc[SELECTED_LABEL, "value"]),
                           s=LEARNABILITY_SELECTED_SIZE, marker=LEARNABILITY_SELECTED_MARKER,
                           color=soften(color, LEARNABILITY_SELECTED_SOFTEN), edgecolor=AX_FACE,
                           linewidth=LEARNABILITY_MARKER_EDGEWIDTH, zorder=7)

            ax.set_xlim(-0.62, 0.62)
            ax.set_xticks([0])
            ax.set_xticklabels(["All models"], fontsize=ANNOTATION_SMALL_SIZE)

            if metric in SPLIT_UNIT_INTERVAL_METRICS:
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            else:
                ax.set_ylim(*limits[task_property(task)])
                ax.yaxis.set_major_locator(metric_tick_locator(metric))
            _two_decimal_ticks(ax.yaxis)

            _polish_axis(ax, "none")
            ax.tick_params(axis="x", length=0, pad=2.6)
            fit_xtick_labels(ax, min_size=4.6, fill=0.98)

            if row == 0:
                add_panel_title(ax, TASK_LABEL[task], dy=PANEL_TITLE_DY + 1.0, weight="bold")
            if column == 0:
                ax.set_ylabel(axis_label, fontsize=AXIS_LABEL_SIZE - 0.6)
                add_panel_label(ax, "ab"[row], dy=PANEL_LABEL_DY + (14.0 if row == 0 else 4.0))

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none",
                   alpha=SPLIT_OBSERVATION_ALPHA + 0.25, markersize=4.6, label=prop)
        for prop, color in PROPERTY_COLOR.items()
    ]
    handles.append(
        tuple(
            plt.Line2D([0], [0], marker=LEARNABILITY_SELECTED_MARKER, color="none",
                       markerfacecolor=soften(property_color, LEARNABILITY_SELECTED_SOFTEN),
                       markeredgecolor=AX_FACE, markeredgewidth=LEARNABILITY_MARKER_EDGEWIDTH,
                       markersize=LEARNABILITY_LEGEND_MARKERSIZE, label=SELECTED_LABEL)
            for property_color in PROPERTY_COLOR.values()
        )
    )
    handles.append(training_mean_legend_handle()[1])

    add_figure_legend(fig, handles,
                      loc="lower center", bbox_to_anchor=(0.5, 0.0),
                      ncol=5, frameon=False, fontsize=ANNOTATION_SIZE,
                      handletextpad=LEARNABILITY_LEGEND_HANDLETEXTPAD,
                      columnspacing=LEARNABILITY_LEGEND_COLUMNSPACING)
    fig.subplots_adjust(left=0.098, right=0.995, top=0.860, bottom=0.150, wspace=0.46, hspace=0.38)

    _save_plot(fig, OUT_ROOT / spec["folder"] / f"{spec['stem']}.png", align_labels=False)
