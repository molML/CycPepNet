from __future__ import annotations
import warnings
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from cycpepnet.analyze._common import (
    CANONICAL_BUTINA_SPLIT,
    TASK_ORDER_CLS,
    TASK_ORDER_REG,
)
from cycpepnet.analyze._results_config import (
    ANNOTATION_SIZE,
    ANNOTATION_SMALL_SIZE,
    EXPECTED_N_SEEDS,
    FIG_WIDTH_FULL,
    NEUTRAL_MID,
    PANEL_LABEL_X,
    PLOTS_DIR,
    RESULTS_ROOT,
    SPINE_COLOR,
    TEXT_COLOR,
)
from cycpepnet.analyze._results_data import representation_of
from cycpepnet.analyze._results_style import (
    _apply_results_plot_style,
    _polish_axis,
    _save_plot,
    _two_decimal_ticks,
)
from cycpepnet.analyze._result_helpers import FAMILY_LABEL, METRIC_HIGHER_IS_BETTER, METRIC_LABEL, PROPERTY_COLOR, TASK_LABEL, add_panel_label, compute_validation_selected_family_ranks, compute_validation_selected_representation_ranks, load_seed_results, method_rank_rows, metric_reference, save_source_table, save_table, task_property, validate_seed_results
from cycpepnet.analyze._figure_learnability import TRAINING_MEAN_LINE
from cycpepnet.analyze._figure_methods import configuration_rank_table
from cycpepnet.analyze._figure_split import significance_label
from cycpepnet.utils import (
    print_completion_message,
    print_section_header,
    setup_plot_style,
    tee_stdout_to_log,
)

# Constants
OUT_DIR         = PLOTS_DIR / "supplementary"
TESTS_CSV       = RESULTS_ROOT / "statistical_tests.csv"
OBSERVATIONS_CSV = "method_distributions.csv"
SUMMARY_CSV     = "method_distributions_box_statistics.csv"
CONTRASTS_CSV   = "method_distributions_drawable_contrasts.csv"

BASELINE_MODEL = "mean"

ANCHORED_SCORES: list[tuple[str, str, str, bool]] = [
    ("ba_anchored",    "balanced_acc", "classification", True),
    ("rmse_anchored",  "rmse",         "regression",     False),
    ("auroc_anchored", "auroc",        "classification", True),
    ("qf3_anchored",   "qf3",          "regression",     True),
]

PRIMARY_LIMITS   = (-1.40, 1.55)
PRIMARY_TICKS    = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
COMPANION_LIMITS = (-1.85, 1.55)
COMPANION_TICKS  = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

# One column of panels per metric
PRIMARY_COLUMNS: list[dict] = [
    {
        "task_type":   "regression",
        "tasks":       list(TASK_ORDER_REG),
        "metric":      "rmse_anchored",
        "test_metric": "rmse",
        "val_metric":  "val_qf3",
        "limits":      PRIMARY_LIMITS,
        "ticks":       PRIMARY_TICKS,
    },
    {
        "task_type":   "classification",
        "tasks":       list(TASK_ORDER_CLS),
        "metric":      "ba_anchored",
        "test_metric": "balanced_acc",
        "val_metric":  "val_balanced_acc",
        "limits":      PRIMARY_LIMITS,
        "ticks":       PRIMARY_TICKS,
    },
]

COMPANION_COLUMNS: list[dict] = [
    {
        "task_type":   "regression",
        "tasks":       list(TASK_ORDER_REG),
        "metric":      "qf3_anchored",
        "test_metric": "qf3",
        "val_metric":  "val_qf3",
        "limits":      COMPANION_LIMITS,
        "ticks":       COMPANION_TICKS,
    },
    {
        "task_type":   "classification",
        "tasks":       list(TASK_ORDER_CLS),
        "metric":      "auroc_anchored",
        "test_metric": "auroc",
        "val_metric":  "val_balanced_acc",
        "limits":      COMPANION_LIMITS,
        "ticks":       COMPANION_TICKS,
    },
]

METRIC_COLUMNS: list[dict] = PRIMARY_COLUMNS + COMPANION_COLUMNS

FIGURES: list[tuple[str, list[dict]]] = [
    ("fig_method_distributions",           PRIMARY_COLUMNS),
    ("fig_method_distributions_companion", COMPANION_COLUMNS),
]

DRAWABLE_CONTRASTS: dict[str, list[tuple[str, str, str, str]]] = {
    "family": [
        ("RF vs XGBoost",           "RF",         "XGBoost",     ""),
        ("LSTM vs Transformer",     "LSTM",       "Transformer", ""),
        ("ChemBERTa vs PeptideCLM", "ChemBERTa",  "PeptideCLM",  ""),
        ("GCN vs MPNN",             "GCN",        "MPNN",        ""),
        ("GIN vs MolCLR",           "GIN",        "MolCLR",      ""),
    ],
    "representation": [
        ("ECFP4c vs ECFP4b",                   "ECFP4c",     "ECFP4b", ""),
        ("fragSMILES vs SMILES (LSTM)",        "fragSMILES", "SMILES", " (LSTM)"),
        ("fragSMILES vs SMILES (Transformer)", "fragSMILES", "SMILES", " (Tr.)"),
    ],
}

BRACKET_SIGNIFICANT_ONLY = True
BRACKET_FIRST_OFFSET     = 0.012
BRACKET_STEP             = 0.052
BRACKET_TICK             = 0.008
BRACKET_COLOR            = "#5F5A54"

RANK_ORDER_COLUMNS: list[tuple[str, list[str], str, str]] = [
    ("regression", list(TASK_ORDER_REG), "rmse", "val_rmse"),
    ("classification", list(TASK_ORDER_CLS), "balanced_acc", "val_balanced_acc"),
]

GROUPINGS: list[tuple[str, object]] = [
    ("family", compute_validation_selected_family_ranks),
    ("representation", compute_validation_selected_representation_ranks),
]

ENDPOINTS   = ["Binding", "Permeability", "Half-life"]
MARKER      = "o"
MARKER_SIZE = 5.0
POINT_ALPHA = 0.35

ROW_PITCH_INCHES      = 0.36
BOX_HEIGHT            = 0.62
SWARM_HALF_HEIGHT     = 0.47
OVERLAP_TOLERANCE     = 0.75
BOX_EDGE              = "#5F5A54"

PRETRAINING_STEM = "fig_pretraining"
PRETRAINING_OBSERVATIONS_CSV = "pretraining_distributions.csv"

PRETRAINING_BLOCKS: list[tuple[str, list[tuple[str, list[str]]]]] = [
    ("ECFP4b", [("", ["RF", "XGBoost"])]),
    ("SMILES", [("Scratch", ["LSTM", "Transformer"]), ("Pretrained", ["ChemBERTa", "PeptideCLM"])]),
    ("Graph",  [("Scratch", ["GCN", "MPNN"]),         ("Pretrained", ["GIN", "MolCLR"])]),
]

PRETRAINING_MODELS: dict[str, list[str]] = {
    "RF":          ["rf_binary_r2_fp"],
    "XGBoost":     ["xgb_binary_r2_fp"],
    "LSTM":        ["lstm"],
    "Transformer": ["transformer"],
}

PRETRAINING_XLABEL: dict[str, str] = {
    "ba_anchored":   r"Balanced accuracy gain over no-skill $\uparrow$",
    "rmse_anchored": r"RMSE reduction over no-skill $\uparrow$",
}

PRETRAINING_LEFT_MARGIN   = 0.250
PRETRAINING_OUTER_X       = 0.052
PRETRAINING_INNER_X       = 0.104
PRETRAINING_BRACKET_TICK  = 0.009
PRETRAINING_LABEL_GAP     = 0.019
PRETRAINING_BRACKET_INSET = 0.16


# Draw the distribution figure and export its observations
def main() -> None:

    setup_plot_style()
    _apply_results_plot_style()

    seed_df = load_seed_results()

    if seed_df.empty:
        print("  No seed-level result rows found. Nothing to plot.")
        print_completion_message()
        return

    validate_seed_results(seed_df)
    seed_df = add_anchored_scores(seed_df)
    validate_metric_columns()

    print_section_header("METHOD PERFORMANCE DISTRIBUTIONS")

    orders       = fig_method_ranks_order(seed_df)
    observations = pd.concat(
        [
            selected_seed_observations(seed_df, group_col, select, column)
            for group_col, select in GROUPINGS
            for column in METRIC_COLUMNS
        ],
        ignore_index=True,
    )

    summary  = box_statistics(observations, orders)
    adjusted = load_adjusted_p()

    save_source_table(observations, OBSERVATIONS_CSV)
    save_table(summary, SUMMARY_CSV)
    save_table(report_drawable_contrasts(adjusted), CONTRASTS_CSV)

    for stem, columns in FIGURES:
        plot_method_distributions(observations, summary, orders, columns, adjusted, stem)

    pretraining = pretraining_observations(seed_df)
    save_source_table(pretraining, PRETRAINING_OBSERVATIONS_CSV)
    plot_pretraining(pretraining)

    print_completion_message()


# Per-seed values rescaled so 0 is the measured baseline and 1 the best configuration
def add_anchored_scores(seed_df: pd.DataFrame) -> pd.DataFrame:

    out = seed_df.copy()

    for target, metric, task_type, higher_better in ANCHORED_SCORES:

        tasks = list(TASK_ORDER_REG) if task_type == "regression" else list(TASK_ORDER_CLS)

        rows = seed_df.loc[
            (seed_df["model"] == BASELINE_MODEL)
            & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
            & (seed_df["task"].isin(tasks))
        ]
        baseline = rows.groupby("task")[metric].mean()

        if set(baseline.index) != set(tasks):
            raise RuntimeError(f"No {BASELINE_MODEL!r} baseline for every task of {metric!r}.")

        pool = configuration_rank_table(seed_df, tasks, task_type, metric)
        best = pool.groupby("task")["test_mean"].agg("max" if higher_better else "min")

        anchor     = out["task"].map(baseline)
        span       = out["task"].map(best) - anchor
        out[target] = (out[metric] - anchor) / span

    return out


# Refuse metric columns that would plot on an implicit or unbounded scale
def validate_metric_columns() -> None:

    for column in METRIC_COLUMNS:

        metric = column["metric"]

        if metric not in METRIC_HIGHER_IS_BETTER:
            raise ValueError(f"Unknown metric direction for {metric!r}.")
        if column.get("limits") is None:
            raise ValueError(f"Metric column {metric!r} needs explicit axis limits.")


# Category order of both rows, as in figure 3
def fig_method_ranks_order(seed_df: pd.DataFrame) -> dict[str, list[str]]:

    specs = {
        group_col: [
            (select(seed_df, tasks, task_type, metric, val_metric), metric)
            for task_type, tasks, metric, val_metric in RANK_ORDER_COLUMNS
        ]
        for group_col, select in GROUPINGS
    }
    rows = method_rank_rows(specs["family"], specs["representation"])

    return {group_col: order for group_col, order, _ in rows}


# Seed-level test values of the configuration selected on validation per (task, group)
def selected_seed_observations(
    seed_df: pd.DataFrame,
    group_col: str,
    select,
    column: dict,
) -> pd.DataFrame:

    task_type  = column["task_type"]
    metric     = column["metric"]
    val_metric = column["val_metric"]

    chosen = select(seed_df, column["tasks"], task_type, metric, val_metric)
    seeds  = seed_df.loc[
        (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (seed_df["task_type"] == task_type)
        & (seed_df["task"].isin(column["tasks"])),
        ["task", "model", "rep", "family", "split_method", "seed", "n_test", metric],
    ]

    obs = chosen[["task", group_col, "model", "rep", "validation_mean", "test_mean"]].merge(
        seeds.drop(columns=[group_col], errors="ignore"),
        on=["task", "model", "rep"],
        how="left",
        validate="one_to_many",
    )

    counts = obs.groupby(["task", group_col])["seed"].nunique()
    if (counts != EXPECTED_N_SEEDS).any():
        raise RuntimeError(f"Selected configurations without {EXPECTED_N_SEEDS} seeds:\n{counts[counts != EXPECTED_N_SEEDS]}")

    seed_means = obs.groupby(["task", group_col])[metric].mean()
    selected   = chosen.set_index(["task", group_col])["test_mean"].reindex(seed_means.index)
    if not np.allclose(seed_means.to_numpy(), selected.to_numpy()):
        raise RuntimeError("Seed-level values do not reproduce the selected test means.")

    return pd.DataFrame(
        {
            "grouping":            group_col,
            "group":               obs[group_col],
            "task":                obs["task"],
            "task_label":          obs["task"].map(TASK_LABEL),
            "endpoint":            obs["task"].map(task_property),
            "task_type":           task_type,
            "split_method":        obs["split_method"],
            "model_family":        obs["family"],
            "model":               obs["model"],
            "representation":      [representation_of(m, r) for m, r in zip(obs["model"], obs["rep"])],
            "configuration":       obs["model"] + " | " + obs["rep"].astype(str),
            "selected_by":         val_metric,
            "validation_seed_mean": obs["validation_mean"],
            "test_seed_mean":      obs["test_mean"],
            "seed":                obs["seed"],
            "n_test":              pd.to_numeric(obs["n_test"]).astype("Int64"),
            "metric":              metric,
            "value":               obs[metric],
        }
    ).sort_values(["grouping", "metric", "group", "task", "seed"], ignore_index=True)


# Box statistics exactly as drawn: quartiles plus whiskers at the minimum and maximum
def box_statistics(observations: pd.DataFrame, orders: dict[str, list[str]]) -> pd.DataFrame:

    rows = []

    for (group_col, metric), block in observations.groupby(["grouping", "metric"], sort=False):
        for group in orders[group_col]:

            sub = block.loc[block["group"] == group]
            if sub.empty:
                continue

            stats = cbook.boxplot_stats(sub["value"].to_numpy(dtype=float), whis=(0, 100))[0]
            rows.append(
                {
                    "grouping":       group_col,
                    "group":          group,
                    "metric":         metric,
                    "n_observations": len(sub),
                    "n_tasks":        sub["task"].nunique(),
                    "n_seeds":        sub["seed"].nunique(),
                    "minimum":        stats["whislo"],
                    "q1":             stats["q1"],
                    "median":         stats["med"],
                    "q3":             stats["q3"],
                    "maximum":        stats["whishi"],
                }
            )

    return pd.DataFrame(rows)


# Offsets across the category axis that keep every marker clear of the others
def swarm_offsets(x: np.ndarray, diameter: float, limit: float) -> np.ndarray:

    offsets = np.zeros(len(x))
    placed: list[tuple[float, float]] = []

    for idx in np.argsort(x, kind="stable"):

        blocked = []
        for px, py in placed:
            dx = abs(x[idx] - px)
            if dx < diameter:
                half = np.sqrt(diameter**2 - dx**2)
                blocked.append((py - half, py + half))

        candidates = [0.0] + [edge for interval in blocked for edge in interval]
        free = [
            c for c in candidates
            if all(not (lo + 1e-6 < c < hi - 1e-6) for lo, hi in blocked)
        ]
        inside = [c for c in free if abs(c) <= limit]

        if inside:
            y = min(inside, key=lambda c: (abs(c), -c))
        else:
            grid = np.linspace(-limit, limit, 81)
            gaps = [min(np.hypot(x[idx] - px, g - py) for px, py in placed) for g in grid]
            y = float(grid[int(np.argmax(gaps))])

        offsets[idx] = y
        placed.append((x[idx], y))

    return offsets


# Marker pairs whose centres are closer than the overlap tolerance, in display units
def count_overlaps(x: np.ndarray, y: np.ndarray, diameter: float) -> int:
    distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    return int(np.triu(distances < OVERLAP_TOLERANCE * diameter, k=1).sum())


# One horizontal box per category with every observation overlaid
def draw_distribution_panel(
    ax: plt.Axes,
    block: pd.DataFrame,
    summary: pd.DataFrame,
    order: list[str],
    column: dict,
) -> int:

    lo, hi = column["limits"]
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xticks(column["ticks"])
    _two_decimal_ticks(ax.xaxis)

    reference = metric_reference(column["metric"])
    if lo < reference < hi:
        ax.axvline(reference, zorder=1, **TRAINING_MEAN_LINE)

    stats = [
        {
            "med": row.median, "q1": row.q1, "q3": row.q3,
            "whislo": row.minimum, "whishi": row.maximum, "fliers": [],
        }
        for row in summary.set_index("group").reindex(order).itertuples()
    ]
    ax.bxp(
        stats,
        positions=range(len(order)),
        widths=BOX_HEIGHT,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "none", "edgecolor": BOX_EDGE, "linewidth": 0.75, "zorder": 4},
        whiskerprops={"color": BOX_EDGE, "linewidth": 0.75, "zorder": 4},
        capprops={"color": BOX_EDGE, "linewidth": 0.75, "zorder": 4},
        medianprops={"color": TEXT_COLOR, "linewidth": 1.3, "zorder": 4, "solid_capstyle": "butt"},
        manage_ticks=False,
    )

    px_per_row = abs(ax.transData.transform((0.0, 1.0))[1] - ax.transData.transform((0.0, 0.0))[1])
    points_to_px = ax.figure.dpi / 72.0
    diameter = np.sqrt(MARKER_SIZE) * 1.35 * points_to_px
    overlaps = 0

    for y, group in enumerate(order):

        sub = block.loc[block["group"] == group].sort_values(["task", "seed"], kind="stable")
        if sub.empty:
            continue

        x_px = ax.transData.transform(np.column_stack([sub["value"], np.zeros(len(sub))]))[:, 0]
        offsets = swarm_offsets(x_px, diameter, SWARM_HALF_HEIGHT * px_per_row)
        overlaps += count_overlaps(x_px, offsets, diameter)
        sub = sub.assign(y=y + offsets / px_per_row)

        for task, points in sub.groupby("task", sort=False):
            ax.scatter(
                points["value"],
                points["y"],
                s=MARKER_SIZE,
                marker=MARKER,
                color=PROPERTY_COLOR.get(task_property(task), NEUTRAL_MID),
                edgecolor="none",
                alpha=POINT_ALPHA,
                zorder=2,
                clip_on=False,
            )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FAMILY_LABEL.get(group, group) for group in order])
    _polish_axis(ax, "none")
    ax.spines["left"].set_color(SPINE_COLOR)

    return overlaps


# One handle per endpoint plus the baseline, since a single shape carries no task identity
def distribution_legend_handles() -> list:

    handles = [
        plt.Line2D([0], [0], marker=MARKER, linestyle="none", markersize=np.sqrt(MARKER_SIZE) * 1.1,
                   markerfacecolor=PROPERTY_COLOR[endpoint], markeredgecolor="none",
                   alpha=POINT_ALPHA + 0.25, label=endpoint)
        for endpoint in ENDPOINTS
    ]
    handles.append(
        plt.Line2D([0], [0], label="No-skill model", **TRAINING_MEAN_LINE)
    )

    return handles

# Adjusted p value of every confirmatory contrast, keyed by metric and comparison label
def load_adjusted_p() -> dict[tuple[str, str], float]:

    if not TESTS_CSV.exists():
        print(f"  WARNING: {TESTS_CSV} is missing; no significance brackets are drawn.")
        return {}

    frame = pd.read_csv(TESTS_CSV)
    frame = frame.loc[frame["p_fdr"].notna()]

    return {(row.metric, row.comparison): float(row.p_fdr) for row in frame.itertuples()}

# Brackets in the right margin, one per drawable contrast, at nested offsets
def draw_significance_brackets(
    ax: plt.Axes,
    order: list[str],
    group_col: str,
    test_metric: str,
    adjusted: dict[tuple[str, str], float],
) -> int:

    transform = ax.get_yaxis_transform()
    positions = {name: index for index, name in enumerate(order)}
    drawn     = 0

    for comparison, left, right, suffix in DRAWABLE_CONTRASTS[group_col]:

        if left not in positions or right not in positions:
            continue

        p_adjusted = adjusted.get((test_metric, comparison))
        if p_adjusted is None:
            continue

        label = significance_label(p_adjusted)
        if BRACKET_SIGNIFICANT_ONLY and label == "n.s.":
            continue

        y_lo, y_hi = sorted((positions[left], positions[right]))
        x = 1.0 + BRACKET_FIRST_OFFSET + drawn * BRACKET_STEP

        ax.plot([x, x], [y_lo, y_hi], transform=transform, color=BRACKET_COLOR,
                linewidth=0.7, clip_on=False, zorder=6)
        for y in (y_lo, y_hi):
            ax.plot([x - BRACKET_TICK, x], [y, y], transform=transform, color=BRACKET_COLOR,
                    linewidth=0.7, clip_on=False, zorder=6)

        ax.text(x + 0.010, (y_lo + y_hi) / 2.0, f"{label}{suffix}", transform=transform,
                ha="left", va="center", rotation=90, rotation_mode="anchor",
                fontsize=ANNOTATION_SMALL_SIZE, color=BRACKET_COLOR, clip_on=False, zorder=6)

        drawn += 1

    return drawn

# Every drawable contrast with its adjusted p value, so the bracket policy is auditable
def report_drawable_contrasts(adjusted: dict[tuple[str, str], float]) -> pd.DataFrame:

    rows = []

    for group_col, contrasts in DRAWABLE_CONTRASTS.items():
        for comparison, left, right, _ in contrasts:
            for column in METRIC_COLUMNS:
                p_adjusted = adjusted.get((column["test_metric"], comparison))
                if p_adjusted is None:
                    continue
                rows.append(dict(
                    grouping=group_col, metric=column["test_metric"], comparison=comparison,
                    row_a=left, row_b=right, p_fdr=p_adjusted,
                    label=significance_label(p_adjusted),
                    drawn=not (BRACKET_SIGNIFICANT_ONLY and significance_label(p_adjusted) == "n.s."),
                ))

    if not rows:
        return pd.DataFrame(columns=["grouping", "metric", "comparison", "row_a", "row_b",
                                     "p_fdr", "label", "drawn"])

    return pd.DataFrame(rows).sort_values(["metric", "p_fdr"]).reset_index(drop=True)

# Rows are groupings, columns are metrics; categories follow figure 3
def plot_method_distributions(
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    orders: dict[str, list[str]],
    columns: list[dict],
    adjusted: dict[tuple[str, str], float],
    stem: str,
) -> None:

    n_rows_per_panel = [len(orders[group_col]) for group_col, _ in GROUPINGS]
    fig_height = ROW_PITCH_INCHES * sum(n_rows_per_panel) + 1.35

    fig = plt.figure(figsize=(FIG_WIDTH_FULL, fig_height))
    grid = fig.add_gridspec(
        len(GROUPINGS),
        len(columns),
        height_ratios=n_rows_per_panel,
        left=0.145,
        right=0.975,
        top=1.0 - 0.22 / fig_height,
        bottom=0.95 / fig_height,
        hspace=0.42 / ROW_PITCH_INCHES / (sum(n_rows_per_panel) / len(GROUPINGS)),
        wspace=0.30,
    )

    panel    = iter("abcdefgh")
    overlaps = 0
    brackets = 0

    for row_idx, (group_col, _) in enumerate(GROUPINGS):
        for col_idx, column in enumerate(columns):

            ax = fig.add_subplot(grid[row_idx, col_idx])
            is_block = (observations["grouping"] == group_col) & (observations["metric"] == column["metric"])
            is_stats = (summary["grouping"] == group_col) & (summary["metric"] == column["metric"])

            overlaps += draw_distribution_panel(
                ax,
                observations.loc[is_block],
                summary.loc[is_stats],
                orders[group_col],
                column,
            )

            brackets += draw_significance_brackets(
                ax, orders[group_col], group_col, column["test_metric"], adjusted,
            )

            add_panel_label(ax, next(panel), x=PANEL_LABEL_X if col_idx == 0 else None)

            if row_idx == len(GROUPINGS) - 1:
                ax.set_xlabel(METRIC_LABEL[column["metric"]])
            if col_idx:
                ax.tick_params(axis="y", labelleft=False)

    print(f"  {stem}: {brackets} significance brackets, {overlaps} marker pairs overlapping "
          f"by more than {1 - OVERLAP_TOLERANCE:.0%} of a diameter")

    fig.legend(handles=distribution_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(ENDPOINTS) + 1, frameon=False)

    _save_plot(fig, OUT_DIR / f"{stem}.png")

# Families pinned to one input, so every row of the figure stands on a single representation
def pretraining_family_ranks(
    seed_df: pd.DataFrame,
    tasks: list[str],
    task_type: str,
    metric: str,
    val_metric: str,
) -> pd.DataFrame:

    keep = [
        family not in PRETRAINING_MODELS or model in PRETRAINING_MODELS[family]
        for family, model in zip(seed_df["family"], seed_df["model"])
    ]

    return compute_validation_selected_family_ranks(seed_df.loc[keep], tasks, task_type, metric, val_metric)


# Seed-level values behind the pretraining figure, on the primary metric of each task type
def pretraining_observations(seed_df: pd.DataFrame) -> pd.DataFrame:

    return pd.concat(
        [
            selected_seed_observations(seed_df, "family", pretraining_family_ranks, column)
            for column in PRIMARY_COLUMNS
        ],
        ignore_index=True,
    )


# One bracket spanning the given rows, labelled along its length, left of the tick labels
def draw_row_bracket(ax: plt.Axes, transform, first: int, last: int, label: str, x: float) -> None:

    y_lo = first - 0.5 + PRETRAINING_BRACKET_INSET
    y_hi = last + 0.5 - PRETRAINING_BRACKET_INSET

    ax.plot([x, x], [y_lo, y_hi], transform=transform, color=BRACKET_COLOR,
            linewidth=0.7, clip_on=False, zorder=6)

    for y in (y_lo, y_hi):
        ax.plot([x, x + PRETRAINING_BRACKET_TICK], [y, y], transform=transform, color=BRACKET_COLOR,
                linewidth=0.7, clip_on=False, zorder=6)

    ax.text(x - PRETRAINING_LABEL_GAP, (y_lo + y_hi) / 2.0, label, transform=transform,
            ha="center", va="center", rotation=90, fontsize=ANNOTATION_SIZE,
            color=TEXT_COLOR, clip_on=False, zorder=6)


# Input type on the outer column, pretraining status on the inner one
def draw_pretraining_brackets(ax: plt.Axes) -> None:

    transform = blended_transform_factory(ax.figure.transFigure, ax.transData)
    first     = 0

    for block, sections in PRETRAINING_BLOCKS:

        rows = sum(len(families) for _, families in sections)
        draw_row_bracket(ax, transform, first, first + rows - 1, block, PRETRAINING_OUTER_X)

        inner = first
        for section, families in sections:
            if section:
                draw_row_bracket(ax, transform, inner, inner + len(families) - 1, section, PRETRAINING_INNER_X)
            inner += len(families)

        first += rows


# Pretrained against scratch families, main text
def plot_pretraining(observations: pd.DataFrame) -> None:

    order   = [family for _, sections in PRETRAINING_BLOCKS for _, families in sections for family in families]
    block   = observations.loc[observations["grouping"] == "family"]
    summary = box_statistics(block, {"family": order})

    fig_height = 0.31 * len(order) + 1.3
    fig, axes  = plt.subplots(1, len(PRIMARY_COLUMNS), figsize=(FIG_WIDTH_FULL, fig_height), sharey=True)
    fig.subplots_adjust(left=PRETRAINING_LEFT_MARGIN, right=0.985, top=1.0 - 0.3 / fig_height,
                        bottom=1.0 / fig_height, wspace=0.08)

    for ax, column, letter in zip(axes, PRIMARY_COLUMNS, "ab"):

        boundary = 0
        for _, sections in PRETRAINING_BLOCKS[:-1]:
            boundary += sum(len(families) for _, families in sections)
            ax.axhline(boundary - 0.5, color=SPINE_COLOR, linewidth=0.8, zorder=1)

        metric = column["metric"]
        draw_distribution_panel(ax, block.loc[block["metric"] == metric],
                                summary.loc[summary["metric"] == metric], order, column)
        ax.set_xlabel(PRETRAINING_XLABEL[metric])
        add_panel_label(ax, letter)

    draw_pretraining_brackets(axes[0])

    handles = distribution_legend_handles()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False,
               fontsize=ANNOTATION_SMALL_SIZE + 0.4, handlelength=1.4, columnspacing=1.2)

    _save_plot(fig, PLOTS_DIR / "main" / f"{PRETRAINING_STEM}.png")


if __name__ == "__main__":
    with tee_stdout_to_log("analyze_distributions.log"):
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible")
        main()
