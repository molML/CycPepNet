from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox, blended_transform_factory
from matplotlib.colors import to_rgb
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT
from cycpepnet.analyze._results_config import ANNOTATION_SIZE, ANNOTATION_SMALL_SIZE, AX_FACE, BASE_REPRESENTATION_COLORS, BASE_REPRESENTATION_LABEL, BASE_REPRESENTATION_ORDER, FIG_WIDTH_FULL, FIG_WIDTH_SUPP, MODEL_COLORS, MODEL_GROUP_MAP, MODEL_ORDER_LIST, NEUTRAL_CONNECTOR, NEUTRAL_LABEL, NEUTRAL_MID, PANEL_LABEL_DY, RANK_SUMMARY_COLOR, RANK_SUMMARY_MARKER, RANK_SUMMARY_SIZE, SPINE_COLOR, TASK_CATEGORY_COLORS, TASK_CATEGORY_MAP, TICK_LABEL_SIZE, TEXT_COLOR
from cycpepnet.analyze._results_data import full_pool, main_pool, representation_of
from cycpepnet.analyze._results_style import _polish_axis, _save_plot
from cycpepnet.analyze._result_helpers import FAMILY_LABEL, JOINT_PCA_BA_MODE, JOINT_PCA_FIT_MODE, MAIN_CLS_TASKS, MAIN_REG_TASKS, METRIC_HIGHER_IS_BETTER, OUT_ROOT, TASK_LABEL, add_panel_label, compute_validation_selected_family_ranks, compute_validation_selected_representation_ranks, method_row_order, save_source_table, save_table

# The two mean ranks behind panels a and b of fig_method_ranks, plotted against each other
def plot_regression_vs_classification_ranks(
    reg_family_ranks: pd.DataFrame,
    cls_family_ranks: pd.DataFrame,
) -> None:

    summary = (
        pd.DataFrame(
            {
                "rank_regression":     reg_family_ranks.groupby("family")["rank"].mean(),
                "rank_classification": cls_family_ranks.groupby("family")["rank"].mean(),
            }
        )
        .dropna()
        .sort_values(["rank_regression", "rank_classification"])
        .reset_index()
    )

    if summary.empty:
        print("  Skipped toc_graphic: no family ranks available.")
        return

    summary["family_label"] = [FAMILY_LABEL.get(name, name) for name in summary["family"]]

    print("  Mean validation-selected family rank, regression vs classification:")
    for row in summary.itertuples():
        print(f"    {row.family_label:<13} {row.rank_regression:.2f} / {row.rank_classification:.2f}")

    save_source_table(summary, "regression_vs_classification_family_ranks.csv")

    values = pd.concat([summary["rank_regression"], summary["rank_classification"]])
    lo     = max(0.5, float(np.floor(values.min() - 0.6)))
    hi     = min(len(summary) + 0.5, float(np.ceil(values.max() + 0.6)))

    fig, ax = plt.subplots(figsize=(3.45, 3.45))

    ax.plot([lo, hi], [lo, hi], color=SPINE_COLOR, linewidth=0.8, linestyle=(0, (4, 4)), zorder=1)
    ax.scatter(
        summary["rank_regression"],
        summary["rank_classification"],
        s=RANK_SUMMARY_SIZE,
        marker=RANK_SUMMARY_MARKER,
        facecolor=AX_FACE,
        edgecolor=RANK_SUMMARY_COLOR,
        linewidth=1.15,
        zorder=3,
    )

    midpoint = (lo + hi) / 2.0

    for row in summary.itertuples():
        label_left = row.rank_regression < midpoint
        ax.annotate(
            row.family_label,
            xy=(row.rank_regression, row.rank_classification),
            xytext=(-5.5 if label_left else 5.5, 0),
            textcoords="offset points",
            ha="right" if label_left else "left",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=TEXT_COLOR,
            clip_on=False,
            zorder=4,
        )

    ticks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Regression rank  (1 = best)")
    ax.set_ylabel("Classification rank  (1 = best)")
    _polish_axis(ax, "both")

    _save_plot(fig, OUT_ROOT / "main" / "toc_graphic.png")

# Marker per model, used only by the supplementary PCA that encodes the model by shape
MODEL_MARKERS = {
    "RF": "o", "XGBoost": "s", "LSTM": "^", "Transformer": "v", "ChemBERTa": "D",
    "PeptideCLM": "P", "GCN": "*", "MPNN": "p", "GIN": "h", "MolCLR": "X",
}

# Marker areas that give every model shape a similar visual weight
MODEL_MARKER_SIZES = {
    "RF": 30, "XGBoost": 26, "LSTM": 32, "Transformer": 32, "ChemBERTa": 24,
    "PeptideCLM": 30, "GCN": 58, "MPNN": 32, "GIN": 32, "MolCLR": 30,
}

CIRCLE_MARKER_SIZE = 30

PCA_MARKER_SCALE = 3.00

PCA_EDGE_DARKEN  = 0.30

SCATTER_LABEL_DARKEN = 0.45

# A colour mixed towards black, for an edge that outlines its own fill
def shade(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color))
    return tuple(rgb * (1.0 - amount))

# One configuration marker: hollow when +PhysChem, otherwise solid with a darker edge of its own colour
def draw_configuration_marker(
    ax: plt.Axes, x: float, y: float, color: str, augmented: bool,
    marker: str = "o", base_size: float = CIRCLE_MARKER_SIZE, zorder: float = 3,
    scale: float = PCA_MARKER_SCALE,
) -> None:
    ax.scatter(
        x, y,
        s=base_size * scale,
        marker=marker,
        facecolor=AX_FACE if augmented else color,
        edgecolor=color if augmented else shade(color, PCA_EDGE_DARKEN),
        linewidth=1.3 if augmented else 0.9,
        zorder=zorder,
    )

# Collapse radius and hashing variants into one representation family
def representation_marker_family(base_rep: str) -> str:
    if base_rep.startswith("ECFP"):
        return "ECFP"
    if base_rep.startswith("MAP"):
        return "MAPC"
    return base_rep

# Preserve radius and PhysChem augmentation when labeling PCA configurations.
def configuration_representation(model: object, rep: object) -> tuple[str, str, bool]:
    raw_rep = "" if pd.isna(rep) else str(rep)
    augmented = raw_rep.endswith("+physchem")
    base_raw = raw_rep.removesuffix("+physchem")
    model_based = representation_of(model, base_raw)
    base_rep = model_based or {
        "morgan_binary_r2": "ECFP4b",
        "morgan_count_r2": "ECFP4c",
        "morgan_binary": "ECFP6b",
        "morgan_count": "ECFP6c",
        "map4c": "MAP4C",
        "map6c": "MAP6C",
        "mxfp": "MXFP",
        "hdf": "HDF",
        "physchem": "PhysChem",
        "smiles": "SMILES",
        "fragsmiles": "fragSMILES",
        "graphs": "Graph",
    }.get(base_raw, base_raw or "Other")
    full_rep = f"{base_rep} + PhysChem" if augmented else base_rep
    return base_rep, full_rep, augmented

# ---------------------------------------------------------------------------
# Figure 3, top row: family and representation rank heatmaps plus the
# configuration scatter. Kept free-standing so the PCA can be added below it.
# ---------------------------------------------------------------------------

HEATMAP_CELL_FONTSIZE   = 6.2

HEATMAP_HEADER_FONTSIZE = 6.0

# Rank every configuration of the scatter pool within each task
def configuration_rank_table(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str,
                             pool=main_pool) -> pd.DataFrame:

    work = pool(
        seed_df.loc[
            (seed_df["task"].isin(tasks))
            & (seed_df["task_type"] == task_type)
            & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
            & (~seed_df["family"].isin(["Baseline", "Other"]))
            & (seed_df["model"] != "mean")
        ]
    )

    agg = (
        work.groupby(["task", "model", "rep"], dropna=False)
        .agg(n_seeds=("seed", "nunique"), test_mean=(metric, "mean"))
        .reset_index()
        .dropna(subset=["test_mean"])
    )

    # Seeds are averaged first, then the seed means are ranked within the task
    agg["rank"] = agg.groupby("task")["test_mean"].rank(
        method="average",
        ascending=not METRIC_HIGHER_IS_BETTER[metric],
    )

    return agg.assign(task_type=task_type, metric=metric)

SCATTER_LABEL_COUNT = 3

# Best and worst configurations by overall mean rank
def rank_extremes(coordinates: pd.DataFrame, count: int = SCATTER_LABEL_COUNT) -> list[str]:

    overall = coordinates["mean_rank_overall"]
    best    = set(overall.nsmallest(count, keep="all").index)
    worst   = set(overall.nlargest(count, keep="all").index)

    return [f"top {count}" if i in best else f"bottom {count}" if i in worst else "" for i in coordinates.index]

# Per-task ranks and mean-rank coordinates of every configuration drawn in panel c; the
# companion passes Q2F3 and AUROC, ranked the same way from their own seed means
def configuration_scatter_data(
    seed_df: pd.DataFrame,
    metrics: tuple[str, str] = ("rmse", "balanced_acc"),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    regression     = configuration_rank_table(seed_df, MAIN_REG_TASKS, "regression", metrics[0])
    classification = configuration_rank_table(seed_df, MAIN_CLS_TASKS, "classification", metrics[1])

    complete = {
        name: {
            key
            for key, count in frame.groupby(["model", "rep"])["task"].nunique().items()
            if count == len(MAIN_REG_TASKS)
        }
        for name, frame in (("regression", regression), ("classification", classification))
    }

    kept     = complete["regression"] & complete["classification"]
    offered  = complete["regression"] | complete["classification"]
    dropped  = sorted(offered - kept)

    if dropped:
        print(f"  Excluded from the configuration scatter (incomplete coverage): {dropped}")

    long = pd.concat([regression, classification], ignore_index=True)
    long = long.loc[[key in kept for key in zip(long["model"], long["rep"])]].copy()

    coordinates = (
        long.pivot_table(index=["model", "rep"], columns="task_type", values="rank", aggfunc="mean")
        .rename(columns={"regression": "mean_rank_regression", "classification": "mean_rank_classification"})
        .reset_index()
    )

    details = [configuration_representation(m, r) for m, r in zip(coordinates["model"], coordinates["rep"])]

    coordinates["family"]                = coordinates["model"].map(MODEL_GROUP_MAP)
    coordinates["family_label"]          = [FAMILY_LABEL.get(f, f) for f in coordinates["family"]]
    coordinates["base_representation"]   = [d[0] for d in details]
    coordinates["representation"]        = [d[1] for d in details]
    coordinates["marker_family"]         = [representation_marker_family(d[0]) for d in details]
    coordinates["physchem_augmented"]    = [d[2] for d in details]
    coordinates["configuration_label"]   = [
        f"{label} / {rep}" for label, rep in zip(coordinates["family_label"], coordinates["representation"])
    ]

    coordinates["figure_label"] = list(coordinates["family_label"])
    coordinates["mean_rank_overall"] = coordinates[["mean_rank_regression", "mean_rank_classification"]].mean(axis=1)
    coordinates["rank_extreme"]      = rank_extremes(coordinates)

    coordinates = coordinates.sort_values(["mean_rank_regression", "mean_rank_classification"]).reset_index(drop=True)

    pool = dict(
        split_method     = CANONICAL_BUTINA_SPLIT,
        n_configurations = int(len(coordinates)),
        n_tasks_each     = len(MAIN_REG_TASKS),
        includes         = "one fingerprint radius, no PhysChem augmentation",
        excludes         = "training-mean predictor and any model outside the benchmark families",
        dropped          = ";".join(f"{m}|{r}" for m, r in dropped) if dropped else "none",
    )

    return long, coordinates, pool

# Draw the all-configuration rank scatter
def draw_configuration_rank_scatter(
    ax: plt.Axes,
    coordinates: pd.DataFrame,
    axis_labels: tuple[str, str] = ("Mean within-task regression rank", "Mean within-task classification rank"),
    encoding: str = "representation",
) -> None:

    x      = coordinates["mean_rank_regression"].to_numpy(dtype=float)
    y      = coordinates["mean_rank_classification"].to_numpy(dtype=float)
    colors = joint_pca_point_colors(coordinates, encoding)

    limit_lo = max(0.0, float(min(x.min(), y.min())) - 2.5)
    limit_hi = float(max(x.max(), y.max())) + 2.5

    ax.plot([limit_lo, limit_hi], [limit_lo, limit_hi],
            color=SPINE_COLOR, linewidth=0.8, linestyle=(0, (4, 4)), zorder=1)

    for x_i, y_i, color, augmented in zip(x, y, colors, coordinates["physchem_augmented"]):
        draw_configuration_marker(ax, x_i, y_i, color, augmented)

    ax.set_xlim(limit_lo, limit_hi)
    ax.set_ylim(limit_lo, limit_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    _polish_axis(ax, "none")

LABEL_HALO = [patheffects.Stroke(linewidth=2.2, foreground=AX_FACE)]

LEADER_HALO = [patheffects.withStroke(linewidth=1.6, foreground=AX_FACE)]

SCATTER_MARKER_EDGE      = float(np.sqrt(CIRCLE_MARKER_SIZE * PCA_MARKER_SCALE)) / 2.0

SCATTER_MARKER_RADIUS    = SCATTER_MARKER_EDGE + 0.63

SCATTER_LEADER_STEPS     = (10.0, 15.0, 22.0, 31.0, 42.0, 56.0)

SCATTER_ELBOW_STEPS      = (7.0, 11.0, 16.0, 22.0)

SCATTER_DIAGONAL_STEPS   = (8.0, 12.0, 16.0)

SCATTER_LABEL_CLEARANCE  = SCATTER_MARKER_EDGE + 1.63

SCATTER_FRAME_CLEARANCE  = 1.5

SCATTER_LABEL_SPACING    = 5.0

SCATTER_LEADER_CLEARANCE = SCATTER_MARKER_EDGE + 1.63

SCATTER_REFINE_SWEEPS    = 8

# Halo beneath a plain, editable label
def annotate_with_halo(ax: plt.Axes, text: str, **kwargs) -> None:

    ax.annotate(text, **{**kwargs, "color": AX_FACE}, path_effects=LABEL_HALO)
    ax.annotate(text, **kwargs)

# Orthogonal leader routes, in points from the marker
def leader_routes() -> list[tuple[list[tuple[float, float]], str, int]]:

    routes = []
    for step in SCATTER_LEADER_STEPS:
        for side in (1, -1):
            routes.append(([(0.0, 0.0), (side * step, 0.0)], "h", side))
            routes.append(([(0.0, 0.0), (0.0, side * step)], "v", side))
            for elbow in SCATTER_ELBOW_STEPS:
                for turn in (1, -1):
                    routes.append(([(0.0, 0.0), (turn * elbow, 0.0), (turn * elbow, side * step)], "v", side))
                    routes.append(([(0.0, 0.0), (0.0, turn * elbow), (side * step, turn * elbow)], "h", side))

    return routes

# Text placement at the end of each route
def label_candidates() -> list[tuple[list[tuple[float, float]], tuple[float, float], str, str]]:

    candidates = []
    for route, ending, side in leader_routes():
        x, y = route[-1]
        if ending == "h":
            candidates.append((route, (x + 2.0 * side, y), "left" if side > 0 else "right", "center"))
            continue
        for ha, inset in (("left", -3.0), ("center", 0.0), ("right", 3.0)):
            candidates.append((route, (x + inset, y + 2.0 * side), ha, "bottom" if side > 0 else "top"))

    for step in SCATTER_DIAGONAL_STEPS:
        for sx in (1, -1):
            for sy in (1, -1):
                d = step / np.sqrt(2.0)
                candidates.append(([(0.0, 0.0), (sx * d, sy * d)], (sx * (d + 1.5), sy * (d + 1.5)),
                                   "left" if sx > 0 else "right", "bottom" if sy > 0 else "top"))

    return candidates

# Whether a segment passes through a box
def segment_crosses_box(box, start: np.ndarray, end: np.ndarray) -> bool:

    delta       = end - start
    enter, exit = 0.0, 1.0
    for direction, distance in ((-delta[0], start[0] - box.x0), (delta[0], box.x1 - start[0]),
                                (-delta[1], start[1] - box.y0), (delta[1], box.y1 - start[1])):
        if direction == 0.0:
            if distance < 0.0:
                return False
            continue
        crossing = distance / direction
        if direction < 0.0:
            enter = max(enter, crossing)
        else:
            exit = min(exit, crossing)
        if enter > exit:
            return False

    return True

# Whether two segments cross
def segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:

    def turn(p, q, r) -> float:
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    return turn(a, b, c) * turn(a, b, d) < 0 and turn(c, d, a) * turn(c, d, b) < 0

# Distance of each point to a segment
def segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    t = np.clip((points - start) @ direction / max(float(direction @ direction), 1e-12), 0.0, 1.0)
    return np.hypot(*(points - (start + t[:, None] * direction)).T)

# Name every configuration clear of the markers and of the other labels
def label_configurations(
    ax: plt.Axes,
    coordinates: pd.DataFrame,
    columns: tuple[str, str] = ("mean_rank_regression", "mean_rank_classification"),
    guides: list = (),
) -> None:

    ax.figure.canvas.draw()
    renderer   = ax.figure.canvas.get_renderer()
    scale      = ax.figure.dpi / 72.0
    clearance  = SCATTER_LABEL_CLEARANCE * scale
    leader_gap = SCATTER_LEADER_CLEARANCE * scale
    margin     = SCATTER_FRAME_CLEARANCE * scale
    spacing    = SCATTER_LABEL_SPACING * scale
    points     = coordinates[list(columns)].to_numpy(dtype=float)
    markers    = ax.transData.transform(points)
    to_data    = ax.transData.inverted()
    frame      = ax.bbox
    lines      = [ax.transData.transform(np.array(guide, dtype=float)) for guide in guides]
    options: dict[int, list] = {}

    def inside(box, pts: np.ndarray) -> int:
        return int(np.sum((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1) & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)))

    def leader(idx: int, route: list) -> np.ndarray:
        route    = np.array(route, dtype=float)
        route[0] = route[1] * SCATTER_MARKER_RADIUS / np.hypot(*route[1])
        return markers[idx] + route * scale

    for idx in range(len(coordinates)):

        probe  = ax.annotate(coordinates["figure_label"].iloc[idx], xy=points[idx], xytext=(0.0, 0.0),
                             textcoords="offset points", fontsize=ANNOTATION_SMALL_SIZE, fontweight="bold")
        others = np.delete(markers, idx, axis=0)
        options[idx] = []

        for candidate in label_candidates():
            route, position, ha, va = candidate
            probe.set_position(position)
            probe.set_ha(ha)
            probe.set_va(va)
            edge = probe.get_window_extent(renderer).padded(margin)
            if not (frame.x0 <= edge.x0 and edge.x1 <= frame.x1 and frame.y0 <= edge.y0 and edge.y1 <= frame.y1):
                continue
            box      = edge.padded(clearance - margin)
            path     = leader(idx, route)
            segments = list(zip(path[:-1], path[1:]))
            length   = sum(np.hypot(*(b - a)) for a, b in segments) / scale
            static   = (
                2.0 * length + 0.04 * length ** 2
                + 10.0 * (len(segments) > 1)
                + 500.0 * inside(box, others)
                + 900.0 * sum(int(np.sum(segment_distances(others, a, b) < leader_gap)) for a, b in segments)
                + 40.0 * sum(segment_crosses_box(box, *line) for line in lines)
            )
            reach = Bbox.union([edge.padded(spacing), Bbox.from_extents(*path.min(axis=0), *path.max(axis=0))])
            options[idx].append((static, edge, candidate, segments, reach))

        probe.remove()

        if not options[idx]:
            print(f"  WARNING: no room to label {coordinates['configuration_label'].iloc[idx]} in the rank scatter.")
            del options[idx]

    def clash(option, chosen: list) -> float:
        return sum(
            1e3 * option[1].overlaps(edge) + 120.0 * option[1].padded(spacing).overlaps(edge)
            + 300.0 * sum(segment_crosses_box(edge, a, b) for a, b in option[3])
            + 300.0 * sum(segment_crosses_box(option[1], a, b) for a, b in segments)
            + 500.0 * sum(segments_cross(a, b, c, d) for a, b in option[3] for c, d in segments)
            for _, edge, _, segments, reach in chosen if option[4].overlaps(reach)
        )

    def total(plan: dict) -> float:
        chosen = list(plan.values())
        return sum(option[0] + clash(option, chosen[i + 1:]) for i, option in enumerate(chosen))

    def place(idx: int, plan: dict) -> tuple:
        rest = [option for other, option in plan.items() if other != idx]
        return min(options[idx], key=lambda o: o[0] + clash(o, rest))

    plan: dict[int, tuple] = {}
    for idx in sorted(options, key=lambda idx: -min(option[0] for option in options[idx])):
        plan[idx] = place(idx, plan)

    for _ in range(SCATTER_REFINE_SWEEPS):
        before = total(plan)
        for idx in plan:
            plan[idx] = place(idx, plan)
        for idx in list(plan):
            rest  = [option for other, option in plan.items() if other != idx]
            share = plan[idx][0] + clash(plan[idx], rest)
            for option in sorted(options[idx], key=lambda o: o[0]):
                if option[0] >= share:
                    break
                trial = {**plan, idx: option}
                for other in [other for other in plan if other != idx and clash(option, [plan[other]]) > 0]:
                    trial[other] = place(other, trial)
                if total(trial) < total(plan) - 1e-9:
                    plan = trial
                    break
        if total(plan) >= before - 1e-9:
            break

    colors = joint_pca_point_colors(coordinates, "representation")

    for idx, (_, _, (route, position, ha, va), _, _) in plan.items():
        ink  = shade(colors[idx], SCATTER_LABEL_DARKEN)
        path = to_data.transform(leader(idx, route))
        ax.plot(path[:, 0], path[:, 1], color=ink, linewidth=0.7, zorder=4,
                solid_capstyle="butt", solid_joinstyle="miter", path_effects=LEADER_HALO)
        annotate_with_halo(
            ax, coordinates["figure_label"].iloc[idx], xy=points[idx], xytext=position,
            textcoords="offset points", ha=ha, va=va, fontsize=ANNOTATION_SMALL_SIZE,
            fontweight="bold", color=ink, zorder=5,
        )

# ---------------------------------------------------------------------------
# Figure 3, top row, performance variant: the two heatmaps carry performance
# rather than mean ranks. Panel c keeps the rank calculation unchanged.
# ---------------------------------------------------------------------------

# Column headers of both performance heatmaps
PERFORMANCE_COLUMNS = [
    ("regression",     "rmse",         "val_rmse",         "Regression\nperformance"),
    ("classification", "balanced_acc", "val_balanced_acc", "Classification\nperformance"),
]

# Two ramps, both light at 0 and dark at 1
RMSE_SCORE_COLORS = ["#F2EFEA", "#A8C9C2", "#5A9E92", "#1E7A78", "#11384A"]

MEAN_BA_COLORS    = ["#EEF1F4", "#B3C4D1", "#6F8FA8", "#2F5A78", "#132B40"]

# A task whose reference configurations all score the same cannot separate methods
DEGENERATE_TASK_SCORE = 0.5

# Per-task best and worst mean RMSE over the 44-configuration reference set
def relative_rmse_reference(seed_df: pd.DataFrame) -> pd.DataFrame:

    reference = configuration_rank_table(seed_df, MAIN_REG_TASKS, "regression", "rmse")

    rows = []
    for task, group in reference.groupby("task"):
        ordered = group.sort_values("test_mean").reset_index(drop=True)
        rows.append(dict(
            task            = task,
            task_label      = TASK_LABEL[task],
            n_reference     = int(len(ordered)),
            min_rmse        = float(ordered["test_mean"].iloc[0]),
            max_rmse        = float(ordered["test_mean"].iloc[-1]),
            min_config      = f"{ordered['model'].iloc[0]}|{ordered['rep'].iloc[0]}",
            max_config      = f"{ordered['model'].iloc[-1]}|{ordered['rep'].iloc[-1]}",
            degenerate      = bool(np.isclose(ordered["test_mean"].iloc[0], ordered["test_mean"].iloc[-1])),
        ))

    return pd.DataFrame(rows).set_index("task")

# Min-max inverted RMSE on the common per-task reference scale
def relative_rmse_score(mean_rmse: float, task: str, reference: pd.DataFrame) -> float:

    row = reference.loc[task]

    if row["degenerate"]:
        return DEGENERATE_TASK_SCORE

    return float((row["max_rmse"] - mean_rmse) / (row["max_rmse"] - row["min_rmse"]))

# Per-task performance of the validation-selected configuration of every group, and the
# six-task means drawn in the heatmap cells
def method_performance_cells(
    seed_df: pd.DataFrame,
    group_col: str,
    compute,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    reference_keys = set(
        zip(
            configuration_rank_table(seed_df, MAIN_REG_TASKS, "regression", "rmse")["model"],
            configuration_rank_table(seed_df, MAIN_REG_TASKS, "regression", "rmse")["rep"],
        )
    )

    per_task: list[pd.DataFrame] = []

    for task_type, metric, val_metric, _ in PERFORMANCE_COLUMNS:

        tasks    = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS
        selected = compute(seed_df, tasks, task_type, metric, val_metric).assign(formulation=task_type)

        if task_type == "regression":

            outside = {
                (model, rep)
                for model, rep in zip(selected["model"], selected["rep"])
                if (model, rep) not in reference_keys
            }
            if outside:
                raise RuntimeError(
                    f"{group_col}: validation-selected configurations outside the "
                    f"relative-RMSE reference set: {sorted(outside)}"
                )

            selected["score"] = [
                relative_rmse_score(value, task, reference)
                for value, task in zip(selected["test_mean"], selected["task"])
            ]
        else:
            ba_reference = minmax_reference(seed_df, tasks, task_type, metric)
            selected["score"] = [
                minmax_score(value, task, ba_reference, higher_is_better=True)
                for value, task in zip(selected["test_mean"], selected["task"])
            ]

        per_task.append(selected)

    long = pd.concat(per_task, ignore_index=True)

    counts = long.groupby(["formulation", group_col])["task"].nunique()
    if not (counts == len(MAIN_REG_TASKS)).all():
        print(f"  WARNING: {group_col} groups without all six tasks:\n{counts[counts != len(MAIN_REG_TASKS)]}")

    cells = (
        long.groupby(["formulation", group_col])
        .agg(value=("score", "mean"), n_tasks=("score", "size"))
        .reset_index()
    )

    return long, cells

# Draw one two-column performance heatmap; returns the two images for the shared colour bars
def text_contrast_crossover(cmap) -> float:

    def relative_luminance(color) -> float:
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in color[:3]]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def ratio(one: float, other: float) -> float:
        return (max(one, other) + 0.05) / (min(one, other) + 0.05)

    pale = relative_luminance(to_rgb(AX_FACE))
    dark = relative_luminance(to_rgb(TEXT_COLOR))

    for step in np.linspace(0.0, 1.0, 401):
        background = relative_luminance(cmap(float(step)))
        if ratio(pale, background) > ratio(dark, background):
            return float(step)

    return 1.0

# Draw one performance heatmap: rows are groups, columns are tasks, cells carry the relative score
def draw_performance_heatmap(
    ax: plt.Axes,
    cells: pd.DataFrame,
    group_col: str,
    order: list[str],
    limits: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
) -> list:

    table = (
        cells.pivot(index=group_col, columns="formulation", values="value")
        .reindex(index=order, columns=["regression", "classification"])
        .to_numpy(dtype=float)
    )

    images     = []
    crossovers = []

    for column, colors in enumerate((RMSE_SCORE_COLORS, MEAN_BA_COLORS)):

        panel = np.full_like(table, np.nan)
        panel[:, column] = table[:, column]
        cmap  = sns.blend_palette(colors, as_cmap=True)

        crossovers.append(text_contrast_crossover(cmap))
        images.append(
            ax.pcolormesh(
                np.arange(-0.5, 2), np.arange(-0.5, len(order)), np.ma.masked_invalid(panel),
                cmap=cmap, vmin=limits[column][0], vmax=limits[column][1], zorder=0,
            )
        )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(len(order) - 0.5, -0.5)

    for y_idx, row in enumerate(table):
        for x_idx, value in enumerate(row):
            if not np.isfinite(value):
                continue
            lo, hi = limits[x_idx]
            ax.text(
                x_idx, y_idx, f"{value:.2f}".replace("-", "\u2212"),
                ha="center", va="center", fontsize=HEATMAP_CELL_FONTSIZE,
                color=AX_FACE,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([title for _, _, _, title in PERFORMANCE_COLUMNS], fontsize=HEATMAP_HEADER_FONTSIZE)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.tick_params(axis="both", length=0, pad=2.4, colors=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color=AX_FACE, linewidth=1.1, clip_on=False)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return images

# ---------------------------------------------------------------------------
# Joint regression + classification PCA, and the complete figure 3
# ---------------------------------------------------------------------------

JOINT_PCA_RMSE_PREFIX = "rmse_score"

JOINT_PCA_BA_PREFIX   = "ba"

PCA_ANCHOR_STYLE = dict(marker="s", s=34.0, facecolor=AX_FACE, linewidth=1.25)

# Per-task best and worst mean value of a metric over the 44-configuration reference set
def minmax_reference(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> pd.DataFrame:

    reference = configuration_rank_table(seed_df, tasks, task_type, metric)

    rows = []
    for task, group in reference.groupby("task"):
        ordered = group.sort_values("test_mean").reset_index(drop=True)
        rows.append(dict(
            task         = task,
            task_label   = TASK_LABEL[task],
            metric       = metric,
            n_reference  = int(len(ordered)),
            min_observed = float(ordered["test_mean"].iloc[0]),
            max_observed = float(ordered["test_mean"].iloc[-1]),
            min_config   = f"{ordered['model'].iloc[0]}|{ordered['rep'].iloc[0]}",
            max_config   = f"{ordered['model'].iloc[-1]}|{ordered['rep'].iloc[-1]}",
            degenerate   = bool(np.isclose(ordered["test_mean"].iloc[0], ordered["test_mean"].iloc[-1])),
        ))

    return pd.DataFrame(rows).set_index("task")

# Min-max score on that reference, 1 at the best and 0 at the worst value observed on the task
def minmax_score(value: float, task: str, reference: pd.DataFrame, higher_is_better: bool) -> float:

    row  = reference.loc[task]
    span = row["max_observed"] - row["min_observed"]

    if row["degenerate"]:
        return DEGENERATE_TASK_SCORE

    score = (value - row["min_observed"]) / span

    return float(score if higher_is_better else 1.0 - score)

# Per-task best and worst mean balanced accuracy over the same 44-configuration set
def balanced_accuracy_reference(seed_df: pd.DataFrame) -> pd.DataFrame:

    reference = minmax_reference(seed_df, MAIN_CLS_TASKS, "classification", "balanced_acc")

    return reference.rename(columns={"min_observed": "min_ba", "max_observed": "max_ba"})

# Min-max scaled balanced accuracy on the common per-task reference scale
def relative_ba_score(mean_ba: float, task: str, reference: pd.DataFrame) -> float:

    row = reference.loc[task]

    if row["degenerate"]:
        return DEGENERATE_TASK_SCORE

    return float((mean_ba - row["min_ba"]) / (row["max_ba"] - row["min_ba"]))

# The 44 x 12 matrix behind the joint PCA, with the untransformed values beside it
def joint_pca_matrix(seed_df: pd.DataFrame, ba_mode: str = JOINT_PCA_BA_MODE,
                     pool=main_pool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if ba_mode not in {"raw", "minmax"}:
        raise ValueError(f"ba_mode must be 'raw' or 'minmax', not {ba_mode!r}")

    reference = relative_rmse_reference(seed_df)

    regression     = configuration_rank_table(seed_df, MAIN_REG_TASKS, "regression", "rmse", pool=pool)
    classification = configuration_rank_table(seed_df, MAIN_CLS_TASKS, "classification", "balanced_acc", pool=pool)

    regression = regression.assign(
        transformed=[relative_rmse_score(value, task, reference)
                     for value, task in zip(regression["test_mean"], regression["task"])]
    )

    if ba_mode == "minmax":
        ba_reference   = balanced_accuracy_reference(seed_df)
        classification = classification.assign(
            transformed=[relative_ba_score(value, task, ba_reference)
                         for value, task in zip(classification["test_mean"], classification["task"])]
        )
    else:
        classification = classification.assign(transformed=classification["test_mean"])

    original = pd.concat([regression, classification], ignore_index=True).rename(
        columns={"test_mean": "original_value"}
    )

    wide_regression = regression.pivot(index=["model", "rep"], columns="task", values="transformed")[MAIN_REG_TASKS]
    wide_classification = classification.pivot(index=["model", "rep"], columns="task", values="transformed")[MAIN_CLS_TASKS]

    wide_regression.columns     = [f"{JOINT_PCA_RMSE_PREFIX}::{task}" for task in wide_regression.columns]
    wide_classification.columns = [f"{JOINT_PCA_BA_PREFIX}::{task}" for task in wide_classification.columns]

    matrix = wide_regression.join(wide_classification, how="inner")

    if matrix.isna().any().any():
        missing = matrix.columns[matrix.isna().any()].tolist()
        raise RuntimeError(f"joint PCA matrix has missing cells in {missing}; nothing is imputed")

    expected = set(zip(regression["model"], regression["rep"])) & set(zip(classification["model"], classification["rep"]))
    if len(matrix) != len(expected):
        raise RuntimeError(f"joint PCA matrix has {len(matrix)} rows but {len(expected)} configurations are complete in both formulations")

    details = [configuration_representation(model, rep) for model, rep in matrix.index]
    families = [MODEL_GROUP_MAP.get(model) for model, _ in matrix.index]

    meta = pd.DataFrame(
        {
            "model":               [model for model, _ in matrix.index],
            "rep":                 [rep for _, rep in matrix.index],
            "family":              families,
            "family_label":        [FAMILY_LABEL.get(f, f) for f in families],
            "base_representation": [d[0] for d in details],
            "representation":      [d[1] for d in details],
            "marker_family":       [representation_marker_family(d[0]) for d in details],
            "physchem_augmented":  [d[2] for d in details],
        }
    )
    meta["configuration_label"] = [
        f"{label} / {rep}" for label, rep in zip(meta["family_label"], meta["representation"])
    ]

    return original, matrix, meta

# Fit the joint PCA on the real configurations plus the synthetic Best and Worst profiles
def compute_joint_pca(
    seed_df: pd.DataFrame,
    ba_mode: str = JOINT_PCA_BA_MODE,
    fit_mode: str = JOINT_PCA_FIT_MODE,
    view: str = "primary",
    pool=main_pool,
) -> dict:

    if fit_mode not in {"with_references", "projected"}:
        raise ValueError(f"fit_mode must be 'with_references' or 'projected', not {fit_mode!r}")

    if view == "companion":
        original, matrix, meta = companion_pca_matrix(seed_df)
        prefixes = (COMPANION_PCA_QF3_PREFIX, COMPANION_PCA_AUROC_PREFIX)
    else:
        original, matrix, meta = joint_pca_matrix(seed_df, ba_mode=ba_mode, pool=pool)
        prefixes = (JOINT_PCA_RMSE_PREFIX, JOINT_PCA_BA_PREFIX)

    observed = matrix.to_numpy(dtype=float)
    best     = observed.max(axis=0)
    worst    = observed.min(axis=0)
    anchors  = np.vstack([best, worst])

    constant = [name for name, column in zip(matrix.columns, observed.T) if np.isclose(column.max(), column.min())]
    if constant:
        print(f"  Note: constant PCA columns contribute nothing after standardization: {constant}")

    fitted_on = np.vstack([observed, anchors]) if fit_mode == "with_references" else observed

    column_means  = fitted_on.mean(axis=0)
    column_scales = fitted_on.std(axis=0, ddof=0)
    column_scales = np.where(np.isclose(column_scales, 0.0), 1.0, column_scales)

    def standardize(values: np.ndarray) -> np.ndarray:
        return (values - column_means) / column_scales

    pca = PCA(n_components=min(fitted_on.shape))
    pca.fit(standardize(fitted_on))

    configuration_scores = pca.transform(standardize(observed))[:, :2]
    anchor_scores        = pca.transform(standardize(anchors))[:, :2]
    components           = pca.components_[:2].copy()

    orientation          = np.where(anchor_scores[0] < 0.0, -1.0, 1.0)
    configuration_scores = configuration_scores * orientation
    anchor_scores        = anchor_scores * orientation
    components           = components * orientation[:, None]

    best_xy, worst_xy = anchor_scores[0], anchor_scores[1]
    fit_matrix        = np.vstack([observed, anchors])

    return dict(
        original      = original,
        matrix        = matrix,
        meta          = meta,
        fit_matrix    = fit_matrix,
        column_means  = column_means,
        column_scales = column_scales,
        best_profile  = best,
        worst_profile = worst,
        scores        = configuration_scores,
        best_xy       = best_xy,
        worst_xy      = worst_xy,
        components    = components,
        evr           = pca.explained_variance_ratio_,
        explained_variance = pca.explained_variance_,
        orientation   = orientation,
        constant_columns = constant,
        ba_mode       = ba_mode,
        fit_mode      = fit_mode,
        n_fitted_rows = int(len(fitted_on)),
        prefixes      = prefixes,
        view          = view,
    )

# Spread and orientation diagnostics for the joint PCA
def joint_pca_diagnostics(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    matrix     = data["matrix"]
    observed   = matrix.to_numpy(dtype=float)
    fit_matrix = data["fit_matrix"]
    prefixes   = data.get("prefixes", (JOINT_PCA_RMSE_PREFIX, JOINT_PCA_BA_PREFIX))
    blocks     = [prefixes[0]] * len(MAIN_REG_TASKS) + [prefixes[1]] * len(MAIN_CLS_TASKS)

    columns = pd.DataFrame(
        {
            "column":                  list(matrix.columns),
            "block":                   blocks,
            "variance_real":          observed.var(axis=0, ddof=0),
            "variance_fit_matrix":    fit_matrix.var(axis=0, ddof=0),
            "min_44_real":             observed.min(axis=0),
            "max_44_real":             observed.max(axis=0),
        }
    )

    block_rows = []
    for block in prefixes:
        mask = np.array([b == block for b in blocks])
        block_rows.append(dict(
            block                       = block,
            n_columns                   = int(mask.sum()),
            summed_variance_real        = float(observed[:, mask].var(axis=0, ddof=0).sum()),
            summed_variance_fit         = float(fit_matrix[:, mask].var(axis=0, ddof=0).sum()),
            PC1_sum_squared_loadings    = float(np.sum(data["components"][0][mask] ** 2)),
            PC2_sum_squared_loadings    = float(np.sum(data["components"][1][mask] ** 2)),
        ))
    blocks_table = pd.DataFrame(block_rows)

    variance = pd.DataFrame(
        {
            "component":          [f"PC{i + 1}" for i in range(len(data["evr"]))],
            "explained_variance": data["explained_variance"],
            "explained_variance_ratio": data["evr"],
            "cumulative_ratio":   np.cumsum(data["evr"]),
        }
    )

    return columns, blocks_table, variance

# Colour of every PCA point: the model family, or the exact representation variant
def joint_pca_point_colors(meta: pd.DataFrame, encoding: str = "model") -> list[str]:

    if encoding == "model":
        return [MODEL_COLORS.get(family, NEUTRAL_MID) for family in meta["family"]]

    if encoding == "representation":
        return [BASE_REPRESENTATION_COLORS.get(rep, NEUTRAL_MID) for rep in meta["base_representation"]]

    raise ValueError(f"encoding must be 'model' or 'representation', not {encoding!r}")

# Draw the joint PCA landscape; the legend is shared with panel c, so none is built here
def draw_joint_pca_panel(ax: plt.Axes, data: dict, encoding: str = "model", shape_by: str | None = None,
                         scale: float = PCA_MARKER_SCALE) -> None:

    scores   = data["scores"]
    meta     = data["meta"]
    best_xy  = data["best_xy"]
    worst_xy = data["worst_xy"]
    evr      = data["evr"]
    colors   = joint_pca_point_colors(meta, encoding)

    zero = dict(color=SPINE_COLOR, linewidth=0.7, linestyle=(0, (4, 4)), zorder=1)
    ax.axhline(0.0, **zero)
    ax.axvline(0.0, **zero)

    ax.plot(
        [worst_xy[0], best_xy[0]], [worst_xy[1], best_xy[1]],
        color=NEUTRAL_CONNECTOR, linewidth=1.0, solid_capstyle="round", zorder=2,
    )

    for (pc1, pc2), color, family, augmented in zip(
        scores, colors, meta["family"], meta["physchem_augmented"]
    ):
        if shape_by == "model":
            draw_configuration_marker(ax, pc1, pc2, color, augmented, scale=scale,
                                      marker=MODEL_MARKERS[family], base_size=MODEL_MARKER_SIZES[family])
        else:
            draw_configuration_marker(ax, pc1, pc2, color, augmented, scale=scale)

    # Neutral squares, unlike any configuration marker, mark the two synthetic profiles
    for xy, name in ((worst_xy, "Worst"), (best_xy, "Best")):
        ax.scatter(xy[0], xy[1], edgecolor=NEUTRAL_LABEL, zorder=6, **PCA_ANCHOR_STYLE)
        annotate_with_halo(
            ax, name, xy=xy, xytext=(0.0, 6.0), textcoords="offset points", ha="center", va="bottom",
            fontsize=ANNOTATION_SIZE, color=NEUTRAL_LABEL, zorder=6,
        )

    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.set_aspect("equal", adjustable="box")
    _polish_axis(ax, "none")

# True when any configuration carries PhysChem descriptors
def any_augmented(meta: pd.DataFrame) -> bool:
    column = meta.get("physchem_augmented")
    return bool(column is not None and column.any())

# Handle of the +PhysChem marker, hollow in every figure that draws one
def physchem_legend_handle() -> plt.Line2D:
    return plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=AX_FACE,
                      markeredgecolor=NEUTRAL_MID, markeredgewidth=1.3, markersize=5.0, label="+ PhysChem (hollow)")

# One coloured handle per model, for the figures that colour by model
def joint_pca_legend_handles(meta: pd.DataFrame) -> list:

    families = list(dict.fromkeys(meta["family"]))
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[family],
                   markeredgecolor=shade(MODEL_COLORS[family], PCA_EDGE_DARKEN), markeredgewidth=0.9,
                   markersize=5.4, label=FAMILY_LABEL.get(family, family))
        for family in MODEL_ORDER_LIST if family in families
    ]
    if any_augmented(meta):
        handles.append(physchem_legend_handle())

    return handles

# One grey handle per model shape, for the PCA that encodes the model by marker
def model_shape_legend_handles(meta: pd.DataFrame) -> list:

    families = set(meta["family"])

    return [
        plt.Line2D([0], [0], marker=MODEL_MARKERS[family], color="none", markerfacecolor=NEUTRAL_MID,
                   markeredgecolor=shade(NEUTRAL_MID, PCA_EDGE_DARKEN), markeredgewidth=0.9,
                   markersize=6.4 if family == "GCN" else 5.0, label=FAMILY_LABEL.get(family, family))
        for family in MODEL_ORDER_LIST if family in families
    ]

# One handle per representation variant, drawn exactly as its unaugmented points are; shape handles, when
# the figure also encodes the model by marker, go between the colours and the closing entries
def representation_pca_legend_handles(meta: pd.DataFrame, shape_handles: list | None = None) -> list:

    present = set(meta["base_representation"])

    handles = [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=BASE_REPRESENTATION_COLORS[rep],
                   markeredgecolor=shade(BASE_REPRESENTATION_COLORS[rep], PCA_EDGE_DARKEN),
                   markeredgewidth=0.9, markersize=5.6,
                   label=BASE_REPRESENTATION_LABEL.get(rep, rep))
        for rep in BASE_REPRESENTATION_ORDER if rep in present
    ]
    closing = [physchem_legend_handle()] if any_augmented(meta) else []

    return handles + list(shape_handles or []) + closing

REPRESENTATION_INPUT_TYPE = {
    "ECFP4b": "Descriptors", "ECFP4c": "Descriptors", "ECFP6b": "Descriptors", "ECFP6c": "Descriptors",
    "MAP4C": "Descriptors", "MAP6C": "Descriptors", "MXFP": "Descriptors", "HDF": "Descriptors",
    "PhysChem": "Descriptors", "SMILES": "Strings", "fragSMILES": "Strings", "Graph": "Graphs",
}

LEGEND_BLOCKS    = ("Descriptors", "Strings", "Graphs", "Models")

LEGEND_BLOCK_GAP = 0.03

# One titled legend block per input type, as in Figure 1, centred together across the figure width
def add_wide_legend(fig: plt.Figure, handles: list, rows: int = 2) -> None:

    input_type = {BASE_REPRESENTATION_LABEL.get(rep, rep): kind for rep, kind in REPRESENTATION_INPUT_TYPE.items()}
    input_type[physchem_legend_handle().get_label()] = "Descriptors"

    blocks: dict[str, list] = {}
    for handle in handles:
        blocks.setdefault(input_type.get(handle.get_label(), "Models"), []).append(handle)
    if "Models" in blocks:
        rows = max(rows, 4)

    legends = []
    for title in LEGEND_BLOCKS:
        if title not in blocks:
            continue
        legend = fig.legend(handles=blocks[title], title=title, loc="upper left", bbox_to_anchor=(0.0, 0.0),
                            ncol=-(-len(blocks[title]) // rows), frameon=False, fontsize=ANNOTATION_SMALL_SIZE,
                            title_fontsize=ANNOTATION_SMALL_SIZE, alignment="left",
                            handletextpad=0.25, columnspacing=0.62, labelspacing=0.34)
        legend.get_title().set_color(TEXT_COLOR)
        legend.get_title().set_fontweight("medium")
        legends.append(legend)

    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    widths = [float(np.ptp(to_figure.transform(legend.get_window_extent())[:, 0])) for legend in legends]
    x = 0.5 - (sum(widths) + LEGEND_BLOCK_GAP * (len(widths) - 1)) / 2
    for legend, width in zip(legends, widths):
        legend.set_bbox_to_anchor((x, 0.0), transform=fig.transFigure)
        x += width + LEGEND_BLOCK_GAP

# The PCA panel on its own canvas, with the encoding legend at its foot
def draw_joint_pca_figure(data: dict, encoding: str = "model", shape_by: str | None = None,
                          panel_label: str | None = "d", scale: float = PCA_MARKER_SCALE) -> plt.Figure:

    if encoding == "model":
        handles = joint_pca_legend_handles(data["meta"])
    else:
        shape_handles = model_shape_legend_handles(data["meta"]) if shape_by == "model" else None
        handles       = representation_pca_legend_handles(data["meta"], shape_handles)

    rows = 3 if shape_by == "model" else 2

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL, 4.70 + 0.30 * (rows - 2)))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.190 + 0.045 * (rows - 2))
    draw_joint_pca_panel(ax, data, encoding=encoding, shape_by=shape_by, scale=scale)
    if panel_label is not None:
        add_panel_label(ax, panel_label, dy=PANEL_LABEL_DY + 2.0)
    add_wide_legend(fig, handles, rows=rows)

    return fig

# ---------------------------------------------------------------------------
# Figure 3: panels a-c above the joint PCA panel d, with one shared legend
# ---------------------------------------------------------------------------

COMPARISON_LEFT    = 0.76

COMPARISON_HEATMAP = 1.075

COMPARISON_GAP     = 0.63

COMPARISON_SIDE    = 2.54

COMPARISON_TOP     = 0.55

COMPARISON_RIGHT   = 0.04

# Panels a-c, plus panel d when a PCA is given
def draw_method_comparison_figure(
    family_cells: pd.DataFrame,
    rep_cells: pd.DataFrame,
    coordinates: pd.DataFrame,
    family_order: list[str],
    rep_order: list[str],
    colorbar_labels: tuple[str, str],
    scatter_labels: tuple[str, str] = ("Mean within-task regression rank", "Mean within-task classification rank"),
    pca: dict | None = None,
) -> plt.Figure:

    width     = FIG_WIDTH_FULL
    rep_left  = COMPARISON_LEFT + COMPARISON_HEATMAP + COMPARISON_GAP
    rep_right = rep_left + COMPARISON_HEATMAP
    row_end   = COMPARISON_TOP + COMPARISON_SIDE
    pca_top   = row_end + 0.95
    height    = pca_top + 3.85 if pca is not None else row_end + 1.15

    fig = plt.figure(figsize=(width, height))

    def place(left: float, top: float, w: float, h: float) -> plt.Axes:
        return fig.add_axes((left / width, 1.0 - (top + h) / height, w / width, h / height))

    ax_family = place(COMPARISON_LEFT, COMPARISON_TOP, COMPARISON_HEATMAP, COMPARISON_SIDE)
    images    = draw_performance_heatmap(ax_family, family_cells, "family", family_order)
    ax_family.set_yticklabels([FAMILY_LABEL.get(name, name) for name in family_order])

    ax_rep = place(rep_left, COMPARISON_TOP, COMPARISON_HEATMAP, COMPARISON_SIDE)
    draw_performance_heatmap(ax_rep, rep_cells, "representation", rep_order)

    ax_scatter = place(width - COMPARISON_RIGHT - COMPARISON_SIDE, COMPARISON_TOP, COMPARISON_SIDE, COMPARISON_SIDE)
    draw_configuration_rank_scatter(ax_scatter, coordinates, scatter_labels)

    for ax, letter in ((ax_family, "a"), (ax_rep, "b"), (ax_scatter, "c")):
        add_panel_label(ax, letter, dy=PANEL_LABEL_DY + 24.0)

    bar_width = (rep_right - COMPARISON_LEFT) * 0.44
    for image, left, label in zip(images, (COMPARISON_LEFT, rep_right - bar_width), colorbar_labels):
        bar = fig.colorbar(image, cax=place(left, row_end + 0.193, bar_width, 0.107), orientation="horizontal")
        bar.set_ticks([0.0, 1.0])
        bar.set_ticklabels(["0.00", "1.00"])
        bar.ax.text(0.5, -1.55, label, transform=bar.ax.transAxes, ha="center", va="top",
                    fontsize=ANNOTATION_SMALL_SIZE, color=TEXT_COLOR)
        bar.ax.tick_params(length=0, labelsize=ANNOTATION_SMALL_SIZE, pad=1.6, colors=TEXT_COLOR)
        bar.outline.set_visible(False)

    if pca is not None:
        ax_pca = place(COMPARISON_LEFT, pca_top, width - COMPARISON_RIGHT - COMPARISON_LEFT, 3.0)
        ax_pca.set_anchor("N")
        draw_joint_pca_panel(ax_pca, pca, encoding="representation")
        add_panel_label(ax_pca, "d", dy=PANEL_LABEL_DY + 2.0)

    add_wide_legend(fig, representation_pca_legend_handles(coordinates))
    lo, hi = ax_scatter.get_xlim()
    label_configurations(ax_scatter, coordinates, guides=[((lo, lo), (hi, hi))])

    if pca is not None:
        scores = pca["meta"].assign(PC1=pca["scores"][:, 0], PC2=pca["scores"][:, 1]).merge(
            coordinates[["model", "rep", "figure_label"]], on=["model", "rep"], how="left", validate="one_to_one"
        )
        (x0, x1), (y0, y1) = ax_pca.get_xlim(), ax_pca.get_ylim()
        label_configurations(ax_pca, scores, columns=("PC1", "PC2"), guides=[
            (pca["worst_xy"], pca["best_xy"]), ((x0, 0.0), (x1, 0.0)), ((0.0, y0), (0.0, y1)),
        ])

    return fig

# Two-way share of the total sum of squares carried by the task and by the model family,
# on the raw metric and on the per-task scale whose extremes are the configuration means
def variance_decomposition(seed_df: pd.DataFrame, tasks: list[str], task_type: str, metric: str) -> pd.DataFrame:

    work = main_pool(
        seed_df.loc[
            (seed_df["task"].isin(tasks))
            & (seed_df["task_type"] == task_type)
            & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
            & (~seed_df["family"].isin(["Baseline", "Other"]))
            & (seed_df["model"] != "mean")
        ]
    ).dropna(subset=[metric])

    configuration_mean = work.groupby(["task", "model", "rep"])[metric].transform("mean")
    lower = configuration_mean.groupby(work["task"]).transform("min")
    upper = configuration_mean.groupby(work["task"]).transform("max")
    work = work.assign(per_task_score=(work[metric] - lower) / (upper - lower))

    rows = []
    for scale, column in (("raw", metric), ("per_task", "per_task_score")):
        values = work[column].to_numpy(dtype=float)
        grand = values.mean()
        total = ((values - grand) ** 2).sum()
        entry = dict(task_type=task_type, metric=metric, scale=scale,
                     n_observations=int(len(work)),
                     n_configurations=int(work.groupby(["model", "rep"]).ngroups))
        for factor in ("task", "family"):
            effect = sum(len(group) * (group.mean() - grand) ** 2 for _, group in work.groupby(factor)[column])
            entry[f"share_{factor}"] = float(effect / total)
        rows.append(entry)

    return pd.DataFrame(rows)

# Figure 3: heatmaps, rank scatter and joint PCA
def plot_method_comparison(seed_df: pd.DataFrame, pca: dict) -> pd.DataFrame:

    reference = relative_rmse_reference(seed_df)
    save_table(reference.reset_index(), "relative_rmse_reference.csv")

    save_table(
        pd.concat([
            variance_decomposition(seed_df, MAIN_CLS_TASKS, "classification", "balanced_acc"),
            variance_decomposition(seed_df, MAIN_REG_TASKS, "regression", "rmse"),
        ], ignore_index=True),
        "variance_decomposition.csv",
    )

    family_long, family_cells = method_performance_cells(
        seed_df, "family", compute_validation_selected_family_ranks, reference
    )
    rep_long, rep_cells = method_performance_cells(
        seed_df, "representation", compute_validation_selected_representation_ranks, reference
    )
    scatter_long, coordinates, pool = configuration_scatter_data(seed_df)

    family_order = method_row_order("family", family_cells["family"])
    rep_order    = method_row_order("representation", rep_cells["representation"])

    fig = draw_method_comparison_figure(
        family_cells, rep_cells, coordinates, family_order, rep_order,
        colorbar_labels=("Relative RMSE score ↑", "Relative BA score ↑"), pca=pca,
    )

    save_source_table(family_long.assign(panel="a"), "method_performance_family_per_task.csv")
    save_source_table(rep_long.assign(panel="b"), "method_performance_representation_per_task.csv")
    save_table(
        pd.concat([
            family_cells.assign(panel="a", grouping="family").rename(columns={"family": "group"}),
            rep_cells.assign(panel="b", grouping="representation").rename(columns={"representation": "group"}),
        ], ignore_index=True),
        "method_performance_heatmap_cells.csv",
    )
    save_source_table(scatter_long.assign(panel="c"), "method_ranks_scatter_per_task.csv")
    save_source_table(coordinates.assign(panel="c", **pool), "method_ranks_scatter_points.csv")
    save_joint_pca_points(pca, "fig_joint_pca_by_representation")

    _save_plot(fig, OUT_ROOT / "main" / "fig_method_comparison.png", align_labels=False)

    return coordinates

# Panel d: the adopted joint PCA, on its own canvas and with no panel title
def plot_joint_pca_figure(seed_df: pd.DataFrame) -> dict:

    data = compute_joint_pca(seed_df)

    columns, blocks, variance = joint_pca_diagnostics(data)

    save_source_table(data["original"], "joint_pca_original_values.csv")
    save_source_table(data["matrix"].reset_index(), "joint_pca_transformed_matrix.csv")
    save_source_table(
        data["meta"].assign(PC1=data["scores"][:, 0], PC2=data["scores"][:, 1],
                            ba_mode=data["ba_mode"], fit_mode=data["fit_mode"], panel="d"),
        "joint_pca_configuration_scores.csv",
    )
    save_source_table(
        pd.DataFrame({
            "profile": ["Best", "Worst"],
            "PC1": [data["best_xy"][0], data["worst_xy"][0]],
            "PC2": [data["best_xy"][1], data["worst_xy"][1]],
            "included_in_fit": data["fit_mode"] == "with_references",
            "placed_by": f"PCA.transform on the {data['n_fitted_rows']}-row fit",
        }),
        "joint_pca_reference_scores.csv",
    )

    rmse_extremes = relative_rmse_reference(seed_df).reset_index().assign(block=JOINT_PCA_RMSE_PREFIX)
    rmse_extremes = rmse_extremes.rename(columns={"min_rmse": "min_observed", "max_rmse": "max_observed"})
    ba_extremes = balanced_accuracy_reference(seed_df).reset_index().assign(block=JOINT_PCA_BA_PREFIX)
    ba_extremes = ba_extremes.rename(columns={"min_ba": "min_observed", "max_ba": "max_observed"})
    save_table(pd.concat([rmse_extremes, ba_extremes], ignore_index=True), "joint_pca_reference_extremes.csv")

    save_table(
        pd.DataFrame({
            "column": list(data["matrix"].columns),
            "block": [JOINT_PCA_RMSE_PREFIX] * len(MAIN_REG_TASKS) + [JOINT_PCA_BA_PREFIX] * len(MAIN_CLS_TASKS),
            "task_label": [TASK_LABEL[t] for t in list(MAIN_REG_TASKS) + list(MAIN_CLS_TASKS)],
            "centring_mean": data["column_means"],
            "scaling_sd": data["column_scales"],
            "best_profile": data["best_profile"],
            "worst_profile": data["worst_profile"],
            "PC1_loading": data["components"][0],
            "PC2_loading": data["components"][1],
        }),
        "joint_pca_loadings.csv",
    )
    save_table(variance, "joint_pca_explained_variance.csv")
    save_table(columns, "joint_pca_column_diagnostics.csv")
    save_table(blocks, "joint_pca_block_diagnostics.csv")

    return data

# Source data of a representation-coloured PCA
def save_joint_pca_points(data: dict, stem: str, shape_by: str | None = None) -> None:
    save_source_table(
        data["meta"].assign(
            PC1=data["scores"][:, 0], PC2=data["scores"][:, 1], view=data["view"],
            fit_mode=data["fit_mode"], encoding="representation",
            point_color=joint_pca_point_colors(data["meta"], "representation"),
            point_marker=[MODEL_MARKERS[family] if shape_by == "model" else "o" for family in data["meta"]["family"]],
        ),
        f"{stem}_points.csv",
    )

# The same PCA fit, on the same canvas, with colour moved from the model to the representation
def plot_joint_pca_by_representation(data: dict, folder: str, stem: str, shape_by: str | None = None,
                                     panel_label: str | None = "d", scale: float = PCA_MARKER_SCALE) -> None:

    _save_plot(draw_joint_pca_figure(data, encoding="representation", shape_by=shape_by,
                                     panel_label=panel_label, scale=scale),
               OUT_ROOT / folder / f"{stem}.png", align_labels=False)

    save_joint_pca_points(data, stem, shape_by)

# Supplementary loadings of the two plotted components, one bar per column of the PCA matrix
def plot_joint_pca_loadings(data: dict, stem: str = "fig_joint_pca_loadings") -> None:

    tasks  = list(MAIN_REG_TASKS) + list(MAIN_CLS_TASKS)
    rows   = list(range(len(MAIN_REG_TASKS))) + [len(MAIN_REG_TASKS) + 0.8 + i for i in range(len(MAIN_CLS_TASKS))]
    colors = [TASK_CATEGORY_COLORS[TASK_CATEGORY_MAP[task]] for task in tasks]
    limit  = 0.05 * np.ceil(np.abs(data["components"]).max() / 0.05) + 0.05

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_SUPP, 3.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.24, wspace=0.12)

    for index, ax in enumerate(axes):
        ax.barh(rows, data["components"][index], height=0.68, color=colors, zorder=2)
        ax.axvline(0.0, color=SPINE_COLOR, linewidth=0.8, zorder=1)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(rows[-1] + 0.7, -0.7)
        ax.set_title(f"PC{index + 1} ({100 * data['evr'][index]:.1f}% of variance)", fontsize=TICK_LABEL_SIZE,
                     color=TEXT_COLOR, pad=6)
        ax.set_xlabel("Loading", fontsize=TICK_LABEL_SIZE, color=TEXT_COLOR)
        ax.tick_params(axis="both", labelsize=ANNOTATION_SMALL_SIZE + 0.6, colors=TEXT_COLOR, length=0)
        ax.grid(axis="x", color=NEUTRAL_CONNECTOR, linewidth=0.4, alpha=0.6, zorder=0)
        ax.grid(axis="y", visible=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SPINE_COLOR)
        add_panel_label(ax, "ab"[index], dy=PANEL_LABEL_DY + 4.0)

    axes[0].set_yticks(rows)
    axes[0].set_yticklabels([TASK_LABEL[task] for task in tasks])

    blended = blended_transform_factory(fig.transFigure, axes[0].transData)
    for first, last, label in ((rows[0], rows[len(MAIN_REG_TASKS) - 1], "Regression"),
                               (rows[len(MAIN_REG_TASKS)], rows[-1], "Classification")):
        fig.text(0.022, (first + last) / 2, label, transform=blended, rotation=90, ha="center", va="center",
                 fontsize=ANNOTATION_SMALL_SIZE + 0.6, fontweight="bold", color=TEXT_COLOR)
        fig.add_artist(plt.Line2D([0.04, 0.04], [first - 0.34, last + 0.34], transform=blended,
                                  color=SPINE_COLOR, linewidth=0.8))

    fig.legend(handles=[Patch(color=TASK_CATEGORY_COLORS[name], label=name) for name in ("Binding", "Permeability", "Half-life")],
               loc="lower center", bbox_to_anchor=(0.59, 0.0), ncol=3, frameon=False,
               fontsize=ANNOTATION_SMALL_SIZE + 0.6, handlelength=1.2, columnspacing=1.6)

    _save_plot(fig, OUT_ROOT / "supplementary" / f"{stem}.png", align_labels=False)

# Figure 3, its supplementary loadings, and the supplementary PCA, which refits on every configuration that was run
def plot_figure_three(seed_df: pd.DataFrame) -> None:
    data = plot_joint_pca_figure(seed_df)
    plot_method_comparison(seed_df, data)
    plot_joint_pca_loadings(data)

    every = compute_joint_pca(seed_df, pool=full_pool)
    print(f"  Supplementary PCA fitted on {len(every['meta'])} configurations.")
    save_table(joint_pca_diagnostics(every)[2], "joint_pca_all_configurations_explained_variance.csv")
    plot_joint_pca_by_representation(every, "supplementary", "fig_joint_pca_by_representation_and_model",
                                     shape_by="model", panel_label=None, scale=1.85)

# ---------------------------------------------------------------------------
# Figure S2: figure 3 on the companion metrics, in the same three pieces
# ---------------------------------------------------------------------------

# Selection and companion metric per formulation
COMPANION_PERFORMANCE_COLUMNS = [
    ("regression",     "rmse",         "val_rmse",         "qf3"),
    ("classification", "balanced_acc", "val_balanced_acc", "auroc"),
]

COMPANION_PCA_QF3_PREFIX   = "qf3_minmax"

COMPANION_PCA_AUROC_PREFIX = "auroc_minmax"

# Per-task companion value of the configuration each group selects on the primary metric
def companion_performance_cells(seed_df: pd.DataFrame, group_col: str, compute) -> tuple[pd.DataFrame, pd.DataFrame]:

    per_task: list[pd.DataFrame] = []

    for task_type, metric, val_metric, companion in COMPANION_PERFORMANCE_COLUMNS:

        tasks    = MAIN_REG_TASKS if task_type == "regression" else MAIN_CLS_TASKS
        selected = compute(seed_df, tasks, task_type, metric, val_metric).rename(
            columns={"test_mean": f"test_mean_{metric}", "rank": f"rank_{metric}"}
        )

        runs = seed_df.loc[
            (seed_df["task"].isin(tasks))
            & (seed_df["task_type"] == task_type)
            & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        ]
        means = (
            runs.groupby(["task", "model", "rep"], dropna=False)
            .agg(value=(companion, "mean"), n_seeds_companion=(companion, "count"))
            .reset_index()
        )

        merged = selected.merge(means, on=["task", "model", "rep"], how="left", validate="one_to_one")
        if merged["value"].isna().any():
            missing = merged.loc[merged["value"].isna(), ["task", group_col, "model", "rep"]]
            raise RuntimeError(f"{group_col}: no {companion} for selected configurations; nothing is imputed:\n{missing}")

        # Both companion metrics improve upwards, so the score is read off the pool directly
        reference      = minmax_reference(seed_df, tasks, task_type, companion)
        merged["mean_companion"] = merged["value"]
        merged["value"] = [
            minmax_score(value, task, reference, higher_is_better=True)
            for value, task in zip(merged["mean_companion"], merged["task"])
        ]

        per_task.append(merged.assign(formulation=task_type, companion_metric=companion))

    long = pd.concat(per_task, ignore_index=True)

    cells = (
        long.groupby(["formulation", group_col])
        .agg(value=("value", "mean"), n_tasks=("value", "size"))
        .reset_index()
    )

    return long, cells

# Panels a-c of figure S2: mean Q2F3 and mean AUROC heatmaps and the companion rank scatter
def plot_method_comparison_companion(seed_df: pd.DataFrame) -> pd.DataFrame:

    family_long, family_cells = companion_performance_cells(seed_df, "family", compute_validation_selected_family_ranks)
    rep_long, rep_cells       = companion_performance_cells(seed_df, "representation", compute_validation_selected_representation_ranks)
    scatter_long, coordinates, pool = configuration_scatter_data(seed_df, metrics=("qf3", "auroc"))

    family_order = method_row_order("family", family_cells["family"])
    rep_order    = method_row_order("representation", rep_cells["representation"])

    fig = draw_method_comparison_figure(
        family_cells, rep_cells, coordinates, family_order, rep_order,
        colorbar_labels=(r"Relative $Q^2_{\mathrm{F3}}$ score ↑", "Relative AUROC score ↑"),
        scatter_labels=(r"Mean within-task regression rank ($Q^2_{\mathrm{F3}}$)",
                        "Mean within-task classification rank (AUROC)"),
    )

    save_source_table(family_long.assign(panel="a"), "method_comparison_companion_family_per_task.csv")
    save_source_table(rep_long.assign(panel="b"), "method_comparison_companion_representation_per_task.csv")
    save_table(
        pd.concat([
            family_cells.assign(panel="a", grouping="family").rename(columns={"family": "group"}),
            rep_cells.assign(panel="b", grouping="representation").rename(columns={"representation": "group"}),
        ], ignore_index=True).assign(
            metric=lambda frame: frame["formulation"].map({"regression": "qf3", "classification": "auroc"}),
            selected_by=lambda frame: frame["formulation"].map({"regression": "val_rmse", "classification": "val_balanced_acc"}),
            colour_min=0.0,
            colour_max=1.0,
        ),
        "method_comparison_companion_heatmap_cells.csv",
    )
    save_source_table(scatter_long.assign(panel="c"), "method_comparison_companion_scatter_per_task.csv")
    save_source_table(coordinates.assign(panel="c", **pool), "method_comparison_companion_scatter_points.csv")

    _save_plot(fig, OUT_ROOT / "supplementary" / "fig_method_comparison_companion.png", align_labels=False)

    return coordinates

# The 44 x 12 matrix behind the companion PCA, with the untransformed values beside it
def companion_pca_matrix(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    _, primary_matrix, meta = joint_pca_matrix(seed_df)
    rows = primary_matrix.index

    originals: list[pd.DataFrame] = []
    blocks: list[pd.DataFrame] = []

    for task_type, tasks, metric, prefix in (
        ("regression",     MAIN_REG_TASKS, "qf3",   COMPANION_PCA_QF3_PREFIX),
        ("classification", MAIN_CLS_TASKS, "auroc", COMPANION_PCA_AUROC_PREFIX),
    ):
        table = configuration_rank_table(seed_df, tasks, task_type, metric)
        table = table.loc[[key in set(rows) for key in zip(table["model"], table["rep"])]].copy()
        wide  = table.pivot(index=["model", "rep"], columns="task", values="test_mean").reindex(rows)[tasks]

        if wide.isna().any().any():
            missing = [(key, task) for task in tasks for key in wide.index[wide[task].isna()]]
            raise RuntimeError(f"companion PCA: no {metric} for {missing}; nothing is imputed")

        low, high = wide.min(axis=0), wide.max(axis=0)
        span      = (high - low).replace(0.0, np.nan)
        scaled    = ((wide - low) / span).fillna(DEGENERATE_TASK_SCORE)

        table["transformed"] = [scaled.loc[(m, r), t] for m, r, t in zip(table["model"], table["rep"], table["task"])]
        originals.append(table.rename(columns={"test_mean": "original_value"}))

        scaled.columns = [f"{prefix}::{task}" for task in tasks]
        blocks.append(scaled)

    matrix = blocks[0].join(blocks[1], how="inner")

    if len(matrix) != len(primary_matrix):
        raise RuntimeError(f"companion PCA has {len(matrix)} rows, the figure 3 PCA {len(primary_matrix)}")

    return pd.concat(originals, ignore_index=True), matrix, meta

# Observed extremes of every companion PCA column, with the configuration that sets each
def companion_pca_extremes(original: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for (metric, task), group in original.groupby(["metric", "task"], sort=False):
        ordered = group.sort_values("original_value").reset_index(drop=True)
        rows.append(dict(
            task         = task,
            task_label   = TASK_LABEL[task],
            metric       = metric,
            block        = COMPANION_PCA_QF3_PREFIX if metric == "qf3" else COMPANION_PCA_AUROC_PREFIX,
            n_reference  = int(len(ordered)),
            min_observed = float(ordered["original_value"].iloc[0]),
            max_observed = float(ordered["original_value"].iloc[-1]),
            min_config   = f"{ordered['model'].iloc[0]}|{ordered['rep'].iloc[0]}",
            max_config   = f"{ordered['model'].iloc[-1]}|{ordered['rep'].iloc[-1]}",
            degenerate   = bool(np.isclose(ordered["original_value"].iloc[0], ordered["original_value"].iloc[-1])),
        ))

    return pd.DataFrame(rows)

# Panel d of figure S2: the companion PCA, fitted afresh, on the canvas of figure 3d
def plot_joint_pca_companion_figure(seed_df: pd.DataFrame) -> dict:

    data = compute_joint_pca(seed_df, view="companion")

    columns, blocks, variance = joint_pca_diagnostics(data)
    tasks = list(MAIN_REG_TASKS) + list(MAIN_CLS_TASKS)

    original_wide = (
        data["original"].pivot(index=["model", "rep"], columns=["metric", "task"], values="original_value")
        .reindex(data["matrix"].index)
    )
    original_wide.columns = [f"{metric}::{task}" for metric, task in original_wide.columns]
    original_wide = original_wide[[f"qf3::{t}" for t in MAIN_REG_TASKS] + [f"auroc::{t}" for t in MAIN_CLS_TASKS]]

    save_source_table(data["original"], "joint_pca_companion_original_values.csv")
    save_source_table(original_wide.reset_index(), "joint_pca_companion_original_matrix.csv")
    save_source_table(data["matrix"].reset_index(), "joint_pca_companion_transformed_matrix.csv")
    save_source_table(
        data["meta"].assign(PC1=data["scores"][:, 0], PC2=data["scores"][:, 1], view="companion",
                            fit_mode=data["fit_mode"], panel="d"),
        "joint_pca_companion_configuration_scores.csv",
    )
    save_source_table(
        pd.DataFrame({
            "profile": ["Best", "Worst"],
            "PC1": [data["best_xy"][0], data["worst_xy"][0]],
            "PC2": [data["best_xy"][1], data["worst_xy"][1]],
            "included_in_fit": data["fit_mode"] == "with_references",
            "placed_by": f"PCA.transform on the {data['n_fitted_rows']}-row fit",
        }),
        "joint_pca_companion_reference_scores.csv",
    )
    save_table(companion_pca_extremes(data["original"]), "joint_pca_companion_reference_extremes.csv")
    save_table(
        pd.DataFrame({
            "column": list(data["matrix"].columns),
            "block": [COMPANION_PCA_QF3_PREFIX] * len(MAIN_REG_TASKS) + [COMPANION_PCA_AUROC_PREFIX] * len(MAIN_CLS_TASKS),
            "task_label": [TASK_LABEL[t] for t in tasks],
            "centring_mean": data["column_means"],
            "scaling_sd": data["column_scales"],
            "best_profile": data["best_profile"],
            "worst_profile": data["worst_profile"],
            "PC1_loading": data["components"][0],
            "PC2_loading": data["components"][1],
        }),
        "joint_pca_companion_loadings.csv",
    )
    save_table(variance, "joint_pca_companion_explained_variance.csv")
    save_table(columns, "joint_pca_companion_column_diagnostics.csv")
    save_table(blocks, "joint_pca_companion_block_diagnostics.csv")

    return data

# Figure S2 in the pieces of figure 3
def plot_figure_three_companion(seed_df: pd.DataFrame) -> None:
    plot_method_comparison_companion(seed_df)
    plot_joint_pca_companion_figure(seed_df)
