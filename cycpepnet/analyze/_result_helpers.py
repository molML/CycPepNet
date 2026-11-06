from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycpepnet.paths import DATA_ROOT, INPUT_RESULTS_ROOT
from cycpepnet.benchmark import BENCHMARK_TASKS, load_task_table
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT, TASK_ORDER_CLS, TASK_ORDER_REG
from cycpepnet.analyze._results_config import EXPECTED_N_SEEDS, MODEL_GROUP_MAP, MODEL_ORDER_LIST, PANEL_LABEL_DX, PANEL_LABEL_DY, PANEL_LABEL_SIZE, PANEL_TITLE_DY, PANEL_TITLE_SIZE, PLOTS_DIR, RESULTS_ROOT, TASK_CATEGORY_COLORS, TASK_CATEGORY_MAP, TASK_DISPLAY, TASK_DISPLAY_CLS, TEXT_COLOR
from cycpepnet.analyze._results_data import main_pool, _load_seed_level_results, _prepare_seed_level_results, representation_of
from cycpepnet.utils import get_processed_dataset_dir, load_split

# Constants
RANDOM_SPLIT    = "random"
OUT_ROOT        = PLOTS_DIR
TABLES_DIR      = RESULTS_ROOT / "tables"
SOURCE_DATA_DIR = RESULTS_ROOT / "source_data"
MAIN_REG_TASKS  = list(TASK_ORDER_REG)
MAIN_CLS_TASKS  = list(TASK_ORDER_CLS)

TASK_LABEL = {
    **{k: v.replace("Binding\n(", "").replace("Permeability\n(", "").replace(")", "") for k, v in TASK_DISPLAY.items()},
    **{k: v.replace("Binding\n(", "").replace("Permeability\n(", "").replace(")", "") for k, v in TASK_DISPLAY_CLS.items()},
}

TASK_LABEL.update(
    {
        "peplife2_pepmsnd_peptherdia_reg": "Half-life",
        "peplife2_pepmsnd_peptherdia_cls": "Half-life",
        "cycpeptmpdb_npmmpd_reg": "Merged permeability",
        "cycpeptmpdb_npmmpd_cls": "Merged permeability",
        "cycpeptmpdb_townsend_reg": "Townsend 2020",
        "cycpeptmpdb_townsend_cls": "Townsend 2020",
        "npmmpd_rzepiela_reg": "Rzepiela 2022",
        "npmmpd_rzepiela_cls": "Rzepiela 2022",
    }
)

PROPERTY_COLOR = {
    "Binding": TASK_CATEGORY_COLORS["Binding"],
    "Permeability": TASK_CATEGORY_COLORS["Permeability"],
    "Half-life": TASK_CATEGORY_COLORS["Half-life"],
}

METRIC_LABEL = {
    "ba_anchored": r"Relative balanced accuracy score $\uparrow$",
    "rmse_anchored": r"Relative RMSE score $\uparrow$",
    "auroc_anchored": r"Relative AUROC score $\uparrow$",
    "qf3_anchored": r"Relative $Q^2_{\mathrm{F3}}$ score $\uparrow$",
    "nrmse": r"RMSE / SD$_\mathrm{train}$",
    "qf3": r"$Q^2_{F3}$",
    "r2": r"$R^2$",
    "rmse": "RMSE",
    "auroc": "AUROC",
    "balanced_acc": "Balanced accuracy",
}

METRIC_HIGHER_IS_BETTER = {
    "ba_anchored": True,
    "rmse_anchored": True,
    "auroc_anchored": True,
    "qf3_anchored": True,
    "nrmse": False,
    "qf3": True,
    "r2": True,
    "rmse": False,
    "auroc": True,
    "balanced_acc": True,
}

# Metrics whose no-skill reference is a train-mean predictor rather than zero
METRIC_REFERENCE_VALUE = {
    "nrmse": 1.0,
    "auroc": 0.5,
    "balanced_acc": 0.5,
}

TASK_BY_NAME             = {task.name: task for task in BENCHMARK_TASKS}

# Proportion metrics (balanced accuracy, sensitivity, specificity) span their full scale
UNIT_INTERVAL_TICKS      = [0.0, 0.25, 0.5, 0.75, 1.0]

# The adopted joint PCA
JOINT_PCA_BA_MODE  = "minmax"

JOINT_PCA_FIT_MODE = "with_references"

# Create output directories for plots and tables
def make_dirs() -> None:
    (OUT_ROOT / "main").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "supplementary").mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load and prepare seed-level framework results
def load_seed_results() -> pd.DataFrame:
    df = _prepare_seed_level_results(_load_seed_level_results(INPUT_RESULTS_ROOT))
    if df.empty:
        return df

    for col in [
        "qf3",
        "rmse",
        "r2",
        "val_qf3",
        "val_rmse",
        "val_r2",
        "auroc",
        "balanced_acc",
        "tpr",
        "tnr",
        "precision",
        "val_auroc",
        "val_balanced_acc",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_split_r2(df)
    df = add_nrmse(df)

    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")

    df["family"] = df["model"].map(MODEL_GROUP_MAP).fillna("Other")
    df["property"] = df["task"].map(task_property)
    return df

# Add NRMSE columns derived from QF3²
def add_nrmse(seed_df: pd.DataFrame) -> pd.DataFrame:

    out = seed_df.copy()

    for source_col, target_col in (("qf3", "nrmse"), ("val_qf3", "val_nrmse")):

        if source_col not in out.columns:
            out[target_col] = np.nan
            continue

        values          = pd.to_numeric(out[source_col], errors="coerce")
        out[target_col] = np.sqrt(np.clip(1.0 - values, 0.0, None))

    return out

# R² is not stored in the run JSONs
def add_split_r2(seed_df: pd.DataFrame) -> pd.DataFrame:

    out = seed_df.copy()

    if "r2" not in out.columns:
        out["r2"] = np.nan
    if "val_r2" not in out.columns:
        out["val_r2"] = np.nan

    regression_tasks = sorted(
        set(out.loc[out["task_type"] == "regression", "task"].dropna())
    )

    for task_name in regression_tasks:

        task = TASK_BY_NAME.get(task_name)

        if task is None:
            continue
        try:
            task_df = load_task_table(task, data_root=DATA_ROOT)[0]
        except (FileNotFoundError, OSError, AssertionError, KeyError, ValueError) as exc:
            print(f"  WARNING: could not derive R² for {task_name}: {exc}")
            continue

        y = task_df[task.target_column].to_numpy(dtype=float)
        split_methods = sorted(
            set(out.loc[out["task"] == task_name, "split_method"].dropna())
        )

        for split_method in split_methods:
            split_path = (
                get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT)
                / "splits"
                / f"{task.name}_{split_method}.json"
            )
            if not split_path.exists():
                continue
            split, _ = load_split(split_path)
            mask = (out["task"] == task_name) & (out["split_method"] == split_method)
            for part, rmse_col, r2_col in [
                ("test", "rmse", "r2"),
                ("val", "val_rmse", "val_r2"),
            ]:
                indices = split.get(part)
                if indices is None or len(indices) < 2 or rmse_col not in out.columns:
                    continue
                target_var = float(np.var(y[indices], ddof=0))
                if not np.isfinite(target_var) or target_var <= 0:
                    continue
                derived = 1.0 - out.loc[mask, rmse_col].pow(2) / target_var
                missing_index = out.index[mask & out[r2_col].isna()]
                out.loc[missing_index, r2_col] = derived.reindex(missing_index)

    return out

# Validate seed results
def validate_seed_results(seed_df: pd.DataFrame) -> None:

    required = {"task", "model", "rep", "split_method", "seed", "task_type"}
    missing  = sorted(required - set(seed_df.columns))

    if missing:
        raise RuntimeError(f"Seed-level results are missing required columns: {missing}")

    main   = seed_df.loc[seed_df["task"].isin(MAIN_REG_TASKS + MAIN_CLS_TASKS)].copy()
    counts = (
        main.groupby(["task", "model", "rep", "split_method"], dropna=False)["seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )

    incomplete = counts.loc[counts["n_seeds"] != EXPECTED_N_SEEDS]

    if not incomplete.empty:
        n_bad = len(incomplete)
        print(f"  WARNING: {n_bad} main-task result groups do not have {EXPECTED_N_SEEDS} seeds.")

# Map a task name to its endpoint property
def task_property(task: str) -> str:

    prop = TASK_CATEGORY_MAP.get(task)

    if prop:
        return prop
    if task.startswith("macrocycledb"):
        return "Binding"
    if task.startswith(("cycpeptmpdb", "npmmpd")):
        return "Permeability"
    if task.startswith(("peplife", "half_life")):
        return "Half-life"
    return "Other"

# Return the no-skill reference for a metric
def metric_reference(metric: str) -> float:
    return METRIC_REFERENCE_VALUE.get(metric, 0.0)

# Add a panel label to an axis
def add_panel_label(ax: plt.Axes, label: str, *, dx: float = -PANEL_LABEL_DX,
                    dy: float = PANEL_LABEL_DY, x: float | None = None) -> None:
    ax.annotate(
        label,
        xy=(0.0 if x is None else x, 1.0),
        xycoords="axes fraction" if x is None else (ax.figure.transFigure, ax.transAxes),
        xytext=(dx if x is None else 0.0, dy),
        textcoords="offset points",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
        va="bottom",
        ha="right" if x is None else "left",
        annotation_clip=False,
    )

# Add a compact panel header above an axis.
def add_panel_title(ax: plt.Axes, title: str, *, x: float = 0.5, dy: float = PANEL_TITLE_DY, ha: str = "center",
                    weight: str = "regular") -> None:
    ax.annotate(
        title,
        xy=(x, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, dy),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontweight=weight,
        color=TEXT_COLOR,
        annotation_clip=False,
    )

# The framework's declared order of a grouping, limited to the groups a figure draws
def method_row_order(group_col: str, present) -> list[str]:
    base  = MODEL_ORDER_LIST if group_col == "family" else REP_RANK_ORDER
    names = set(present)
    return [name for name in base if name in names]

# Groups present in a rank row, in the declared order of their grouping
def rank_row_order(
    panel_specs: list[tuple[pd.DataFrame, str]],
    group_col: str,
) -> list[str]:
    present: set[str] = set()
    for ranks, _ in panel_specs:
        if not ranks.empty and group_col in ranks.columns:
            present.update(ranks[group_col].dropna())
    return method_row_order(group_col, present)

# Rank rows for the two groupings the framework compares
def method_rank_rows(
    family_specs: list[tuple[pd.DataFrame, str]],
    representation_specs: list[tuple[pd.DataFrame, str]],
) -> list[tuple[str, list[str], list[tuple[pd.DataFrame, str]]]]:
    return [
        ("family", rank_row_order(family_specs, "family"), family_specs),
        ("representation", rank_row_order(representation_specs, "representation"), representation_specs),
    ]

# Shrink x tick labels until they fit inside their own column
def fit_xtick_labels(ax: plt.Axes, min_size: float = 5.4, fill: float = 0.88) -> None:

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    labels = [
        label
        for label in ax.get_xticklabels()
        if label.get_text() and label.get_visible()
    ]
    ticks = ax.get_xticks()

    if len(labels) < 2 or len(ticks) < 2:
        return

    to_display = ax.transData.transform
    step = abs(to_display((ticks[1], 0.0))[0] - to_display((ticks[0], 0.0))[0])
    available = step * fill

    def widest() -> float:
        return max(label.get_window_extent(renderer).width for label in labels)

    if widest() <= available:
        return

    scaled = max(min_size, labels[0].get_fontsize() * available / widest() * 0.96)
    for label in labels:
        label.set_fontsize(scaled)

    fig.canvas.draw()
    if widest() <= available:
        return

    for label in labels:
        label.set_rotation(30)
        label.set_ha("right")
        label.set_rotation_mode("anchor")

# Save a manuscript table
def save_table(df: pd.DataFrame, name: str) -> None:

    out = TABLES_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out, index=False)

    print(f"  Saved: {out}")

# Save a source-data table
def save_source_table(df: pd.DataFrame, name: str) -> None:

    out = SOURCE_DATA_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out, index=False)

    print(f"  Saved: {out}")

# Response actually modelled by each regression task
TASK_RESPONSE_SCALE: dict[str, str] = {
    "macrocycledb_oprm1_ki_reg":       r"log$_{10}$($K_\mathrm{i}$ / nM)",
    "macrocycledb_mc4r_ki_reg":        r"log$_{10}$($K_\mathrm{i}$ / nM)",
    "macrocycledb_sstr2_ic50_reg":     r"log$_{10}$(IC$_{50}$ / nM)",
    "npmmpd_reg":                      r"log$_{10}$($P_\mathrm{eff}$ / cm s$^{-1}$)",
    "cycpeptmpdb_reg":                 r"log$_{10}$($P_\mathrm{app}$ / cm s$^{-1}$)",
    "peplife2_pepmsnd_peptherdia_reg": r"log$_{10}$($t_{1/2}$ / h)",
}

# Family names carrying a version suffix in figure labels
FAMILY_LABEL = {
    "ChemBERTa":  "ChemBERTa-2",
    "PeptideCLM": "PeptideCLM-2",
}

# Total molecule count per task (train+val+test on the canonical split), exported
def task_dataset_sizes(seed_df: pd.DataFrame) -> dict[str, int]:
    cols = [c for c in ("n_train", "n_val", "n_test") if c in seed_df.columns]
    if not cols:
        return {}
    work = seed_df.loc[seed_df["split_method"] == CANONICAL_BUTINA_SPLIT]
    totals = work[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    return work.assign(_n=totals).groupby("task")["_n"].max().astype(int).to_dict()

REP_RANK_ORDER  = ["ECFP4b", "ECFP4c", "MAP4C", "MXFP", "HDF", "PhysChem", "SMILES", "fragSMILES", "Graph"]

# Direction of a validation criterion, so selecting on it cannot be inverted by mistake
def validation_direction(metric: str, val_metric: str) -> bool:

    base = val_metric.removeprefix("val_")

    if base in METRIC_HIGHER_IS_BETTER and METRIC_HIGHER_IS_BETTER[base] != METRIC_HIGHER_IS_BETTER[metric]:
        raise ValueError(
            f"{val_metric} and {metric} improve in opposite directions; selecting on "
            f"{val_metric} with the direction of {metric} would choose the worst configuration"
        )

    return METRIC_HIGHER_IS_BETTER[metric]

# Validation-selected family ranks: best config per family chosen on validation, ranked on test
def compute_validation_selected_family_ranks(
    seed_df: pd.DataFrame,
    tasks: list[str],
    task_type: str,
    metric: str,
    val_metric: str,
) -> pd.DataFrame:
    work = seed_df.loc[
        (seed_df["task"].isin(tasks))
        & (seed_df["task_type"] == task_type)
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (~seed_df["family"].isin(["Baseline", "Other"]))
    ].copy()
    work = main_pool(work)
    configs = (
        work.groupby(["task", "family", "model", "rep"], dropna=False)
        .agg(validation_mean=(val_metric, "mean"), test_mean=(metric, "mean"))
        .reset_index()
        .dropna(subset=["validation_mean", "test_mean"])
    )
    validation_groups = configs.groupby(["task", "family"])["validation_mean"]
    selected_indices = (
        validation_groups.idxmax()
        if validation_direction(metric, val_metric)
        else validation_groups.idxmin()
    )
    chosen = configs.loc[selected_indices].copy()
    chosen["rank"] = chosen.groupby("task")["test_mean"].rank(
        method="average",
        ascending=not METRIC_HIGHER_IS_BETTER[metric],
    )
    chosen["metric"] = metric
    chosen["selected_by"] = val_metric
    chosen["task_type"] = task_type
    chosen["property"] = chosen["task"].map(task_property)
    chosen["dataset_size"] = chosen["task"].map(task_dataset_sizes(seed_df))
    return chosen

# Validation-selected representation ranks
def compute_validation_selected_representation_ranks(
    seed_df: pd.DataFrame,
    tasks: list[str],
    task_type: str,
    metric: str,
    val_metric: str,
) -> pd.DataFrame:

    work = seed_df.loc[
        (seed_df["task"].isin(tasks))
        & (seed_df["task_type"] == task_type)
        & (seed_df["split_method"] == CANONICAL_BUTINA_SPLIT)
        & (~seed_df["family"].isin(["Baseline", "Other"]))
    ].copy()

    work                   = main_pool(work)
    work["representation"] = [representation_of(m, r) for m, r in zip(work["model"], work["rep"])]
    work                   = work.dropna(subset=["representation"])

    configs = (
        work.groupby(["task", "representation", "model", "rep"], dropna=False)
        .agg(validation_mean=(val_metric, "mean"), test_mean=(metric, "mean"))
        .reset_index()
        .dropna(subset=["validation_mean", "test_mean"])
    )

    validation_groups = configs.groupby(["task", "representation"])["validation_mean"]
    selected_indices  = (
        validation_groups.idxmax()
        if validation_direction(metric, val_metric)
        else validation_groups.idxmin()
    )

    chosen         = configs.loc[selected_indices].copy()
    chosen["rank"] = chosen.groupby("task")["test_mean"].rank(
        method="average",
        ascending=not METRIC_HIGHER_IS_BETTER[metric],
    )

    chosen["metric"]      = metric
    chosen["selected_by"] = val_metric
    chosen["task_type"]   = task_type
    chosen["property"]   = chosen["task"].map(task_property)

    return chosen

# Metrics drawn on exactly 0-1; any other metric keeps a scale that shows all of its values
SPLIT_UNIT_INTERVAL_METRICS = {"balanced_acc", "auroc"}

# Metrics that carry no endpoint units, so every task can be read on one shared axis
SPLIT_DIMENSIONLESS_METRICS = {"qf3"}

# Scale recorded for each observation of a dimensionless metric
SPLIT_METRIC_SCALE = {
    "balanced_acc": "balanced accuracy, 0-1",
    "auroc":        "AUROC, 0-1",
    "qf3":          "Q2F3, dimensionless",
}

SPLIT_OBSERVATION_SIZE    = 2.6

SPLIT_OBSERVATION_ALPHA   = 0.28

SPLIT_BOX_COLOR           = "#5F5A53"

SPLIT_BOX_LINEWIDTH       = 0.75

SPLIT_MEDIAN_LINEWIDTH    = 1.35
