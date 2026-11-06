from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cycpepnet.paths import DATA_ROOT
from cycpepnet.benchmark import load_task_table
from cycpepnet.analyze._results_config import ANNOTATION_SIZE, AXIS_LABEL_SIZE, FIG_WIDTH_SUPP, RESULTS_ROOT
from cycpepnet.analyze._results_style import _polish_axis, _save_plot
from cycpepnet.analyze._result_helpers import (
    MAIN_REG_TASKS, OUT_ROOT, PROPERTY_COLOR, TASK_BY_NAME, TASK_LABEL,
    TASK_RESPONSE_SCALE, add_panel_label, task_property,
)
from cycpepnet.analyze._common import CANONICAL_BUTINA_SPLIT
from cycpepnet.utils import get_processed_dataset_dir, load_split

PANEL_LETTERS   = "abcdef"
TITLE_PAD       = 13.0
PREDICTION_SIZE = 9
PREDICTION_ALPHA = 0.30
REFERENCE_COLOR = "0.45"
REFERENCE_WIDTH = 0.9


def regression_selected_configurations() -> dict[str, str]:

    learn = pd.read_csv(RESULTS_ROOT / "learnability.csv")
    learn = learn.loc[(learn["task_type"] == "regression") & (learn["metric"] == "rmse")]

    return dict(zip(learn["task"], learn["best_model"]))


def load_seed_predictions(task: str, model: str, n_seeds: int, roots: list[Path]) -> pd.DataFrame | None:

    measured, predictions = None, []

    for seed in range(n_seeds):

        found = [root / "regression" / "main" / task / f"{model}__seed{seed:02d}.npz" for root in roots]
        found = [path for path in found if path.exists()]

        if not found:
            return None

        payload = np.load(found[0], allow_pickle=True)
        truth   = np.asarray(payload["y_true"], dtype=float)

        if measured is None:
            measured = truth
        elif not np.array_equal(measured, truth):
            raise ValueError(f"Test compounds differ between seeds for {task} / {model}")

        predictions.append(np.asarray(payload["y_pred"], dtype=float))

    return pd.DataFrame({"measured": measured, "predicted": np.mean(predictions, axis=0)})


def task_training_mean(task: str) -> float:

    spec     = TASK_BY_NAME[task]
    y        = load_task_table(spec, data_root=DATA_ROOT)[0][spec.target_column].to_numpy(dtype=float)
    path     = get_processed_dataset_dir(spec.dataset, data_root=DATA_ROOT) / "splits" / f"{task}_{CANONICAL_BUTINA_SPLIT}.json"
    split, _ = load_split(path)

    return float(np.mean(y[split["train"]]))


def plot_predicted_vs_measured(seed_df: pd.DataFrame, n_seeds: int = 5,
                               predictions_root: Path | None = None) -> None:

    roots     = [RESULTS_ROOT] if predictions_root is None else [Path(predictions_root), RESULTS_ROOT]
    chosen    = regression_selected_configurations()
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_SUPP, FIG_WIDTH_SUPP * 0.68))
    skipped   = []

    for index, (ax, task) in enumerate(zip(axes.ravel(), MAIN_REG_TASKS)):

        model = chosen.get(task)
        table = load_seed_predictions(task, model, n_seeds, roots) if model else None

        if table is None:
            skipped.append(f"{task} ({model})")
            ax.set_axis_off()
            continue

        measured  = table["measured"].to_numpy(dtype=float)
        predicted = table["predicted"].to_numpy(dtype=float)

        low  = min(measured.min(), predicted.min())
        high = max(measured.max(), predicted.max())
        pad  = 0.05 * (high - low)
        low, high = low - pad, high + pad

        ax.plot([low, high], [low, high], ls="--", lw=REFERENCE_WIDTH, color=REFERENCE_COLOR, zorder=1)
        ax.axhline(task_training_mean(task), ls=":", lw=REFERENCE_WIDTH, color=REFERENCE_COLOR, zorder=1)
        ax.scatter(measured, predicted, s=PREDICTION_SIZE, alpha=PREDICTION_ALPHA,
                   color=PROPERTY_COLOR[task_property(task)], linewidths=0, zorder=2)

        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(TASK_LABEL.get(task, task), fontsize=ANNOTATION_SIZE + 1, pad=TITLE_PAD)
        ax.annotate(TASK_RESPONSE_SCALE[task], xy=(0.5, 1.02), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=ANNOTATION_SIZE - 0.7)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        _polish_axis(ax, grid_axis="both")
        ax.grid(False)
        ax.xaxis.label.set_size(AXIS_LABEL_SIZE - 2.0)
        ax.yaxis.label.set_size(AXIS_LABEL_SIZE - 2.0)
        add_panel_label(ax, PANEL_LETTERS[index], dy=TITLE_PAD)

    fig.legend(
        handles=[
            Line2D([], [], ls="--", lw=REFERENCE_WIDTH, color=REFERENCE_COLOR, label="Perfect prediction"),
            Line2D([], [], ls=":",  lw=REFERENCE_WIDTH, color=REFERENCE_COLOR, label="Training mean"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
        fontsize=ANNOTATION_SIZE, handletextpad=0.3, columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.14, wspace=0.26, hspace=0.42)

    _save_plot(fig, OUT_ROOT / "supplementary" / "sfig_predicted_vs_measured.png")

    if skipped:
        print(f"  Skipped, incomplete seeds: {'; '.join(skipped)}")
