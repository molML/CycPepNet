from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import seaborn as sns
from cycpepnet.analyze._results_config import (
    AXIS_LABEL_SIZE,
    AX_FACE,
    FIG_FACE,
    GRID_COLOR,
    LEGEND_EDGE_COLOR,
    SAVE_DPI,
    SPINE_COLOR,
    TEXT_COLOR,
    TICK_LABEL_SIZE,
)

# Plotting style functions
PRIMARY_FAMILIES: list[str] = [
    "Roboto",
    "Roboto Light",
]
FALLBACK_FAMILIES: list[str] = [
    "Nimbus Sans",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
]

_FONT_SEARCH_DIRS: list[Path] = [
    Path("assets/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local" / "share" / "fonts",
]

_FONTS_REGISTERED = False


# Register every font file found in the local search dirs
def _register_local_fonts() -> bool:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return True

    found = False
    for directory in _FONT_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for font_path in sorted(directory.rglob("*")):
            if font_path.suffix.lower() not in {".otf", ".ttf", ".ttc"}:
                continue
            try:
                fm.fontManager.addfont(str(font_path))
                found = True
            except (RuntimeError, OSError):
                continue

    _FONTS_REGISTERED = True
    return found


# Build the font stack with installed primary faces first
def _resolve_font_stack() -> list[str]:
    _register_local_fonts()
    installed = {f.name for f in fm.fontManager.ttflist}
    available = [name for name in PRIMARY_FAMILIES if name in installed]
    remaining = [name for name in PRIMARY_FAMILIES if name not in available]
    return available + remaining + FALLBACK_FAMILIES


# Shared publication style for every result figure
def _apply_results_plot_style() -> None:

    font_stack = _resolve_font_stack()
    resolved = fm.FontProperties(family=font_stack).get_name()

    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.facecolor": FIG_FACE,
            "axes.facecolor": AX_FACE,
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.labelcolor": TEXT_COLOR,
            "ytick.labelcolor": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.42,
            "grid.alpha": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": font_stack,
            "mathtext.fontset": "custom",
            "mathtext.default": "regular",
            "mathtext.rm": resolved,
            "mathtext.it": f"{resolved}:italic",
            "mathtext.bf": f"{resolved}:bold",
            "mathtext.sf": resolved,
            "mathtext.cal": resolved,
            "font.size": 7.8,
            "axes.linewidth": 0.55,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelpad": 4.0,
            "axes.titlesize": 8.4,
            "axes.titlepad": 5.0,
            "axes.titleweight": "regular",
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.55,
            "legend.fontsize": 6.8,
            "legend.title_fontsize": 7.0,
            "legend.handlelength": 1.18,
            "legend.handletextpad": 0.35,
            "legend.columnspacing": 0.70,
            "legend.labelspacing": 0.25,
            "legend.borderaxespad": 0.18,
            "legend.borderpad": 0.18,
            "legend.facecolor": FIG_FACE,
            "legend.edgecolor": LEGEND_EDGE_COLOR,
            "legend.framealpha": 0.96,
            "figure.dpi": 150,
            "savefig.dpi": SAVE_DPI,
            "savefig.facecolor": FIG_FACE,
            "savefig.edgecolor": FIG_FACE,
            "savefig.bbox": "tight",
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


# Axis styling helper for all plots
def _polish_axis(ax: plt.Axes, grid_axis: str = "y") -> None:

    ax.set_facecolor(AX_FACE)
    ax.set_axisbelow(True)

    if grid_axis == "both":
        ax.grid(True, axis="both", color=GRID_COLOR, linewidth=0.42, alpha=1.0)
    elif grid_axis in ("x", "y"):
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.42, alpha=1.0)
        ax.grid(False, axis="x" if grid_axis == "y" else "y")
    else:
        ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)

    ax.tick_params(length=0, pad=2.4, colors=TEXT_COLOR, labelsize=TICK_LABEL_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)


# Two-decimal tick labels
def _two_decimal_ticks(axis: Axis) -> None:

    axis.set_major_formatter(lambda value, _: f"{round(value, 2) + 0.0:.2f}".replace("-", "\N{MINUS SIGN}"))


# Legend styling helper for all plots
def _style_legend(legend) -> None:

    if legend is None:
        return

    legend.set_frame_on(False)
    legend.handletextpad = min(float(getattr(legend, "handletextpad", 0.35)), 0.40)
    legend.columnspacing = min(float(getattr(legend, "columnspacing", 0.70)), 0.82)
    legend.labelspacing = min(float(getattr(legend, "labelspacing", 0.25)), 0.32)
    legend.borderaxespad = min(float(getattr(legend, "borderaxespad", 0.18)), 0.24)
    legend.borderpad = min(float(getattr(legend, "borderpad", 0.18)), 0.24)

    for text in legend.get_texts():
        text.set_fontsize(min(float(text.get_fontsize()), 6.8))
        text.set_color(TEXT_COLOR)

    title = legend.get_title()
    if title is not None and title.get_text():
        title.set_fontsize(min(float(title.get_fontsize()), 7.0))
        title.set_fontweight("regular")
        title.set_color(TEXT_COLOR)

    frame = legend.get_frame()
    frame.set_facecolor(FIG_FACE)
    frame.set_edgecolor(LEGEND_EDGE_COLOR)
    frame.set_linewidth(0.55)
    frame.set_alpha(0.96)


# Pull bottom-centered figure legends up to a fixed gap below the axes content
def _tuck_bottom_legends(fig: plt.Figure, gap_inches: float = 0.13) -> None:
    if not fig.legends:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    axis_bottoms = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        axis_bottoms.append(inv.transform((bbox.x0, bbox.y0))[1])

    for child in fig.get_children():
        if not isinstance(child, Text) or not child.get_visible() or not child.get_text():
            continue
        extent = child.get_window_extent(renderer)
        if extent.width < extent.height:
            continue
        axis_bottoms.append(inv.transform((extent.x0, extent.y0))[1])

    if not axis_bottoms:
        return

    gap = gap_inches / fig.get_figheight()
    target_top = min(axis_bottoms) - gap

    for legend in fig.legends:
        extent = legend.get_window_extent(renderer)
        bottom = inv.transform((extent.x0, extent.y0))[1]
        top = inv.transform((extent.x0, extent.y1))[1]
        x, y = inv.transform(legend.get_bbox_to_anchor().p0)
        hangs_from_top = abs(top - y) < abs(bottom - y)
        legend.set_bbox_to_anchor((x, target_top if hangs_from_top else target_top - (top - bottom)),
                                  transform=fig.transFigure)

# Plot saving helper
def _save_plot(
    fig: plt.Figure,
    out_path: Path,
    pad_inches: float = 0.07,
    align_labels: bool = True,
) -> None:
    for legend in fig.legends:
        _style_legend(legend)
    for ax in fig.axes:
        _style_legend(ax.get_legend())

    _tuck_bottom_legends(fig)

    if align_labels:
        try:
            fig.align_labels()
        except (RuntimeError, ValueError):
            pass

    save_kwargs = dict(bbox_inches=_export_bbox(fig, pad_inches), facecolor=FIG_FACE)

    raster = _format_path(out_path, out_path.suffix)
    fig.savefig(raster, dpi=SAVE_DPI, **save_kwargs)

    plt.close(fig)
    print(f"  Saved: {raster}")


# One subdirectory per file format, so that png/ stays separate from any other export under a figure folder
def _format_path(out_path: Path, suffix: str) -> Path:
    path = out_path.parent / suffix.lstrip(".") / out_path.with_suffix(suffix).name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Export box that keeps the declared figure width but trims unused height
def _export_bbox(fig: plt.Figure, pad_inches: float) -> Bbox:
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()

    if tight is None:
        return Bbox.from_extents(0.0, 0.0, width, height)

    y0 = max(0.0, min(tight.y0 - pad_inches, height))
    y1 = min(height, max(tight.y1 + pad_inches, 0.0))

    if y1 - y0 <= 0.0:
        y0, y1 = 0.0, height

    return Bbox.from_extents(0.0, y0, width, y1)
