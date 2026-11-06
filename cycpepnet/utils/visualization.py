from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns

_FONT_SEARCH_DIRS = [
    Path('assets/fonts'),
    Path.home() / '.fonts',
    Path.home() / '.local' / 'share' / 'fonts',
]

# Register every font file found in the local search dirs
def _register_local_fonts() -> None:
    for directory in _FONT_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for font_path in sorted(directory.rglob('*')):
            if font_path.suffix.lower() not in {'.otf', '.ttf', '.ttc'}:
                continue
            try:
                fm.fontManager.addfont(str(font_path))
            except (RuntimeError, OSError):
                continue

# Setup global plot style
def setup_plot_style() -> None:
    _register_local_fonts()
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    font_stack = [
        'Roboto', 'Roboto Medium', 'Roboto Light',
        'Nimbus Sans', 'Helvetica Neue', 'Helvetica', 'Arial',
        'Liberation Sans', 'DejaVu Sans',
    ]
    plt.rcParams['font.sans-serif'] = font_stack
    resolved = fm.FontProperties(family=font_stack).get_name()
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['mathtext.rm'] = resolved
    plt.rcParams['mathtext.it'] = f'{resolved}:italic'
    plt.rcParams['mathtext.bf'] = f'{resolved}:bold'
    plt.rcParams['mathtext.sf'] = resolved

# Print header for visualization section
def print_visualization_header(width: int = 70) -> None:
    print("="*width)
    print("  GENERATING VISUALIZATIONS")
    print("="*width + "\n")

# Define a consistent color palette
def get_color_palette(n_colors: Optional[int] = None) -> Tuple[str, List[str]]:

    main_color = '#1E8C88'  # Teal
    base_colors = [
        '#1E8C88',  # Teal
        '#DD7A3A',  # Orange
        '#74B6A4',  # Sage
        '#F6A95C',  # Amber
        '#15697E',  # Petrol
    ]

    if n_colors is None or n_colors <= len(base_colors):
        return main_color, base_colors

    diverse_palette = sns.color_palette("husl", n_colors=n_colors - 2)
    diverse_colors  = [mcolors.rgb2hex(color) for color in diverse_palette]

    extended_colors = [base_colors[0], base_colors[1]] + diverse_colors

    return main_color, extended_colors

# Format percentage label for pie chart slices
def autopct_format(pct: float, threshold: float = 5.0) -> str:
    return f'{pct:.1f}%' if pct > threshold else ''

# Create pie chart
def create_pie_chart(stats_df: pd.DataFrame, value_column: str, count_column: str, percentage_column: str, output_path: Path, figsize: Tuple[int, int] = (12, 8)) -> None:

    _, colors = get_color_palette(n_colors=len(stats_df))
    fig, ax   = plt.subplots(figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        stats_df[count_column],
        autopct    = autopct_format,
        startangle = 90,
        colors     = colors,
        wedgeprops = {'edgecolor': 'white', 'linewidth': 2}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    legend_labels = [f'{val}: {count:,} entries ({pct:.1f}%)'
                     for val, count, pct in zip(stats_df[value_column],
                                                 stats_df[count_column],
                                                 stats_df[percentage_column])]

    ax.legend(wedges, legend_labels,
              loc            = 'center left',
              bbox_to_anchor = (1, 0, 0.5, 1),
              fontsize       = 11,
              frameon        = False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

# Create horizontal bar chart
def create_horizontal_bar_chart(
        stats_df         : pd.DataFrame,
        value_column     : str,
        count_column     : str,
        percentage_column: str,
        output_path      : Path,
        figsize          : Tuple[int, int] = (20, 8),
        top_n            : Optional[int] = None
        ) -> None:

    main_color, _ = get_color_palette()
    plot_df       = stats_df.head(top_n) if top_n else stats_df
    fig, ax       = plt.subplots(figsize=figsize)
    ax.barh(range(len(plot_df)), plot_df[count_column], color=main_color)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df[value_column])
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_axisbelow(True)

    for i, (count, pct) in enumerate(zip(plot_df[count_column], plot_df[percentage_column])):
        ax.text(count, i, f' {count:,} ({pct:.1f}%)', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


# Create histogram (with optional vertical lines)
def create_histogram(
        data          : pd.Series,
        xlabel        : str,
        output_path   : Path,
        bins          : int = 30,
        figsize       : Tuple[int, int] = (12, 8),
        vertical_lines: Optional[List[Tuple[float, str, str]]] = None
        ) -> None:

    main_color, _ = get_color_palette()
    fig, ax       = plt.subplots(figsize=figsize)

    ax.hist(data, bins=bins, color=main_color, alpha=0.7, edgecolor='white', linewidth=1.5)

    if vertical_lines:

        for value, label, color in vertical_lines:
            ax.axvline(x=value, color=color, linestyle='--', linewidth=2, label=label)

        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
