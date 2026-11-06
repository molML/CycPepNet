from __future__ import annotations
from cycpepnet.analyze._figure_ablation import export_representation_ablation_deltas
from cycpepnet.analyze._figure_composition import (
    export_composition_relative_to_training_mean,
    export_permeability_composition_deltas,
    plot_permeability_composition_lineage,
    plot_permeability_composition_panels,
    plot_permeability_composition_lineage_companion,
)
from cycpepnet.analyze._figure_learnability import plot_task_learnability
from cycpepnet.analyze._figure_methods import (
    plot_figure_three,
    plot_figure_three_companion,
    plot_regression_vs_classification_ranks,
)
from cycpepnet.analyze._figure_predictions import plot_predicted_vs_measured
from cycpepnet.analyze._figure_sensitivity import plot_sensitivity_specificity
from cycpepnet.analyze._figure_split import plot_random_vs_butina, plot_random_vs_butina_companion
from cycpepnet.analyze._result_helpers import (
    MAIN_CLS_TASKS,
    MAIN_REG_TASKS,
    compute_validation_selected_family_ranks,
    load_seed_results,
    make_dirs,
    validate_seed_results,
)
from cycpepnet.analyze._results_style import _apply_results_plot_style
from cycpepnet.utils import print_completion_message, print_section_header, setup_plot_style

# Run the complete result-analysis workflow
def main() -> None:

    setup_plot_style()
    _apply_results_plot_style()
    make_dirs()

    seed_df = load_seed_results()

    if seed_df.empty:
        print("  No seed-level result rows found. Nothing to plot.")
        print_completion_message()
        return

    validate_seed_results(seed_df)

    print_section_header("MAIN FIGURES")
    plot_task_learnability(seed_df)
    plot_figure_three(seed_df)
    plot_random_vs_butina(seed_df)
    plot_permeability_composition_panels(seed_df)
    plot_regression_vs_classification_ranks(
        compute_validation_selected_family_ranks(seed_df, MAIN_REG_TASKS, "regression", "rmse", "val_rmse"),
        compute_validation_selected_family_ranks(seed_df, MAIN_CLS_TASKS, "classification", "balanced_acc", "val_balanced_acc"),
    )

    print_section_header("SUPPLEMENTARY FIGURES")
    plot_task_learnability(seed_df, view="companion")
    plot_figure_three_companion(seed_df)
    plot_random_vs_butina_companion(seed_df)
    export_permeability_composition_deltas(seed_df)
    export_composition_relative_to_training_mean(seed_df)
    plot_permeability_composition_lineage(seed_df)
    plot_permeability_composition_lineage_companion(seed_df)
    plot_sensitivity_specificity(seed_df)
    plot_predicted_vs_measured(seed_df)

    print_section_header("TABLES")
    export_representation_ablation_deltas(seed_df)

    print_completion_message()


if __name__ == "__main__":
    main()
