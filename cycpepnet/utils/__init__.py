from importlib import import_module

# Public helpers grouped by the submodule that defines them; a submodule is imported only when one of its names is used
_SUBMODULE_EXPORTS: dict[str, list[str]] = {
    "console": [
        "print_completion_message",
        "print_distribution_stats",
        "print_numeric_stats",
        "print_section_header",
        "print_subsection_header",
        "tee_stdout_to_log",
    ],
    "io": [
        "save_processed_data",
        "apply_with_progress",
    ],
    "data_paths": [
        "get_dataset_property",
        "get_processed_csv_path",
        "get_processed_dataset_dir",
    ],
    "curation": [
        "standardize_smiles",
    ],
    "fragsmiles": [
        "smiles_to_fragsmiles",
        "fragsmiles_batch",
        "save_fragsmiles",
    ],
    "peptides": [
        "get_ring_atoms_count_from_smiles",
        "calculate_amide_ratio",
        "sequence_to_smiles",
    ],
    "visualization": [
        "setup_plot_style",
        "print_visualization_header",
        "get_color_palette",
        "create_pie_chart",
        "create_horizontal_bar_chart",
        "create_histogram",
    ],
    "representations": [
        "save_morgan_fingerprints",
        "save_morgan_count_fingerprints",
        "save_morgan_r2_binary_fingerprints",
        "save_morgan_r2_count_fingerprints",
        "save_map6c_fingerprints",
        "save_map4c_fingerprints",
        "save_mxfp_fingerprints",
        "save_hdf_fingerprints",
        "save_pyg_graphs",
        "load_morgan_binary_fingerprints",
        "load_morgan_count_fingerprints",
        "load_morgan_r2_binary_fingerprints",
        "load_morgan_r2_count_fingerprints",
        "load_map6c_fingerprints",
        "load_map4c_fingerprints",
        "load_mxfp_fingerprints",
        "load_hdf_fingerprints",
        "load_pyg_graphs",
        "get_atom_feature_dim",
        "get_bond_feature_dim",
        "apply_physicochemical_properties",
        "PHYSICOCHEMICAL_COLUMNS",
    ],
    "metrics": [
        "rmse",
        "qf3_squared",
        "evaluate",
    ],
    "splitting": [
        "butina_cluster",
        "butina_cluster_count",
        "butina_cluster_map6c",
        "assign_clusters_to_splits",
        "butina_split",
        "random_split",
        "save_split",
        "load_split",
        "validate_split",
        "assert_valid_split",
        "build_metadata",
    ],
}

_EXPORTS = {name: module for module, names in _SUBMODULE_EXPORTS.items() for name in names}

__all__ = list(_EXPORTS)

# Import a public helper from its submodule the first time it is requested
def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")

    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value

    return value
