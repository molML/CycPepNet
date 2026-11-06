from cycpepnet.benchmark import BENCHMARK_TASKS
from cycpepnet.paths import RESULTS_ROOT as RESULTS_ROOT, PLOTS_DIR as PLOTS_DIR

EXPECTED_N_SEEDS = 5
_ACTIVE_TASKS = {task.name for task in BENCHMARK_TASKS}
MAIN_ROW_MODELS: list[str] = [

    # Fingerprint-based models
    "rf_binary_fp",
    "rf_binary",
    "rf_count_fp",
    "rf_count",
    "rf_binary_r2_fp",
    "rf_binary_r2",
    "rf_count_r2_fp",
    "rf_count_r2",
    "rf_map6c_fp",
    "rf_map6c",
    "rf_map4c_fp",
    "rf_map4c",
    "rf_mxfp_fp",
    "rf_mxfp",
    "rf_hdf_fp",
    "rf_hdf",
    "rf_physchem",
    "xgb_binary_fp",
    "xgb_binary",
    "xgb_count_fp",
    "xgb_count",
    "xgb_binary_r2_fp",
    "xgb_binary_r2",
    "xgb_count_r2_fp",
    "xgb_count_r2",
    "xgb_map6c_fp",
    "xgb_map6c",
    "xgb_map4c_fp",
    "xgb_map4c",
    "xgb_mxfp_fp",
    "xgb_mxfp",
    "xgb_hdf_fp",
    "xgb_hdf",
    "xgb_physchem",

    # String-based models
    "lstm",
    "lstm_fragsmiles",
    "transformer",
    "transformer_fragsmiles",

    # Graph-based models
    "gcn",
    "mpnn",

    # Pretrained chemical language models
    "chemberta2",
    "peptide_clm2",

    # Pretrained GNNs
    "pretrained_gin",
    "molclr",

]

# REGRESSION TASKS

TASK_DISPLAY = {
    # Main tasks
    "macrocycledb_oprm1_ki_reg":       "Binding\n(OPRM1)",
    "macrocycledb_mc4r_ki_reg":        "Binding\n(MC4R)",
    "macrocycledb_sstr2_ic50_reg":     "Binding\n(SSTR2)",
    "npmmpd_reg":                      "Permeability\n(NPMMPD)",
    "cycpeptmpdb_reg":                 "Permeability\n(CycPeptMPDB)",
    "peplife2_pepmsnd_peptherdia_reg": "Half-life",

    # Supplementary tasks
    "npmmpd_rzepiela_reg":             "Permeability\n(NPMMPD Rzepiela 2022)",
    "cycpeptmpdb_townsend_reg":        "Permeability\n(CycPeptMPDB Townsend 2020)",
    "cycpeptmpdb_npmmpd_reg":          "Permeability\n(Merged)",
}

# CLASSIFICATION TASKS

# Primary classification tasks

TASK_DISPLAY_CLS = {

    # Main tasks
    "macrocycledb_oprm1_ki_cls":       "Binding\n(OPRM1)",
    "macrocycledb_mc4r_ki_cls":        "Binding\n(MC4R)",
    "macrocycledb_sstr2_ic50_cls":     "Binding\n(SSTR2)",
    "npmmpd_cls":                      "Permeability\n(NPMMPD)",
    "cycpeptmpdb_cls":                 "Permeability\n(CycPeptMPDB)",
    "peplife2_pepmsnd_peptherdia_cls": "Half-life",

    # Supplementary tasks
    "npmmpd_rzepiela_cls":             "Permeability\n(NPMMPD Rzepiela 2022)",
    "cycpeptmpdb_npmmpd_cls":          "Permeability\n(Merged)",
    "cycpeptmpdb_townsend_cls":        "Permeability\n(CycPeptMPDB Townsend 2020)",
}

MODEL_GROUP_MAP: dict[str, str] = {}

# Assign each model key to its visual family.
for _m in list(MAIN_ROW_MODELS) + ["mean"]:
    if _m.startswith("rf_"):
        MODEL_GROUP_MAP[_m] = "RF"
    elif _m.startswith("xgb_"):
        MODEL_GROUP_MAP[_m] = "XGBoost"
    elif _m in ("lstm", "lstm_fragsmiles"):
        MODEL_GROUP_MAP[_m] = "LSTM"
    elif _m in ("transformer", "transformer_fragsmiles"):
        MODEL_GROUP_MAP[_m] = "Transformer"
    elif _m == "gcn":
        MODEL_GROUP_MAP[_m] = "GCN"
    elif _m == "mpnn":
        MODEL_GROUP_MAP[_m] = "MPNN"
    elif _m == "chemberta2":
        MODEL_GROUP_MAP[_m] = "ChemBERTa"
    elif _m == "peptide_clm2":
        MODEL_GROUP_MAP[_m] = "PeptideCLM"
    elif _m == "pretrained_gin":
        MODEL_GROUP_MAP[_m] = "GIN"
    elif _m == "molclr":
        MODEL_GROUP_MAP[_m] = "MolCLR"
    elif _m == "mean":
        MODEL_GROUP_MAP[_m] = "Baseline"

MODEL_COLORS: dict[str, str] = {
    "RF":          "#086165",
    "XGBoost":     "#72BBBF",
    "LSTM":        "#734A0F",
    "Transformer": "#AB6E16",
    "ChemBERTa":   "#D0A362",
    "PeptideCLM":  "#E2C69E",
    "GCN":         "#113747",
    "MPNN":        "#4B7588",
    "GIN":         "#87A3AF",
    "MolCLR":      "#BACAD1",
    "Baseline":    "#9B948A",
    "Unknown":     "#6F6A62",
}

TASK_CATEGORY_MAP: dict[str, str] = {
    "macrocycledb_oprm1_ki_reg":       "Binding",
    "macrocycledb_mc4r_ki_reg":        "Binding",
    "macrocycledb_sstr2_ic50_reg":     "Binding",
    "peplife2_pepmsnd_peptherdia_reg": "Half-life",
    "cycpeptmpdb_reg":                 "Permeability",
    "npmmpd_reg":                      "Permeability",
    "cycpeptmpdb_npmmpd_reg":          "Permeability",
    "cycpeptmpdb_townsend_reg":        "Permeability",
    "npmmpd_rzepiela_reg":             "Permeability",
    "macrocycledb_oprm1_ki_cls":       "Binding",
    "macrocycledb_mc4r_ki_cls":        "Binding",
    "macrocycledb_sstr2_ic50_cls":     "Binding",
    "peplife2_pepmsnd_peptherdia_cls": "Half-life",
    "cycpeptmpdb_cls":                 "Permeability",
    "npmmpd_cls":                      "Permeability",
    "cycpeptmpdb_npmmpd_cls":          "Permeability",
    "cycpeptmpdb_townsend_cls":        "Permeability",
    "npmmpd_rzepiela_cls":             "Permeability",
}

# Category palette
TASK_CATEGORY_COLORS: dict[str, str] = {
    "Binding":      "#DD7A3A",
    "Permeability": "#1E8C88",
    "Half-life":    "#1D4E6B",
}

# Representations excluded from the main analysis pool, compared only by the paired ablations
HIGHER_RADIUS_REPS: set[str] = {
    "morgan_binary",
    "morgan_count",
    "map6c",
}

# Model-family display order
MODEL_ORDER_LIST = [
    "RF",
    "XGBoost",
    "LSTM",
    "Transformer",
    "ChemBERTa",
    "PeptideCLM",
    "GCN",
    "MPNN",
    "GIN",
    "MolCLR",
]

# Plot style constants
FIG_FACE          = "#FFFFFF"
AX_FACE           = "#FFFFFF"
GRID_COLOR        = "#ECEAE6"
SPINE_COLOR       = "#C7C1B8"
TEXT_COLOR        = "#23201C"
LEGEND_EDGE_COLOR = "#DDD6CE"

# Shared neutral and comparison colors
NEUTRAL_MID              = "#8E8A84"
NEUTRAL_LABEL            = "#6F6A62"
NEUTRAL_CONNECTOR        = "#BDB8B0"
RANK_SUMMARY_COLOR       = "#1C2E44"

# One mean-rank marker for every rank figure: an open ring, never a diamond
RANK_SUMMARY_MARKER      = "o"
RANK_SUMMARY_SIZE        = 34.0
# Export resolution for raster figures
SAVE_DPI = 600

BASE_REPRESENTATION_COLORS: dict[str, str] = {
    "ECFP4b":     "#92AEED",
    "ECFP4c":     "#5C7F94",
    "ECFP6b":     "#3F5A6A",
    "ECFP6c":     "#8DDDFF",
    "MAP4C":      "#867E01",
    "MAP6C":      "#B2A936",
    "MXFP":       "#5EAE9F",
    "HDF":        "#443B12",
    "PhysChem":   "#EEC57C",
    "SMILES":     "#7B5435",
    "fragSMILES": "#C48A61",
    "Graph":      "#66408A",
}

BASE_REPRESENTATION_ORDER: list[str] = list(BASE_REPRESENTATION_COLORS)

BASE_REPRESENTATION_LABEL: dict[str, str] = {
    "ECFP4b": "ECFP4 binary",
    "ECFP4c": "ECFP4 count",
    "ECFP6b": "ECFP6 binary",
    "ECFP6c": "ECFP6 count",
}

FIG_WIDTH_FULL          = 6.50
FIG_WIDTH_SUPP          = 6.50

TICK_LABEL_SIZE         = 7.4
AXIS_LABEL_SIZE         = 8.2

PANEL_LABEL_SIZE        = 9.2
PANEL_TITLE_SIZE        = 6.9
ANNOTATION_SIZE         = 6.7
ANNOTATION_SMALL_SIZE   = 6.2

PANEL_LABEL_DX          = 3.0
PANEL_LABEL_DY          = 2.0
PANEL_LABEL_X           = 0.012
PANEL_TITLE_DY          = 3.5
BASELINE_LINEWIDTH      = 1.0
BASELINE_LINESTYLE      = (0, (2.6, 1.8))
