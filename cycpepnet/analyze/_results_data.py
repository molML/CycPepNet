from pathlib import Path
import pandas as pd
from cycpepnet.paths import INPUT_RESULTS_ROOT
from cycpepnet.utils.results import read_run_records
from cycpepnet.analyze._results_config import _ACTIVE_TASKS, HIGHER_RADIUS_REPS, MAIN_ROW_MODELS

# Load and prepare seed-level results for plotting
def _base_rep(rep: object) -> str:
    if pd.isna(rep):
        return ""
    return str(rep).removesuffix("+physchem")

# Check if a representation is a higher-radius variant of a canonical representation
def _is_higher_radius_rep(rep: object) -> bool:
    return _base_rep(rep) in HIGHER_RADIUS_REPS

# Restrict the results DataFrame to the main analysis pool
def main_pool(df: pd.DataFrame) -> pd.DataFrame:
    if "rep" not in df.columns:
        return df.copy()
    excluded = df["rep"].map(_is_higher_radius_rep) | df["rep"].astype(str).str.contains(r"\+physchem", na=False)
    if "model" in df.columns:
        excluded &= df["model"].ne("mean")
    return df.loc[~excluded].copy()

# Every configuration that was run, including both fingerprint radii and the augmented variants
def full_pool(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

# Models grouped by the molecular representation
GRAPH_MODELS      = {"gcn", "mpnn", "pretrained_gin", "molclr"}
SMILES_MODELS     = {"lstm", "transformer", "chemberta2", "peptide_clm2"}
FRAGSMILES_MODELS = {"lstm_fragsmiles", "transformer_fragsmiles"}

# Canonical (model, rep) -> input-representation label
def representation_of(model: object, rep: object) -> str | None:
    if not isinstance(model, str):
        return None
    if model in GRAPH_MODELS:
        return "Graph"
    if model in FRAGSMILES_MODELS:
        return "fragSMILES"
    if model in SMILES_MODELS:
        return "SMILES"
    if model in ("rf_physchem", "xgb_physchem"):
        return "PhysChem"
    rep = str(rep)
    if "+physchem" in rep:
        return None
    return {"morgan_binary_r2": "ECFP4b", "morgan_count_r2": "ECFP4c",
            "map4c": "MAP4C", "mxfp": "MXFP", "hdf": "HDF"}.get(rep)


# Fill missing task types from task suffixes and metric presence
def _fill_task_type(df: pd.DataFrame) -> pd.DataFrame:

    if "task_type" not in df.columns:
        df["task_type"] = pd.NA

    if "task" in df.columns:
        from_name = pd.Series(pd.NA, index=df.index, dtype="object")
        from_name.loc[df["task"].astype(str).str.endswith("_reg")] = "regression"
        from_name.loc[df["task"].astype(str).str.endswith("_cls")] = "classification"
        mask = df["task_type"].isna() & from_name.notna()
        if mask.any():
            df.loc[mask, "task_type"] = from_name.loc[mask]

    regression_cols = [c for c in ("rmse", "qf3", "val_rmse", "val_qf3") if c in df.columns]

    classification_cols = [
        c
        for c in (
            "auroc", "tpr", "tnr", "precision", "balanced_acc",
            "val_auroc", "val_tpr", "val_tnr", "val_precision", "val_balanced_acc",
        )
        if c in df.columns
    ]

    if regression_cols:
        mask = df["task_type"].isna() & df[regression_cols].notna().any(axis=1)
        if mask.any():
            df.loc[mask, "task_type"] = "regression"

    if classification_cols:
        mask = df["task_type"].isna() & df[classification_cols].notna().any(axis=1)
        if mask.any():
            df.loc[mask, "task_type"] = "classification"

    missing = int(df["task_type"].isna().sum())
    if missing:
        print(f"  WARNING: {missing} seed rows still have missing task_type and may be skipped.")

    return df

# Load individual result JSON files and reshape them to the plotting schema
def _load_seed_level_results(results_root: Path = INPUT_RESULTS_ROOT) -> pd.DataFrame:

    records = read_run_records(results_root)
    df      = pd.DataFrame(records)

    if df.empty:
        print("  Skipped seed-level plots: result JSON files were empty.")
        return df

    df = df.rename(columns={"model_key": "model"}).copy()
    missing = sorted({"task", "model"} - set(df.columns))
    if missing:
        print(f"  Skipped seed-level results: missing required columns {missing}.")
        return pd.DataFrame()

    active_models = set(MAIN_ROW_MODELS) | {"mean"}
    inactive_model_mask = df["model"].notna() & ~df["model"].isin(active_models)
    if inactive_model_mask.any():
        inactive = sorted(df.loc[inactive_model_mask, "model"].unique())
        print(f"  Ignoring inactive seed-level result models: {inactive}")
    df = df.loc[df["model"].isin(active_models)].copy()

    inactive_task_mask = df["task"].notna() & ~df["task"].isin(_ACTIVE_TASKS)
    if inactive_task_mask.any():
        inactive = sorted(df.loc[inactive_task_mask, "task"].unique())
        print(f"  Ignoring inactive seed-level result tasks: {inactive}")
    df = df.loc[df["task"].isin(_ACTIVE_TASKS)].copy()

    metric_cols = [
        "auroc", "tpr", "tnr", "precision", "balanced_acc",
        "val_auroc", "val_tpr", "val_tnr", "val_precision", "val_balanced_acc",
        "val_rmse", "val_qf3", "rmse", "qf3",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[f"{col}_mean"] = df[col]
            df[f"{col}_std"]  = 0.0

    df = _fill_task_type(df)

    return df

# Validate that the seed-level frame has the columns plotting needs
def _prepare_seed_level_results(df: pd.DataFrame) -> pd.DataFrame:
    required = {"task", "model", "rep", "split_method", "seed"}
    if df.empty or not required.issubset(df.columns):
        return df.copy()
    return df.copy()
