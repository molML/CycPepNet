from pathlib import Path
from cycpepnet.paths import DATA_ROOT

# Endpoint property folder of each processed dataset
DATASET_PROPERTIES: dict[str, str] = {
    "cycpeptmpdb"                : "permeability",
    "npmmpd"                     : "permeability",
    "cycpeptmpdb_npmmpd"         : "permeability",
    "macrocycledb"               : "binding",
    "peplife2"                   : "half-life",
    "pepmsnd"                    : "half-life",
    "peptherdia"                 : "half-life",
    "peplife2_pepmsnd_peptherdia": "half-life",
}

# Endpoint property of a processed dataset
def get_dataset_property(dataset_name: str) -> str:
    try:
        return DATASET_PROPERTIES[dataset_name]
    except KeyError as exc:
        raise KeyError(f"Unknown processed dataset '{dataset_name}'") from exc

# Folder that holds a processed dataset, its representations and its splits
def get_processed_dataset_dir(
    dataset_name: str,
    data_root: Path = DATA_ROOT,
    create: bool = False,
) -> Path:

    dataset_dir = data_root / get_dataset_property(dataset_name) / dataset_name

    if create:
        dataset_dir.mkdir(parents=True, exist_ok=True)

    return dataset_dir

# Curated CSV of a processed dataset
def get_processed_csv_path(
    dataset_name: str,
    data_root: Path = DATA_ROOT,
) -> Path:
    return get_processed_dataset_dir(dataset_name, data_root=data_root) / f"{dataset_name}.csv"
