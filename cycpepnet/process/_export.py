import argparse
import pandas as pd
from pathlib import Path
from cycpepnet.process._shared import EXPORTS, export_representations
from cycpepnet.utils.data_paths import DATASET_PROPERTIES, get_processed_csv_path

# Generate molecular representations for curated tables already present under the data root
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate molecular representations for curated datasets.")
    parser.add_argument("--dataset",         choices=("all", *DATASET_PROPERTIES), default="all")
    parser.add_argument("--representations", nargs="+", choices=[Path(name).stem for _, name in EXPORTS])
    args = parser.parse_args()

    names = DATASET_PROPERTIES if args.dataset == "all" else (args.dataset,)

    for name in names:
        path = get_processed_csv_path(name)
        export_representations(pd.read_csv(path), path.parent, args.representations)


if __name__ == "__main__":
    main()
