import os
from pathlib import Path

# Default input and output roots; reproduce.py and prepare.py redirect them through environment variables
DATA_ROOT          = Path(os.environ.get("CYCPEPNET_DATA_ROOT", "data/processed"))
INPUT_RESULTS_ROOT = Path(os.environ.get("CYCPEPNET_RESULTS_ROOT", "results"))
RESULTS_ROOT       = Path(os.environ.get("CYCPEPNET_OUTPUT_ROOT", "results"))
PLOTS_DIR          = Path(os.environ.get("CYCPEPNET_PLOTS_ROOT", "plots"))
