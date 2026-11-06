from typing import Callable
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Apply function with progress bar
def apply_with_progress(

    series: pd.Series,
    func  : Callable,
    desc  : str = "Processing",

) -> pd.Series:

    tqdm.pandas(desc=f"  {desc}", ncols=100)
    return series.progress_apply(func)


# Save processed DataFrame to CSV
def save_processed_data(

    df         : pd.DataFrame,
    output_dir : Path,
    filename   : str,
    verbose    : bool = True,

) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists():
        if verbose:
            print(f"  Processed data already exists, skipping: {output_path}")
        return output_path

    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n  Saved: {output_path}  ({df.shape[0]:,} rows × {df.shape[1]:,} cols)")

    return output_path
