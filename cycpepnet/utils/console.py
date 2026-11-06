import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TextIO
import pandas as pd

# File-like object that duplicates every write to several streams
class _Tee:

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    # Write the same text to every stream
    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    # Flush every stream
    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    # Report terminal status of the primary stream
    def isatty(self) -> bool:
        first = self._streams[0]
        return first.isatty() if hasattr(first, "isatty") else False

    # File descriptor of the primary stream
    def fileno(self) -> int:
        return self._streams[0].fileno()

    # Encoding of the primary stream
    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")

# Mirror stdout to a log file (default folder: $CYCPEPNET_LOGS_ROOT or logs/)
@contextmanager
def tee_stdout_to_log(
    log_name: str,
    logs_dir: Path | str | None = None,
) -> Iterator[None]:

    if logs_dir is None:
        logs_dir = os.environ.get("CYCPEPNET_LOGS_ROOT", "logs")

    log_path = Path(logs_dir) / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with open(log_path, "w", buffering=1, encoding="utf-8") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout

# Print a framed section title
def print_section_header(title: str, width: int = 70, char: str = "=") -> None:
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)

# Print a framed subsection title
def print_subsection_header(title: str, width: int = 70, char: str = "─") -> None:
    print(char * width)
    print(f"  {title}")
    print(char * width)

# Print and return the value counts of a categorical column
def print_distribution_stats(
    df: pd.DataFrame,
    column: str,
    title: str,
    top_n: Optional[int] = None,
    width: int = 70,
) -> pd.DataFrame:

    print("─" * width)
    print(f"  {title}")
    print("─" * width)

    value_counts = df[column].value_counts(dropna=False)

    stats_df = pd.DataFrame(
        {
            column: value_counts.index.fillna("Unknown/Unspecified"),
            "Count": value_counts.values,
            "Percentage": (value_counts.values / len(df) * 100).round(2),
        }
    )

    print("")

    if top_n:
        print(f"  • Total Unique {column}: {df[column].nunique():,}\n")
        print(stats_df.head(top_n).to_string(index=False))
    else:
        print(stats_df.to_string(index=False))

    print("")

    return stats_df

# Print and return min, max, mean and median of a numeric column
def print_numeric_stats(
    df: pd.DataFrame,
    column: str,
    title: str,
    unit: str = "",
    decimals: int = 2,
    width: int = 70,
) -> pd.DataFrame:

    print("─" * width)
    print(f"  {title}")
    print("─" * width)

    col_data = df[column].dropna()
    unit_str = f" {unit}" if unit else ""
    fmt      = f".{decimals}f"

    stats_df = pd.DataFrame(
        {
            "Statistic": ["Min", "Max", "Mean", "Median"],
            "Value": [
                f"{col_data.min():{fmt}}{unit_str}",
                f"{col_data.max():{fmt}}{unit_str}",
                f"{col_data.mean():{fmt}}{unit_str}",
                f"{col_data.median():{fmt}}{unit_str}",
            ],
        }
    )

    print("")
    print(stats_df.to_string(index=False))
    print("")

    return stats_df

# Print the closing line of a script
def print_completion_message() -> None:
    print("\nDone.")
