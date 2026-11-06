import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Analysis modules in execution order
STAGES = (
    "analyze_tasks",
    "analyze_splits",
    "analyze_uncertainty",
    "statistical_tests",
    "analyze_ranking",
    "analyze_results",
    "analyze_distributions",
)

# Regenerate every table and figure from saved run records into a separate output root
def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the published tables and figures from saved run records.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--data-root",    type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root",  type=Path, default=Path("outputs/reproduction"))
    args = parser.parse_args()

    source = args.results_root.resolve()
    output = args.output_root.resolve()

    if output == source or source in output.parents or output in source.parents:
        parser.error("Input results and output directory must be separate, non-nested paths.")

    if not any(source.glob("regression/**/*.json")):
        parser.error(f"No regression run records found in {source}.")

    env = dict(
        os.environ,
        MPLBACKEND             = "Agg",
        CYCPEPNET_RESULTS_ROOT = str(source),
        CYCPEPNET_DATA_ROOT    = str(args.data_root.resolve()),
        CYCPEPNET_OUTPUT_ROOT  = str(output / "results"),
        CYCPEPNET_PLOTS_ROOT   = str(output / "figures"),
        CYCPEPNET_LOGS_ROOT    = str(output / "logs"),
    )

    output.mkdir(parents=True, exist_ok=True)
    marker = output / "reproduction.json"
    marker.unlink(missing_ok=True)

    for stage in STAGES:
        print(f"\n=== {stage} ===", flush=True)
        extra = ["--splits-summary"] if stage == "analyze_tasks" else []
        subprocess.run([sys.executable, "-m", f"cycpepnet.analyze.{stage}", *extra], env=env, check=True)

    status = {"stages": list(STAGES), "status": "complete", "python": sys.version.split()[0]}
    marker.write_text(json.dumps(status, indent=2) + "\n")

    print(f"\nReproduction complete: {output}")


if __name__ == "__main__":
    main()
