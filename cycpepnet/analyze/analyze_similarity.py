import csv
import json
import warnings
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from cycpepnet.benchmark import BENCHMARK_TASKS, DATA_ROOT, load_task_data
from cycpepnet.utils.splitting import butina_cluster, butina_cluster_count, butina_cluster_map6c
from cycpepnet.utils import (
    get_processed_dataset_dir,
    get_color_palette,
    load_morgan_binary_fingerprints,
    load_morgan_count_fingerprints,
    load_map6c_fingerprints,
    load_map4c_fingerprints,
    print_completion_message,
    print_section_header,
    setup_plot_style,
    tee_stdout_to_log,
    load_morgan_r2_binary_fingerprints,
    load_morgan_r2_count_fingerprints,
)

# Output directory for all cutoff analysis plots and JSON results
CUTOFF_PLOTS_ROOT = Path("plots/cutoff_analysis")

# Kneedle summaries substantiate the 0.45 threshold reported in the Methods
CUTOFF_RESULTS_ROOT = Path("results")

# Cutoff range
CUTOFF_RANGE = np.arange(0.2, 0.9, 0.05)

# Fixed clustering cutoff for all fingerprint types
FIXED_CUTOFF = 0.45

# Fingerprint type definitions
FP_TYPES = {
    "ECFP6_binary": ("morgan_fp.npy",         load_morgan_binary_fingerprints,       "tanimoto_binary"),
    "ECFP6_count":  ("morgan_count_fp.npy",   load_morgan_count_fingerprints,        "tanimoto_count"),
    "ECFP4_binary": ("morgan_r2_fp.npy",      load_morgan_r2_binary_fingerprints,    "tanimoto_binary"),
    "ECFP4_count":  ("morgan_r2_count_fp.npy",load_morgan_r2_count_fingerprints,     "tanimoto_count"),
    "MAP6C":        ("map6c_fp.npy",          load_map6c_fingerprints,               "smc"),
    "MAP4C":        ("map4c_fp.npy",          load_map4c_fingerprints,               "smc"),
}

# One entry per unique molecule pool (dataset + filters + exclude_filters + exclude_zeros)
def _unique_molecule_pools(tasks) -> list:
    seen = set()
    result = []
    for t in tasks:
        key = (
            t.dataset,
            tuple(sorted(t.filters.items()))          if t.filters          else (),
            tuple(
                (col, tuple(vals))
                for col, vals in sorted(t.exclude_filters.items())
            )                                          if t.exclude_filters  else (),
            t.exclude_zeros,
        )
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result

SUBSET_TASKS = _unique_molecule_pools(BENCHMARK_TASKS)

# Strip the trailing _reg/_cls task-type suffix so plot folders stay clean
def task_plot_dirname(task_name: str) -> str:
    if task_name.endswith("_reg") or task_name.endswith("_cls"):
        return task_name.rsplit("_", 1)[0]
    return task_name

# Compute pairwise Tanimoto distances for binary fingerprints
def _distances_binary(fps: np.ndarray) -> np.ndarray:
    n     = len(fps)
    dists = []
    for i in range(1, n):
        for j in range(i):
            inter = np.sum(fps[i] & fps[j])
            union = np.sum(fps[i] | fps[j])
            dists.append(1.0 - (inter / union if union > 0 else 0.0))

    return np.array(dists)

# Compute pairwise Tanimoto distances for count fingerprints
def _distances_count(fps: np.ndarray) -> np.ndarray:
    n     = len(fps)
    dists = []
    for i in range(1, n):
        for j in range(i):
            num = np.sum(np.minimum(fps[i], fps[j]))
            den = np.sum(np.maximum(fps[i], fps[j]))
            dists.append(1.0 - (num / den if den > 0 else 0.0))

    return np.array(dists)

# Compute pairwise Manhattan distances
def _distances_manhattan_normalized(fps: np.ndarray) -> np.ndarray:

    fps_f   = fps.astype(np.float64)
    fp_min  = fps_f.min(axis=0)
    fp_max  = fps_f.max(axis=0)
    fp_rng  = fp_max - fp_min

    fp_rng[fp_rng == 0.0] = 1.0

    fps_n   = (fps_f - fp_min) / fp_rng
    dim     = fps_n.shape[1]
    n       = len(fps_n)
    dists   = []

    for i in range(1, n):
        for j in range(i):
            dists.append(float(np.sum(np.abs(fps_n[i] - fps_n[j])) / dim))

    return np.array(dists, dtype=np.float64)

# Compute pairwise SMC distances for MAP6C fingerprints
def _distances_smc(fps: np.ndarray) -> np.ndarray:
    n     = len(fps)
    dists = []
    for i in range(1, n):
        for j in range(i):
            dists.append(1.0 - np.sum(fps[i] == fps[j]) / len(fps[i]))

    return np.array(dists)

# Returns pairwise distances for a given fingerprint array and metric
def compute_distances(fps: np.ndarray, metric: str) -> np.ndarray:

    if metric == "tanimoto_binary":
        return _distances_binary(fps)

    elif metric == "tanimoto_count":
        return _distances_count(fps)

    elif metric == "manhattan_normalized":
        return _distances_manhattan_normalized(fps)

    else:
        return _distances_smc(fps)

# Returns Butina-compatible cluster function for the given metric
def clustering_func_for(metric: str):

    if metric == "tanimoto_binary":
        return butina_cluster

    elif metric == "tanimoto_count":
        return butina_cluster_count

    elif metric == "manhattan_normalized":
        return butina_cluster

    else:
        return butina_cluster_map6c

# Analyze how clustering metrics change across cutoffs
def analyze_cutoff_sensitivity(fps: np.ndarray, cutoffs: np.ndarray, cluster_fn, precomputed_dists: np.ndarray = None) -> dict:
    results = {"cutoffs": cutoffs.tolist(), "n_clusters": [], "largest_cluster": [], "singletons": []}
    for cutoff in cutoffs:
        clusters = cluster_fn(fps, cutoff=cutoff, precomputed_dists=precomputed_dists) \
                   if precomputed_dists is not None else cluster_fn(fps, cutoff=cutoff)
        results["n_clusters"].append(len(clusters))
        results["largest_cluster"].append(len(clusters[0]) if clusters else 0)
        results["singletons"].append(sum(1 for c in clusters if len(c) == 1))
    return results

# Recommend optimal cutoff using Kneedle algorithm
def recommend_cutoff(results: dict) -> float:

    cutoffs        = np.array(results["cutoffs"])
    n_clusters     = np.array(results["n_clusters"])
    x_norm         = (cutoffs - cutoffs.min()) / (cutoffs.max() - cutoffs.min())
    y_norm         = (n_clusters - n_clusters.min()) / (n_clusters.max() - n_clusters.min() + 1e-9)
    x0, y0, x1, y1 = x_norm[0], y_norm[0], x_norm[-1], y_norm[-1]
    numerator      = np.abs((y1 - y0) * x_norm - (x1 - x0) * y_norm + x1 * y0 - y1 * x0)
    denominator    = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)

    return float(cutoffs[np.argmax(numerator / (denominator + 1e-9))])

# Plot similarity distributions for all FP types
def plot_similarity_distribution(distances_dict: dict[str, np.ndarray], label: str, output_path: Path, kneedles: dict[str, float]):

    fp_names  = list(distances_dict.keys())
    n_fps     = len(fp_names)
    n_cols    = 3
    n_rows    = (n_fps + n_cols - 1) // n_cols
    _, colors = get_color_palette(n_fps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes      = np.array(axes).flatten()

    for idx, fp_name in enumerate(fp_names):

        ax   = axes[idx]
        sims = 1 - distances_dict[fp_name]

        ax.hist(sims, bins=50, alpha=0.7, color=colors[idx], edgecolor="white", linewidth=0.5)

        kneedle_val = kneedles.get(fp_name)

        if kneedle_val is not None:
            ax.axvline(1 - kneedle_val, color="#8B0000", linestyle=":", linewidth=2, label=f"Kneedle {kneedle_val:.2f}")

        ax.set_xlabel("Similarity", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(fp_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(n_fps, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{label} (Similarity Distributions)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Plot cutoff sensitivity for all FP types (only Butina-compatible ones)
def plot_cutoff_sensitivity(sensitivity_dict: dict[str, dict], label: str, output_path: Path, kneedles: dict[str, float]):

    fp_names  = [fp for fp in sensitivity_dict if sensitivity_dict[fp] is not None]

    if not fp_names:
        return

    _, colors = get_color_palette(len(fp_names))
    markers   = ["o", "d", "s", "^", "v", "P", "X", "D", "h"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax_idx, (metric_key, title, ylabel) in enumerate([
        ("n_clusters",      "Total Clusters vs Cutoff",     "Number of Clusters"),
        ("largest_cluster", "Largest Cluster vs Cutoff",    "Largest Cluster Size"),
        ("singletons",      "Singleton Clusters vs Cutoff", "Number of Singletons"),
    ]):
        ax = axes[ax_idx]
        for i, fp_name in enumerate(fp_names):
            res = sensitivity_dict[fp_name]
            ax.plot(np.array(res["cutoffs"]), res[metric_key], f"{markers[i]}-",
                    color=colors[i], linewidth=2, markersize=5, label=fp_name)

        ax.set_xlabel("Distance Cutoff", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ax_idx == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"{label} - Cutoff Sensitivity", fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Load all available FP types for a dataset directory given a row index array
def load_all_fps(dataset_dir: Path, idx: np.ndarray) -> dict[str, np.ndarray]:
    fps = {}
    for fp_name, (filename, loader, _) in FP_TYPES.items():
        path = dataset_dir / filename
        if path.exists():
            fps[fp_name] = loader(path)[idx]
        else:
            print(f"  Warning: {fp_name} not found ({path.name}), skipping")
    return fps

# Subsample fingerprint dict if larger than max_n
def _subsample(fps_dict: dict[str, np.ndarray], max_n: int = 2000) -> dict[str, np.ndarray]:
    n = next(iter(fps_dict.values())).shape[0]
    if n <= max_n:
        return fps_dict
    indices = np.random.default_rng(42).choice(n, max_n, replace=False)
    return {k: v[indices] for k, v in fps_dict.items()}

# Run distances + sensitivity + kneedle for all FP types.
def run_analysis(fps_dict: dict[str, np.ndarray]) -> tuple[dict, dict, dict]:
    distances   = {}
    sensitivity = {}
    kneedles    = {}
    for fp_name, fps in fps_dict.items():
        metric     = FP_TYPES[fp_name][2]
        cluster_fn = clustering_func_for(metric)
        dists      = compute_distances(fps, metric)
        distances[fp_name] = dists

        sens                 = analyze_cutoff_sensitivity(fps, CUTOFF_RANGE, cluster_fn, precomputed_dists=dists)
        sensitivity[fp_name] = sens
        kneedles[fp_name]    = recommend_cutoff(sens)

    return distances, sensitivity, kneedles

# Perform complete similarity analysis for a single framework task
def analyze_task(task) -> dict | None:
    print("")
    try:
        df, _, _, orig_idx, _ = load_task_data(task, DATA_ROOT)
    except FileNotFoundError as e:
        print(f"  Skipped {task.name}: {e}")
        return None

    dataset_dir             = get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT)
    fps_dict                = load_all_fps(dataset_dir, orig_idx)

    if not fps_dict:
        print(f"  Skipped {task.name}: no fingerprints found")
        return None

    fps_dict                         = _subsample(fps_dict)
    n_sub                            = next(iter(fps_dict.values())).shape[0]
    distances, sensitivity, kneedles = run_analysis(fps_dict)

    plots_dir = CUTOFF_PLOTS_ROOT / task_plot_dirname(task.name)
    plots_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    plot_similarity_distribution(distances, task.name.upper(),
                                  plots_dir / "similarity_distributions.png", kneedles)
    plot_cutoff_sensitivity(sensitivity, task.name.upper(),
                             plots_dir / "cutoff_sensitivity.png", kneedles)

    print(f"\n  {task.name.upper()}  (n={len(df)}, subsampled={n_sub}):")
    for fp_name, k in kneedles.items():
        val = f"{k:.2f}" if k is not None else "N/A (L1, no Butina)"
        print(f"    Kneedle {fp_name:<16}: {val}")

    return {"task": task.name, "kneedles": kneedles}

# Print Kneedle table + medians for a list of results
def _print_kneedle_table(results: list[dict], title: str, use_for_cutoff: bool = False):

    fp_names = list(FP_TYPES.keys())
    print(f"\n  {title}")
    header = f"  {'Dataset/Task':<35}" + "".join(f"  {n:<12}" for n in fp_names)
    print(header)
    print("  " + "-" * (len(header) - 2))

    kneedles_per_fp: dict[str, list] = {fp: [] for fp in fp_names}

    for result in results:

        row = f"  {result['task']:<35}"

        for fp_name in fp_names:
            k = result["kneedles"].get(fp_name)

            if k is not None:
                row += f"  {k:<12.2f}"
                kneedles_per_fp[fp_name].append(k)

            else:
                row += f"  {'N/A':<12}"

        print(row)

    print("\n  Medians:")
    for fp_name in fp_names:

        vals = kneedles_per_fp[fp_name]

        if vals:
            marker = "  ← used for FIXED_CUTOFF" if (use_for_cutoff and fp_name == "ECFP6_binary") else ""
            print(f"    {fp_name:<16}: {np.median(vals):.2f}{marker}")

# Save all kneedle results to a single CSV
def _save_results_csv(task_results: list[dict], out_path: Path) -> None:

    fp_names = list(FP_TYPES.keys())
    rows     = []

    for result in task_results:
        row = {"name": result["task"]}
        for fp in fp_names:
            k = result["kneedles"].get(fp)
            row[fp] = round(k, 4) if k is not None else "N/A"
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name"] + fp_names

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Kneedle cutoffs CSV saved → {out_path}")


# Main function to run similarity analysis on all framework task subsets
def main():

    task_results = []
    print("")
    print_section_header("Similarity Analysis (Framework Task Subsets)")
    for task in SUBSET_TASKS:
        try:
            result = analyze_task(task)
            if result is not None:
                task_results.append(result)
        except Exception as e:
            print(f"  Error analyzing {task.name}: {e}")

    print_section_header("Summary (Kneedle Cutoffs)")
    _print_kneedle_table(task_results, "Framework Task Subsets:", use_for_cutoff=True)

    _save_kneedle_results(task_results)
    _save_results_csv(task_results, CUTOFF_RESULTS_ROOT / "kneedle_cutoffs.csv")

# Save all kneedle results to a single JSON
def _save_kneedle_results(task_results: list[dict]) -> None:

    fp_names  = list(FP_TYPES.keys())
    out: dict = {"fixed_cutoff": FIXED_CUTOFF, "tasks": {}}

    for result in task_results:
        out["tasks"][result["task"]] = {fp: result["kneedles"].get(fp) for fp in fp_names}

    medians: dict[str, float] = {}
    for fp in fp_names:
        vals = [r["kneedles"][fp] for r in task_results if r["kneedles"].get(fp) is not None]
        if vals:
            medians[fp] = float(np.median(vals))

    out["task_medians"] = medians

    out_path = CUTOFF_RESULTS_ROOT / "cutoff_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Cutoff analysis saved → {out_path}")

# Targeted Kneedle validation on the 8 tasks
def validate_cutoff() -> None:

    target_names = {
        "cycpeptmpdb_reg",              "cycpeptmpdb_npmmpd_reg",
        "npmmpd_reg",                   "macrocycledb_oprm1_ki_reg",
        "macrocycledb_sstr2_ic50_reg",  "macrocycledb_mc4r_ki_reg",
    }

    tasks = [t for t in SUBSET_TASKS if t.name in target_names]

    print_section_header(f"Cutoff Validation ({len(tasks)} Key Tasks)")

    results = []

    for task in tasks:

        df, _, _, orig_idx, _ = load_task_data(task, DATA_ROOT)
        fps_dict              = load_all_fps(get_processed_dataset_dir(task.dataset, data_root=DATA_ROOT), orig_idx)
        fps_dict              = _subsample(fps_dict)
        _, _, kneedles        = run_analysis(fps_dict)

        results.append({"task": task.name, "n": len(df), "kneedles": kneedles})

        print(f"\n  {task.name}  (n={len(df)}):")

        for fp, k in kneedles.items():
            flag = "  <-- DIFF" if (k is not None and abs(k - FIXED_CUTOFF) >= 0.05) else ""
            val  = f"{k:.3f}" if k is not None else "N/A"
            print(f"    {fp:<20}: {val}{flag}")

    print()
    print_section_header("Median Kneedle Cutoffs")

    for fp in FP_TYPES:

        vals = [r["kneedles"][fp] for r in results if r["kneedles"].get(fp) is not None]

        if vals:
            med  = float(np.median(vals))
            flag = "  <-- MISMATCH" if abs(med - FIXED_CUTOFF) >= 0.05 else ""
            print(f"  {fp:<20}: median={med:.3f}  fixed={FIXED_CUTOFF}{flag}")
        else:
            print(f"  {fp:<20}: N/A (L1, no Butina)")

    _save_kneedle_results(results)


if __name__ == "__main__":

    with tee_stdout_to_log("analyze_similarity.log"):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--validate-cutoff", action="store_true",
            help="Run targeted Kneedle validation on 8 key tasks only (fast)",
        )

        args = parser.parse_args()

        if args.validate_cutoff:
            validate_cutoff()
        else:
            main()

        print_completion_message()
