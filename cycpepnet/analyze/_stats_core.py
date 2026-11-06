import numpy as np
import pandas as pd
from scipy import stats


# Holm-Bonferroni correction for multiple hypothesis testing
def _holm_correction(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []

    clean = np.array([1.0 if pd.isna(p) else float(p) for p in pvals], dtype=float)
    clean = np.nan_to_num(clean, nan=1.0, posinf=1.0, neginf=1.0)

    order     = np.argsort(clean)
    corrected = np.zeros(n)

    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, clean[idx] * (n - rank))

    for i in range(1, n):
        corrected[order[i]] = max(corrected[order[i]], corrected[order[i - 1]])

    return corrected.tolist()

# Benjamini-Hochberg FDR correction for multiple hypothesis testing
def _benjamini_hochberg(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []

    clean = np.array([1.0 if pd.isna(p) else float(p) for p in pvals], dtype=float)
    clean = np.nan_to_num(clean, nan=1.0, posinf=1.0, neginf=1.0)

    order  = np.argsort(clean)
    ranked = clean[order]
    adj    = np.empty(n)
    prev   = 1.0

    for i in range(n - 1, -1, -1):
        rank   = i + 1
        prev   = min(prev, ranked[i] * n / rank)
        adj[i] = prev

    out        = np.empty(n)
    out[order] = adj

    return out.tolist()

# Friedman omnibus + Nemenyi post-hoc on a (blocks x methods) matrix
def _friedman_nemenyi(matrix: pd.DataFrame, higher_is_better: bool = True) -> dict | None:
    clean = matrix.dropna(axis=0, how="any")

    n_blocks  = int(clean.shape[0])
    methods   = list(clean.columns)
    k         = len(methods)

    if n_blocks < 3 or k < 3:
        return None

    ranked     = clean.rank(axis=1, ascending=not higher_is_better, method="average")
    mean_ranks = ranked.mean(axis=0)

    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(
            *[clean[m].to_numpy() for m in methods]
        )
    except ValueError:
        return None

    q_alpha = float(stats.studentized_range.ppf(1.0 - SIGNIFICANCE_ALPHA, k, np.inf)) / np.sqrt(2.0)
    cd      = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    pairwise: list[dict] = []
    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    for i in range(k):
        for j in range(i + 1, k):
            ma, mb    = methods[i], methods[j]
            rank_diff = float(mean_ranks[ma] - mean_ranks[mb])
            z         = abs(rank_diff) / se if se > 0 else 0.0
            p_nemenyi = float(stats.studentized_range.sf(z * np.sqrt(2.0), k, np.inf))
            pairwise.append({
                "method_a":   ma,
                "method_b":   mb,
                "mean_rank_a": float(mean_ranks[ma]),
                "mean_rank_b": float(mean_ranks[mb]),
                "rank_diff":  rank_diff,
                "nemenyi_p":  min(1.0, p_nemenyi),
                "significant": "significant" if abs(rank_diff) > cd else "non-significant",
            })

    return {
        "n_blocks":      n_blocks,
        "n_methods":     k,
        "friedman_stat": float(friedman_stat),
        "friedman_p":    float(friedman_p),
        "critical_difference": float(cd),
        "mean_ranks":    {m: float(mean_ranks[m]) for m in methods},
        "pairwise":      pairwise,
    }

SIGNIFICANCE_ALPHA = 0.05

# Binary significance label at the conventional α=0.05 threshold
def _significance(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "non-significant"
    try:
        return "significant" if float(p) < SIGNIFICANCE_ALPHA else "non-significant"
    except (TypeError, ValueError):
        return "non-significant"
