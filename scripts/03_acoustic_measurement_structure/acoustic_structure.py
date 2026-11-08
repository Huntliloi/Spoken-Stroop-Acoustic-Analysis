"""PCA of overall healthy profiles and participant-specific CW-W changes.

The 49-feature set is frozen before this analysis. Parallel analysis permutes
each standardized feature independently across the 232 participants, with the
95th percentile of null eigenvalues as the retention threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PACKAGE / "scripts"))
from release_paths import output_root
ROOT = output_root(PACKAGE)
DATA = PACKAGE / "data" / "public"
OUTPUT = ROOT / "03_acoustic_measurement_structure"
META = {"Subject_ID", "Task", "Item_Index", "Absolute_Word_Start", "Absolute_Word_End"}


def varimax(loadings: np.ndarray, iterations: int = 100,
            tolerance: float = 1e-7) -> np.ndarray:
    p, k = loadings.shape
    rotation = np.eye(k)
    previous = 0.0
    for _ in range(iterations):
        rotated = loadings @ rotation
        u, singular, vh = np.linalg.svd(
            loadings.T @ (rotated**3 - rotated @
                          np.diag(np.diag(rotated.T @ rotated)) / p))
        rotation = u @ vh
        objective = singular.sum()
        if previous and objective / previous < 1 + tolerance:
            break
        previous = objective
    return loadings @ rotation


def run_space(label: str, data: pd.DataFrame, repeats: int, seed: int) -> None:
    values = data.to_numpy(float)
    if values.shape != (232, 49) or not np.isfinite(values).all():
        raise ValueError(f"Expected a complete 232 x 49 matrix for {label}")
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    correlation = np.corrcoef(z, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    rng = np.random.default_rng(seed)
    null = np.empty((repeats, 49))
    for iteration in range(repeats):
        permuted = np.column_stack([rng.permutation(z[:, j]) for j in range(49)])
        null[iteration] = np.linalg.eigvalsh(np.corrcoef(permuted, rowvar=False))[::-1]
    threshold = np.quantile(null, .95, axis=0)
    k = int((eigenvalues > threshold).sum())
    unrotated = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
    rotated = varimax(unrotated)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    correlation_table = pd.DataFrame(correlation, index=data.columns, columns=data.columns)
    correlation_table.to_csv(OUTPUT / f"{label}_correlation.csv", index_label="Feature")
    pd.DataFrame({"Component": np.arange(1, 50), "Observed": eigenvalues,
                  "Null95": threshold, "Retained": eigenvalues > threshold}).to_csv(
        OUTPUT / f"{label}_parallel_analysis.csv", index=False)
    pd.DataFrame(rotated, index=data.columns,
                 columns=[f"PC{i}" for i in range(1, k + 1)]).to_csv(
        OUTPUT / f"{label}_rotated_loadings.csv", index_label="Feature")
    print(label, "participants", len(data), "components", k,
          "variance explained", float(eigenvalues[:k].sum() / 49))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()
    parts = [pd.read_csv(DATA / f"{cohort}_item_retained_transformed.csv",
                         dtype={"Subject_ID": str}) for cohort in ("hc158", "hc74")]
    items = pd.concat(parts, ignore_index=True)
    features = [column for column in items if column not in META]
    if len(features) != 49 or items.Subject_ID.nunique() != 232:
        raise ValueError("Unexpected healthy feature table")
    overall = items.groupby("Subject_ID")[features].mean()
    task_means = items.groupby(["Subject_ID", "Task"])[features].mean()
    cw = task_means.xs("STT", level="Task")
    w = task_means.xs("Hanzi", level="Task").loc[cw.index]
    change = cw - w
    run_space("overall", overall, args.permutations, 20261019)
    run_space("cw_minus_w", change, args.permutations, 20261024)


if __name__ == "__main__":
    main()
