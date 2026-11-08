"""Print fixed-feature-set median item ICCs and participant-bootstrap CIs.

The 90 and QC60 results use their original extraction scales, matching the
previous ICC report. The final 49 use the frozen QC analysis transformations.
The feature sets are held fixed in the bootstrap; this is not a test of the
feature-selection process or of differences among the three nested sets.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import manual_auto_icc as base
import retained_features_icc as final


N_BOOT = 2000
RNG = np.random.default_rng(20261023)
TASKS = ["Pooled", "Hanzi", "Dots", "STT"]


def icc_draws(df: pd.DataFrame, features: list[str], subjects: np.ndarray,
              weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ICC(A,1) for each feature, using one common cluster resample matrix."""
    x = df[[f"{name}_manual" for name in features]].to_numpy(dtype=float)
    y = df[[f"{name}_auto" for name in features]].to_numpy(dtype=float)
    codes = pd.Categorical(df.Subject_ID, categories=subjects).codes
    if np.any(codes < 0):
        raise ValueError("Unknown participant ID")
    moments = np.zeros((5, len(subjects), len(features)), dtype=float)
    for j in range(len(features)):
        valid = np.isfinite(x[:, j]) & np.isfinite(y[:, j])
        xv, yv, c = x[valid, j], y[valid, j], codes[valid]
        center = np.median(np.r_[xv, yv])
        s, d = (xv - center) + (yv - center), xv - yv
        for k, measure in enumerate((np.ones(len(s)), s, s * s, d, d * d)):
            moments[k, :, j] = np.bincount(c, weights=measure, minlength=len(subjects))
    point = base.icc_from_moments(*moments.sum(axis=1))
    boot_moments = [weights @ moments[k] for k in range(5)]
    draws = base.icc_from_moments(*boot_moments)
    return point, draws


def describe(point: np.ndarray, draws: np.ndarray,
             feature_indices: list[int]) -> dict[str, float | int]:
    chosen = np.asarray(feature_indices, dtype=int)
    median_draws = np.nanmedian(draws[:, chosen], axis=1)
    low, high = np.nanpercentile(median_draws, [2.5, 97.5])
    return {
        "N_features": int(len(chosen)),
        "Median_ICC": float(np.nanmedian(point[chosen])),
        "CI_95_low": float(low),
        "CI_95_high": float(high),
        "N_valid_bootstrap": int(np.isfinite(median_draws).sum()),
    }


def main() -> None:
    manual = pd.read_csv(base.MANUAL, dtype={"Subject_ID": str}, low_memory=False)
    auto = pd.read_csv(base.AUTO, dtype={"Subject_ID": str}, low_memory=False)
    paired = manual.merge(auto, on=base.KEY, how="inner", suffixes=("_manual", "_auto"),
                          validate="one_to_one")
    if len(paired) != len(auto):
        raise ValueError("The paired HC158 item set has changed")
    subjects = np.sort(manual.Subject_ID.unique())
    weights = RNG.multinomial(len(subjects), np.repeat(1 / len(subjects), len(subjects)),
                              size=N_BOOT)
    raw_features = [c for c in manual if c not in base.KEY + [
        "Absolute_Word_Start", "Absolute_Word_End"]]
    qc62 = set(pd.read_csv(base.QC60)["openSMILE variable"])
    final51, rules = final.select_retained_features()
    if len(raw_features) != 90 or len(qc62) != 60 or len(final51) != 49:
        raise ValueError("Unexpected feature set size")
    if not (set(final51) <= qc62 <= set(raw_features)):
        raise ValueError("Feature sets are not nested as expected")
    raw_indices = list(range(len(raw_features)))
    qc62_indices = [j for j, name in enumerate(raw_features) if name in qc62]
    transformed_columns = {}
    for name in final51:
        for source in ["manual", "auto"]:
            col = f"{name}_{source}"
            transformed_columns[col] = final.transform(paired[col], name, rules.loc[name])
    transformed = pd.concat([paired[base.KEY],
                             pd.DataFrame(transformed_columns, index=paired.index)], axis=1)
    prior_raw = pd.read_csv(base.HERE / "manual_vs_auto_item_icc_90_raw.csv")
    prior_final = pd.read_csv(base.HERE / "manual_vs_auto_item_icc_49_transformed.csv")
    result = {}
    for task in TASKS:
        raw_subset = paired if task == "Pooled" else paired.loc[paired.Task.eq(task)]
        final_subset = transformed if task == "Pooled" else transformed.loc[
            transformed.Task.eq(task)]
        raw_point, raw_draws = icc_draws(raw_subset, raw_features, subjects, weights)
        final_point, final_draws = icc_draws(final_subset, final51, subjects, weights)
        check_raw = prior_raw.loc[prior_raw.Task.eq(task)].set_index("Feature").loc[
            raw_features, "ICC_A1"].to_numpy()
        check_final = prior_final.loc[prior_final.Task.eq(task)].set_index("Feature").loc[
            final51, "ICC_A1"].to_numpy()
        if not np.allclose(raw_point, check_raw, atol=1e-9, rtol=1e-8, equal_nan=True):
            raise ValueError(f"Raw ICC point estimates do not reproduce for {task}")
        if not np.allclose(final_point, check_final, atol=1e-9, rtol=1e-8,
                           equal_nan=True):
            raise ValueError(f"Final ICC point estimates do not reproduce for {task}")
        result[task] = {
            "All_90_raw": describe(raw_point, raw_draws, raw_indices),
            "QC_60_raw": describe(raw_point, raw_draws, qc62_indices),
            "Final_49_transformed": describe(final_point, final_draws,
                                             list(range(len(final51)))),
            "Final_49_raw_for_scale_check": describe(
                raw_point, raw_draws,
                [j for j, name in enumerate(raw_features) if name in final51]),
        }
    rows = []
    labels = {
        "All_90_raw": "All 90",
        "QC_60_raw": "QC 60",
        "Final_49_raw_for_scale_check": "Final 49, raw",
        "Final_49_transformed": "Final 49, transformed",
    }
    for key, label in labels.items():
        for task in ["Hanzi", "Dots", "STT", "Pooled"]:
            row = result[task][key]
            rows.append({"Feature_set": label, "Task": task, **row})
    pd.DataFrame(rows).to_csv(
        base.HERE / "median_icc_90_60_49_summary.csv", index=False,
        encoding="utf-8-sig")
    (base.HERE / "median_icc_90_60_49_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
