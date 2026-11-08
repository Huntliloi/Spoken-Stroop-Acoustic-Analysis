"""Compare 90 original acoustic measurements from manual vs automatic timestamps.

Primary endpoint: ICC(A,1), two-way mixed, single-measure absolute agreement,
within each task at the paired item level. Confidence intervals resample
participants, not dependent items. No imputation, transform, or outlier removal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "public"
HERE = ROOT / "results" / "recomputed" / "01_automated_measurement_validation"
MANUAL = SOURCE / "hc158_item_manual_raw.csv"
AUTO = SOURCE / "hc158_item_raw.csv"
QC62 = ROOT / "results" / "02_acoustic_measurement_qc" / "frozen_transformation_rules_62.csv"
KEY = ["Subject_ID", "Task", "Item_Index"]
TASKS = ["Hanzi", "Dots", "STT"]
N_BOOT = 2000
RNG = np.random.default_rng(20261023)


def icc_from_moments(n, sum_s, sum_s2, sum_d, sum_d2):
    """Two-column ICC(A,1) from moments of s=x+y and d=x-y."""
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vs = (sum_s2 - sum_s**2 / n) / (n - 1)
        vd = (sum_d2 - sum_d**2 / n) / (n - 1)
        md = sum_d / n
        msr, mse, msc = vs / 2, vd / 2, n * md**2 / 2
        denominator = msr + mse + (2 / n) * (msc - mse)
        value = (msr - mse) / denominator
    # Constant identical pairs have zero denominator: agreement is undefined.
    valid = (n >= 3) & np.isfinite(value) & (denominator > 0)
    return np.where(valid, value, np.nan)


def bootstrap_icc(x: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                  all_subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Item ICC and participant-cluster percentile CIs for all features at once."""
    n_rows, n_features = x.shape
    if y.shape != (n_rows, n_features):
        raise ValueError("Shape mismatch")
    codes = pd.Categorical(subjects, categories=all_subjects).codes
    if np.any(codes < 0):
        raise ValueError("Unknown participant in paired rows")
    n_subj = len(all_subjects)
    moments = np.zeros((5, n_subj, n_features), dtype=float)
    for j in range(n_features):
        valid = np.isfinite(x[:, j]) & np.isfinite(y[:, j])
        if not valid.any():
            continue
        xv, yv = x[valid, j], y[valid, j]
        center = np.median(np.r_[xv, yv])
        s, d, c = (xv - center) + (yv - center), xv - yv, codes[valid]
        for k, w in enumerate((np.ones(len(s)), s, s * s, d, d * d)):
            moments[k, :, j] = np.bincount(c, weights=w, minlength=n_subj)
    total = moments.sum(axis=1)
    point = icc_from_moments(*total)
    weights = RNG.multinomial(n_subj, np.repeat(1 / n_subj, n_subj), size=N_BOOT)
    boot_moments = [weights @ moments[k] for k in range(5)]
    draws = icc_from_moments(*boot_moments)
    with np.errstate(all="ignore"):
        ci = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    enough = np.isfinite(draws).mean(axis=0) >= .95
    ci[:, ~enough] = np.nan
    return point, ci[0], ci[1]


def measure(df: pd.DataFrame, features: list[str], all_subjects: np.ndarray,
            level: str, task: str, qc62: set[str]) -> pd.DataFrame:
    x = df[[f"{name}_manual" for name in features]].to_numpy(dtype=float)
    y = df[[f"{name}_auto" for name in features]].to_numpy(dtype=float)
    point, lower, upper = bootstrap_icc(x, y, df.Subject_ID.to_numpy(), all_subjects)
    rows = []
    for j, name in enumerate(features):
        paired = np.isfinite(x[:, j]) & np.isfinite(y[:, j])
        manual_finite = np.isfinite(x[:, j])
        auto_finite = np.isfinite(y[:, j])
        xv, yv = x[paired, j], y[paired, j]
        if len(xv) >= 3:
            err = yv - xv
            rho = float(spearmanr(xv, yv).statistic) if np.std(xv) and np.std(yv) else np.nan
            rmse = float(np.sqrt(np.mean(err**2)))
            mad = float(np.median(np.abs(err)))
            bias = float(np.mean(err))
            sd_manual = float(np.std(xv, ddof=1))
            std_bias = bias / sd_manual if sd_manual else np.nan
            both_zero = float(np.mean((xv == 0) & (yv == 0)))
            discordant_zero = float(np.mean((xv == 0) != (yv == 0)))
        else:
            rho = rmse = mad = bias = std_bias = both_zero = discordant_zero = np.nan
        rows.append({
            "Feature": name, "Task": task, "Level": level,
            "QC62": int(name in qc62), "N_paired": int(paired.sum()),
            "N_manual_finite": int(manual_finite.sum()),
            "N_auto_finite": int(auto_finite.sum()),
            "N_manual_only_finite": int((manual_finite & ~auto_finite).sum()),
            "N_auto_only_finite": int((~manual_finite & auto_finite).sum()),
            "N_participants": int(df.loc[paired, "Subject_ID"].nunique()),
            "ICC_A1": float(point[j]), "ICC_95_low": float(lower[j]),
            "ICC_95_high": float(upper[j]), "Spearman_rho": rho,
            "Mean_auto_minus_manual": bias, "Standardized_mean_bias": std_bias,
            "Median_absolute_difference": mad, "RMSE": rmse,
            "Both_zero_fraction": both_zero, "Discordant_zero_fraction": discordant_zero,
        })
    return pd.DataFrame(rows)


def paired_person_means(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Average each feature over exactly the same valid items for both methods."""
    frames = []
    for name in features:
        cols = ["Subject_ID", "Task", f"{name}_manual", f"{name}_auto"]
        part = df[cols].copy()
        part = part.loc[np.isfinite(part[f"{name}_manual"]) & np.isfinite(part[f"{name}_auto"])]
        group = part.groupby(["Subject_ID", "Task"], as_index=False)[
            [f"{name}_manual", f"{name}_auto"]].mean()
        frames.append(group.set_index(["Subject_ID", "Task"]))
    return pd.concat(frames, axis=1).reset_index()


def rate_counts(data: pd.DataFrame) -> dict:
    out = {}
    for task in ["Pooled", *TASKS]:
        sub = data.loc[data.Task.eq(task)]
        valid = sub.ICC_A1.dropna()
        out[task] = {
            "N_estimable": int(len(valid)), "Median_ICC": float(valid.median()),
            "N_ICC_ge_075": int((valid >= .75).sum()),
            "N_ICC_ge_090": int((valid >= .90).sum()),
            "N_ICC_lt_050": int((valid < .50).sum()),
        }
    return out


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    manual = pd.read_csv(MANUAL, dtype={"Subject_ID": str}, low_memory=False)
    auto = pd.read_csv(AUTO, dtype={"Subject_ID": str}, low_memory=False)
    if manual.duplicated(KEY).any() or auto.duplicated(KEY).any():
        raise ValueError("Duplicate participant/task/item keys")
    features = [c for c in manual.columns if c not in KEY + ["Absolute_Word_Start", "Absolute_Word_End"]]
    if len(features) != 90 or not set(features).issubset(auto.columns):
        raise ValueError("The expected 90 features are not shared")
    qc62 = set(pd.read_csv(QC62)["openSMILE variable"].astype(str))
    if len(qc62) != 62 or not qc62.issubset(features):
        raise ValueError("QC62 names do not match original features")
    all_subjects = np.sort(manual.Subject_ID.unique())
    if len(all_subjects) != 158 or set(all_subjects) != set(auto.Subject_ID.unique()):
        raise ValueError("HC158 participant identity changed")
    outer = manual[KEY].merge(auto[KEY], on=KEY, how="outer", indicator=True)
    missing = outer.loc[outer._merge.ne("both"), KEY + ["_merge"]].copy()
    if len(missing) != 11 or not missing._merge.eq("left_only").all():
        raise ValueError("Unexpected pairing coverage")
    paired = manual.merge(auto, on=KEY, how="inner", suffixes=("_manual", "_auto"),
                          validate="one_to_one")
    if len(paired) != 5677:
        raise ValueError("Unexpected paired row count")
    for feat in features:
        for method in ["manual", "auto"]:
            col = f"{feat}_{method}"
            paired[col] = pd.to_numeric(paired[col], errors="coerce")
            paired.loc[~np.isfinite(paired[col]), col] = np.nan
    item_rows = []
    for task in ["Pooled", *TASKS]:
        frame = paired if task == "Pooled" else paired.loc[paired.Task.eq(task)]
        item_rows.append(measure(frame, features, all_subjects, "Item", task, qc62))
    item = pd.concat(item_rows, ignore_index=True)
    means = paired_person_means(paired, features)
    person_rows = [measure(means.loc[means.Task.eq(task)], features, all_subjects,
                           "Person_task_mean", task, qc62) for task in TASKS]
    person = pd.concat(person_rows, ignore_index=True)
    # Timing disagreement is descriptive evidence and is never used to select ICC rows.
    timing = pd.DataFrame({
        "Subject_ID": paired.Subject_ID, "Task": paired.Task, "Item_Index": paired.Item_Index,
        "Manual_start": paired.Absolute_Word_Start_manual,
        "Auto_start": paired.Absolute_Word_Start_auto,
        "Manual_end": paired.Absolute_Word_End_manual,
        "Auto_end": paired.Absolute_Word_End_auto,
    })
    timing["Start_abs_delta_s"] = (timing.Auto_start - timing.Manual_start).abs()
    timing["End_abs_delta_s"] = (timing.Auto_end - timing.Manual_end).abs()
    timing["Max_boundary_abs_delta_s"] = timing[["Start_abs_delta_s", "End_abs_delta_s"]].max(axis=1)
    outliers = timing.sort_values("Max_boundary_abs_delta_s", ascending=False).head(50)
    timing_by_task = {}
    duration_timing_sensitivity = {}
    for task in TASKS:
        t = timing.loc[timing.Task.eq(task)]
        timing_by_task[task] = {
            "N": len(t), "Median_max_boundary_delta_s": float(t.Max_boundary_abs_delta_s.median()),
            "N_any_over_100ms": int(t.Max_boundary_abs_delta_s.gt(.1).sum()),
            "N_any_over_1s": int(t.Max_boundary_abs_delta_s.gt(1).sum()),
        }
        keep_ids = t.loc[t.Max_boundary_abs_delta_s.le(1), KEY]
        restricted = paired.loc[paired.Task.eq(task)].merge(keep_ids, on=KEY, how="inner",
                                                              validate="one_to_one")
        x = restricted[["Duration_manual"]].to_numpy(dtype=float)
        y = restricted[["Duration_auto"]].to_numpy(dtype=float)
        restricted_icc = bootstrap_icc(x, y, restricted.Subject_ID.to_numpy(),
                                       all_subjects)[0][0]
        duration_timing_sensitivity[task] = {
            "N_pairs_max_boundary_delta_le_1s": len(restricted),
            "Duration_ICC_A1_descriptive": float(restricted_icc),
        }
    summary = {
        "Manual_rows": len(manual), "Auto_rows": len(auto), "Paired_rows": len(paired),
        "Manual_only_rows": len(missing), "Participants": len(all_subjects),
        "Item_level": rate_counts(item),
        "Item_level_QC62": rate_counts(item.loc[item.QC62.eq(1)]),
        "Person_task_mean": {task: rate_counts(person.loc[person.Task.eq(task)])[task]
                             for task in TASKS},
        "Timing": {
            "Start_abs_median_s": float(timing.Start_abs_delta_s.median()),
            "Start_abs_p95_s": float(timing.Start_abs_delta_s.quantile(.95)),
            "Start_abs_p99_s": float(timing.Start_abs_delta_s.quantile(.99)),
            "End_abs_median_s": float(timing.End_abs_delta_s.median()),
            "End_abs_p95_s": float(timing.End_abs_delta_s.quantile(.95)),
            "End_abs_p99_s": float(timing.End_abs_delta_s.quantile(.99)),
            "N_start_over_100ms": int(timing.Start_abs_delta_s.gt(.1).sum()),
            "N_end_over_100ms": int(timing.End_abs_delta_s.gt(.1).sum()),
            "N_any_over_1s": int(timing.Max_boundary_abs_delta_s.gt(1).sum()),
            "By_task": timing_by_task,
        },
        "Duration_boundary_le_1s_sensitivity_not_primary": duration_timing_sensitivity,
        "ICC_model": "ICC(A,1), two-way mixed, single measure, absolute agreement",
        "ICC_CI": "95% percentile bootstrap, 2000 participant-cluster resamples",
    }
    item.to_csv(HERE / "manual_vs_auto_item_icc_90_raw.csv", index=False, encoding="utf-8-sig")
    person.to_csv(HERE / "manual_vs_auto_person_mean_icc_90_raw.csv", index=False, encoding="utf-8-sig")
    (HERE / "item_level_icc_90.json").write_text(
        item.to_json(orient="records", force_ascii=False), encoding="utf-8")
    (HERE / "person_task_mean_icc_90.json").write_text(
        person.to_json(orient="records", force_ascii=False), encoding="utf-8")
    missing.to_csv(HERE / "unpaired_manual_rows_11.csv", index=False, encoding="utf-8-sig")
    outliers.to_csv(HERE / "largest_boundary_differences_50.csv", index=False,
                    encoding="utf-8-sig")
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
