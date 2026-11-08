"""Within-subject word-level manual/auto correlations for four feature sets.

For each subject × task × feature, Pearson and Spearman correlations use the
same finite paired words (minimum 8 of 12), provided both methods vary.
Subject-level correlations are first median-aggregated within each task ×
feature. The publication-style table then summarizes these feature medians;
ALL pools the three task × feature medians, not raw words or subjects.

Four sets are reported: all 90 raw; QC60 raw; final 49 raw; and final 49 on
their frozen downstream-analysis scales. No interpolation or exclusion of
boundary outliers is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import manual_auto_icc as base
import retained_features_icc as final


HERE = base.ROOT / "01_automated_measurement_validation"
MIN_PAIRED_WORDS = 8
TASK_NAMES = {"Hanzi": "W", "Dots": "C", "STT": "CW"}
SET_NAMES = {
    "All_90_raw": "All 90",
    "QC_60_raw": "QC 60",
    "Final_49_raw": "Final 49, raw",
    "Final_49_transformed": "Final 49, transformed",
}


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denominator = np.linalg.norm(x0) * np.linalg.norm(y0)
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(np.clip(np.dot(x0, y0) / denominator, -1, 1))


def subject_task_correlations(paired: pd.DataFrame, features: list[str],
                              scale: str) -> pd.DataFrame:
    records = []
    for (subject, task), group in paired.groupby(["Subject_ID", "Task"], sort=True):
        m = group[[f"{name}_manual" for name in features]].to_numpy(dtype=float)
        a = group[[f"{name}_auto" for name in features]].to_numpy(dtype=float)
        for j, name in enumerate(features):
            valid = np.isfinite(m[:, j]) & np.isfinite(a[:, j])
            count = int(valid.sum())
            r = rho = np.nan
            if count >= MIN_PAIRED_WORDS:
                x, y = m[valid, j], a[valid, j]
                if np.ptp(x) > 0 and np.ptp(y) > 0:
                    r = correlation(x, y)
                    rho = correlation(rankdata(x, method="average"),
                                      rankdata(y, method="average"))
            records.append((scale, subject, task, TASK_NAMES[task], name,
                            count, r, rho))
    return pd.DataFrame.from_records(records, columns=[
        "Scale", "Subject_ID", "Task", "Scope", "Feature",
        "N_paired_words", "Pearson_r", "Spearman_rho"])


def feature_medians(subject_rows: pd.DataFrame) -> pd.DataFrame:
    return subject_rows.groupby(["Scale", "Scope", "Feature"], as_index=False).agg(
        Pearson_subject_median=("Pearson_r", "median"),
        Spearman_subject_median=("Spearman_rho", "median"),
        N_valid_subjects_pearson=("Pearson_r", "count"),
        N_valid_subjects_spearman=("Spearman_rho", "count"),
    )


def summarize(table: pd.DataFrame, features: list[str], scale: str,
              set_label: str) -> pd.DataFrame:
    selected = table.loc[table.Feature.isin(features) & table.Scale.eq(scale)]
    records = []
    for scope in ["ALL", "W", "C", "CW"]:
        subset = selected if scope == "ALL" else selected.loc[selected.Scope.eq(scope)]
        pr = subset.Pearson_subject_median.dropna()
        sr = subset.Spearman_subject_median.dropna()
        if pr.empty or sr.empty:
            raise ValueError(f"No valid correlations for {scale}, {scope}")
        records.append({
            "Set": set_label, "Scope": scope,
            "Pearson_median": float(pr.median()),
            "Pearson_Q1": float(pr.quantile(.25)),
            "Pearson_Q3": float(pr.quantile(.75)),
            "Spearman_median": float(sr.median()),
            "Spearman_Q1": float(sr.quantile(.25)),
            "Spearman_Q3": float(sr.quantile(.75)),
            "Pearson_ge_070_n": int(pr.ge(.7).sum()),
            "Pearson_ge_070_pct": float(100 * pr.ge(.7).mean()),
            "Spearman_ge_070_n": int(sr.ge(.7).sum()),
            "Spearman_ge_070_pct": float(100 * sr.ge(.7).mean()),
            "Pearson_ge_080_n": int(pr.ge(.8).sum()),
            "Pearson_ge_080_pct": float(100 * pr.ge(.8).mean()),
            "Spearman_ge_080_n": int(sr.ge(.8).sum()),
            "Spearman_ge_080_pct": float(100 * sr.ge(.8).mean()),
            "N_valid_task_features_pearson": int(len(pr)),
            "N_valid_task_features_spearman": int(len(sr)),
            "N_expected_task_features": len(features) * (3 if scope == "ALL" else 1),
            "Median_valid_subjects_per_task_feature_pearson": float(
                subset.N_valid_subjects_pearson.loc[subset.Pearson_subject_median.notna()].median()),
            "Median_valid_subjects_per_task_feature_spearman": float(
                subset.N_valid_subjects_spearman.loc[subset.Spearman_subject_median.notna()].median()),
        })
    return pd.DataFrame(records)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    manual = pd.read_csv(base.MANUAL, dtype={"Subject_ID": str}, low_memory=False)
    auto = pd.read_csv(base.AUTO, dtype={"Subject_ID": str}, low_memory=False)
    paired = manual.merge(auto, on=base.KEY, how="inner", suffixes=("_manual", "_auto"),
                          validate="one_to_one")
    if len(paired) != len(auto) or paired.groupby(["Subject_ID", "Task"]).size().min() < 8:
        raise ValueError("The paired item structure differs from the HC158 ICC analysis")
    all90 = [c for c in manual if c not in base.KEY + [
        "Absolute_Word_Start", "Absolute_Word_End"]]
    qc62 = set(pd.read_csv(base.QC60)["openSMILE variable"])
    final51, rules = final.select_retained_features()
    if len(all90) != 90 or len(qc62) != 60 or len(final51) != 49:
        raise ValueError("Incorrect feature-set size")
    if not (set(final51) <= qc62 <= set(all90)):
        raise ValueError("Feature sets are not nested")
    qc62_ordered = [name for name in all90 if name in qc62]

    # Build a separate transformed copy; original measurements stay untouched.
    transformed_columns = {}
    for name in final51:
        for method in ("manual", "auto"):
            col = f"{name}_{method}"
            transformed_columns[col] = final.transform(paired[col], name, rules.loc[name])
    transformed = pd.concat([paired[base.KEY], pd.DataFrame(transformed_columns)], axis=1)
    raw_subject = subject_task_correlations(paired, all90, "raw")
    transformed_subject = subject_task_correlations(
        transformed, final51, "transformed")
    all_subject = pd.concat([raw_subject, transformed_subject], ignore_index=True)
    all_feature = feature_medians(all_subject)
    summary = pd.concat([
        summarize(all_feature, all90, "raw", SET_NAMES["All_90_raw"]),
        summarize(all_feature, qc62_ordered, "raw", SET_NAMES["QC_60_raw"]),
        summarize(all_feature, final51, "raw", SET_NAMES["Final_49_raw"]),
        summarize(all_feature, final51, "transformed",
                  SET_NAMES["Final_49_transformed"]),
    ], ignore_index=True)

    # Monotone QC transformations preserve Spearman rho unless the -201 floor
    # is recoded to missing. Cross-check one raw feature with scipy separately
    # during validation; the per-subject rows are retained for auditability.
    raw_final = all_subject.loc[all_subject.Scale.eq("raw")
                                & all_subject.Feature.isin(final51)]
    transformed_final = all_subject.loc[all_subject.Scale.eq("transformed")]
    monotone = raw_final.merge(transformed_final, on=["Subject_ID", "Task", "Feature"],
                               suffixes=("_raw", "_transformed"), validate="one_to_one")
    same_pairs = monotone.N_paired_words_raw.eq(monotone.N_paired_words_transformed)
    comparable = same_pairs & monotone.Spearman_rho_raw.notna() & (
        monotone.Spearman_rho_transformed.notna())
    if not np.allclose(monotone.loc[comparable, "Spearman_rho_raw"],
                       monotone.loc[comparable, "Spearman_rho_transformed"], atol=1e-12):
        raise ValueError("Monotone-transform Spearman invariant check failed")
    audit = {
        "Participants": int(paired.Subject_ID.nunique()),
        "Paired_items": len(paired),
        "Subject_task_groups": int(paired.groupby(["Subject_ID", "Task"]).ngroups),
        "Min_paired_words": MIN_PAIRED_WORDS,
        "Subject_task_group_size_distribution": {
            str(k): int(v) for k, v in paired.groupby(["Subject_ID", "Task"]).size()
            .value_counts().sort_index().items()},
        "Spearman_monotone_check_count": int(comparable.sum()),
        "Spearman_different_pair_count_due_to_floor": int((~same_pairs).sum()),
    }
    all_subject.to_csv(HERE / "within_subject_word_correlations_subject_task_feature.csv",
                       index=False, encoding="utf-8-sig")
    all_feature.to_csv(HERE / "within_subject_word_correlations_task_feature_medians.csv",
                       index=False, encoding="utf-8-sig")
    summary.to_csv(HERE / "within_subject_word_correlations_four_sets.csv",
                   index=False, encoding="utf-8-sig")
    (HERE / "within_subject_word_correlations_four_sets.json").write_text(
        summary.to_json(orient="records", force_ascii=False), encoding="utf-8")
    summary.loc[summary.Scope.eq("ALL")].to_csv(
        HERE / "publication_all_correlation_summary.csv", index=False,
        encoding="utf-8-sig")
    (HERE / "within_subject_word_correlations_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
