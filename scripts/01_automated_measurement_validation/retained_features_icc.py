"""ICC for the exact final 49 features, on their frozen analysis scales.

Uses the same 49 features as HC232 LME and applies the fixed QC transformation
rules to *both* manual- and auto-timestamp feature tables. The primary ICC is
item-level ICC(A,1) within task; participant means are supplementary. Raw-
scale ICCs from the full 90-feature analysis are retained for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson

import manual_auto_icc as base


PACKAGE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PACKAGE / "scripts"))
from release_paths import output_root
ROOT = output_root(PACKAGE)
HERE = ROOT / "01_automated_measurement_validation"
REDUCTION = ROOT / "03_acoustic_measurement_structure" / "redundancy_selection.csv"
RULES = ROOT / "02_acoustic_measurement_qc" / "frozen_transformation_rules_60.csv"
FINAL_HC158 = PACKAGE / "data" / "public" / "hc158_item_qualified_transformed.csv"
FLOOR = {
    "F1amplitudeLogRelF0_sma3nz_amean",
    "F2amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_amean",
}


def select_retained_features():
    selected = json.loads((PACKAGE / "data/public/selected_features.json").read_text(encoding="utf-8"))
    rules = pd.read_csv(RULES).set_index("openSMILE variable")
    return selected, rules.loc[selected]


def transform(series: pd.Series, feature: str, rule: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    if feature in FLOOR:
        values[np.isclose(values, -201.0, rtol=1e-5, atol=1e-8)] = np.nan
    method = rule["Transformation"]
    if method == "Raw":
        result = values
    elif method == "log1p":
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError(f"Negative log1p input: {feature}")
        result = np.log1p(values)
    elif method == "Yeo-Johnson":
        result = np.full(values.shape, np.nan)
        valid = np.isfinite(values)
        result[valid] = yeojohnson(values[valid], lmbda=float(rule["Yeo-Johnson lambda"]))
    else:
        raise ValueError(f"Unknown transform {method}")
    return pd.Series(result, index=series.index)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    features, rules = select_retained_features()
    manual = pd.read_csv(base.MANUAL, dtype={"Subject_ID": str}, low_memory=False)
    auto = pd.read_csv(base.AUTO, dtype={"Subject_ID": str}, low_memory=False)
    paired = manual.merge(auto, on=base.KEY, how="inner", suffixes=("_manual", "_auto"),
                          validate="one_to_one")
    if len(paired) != len(auto):
        raise ValueError("HC158 item pairing changed")
    for feature in features:
        for method in ["manual", "auto"]:
            col = f"{feature}_{method}"
            paired[col] = transform(paired[col], feature, rules.loc[feature])
    # Independent cross-check against the frozen HC158 analytic table.
    final_auto = pd.read_csv(FINAL_HC158, dtype={"Subject_ID": str}, low_memory=False)
    check = paired[base.KEY + [f"{x}_auto" for x in features]].merge(
        final_auto[base.KEY + features], on=base.KEY, how="inner", validate="one_to_one")
    if len(check) != len(auto):
        raise ValueError("Final analytic table has different item keys")
    for feature in features:
        a = check[f"{feature}_auto"].to_numpy(dtype=float)
        b = check[feature].to_numpy(dtype=float)
        if not np.allclose(a, b, atol=1e-10, rtol=1e-9, equal_nan=True):
            raise ValueError(f"Transformation mismatch with frozen analysis table: {feature}")
    subjects = np.sort(manual.Subject_ID.unique())
    item = pd.concat([
        base.measure(paired if task == "Pooled" else paired.loc[paired.Task.eq(task)],
                     features, subjects, "Item_transformed", task, set(features))
        for task in ["Pooled", *base.TASKS]
    ], ignore_index=True)
    person_means = base.paired_person_means(paired, features)
    person = pd.concat([
        base.measure(person_means.loc[person_means.Task.eq(task)], features, subjects,
                     "Person_task_mean_transformed", task, set(features))
        for task in base.TASKS
    ], ignore_index=True)
    raw_item = pd.read_csv(HERE / "manual_vs_auto_item_icc_90_raw.csv")
    raw_person = pd.read_csv(HERE / "manual_vs_auto_person_mean_icc_90_raw.csv")
    cols = ["Feature", "Task", "ICC_A1", "ICC_95_low", "ICC_95_high"]
    item = item.merge(raw_item[cols].rename(columns={
        "ICC_A1": "Raw_ICC_A1", "ICC_95_low": "Raw_ICC_95_low", "ICC_95_high": "Raw_ICC_95_high",
    }), on=["Feature", "Task"], how="left", validate="one_to_one")
    person = person.merge(raw_person[cols].rename(columns={
        "ICC_A1": "Raw_ICC_A1", "ICC_95_low": "Raw_ICC_95_low", "ICC_95_high": "Raw_ICC_95_high",
    }), on=["Feature", "Task"], how="left", validate="one_to_one")
    if len(item) != 196 or len(person) != 147 or item.Raw_ICC_A1.isna().any():
        raise ValueError("Final 49 ICC result alignment failed")
    per_task = base.rate_counts(item)
    per_task_raw = base.rate_counts(item.rename(columns={"ICC_A1": "Transformed_ICC_A1",
                                                         "Raw_ICC_A1": "ICC_A1"}))
    person_summary = {task: base.rate_counts(person.loc[person.Task.eq(task)])[task]
                      for task in base.TASKS}
    task_wide = item.loc[item.Task.ne("Pooled")].pivot(index="Feature", columns="Task", values="ICC_A1")
    low = task_wide.loc[task_wide.lt(.5).any(axis=1)].copy()
    low = low.reset_index().sort_values("Feature")
    item.to_csv(HERE / "manual_vs_auto_item_icc_49_transformed.csv", index=False)
    person.to_csv(HERE / "final49_transformed_person_task_icc.csv", index=False)
    print("Computed 49-feature transformed ICCs:", len(item), "item rows,", len(person), "person-task rows")


if __name__ == "__main__":
    main()
