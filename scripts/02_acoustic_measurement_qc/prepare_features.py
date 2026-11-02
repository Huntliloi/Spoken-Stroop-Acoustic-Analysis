"""Apply frozen measurement-QC transformations and 62-to-51 selection.

Input tables contain the un-imputed, item-level measurements. Missing values
remain missing. Yeo-Johnson lambdas are read from the published rule table.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "public"
RESULTS = ROOT / "results"
FLOOR = {"F1amplitudeLogRelF0_sma3nz_amean",
         "F2amplitudeLogRelF0_sma3nz_amean",
         "F3amplitudeLogRelF0_sma3nz_amean"}


def transform(raw: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    for feature, row in rules.iterrows():
        values = pd.to_numeric(raw[feature], errors="coerce").to_numpy(float, copy=True)
        values[~np.isfinite(values)] = np.nan
        if feature in FLOOR:
            values[np.isclose(values, -201, rtol=1e-5, atol=1e-8)] = np.nan
        method = row["Transformation"]
        if method == "Raw":
            result = values
        elif method == "log1p":
            if np.any(values[np.isfinite(values)] < 0):
                raise ValueError(f"Negative input to log1p: {feature}")
            result = np.log1p(values)
        elif method == "Yeo-Johnson":
            result = np.full_like(values, np.nan)
            valid = np.isfinite(values)
            result[valid] = yeojohnson(values[valid], lmbda=float(row["Yeo-Johnson lambda"]))
        else:
            raise ValueError(f"Unknown transform: {method}")
        out[feature] = result
    meta = ["Subject_ID", "Task", "Item_Index", "Absolute_Word_Start",
            "Absolute_Word_End"]
    return out[[*meta, *rules.index]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Save recomputed tables under results/recomputed")
    args = parser.parse_args()
    rules = pd.read_csv(RESULTS / "02_acoustic_measurement_qc" / "frozen_transformation_rules_62.csv").set_index("openSMILE variable")
    reduction = pd.read_csv(RESULTS / "03_acoustic_measurement_structure" / "redundancy_selection_62_to_51.csv")
    excluded = set(reduction.loc[reduction["Keep for PCA"].eq("No"), "openSMILE variable"])
    selected = [feature for feature in rules.index if feature not in excluded]
    if len(rules) != 62 or len(selected) != 51:
        raise ValueError("Expected 62 QC and 51 selected features")
    for cohort in ("hc158", "hc74"):
        raw = pd.read_csv(DATA / f"{cohort}_item_raw.csv", dtype={"Subject_ID": str})
        out62 = transform(raw, rules)
        meta = [c for c in out62 if c not in rules.index]
        out51 = out62[[*meta, *selected]]
        for n, calculated in ((62, out62), (51, out51)):
            frozen = pd.read_csv(DATA / f"{cohort}_item_{n}_transformed.csv", dtype={"Subject_ID": str})
            if list(calculated) != list(frozen) or not np.allclose(
                calculated.select_dtypes("number"), frozen.select_dtypes("number"),
                equal_nan=True, rtol=1e-10, atol=1e-10):
                raise AssertionError(f"Frozen table mismatch: {cohort}/{n}")
            if args.write:
                output = RESULTS / "recomputed"
                output.mkdir(parents=True, exist_ok=True)
                calculated.to_csv(output / f"{cohort}_item_{n}_transformed.csv", index=False)
        print(cohort, "transformation verified")


if __name__ == "__main__":
    main()
