"""Recompute six cancer acoustic anomaly scores from the healthy reference.

For each feature, fit the published joint LME on HC232 item data. A feature
flagged as unstable in the archived diagnostics uses the fixed-effect OLS
fallback. Participant-level task means of standardized fixed-effect residuals
are contrasted, then referenced to the corresponding HC232 distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson
import statsmodels.formula.api as smf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '04_age_task_lme'))
import joint_lme as lme

PACKAGE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PACKAGE / "scripts"))
from release_paths import output_root
ROOT = output_root(PACKAGE)
DATA = PACKAGE / "data" / "public"
RESULTS = ROOT
FLOOR = {"F1amplitudeLogRelF0_sma3nz_amean",
         "F2amplitudeLogRelF0_sma3nz_amean",
         "F3amplitudeLogRelF0_sma3nz_amean"}
CONTRASTS = (("W", "Hanzi", None), ("C", "Dots", None),
             ("CW", "STT", None), ("C-W", "Dots", "Hanzi"),
             ("CW-C", "STT", "Dots"), ("CW-W", "STT", "Hanzi"))


def transform(values: pd.Series, feature: str, rule: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float, copy=True)
    x[~np.isfinite(x)] = np.nan
    if feature in FLOOR:
        x[np.isclose(x, -201, rtol=1e-5, atol=1e-8)] = np.nan
    method = rule["Transformation"]
    if method == "Raw":
        y = x
    elif method == "log1p":
        if np.any(x[np.isfinite(x)] < 0):
            raise ValueError(f"Negative log1p value: {feature}")
        y = np.log1p(x)
    elif method == "Yeo-Johnson":
        y = np.full_like(x, np.nan)
        valid = np.isfinite(x)
        y[valid] = yeojohnson(x[valid], lmbda=float(rule["Yeo-Johnson lambda"]))
    else:
        raise ValueError(method)
    return pd.Series(y, index=values.index)


def residual_wide(fit, frame: pd.DataFrame, feature: str,
                  mean: float, scale: float) -> pd.DataFrame:
    observed = (frame[feature].astype(float) - mean) / scale
    predicted = pd.Series(np.asarray(fit.predict(frame), float), index=frame.index)
    part = frame[["Subject_ID", "Task"]].copy()
    part["Residual"] = observed - predicted
    return part.groupby(["Subject_ID", "Task"]).Residual.mean().unstack()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=49,
                        help="Use 1 for a quick smoke test; default is the full 49")
    args = parser.parse_args()
    names = lme.load_feature_names()[:args.max_features]
    healthy, context = lme.load_data(names)
    meta = pd.read_csv(DATA / "cancer_participants.csv", dtype={"Subject_ID": str})
    raw = pd.read_csv(DATA / "cancer_item_raw.csv", dtype={"Subject_ID": str})
    assert meta.Subject_ID.nunique() == raw.Subject_ID.nunique() == 92
    rules = pd.read_csv(RESULTS / "02_acoustic_measurement_qc" / "frozen_transformation_rules_60.csv").set_index("openSMILE variable")
    diagnostics = pd.read_csv(RESULTS / "04_age_task_lme" / "joint_lme_fit_diagnostics_retained_features.csv").set_index("Feature")
    clinical = raw[["Subject_ID", "Task", "Item_Index"]].copy()
    for feature in names:
        clinical[feature] = transform(raw[feature], feature, rules.loc[feature])
    clinical = clinical.merge(meta[["Subject_ID", "Age", "Female", "Education_years_approx"]],
                              on="Subject_ID", validate="many_to_one")
    clinical["Age_c"] = clinical.Age - context["age_mean"]
    clinical["Education_c"] = clinical.Education_years_approx - context["education_mean"]
    clinical["SexFemale"] = clinical.Female.astype(float)
    clinical["TaskH"] = clinical.Task.eq("Hanzi").astype(float)
    clinical["TaskS"] = clinical.Task.eq("STT").astype(float)
    clinical["PersonKey"] = "Cancer::" + clinical.Subject_ID
    clinical["Item_Index"] = pd.to_numeric(clinical.Item_Index).astype(int)
    detail, reference = [], []
    for index, feature in enumerate(names, 1):
        print(f"Reference model {index}/{len(names)}: {feature}", flush=True)
        if not bool(diagnostics.loc[feature, "Inference_ok"]):
            hdata = healthy[healthy[feature].notna()].copy()
            mean = float(hdata[feature].mean())
            scale = float(hdata[feature].std(ddof=0))
            hdata["Y_z"] = (hdata[feature] - mean) / scale
            fit = smf.ols(lme.FORMULA, hdata).fit()
            method = "fixed_effect_OLS_fallback"
        else:
            fit, hdata, scale, method, _, warnings = lme.fit_one(healthy, feature, context)
            if not fit.converged or any("Hessian matrix" in warning for warning in warnings):
                raise RuntimeError(f"Unstable fit: {feature}")
            mean = float(hdata[feature].mean())
        hwide = residual_wide(fit, hdata, feature, mean, scale)
        cwide = residual_wide(fit, clinical[clinical[feature].notna()], feature, mean, scale)
        for condition, task_a, task_b in CONTRASTS:
            hc = (hwide[task_a] if task_b is None else hwide[task_a] - hwide[task_b]).dropna()
            ca = cwide[task_a] if task_b is None else cwide[task_a] - cwide[task_b]
            if len(hc) != 232:
                raise ValueError(f"Incomplete healthy reference: {feature}/{condition}")
            mu, sd = float(hc.mean()), float(hc.std(ddof=1))
            reference.append({"Feature": feature, "Condition": condition,
                              "HC_mean": mu, "HC_SD": sd, "Estimator": method})
            for subject_id, z in ((ca - mu) / sd).dropna().items():
                detail.append({"Subject_ID": subject_id, "Feature": feature,
                               "Condition": condition, "SignedZ": float(z), "AbsZ": float(abs(z))})
    output = RESULTS / "06_cancer_fact_cog"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reference).to_csv(output / "hc232_reference_parameters_retained_features.csv", index=False)
    ztable = pd.DataFrame(detail)
    ztable.to_csv(output / "cancer_feature_level_z.csv", index=False)
    scores = ztable.groupby(["Subject_ID", "Condition"]).AbsZ.agg(["mean", "count"]).reset_index()
    if args.max_features == 49 and (len(scores) != 92 * 6 or not scores["count"].eq(49).all()):
        raise ValueError("Expected six complete 49-feature scores for each of 92 participants")
    scores.to_csv(output / "cancer_six_scores.csv", index=False)



if __name__ == "__main__":
    main()
