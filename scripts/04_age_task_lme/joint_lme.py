"""Joint item-level LME for 51 acoustic features in 232 healthy adults.

Age is centered in years. The model adjusts for sex, education, item position,
and task; it includes participant random intercepts and task slopes. Three
families of 153 contrasts each receive BH-FDR correction. No imputation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMParams
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results" / "04_age_task_lme"
TASKS = ("Dots", "Hanzi", "STT")
PAIRS = (("Dots", "Hanzi"), ("STT", "Dots"), ("STT", "Hanzi"))
PREVIEW_FEATURES = ("Duration", "Gap_Before", "loudness_sma3_stddevNorm")
META = {
    "Subject_ID", "Task", "Item_Index", "Absolute_Word_Start",
    "Absolute_Word_End", "Auto_Word",
}
FORMULA = (
    "Y_z ~ Age_c * (TaskH + TaskS) "
    "+ SexFemale + Education_c + C(Item_Index)"
)
ESTIMATE_COL = "Estimate_SD_per1y_or_at_mean_age"
RE_FORMULA = "~TaskH+TaskS"


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    return s[:-2] if re.fullmatch(r"\d+\.0", s) else s


def load_feature_names() -> list[str]:
    reduction = pd.read_csv(ROOT / "results" / "03_acoustic_measurement_structure" / "redundancy_selection_62_to_51.csv")
    excluded = set(reduction.loc[reduction["Keep for PCA"].eq("No"), "openSMILE variable"])
    header = pd.read_csv(ROOT / "data" / "public" / "hc158_item_62_transformed.csv", nrows=0).columns
    meta = {"Subject_ID", "Task", "Item_Index", "Absolute_Word_Start", "Absolute_Word_End"}
    keep = [name for name in header if name not in meta and name not in excluded]
    if len(keep) != 51:
        raise ValueError("Expected 51 selected features")
    return keep


def load_demographics() -> pd.DataFrame:
    info = pd.read_csv(ROOT / "data" / "public" / "healthy_participants.csv",
                       dtype={"Subject_ID": str})
    return pd.DataFrame({"Subject_ID": info.Subject_ID, "Age": info.Age,
                         "Sex": np.where(info.SexFemale.eq(1), "F", "M"),
                         "Education": info.EducationYears,
                         "Cohort": info.SourceCohort})


def load_data(features: list[str]) -> tuple[pd.DataFrame, dict]:
    pieces = []
    for cohort, expected in (("HC158", 158), ("HC74", 74)):
        path = ROOT / "data" / "public" / f"{cohort.lower()}_item_62_transformed.csv"
        df = pd.read_csv(path, dtype={"Subject_ID": "string"}, low_memory=False)
        if not set(features).issubset(df.columns):
            raise ValueError("Invalid model input or fit diagnostics")
        df["Subject_ID"] = df["Subject_ID"].map(normalize_id)
        if df["Subject_ID"].nunique() != expected:
            raise ValueError("Invalid model input or fit diagnostics")
        df["Cohort"] = cohort
        pieces.append(df[["Subject_ID", "Cohort", "Task", "Item_Index", *features]])
    acoustic = pd.concat(pieces, ignore_index=True)
    if not set(acoustic["Task"]).issubset(TASKS):
        raise ValueError("Invalid model input or fit diagnostics")
    if acoustic.duplicated(["Cohort", "Subject_ID", "Task", "Item_Index"]).any():
        raise ValueError("Invalid model input or fit diagnostics")
    info = load_demographics()
    df = acoustic.merge(info, on=["Cohort", "Subject_ID"], how="left", validate="many_to_one")
    if df[["Age", "Sex", "Education"]].isna().any().any():
        raise ValueError("Invalid model input or fit diagnostics")
    if not df["Sex"].isin(["M", "F"]).all():
        raise ValueError("Invalid model input or fit diagnostics")
    df["Item_Index"] = pd.to_numeric(df["Item_Index"], errors="raise").astype(int)
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="raise").replace([np.inf, -np.inf], np.nan)
    person = df[["Cohort", "Subject_ID", "Age", "Education", "Sex"]].drop_duplicates()
    if len(person) != 232:
        raise ValueError("Invalid model input or fit diagnostics")
    age_mean = float(person["Age"].mean())
    education_mean = float(person["Education"].mean())
    df["Age_c"] = df["Age"] - age_mean
    df["Education_c"] = df["Education"] - education_mean
    df["TaskH"] = df["Task"].eq("Hanzi").astype(float)
    df["TaskS"] = df["Task"].eq("STT").astype(float)
    df["SexFemale"] = df["Sex"].eq("F").astype(float)
    df["PersonKey"] = df["Subject_ID"]
    context = {
        "age_mean": age_mean,
        "education_mean": education_mean,
        "female_weight": float(person["Sex"].eq("F").mean()),
        "n_subjects": len(person),
        "n_rows": len(df),
    }
    return df, context


def contrast_vector(names: list[str], coefficients: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(names), dtype=float)
    for key, value in coefficients.items():
        if key not in names:
            raise KeyError("Missing expected model coefficient")
        out[names.index(key)] = value
    return out


def effect_coefficients(kind: str, task: str, context: dict) -> dict[str, float]:
    if kind == "single_task_age":
        out = {"Age_c": 1.0}
        if task == "Hanzi":
            out["Age_c:TaskH"] = 1.0
        elif task == "STT":
            out["Age_c:TaskS"] = 1.0
        return out
    left, right = task.split("-")
    symbol = {"Dots": {}, "Hanzi": {"TaskH": 1}, "STT": {"TaskS": 1}}
    raw = {}
    for name, sign in ((left, 1), (right, -1)):
        for key, val in symbol[name].items():
            raw[key] = raw.get(key, 0.0) + sign * val
    if kind == "age_by_task":
        return {"Age_c:" + key: val for key, val in raw.items()}
    if kind == "task_contrast":
        return dict(raw)
    raise ValueError(kind)


def estimate_contrast(result, coefficients: dict[str, float]) -> dict:
    names = list(result.fe_params.index)
    vector = contrast_vector(names, coefficients)
    beta = np.asarray(result.fe_params, dtype=float)
    covariance = result.cov_params().loc[names, names].to_numpy(dtype=float)
    estimate = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    if not math.isfinite(variance) or variance <= 0:
        return {ESTIMATE_COL: estimate, "SE": np.nan,
                "CI_low": np.nan, "CI_high": np.nan, "p_raw": np.nan}
    se = math.sqrt(variance)
    return {
        ESTIMATE_COL: estimate,
        "SE": se,
        "CI_low": estimate - 1.96 * se,
        "CI_high": estimate + 1.96 * se,
        "p_raw": float(2 * norm.sf(abs(estimate / se))),
    }


def fit_one(df: pd.DataFrame, feature: str, context: dict):
    data = df.loc[df[feature].notna()].copy()
    scale = float(data[feature].std(ddof=0))
    if len(data) < 500 or data["PersonKey"].nunique() < 100 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid model input or fit diagnostics")
    data["Y_z"] = (data[feature] - float(data[feature].mean())) / scale
    if data.groupby("Task")["PersonKey"].nunique().reindex(TASKS, fill_value=0).min() < 100:
        raise ValueError("Invalid model input or fit diagnostics")
    model = smf.mixedlm(FORMULA, data, groups=data["PersonKey"], re_formula=RE_FORMULA)
    attempts = []
    suspect_full = None
    for method in ("lbfgs", "bfgs"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                result = model.fit(reml=False, method=method, maxiter=300, disp=False)
            except Exception as exc:
                attempts.append(f"{method}: {type(exc).__name__}: {exc}")
                continue
        if result.converged and np.isfinite(result.fe_params).all():
            messages = [str(w.message) for w in caught]
            if not any("Hessian matrix" in msg or "covariance is singular" in msg
                       for msg in messages):
                return result, data, scale, method, "unstructured", messages
            suspect_full = result, data, scale, method, "unstructured", messages
            attempts.append(f"unstable fit")
            break
        attempts.append(f"unstable fit")
    diagonal = MixedLMParams.from_components(
        fe_params=np.ones(model.k_fe), cov_re=np.eye(model.k_re))
    for method in ("lbfgs", "bfgs"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                result = model.fit(reml=False, method=method, maxiter=300,
                                   free=diagonal, disp=False)
            except Exception as exc:
                attempts.append(f"diagonal/{method}: {type(exc).__name__}: {exc}")
                continue
        if result.converged and np.isfinite(result.fe_params).all():
            messages = [str(w.message) for w in caught]
            if not any("Hessian matrix" in msg for msg in messages):
                return result, data, scale, method, "diagonal", messages
            attempts.append(f"unstable fit")
        else:
            attempts.append(f"unstable fit")
    simpler = smf.mixedlm(FORMULA, data, groups=data["PersonKey"], re_formula="~TaskS")
    for method in ("lbfgs", "bfgs"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                result = simpler.fit(reml=False, method=method, maxiter=300, disp=False)
            except Exception as exc:
                attempts.append(f"STT_slope/{method}: {type(exc).__name__}: {exc}")
                continue
        if result.converged and np.isfinite(result.fe_params).all():
            messages = [str(w.message) for w in caught]
            if not any("Hessian matrix" in msg for msg in messages):
                return result, data, scale, method, "STT_slope", messages
            attempts.append(f"unstable fit")
        else:
            attempts.append(f"unstable fit")
    if suspect_full is not None:
        return suspect_full
    raise RuntimeError("; ".join(attempts))


def add_fdr(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["q_family_153"] = np.nan
    table["q_contrast_51"] = np.nan
    for kind, indices in table.groupby("Effect_type").groups.items():
        positions = list(indices)
        p = table.loc[positions, "p_raw"].where(
            table.loc[positions, "Inference_ok"]).fillna(1.0).to_numpy(dtype=float)
        table.loc[positions, "q_family_153"] = multipletests(p, method="fdr_bh")[1]
        for label, indexes in table.loc[positions].groupby("Contrast").groups.items():
            ids = list(indexes)
            pp = table.loc[ids, "p_raw"].where(
                table.loc[ids, "Inference_ok"]).fillna(1.0).to_numpy(dtype=float)
            table.loc[ids, "q_contrast_51"] = multipletests(pp, method="fdr_bh")[1]
    table.loc[table["p_raw"].isna() | ~table["Inference_ok"],
              ["q_family_153", "q_contrast_51"]] = np.nan
    table["FDR_significant"] = table["q_family_153"] < 0.05
    return table



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "recomputed" / "04_age_task_lme")
    parser.add_argument("--max-features", type=int, default=None)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    features = load_feature_names()
    if args.max_features is not None:
        if args.max_features <= 0:
            raise ValueError("max-features must be positive")
        features = features[:args.max_features]
    df, context = load_data(features)
    all_results, diagnostics = [], []
    for index, feature in enumerate(features, 1):
        print(f"[{index}/{len(features)}] {feature}", flush=True)
        missing = int(df[feature].isna().sum())
        diag = {"Feature": feature, "N_rows_total": len(df),
                "Missing_cells": missing, "Missing_rate": missing / len(df),
                "Converged": False, "Inference_ok": False, "Error": ""}
        try:
            result, model_data, scale, method, structure, fit_warnings = fit_one(df, feature, context)
            diag.update({"Converged": True, "Optimizer": method, "N_rows_used": len(model_data),
                         "N_subjects_used": model_data["PersonKey"].nunique(),
                         "Raw_feature_SD": scale, "Log_likelihood": float(result.llf),
                         "Random_cov_min_eigenvalue": float(np.linalg.eigvalsh(result.cov_re).min()),
                         "Random_covariance": structure,
                         "Warnings": " | ".join(fit_warnings)})
            diag["Inference_ok"] = not any("Hessian matrix" in w for w in fit_warnings)
        except Exception as exc:
            diag["Error"] = f"{type(exc).__name__}: {exc}"
            print("FAILED", diag["Error"], flush=True)
            result = None
        diagnostics.append(diag)
        for kind, contrasts in (("single_task_age", TASKS),
                                ("task_contrast", tuple(f"{a}-{b}" for a, b in PAIRS)),
                                ("age_by_task", tuple(f"{a}-{b}" for a, b in PAIRS))):
            for contrast in contrasts:
                row = {"Feature": feature, "Effect_type": kind, "Contrast": contrast,
                       "N_rows_used": diag.get("N_rows_used", np.nan),
                       "N_subjects_used": diag.get("N_subjects_used", np.nan),
                       "Random_covariance": diag.get("Random_covariance", "failed"),
                       "Inference_ok": bool(diag["Inference_ok"])}
                if result is None:
                    row.update({ESTIMATE_COL: np.nan, "SE": np.nan,
                                "CI_low": np.nan, "CI_high": np.nan, "p_raw": np.nan})
                else:
                    row.update(estimate_contrast(result, effect_coefficients(kind, contrast, context)))
                all_results.append(row)
    table = add_fdr(pd.DataFrame(all_results))
    for kind, filename in (("single_task_age", "joint_lme_age_within_task_51_features.csv"),
                           ("task_contrast", "joint_lme_task_contrasts_51_features.csv"),
                           ("age_by_task", "joint_lme_age_by_task_51_features.csv")):
        subset = table.loc[table.Effect_type.eq(kind)].sort_values(
            ["q_family_153", "Feature", "Contrast"], na_position="last")
        subset.to_csv(output / filename, index=False)
    pd.DataFrame(diagnostics).to_csv(output / "joint_lme_fit_diagnostics_51_features.csv", index=False)
    (output / "run_config.json").write_text(json.dumps({
        "model_formula": FORMULA, "random_formula": RE_FORMULA,
        "features": features, **context}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
