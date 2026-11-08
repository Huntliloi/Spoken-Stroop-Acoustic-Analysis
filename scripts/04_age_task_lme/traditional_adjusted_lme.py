"""HC232 51-feature joint LME, additionally adjusted for task-level behavior.

Run: python run_lme_adjusted_traditional.py

The original joint-model machinery, transformations, random-effects fallback,
contrasts and FDR families are imported unchanged from run_lme_51_features.py.
Only two centered/scaled participant-by-task fixed-effect covariates are added:
accuracy and total completion time. No source file is modified.
"""

from __future__ import annotations

import importlib.util
import json
import mimetypes
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Windows sandbox denies the MIME registry lookup performed while importing
# openpyxl. XLSX reading needs the built-in MIME table, not that registry data.
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results" / "recomputed" / "traditional_adjusted"
PARENT = ROOT / "results" / "04_age_task_lme"
SOURCE = ROOT / "scripts" / "04_age_task_lme" / "joint_lme.py"
FORMULA = (
    "Y_z ~ Age_c * (TaskH + TaskS) + SexFemale + Education_c "
    "+ Accuracy_z + Completion_z + C(Item_Index)"
)


def load_original_module():
    spec = importlib.util.spec_from_file_location("original_joint_lme", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original model: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_behavior(base) -> tuple[pd.DataFrame, dict]:
    parts = []
    for cohort, expected in (("HC158", 158), ("HC74", 74)):
        path = ROOT / "data" / "public" / "healthy_traditional.csv"
        source = pd.read_csv(path, dtype={"Subject_ID": "string"})
        cohort_ids = pd.read_csv(ROOT / "data" / "public" /
                                 f"{cohort.lower()}_item_51_transformed.csv",
                                 usecols=["Subject_ID"], dtype={"Subject_ID": str}).Subject_ID.unique()
        source = source[source.Subject_ID.isin(cohort_ids)].copy()
        source["Subject_ID"] = source["Subject_ID"].map(base.normalize_id)
        if source["Subject_ID"].nunique() != expected:
            raise ValueError(f"{path}: expected {expected} subjects")
        if source.duplicated(["Subject_ID", "Task"]).any():
            raise ValueError(f"{path}: duplicate subject-task rows")
        if set(source["Task"]) != set(base.TASKS):
            raise ValueError(f"{path}: unexpected task set")
        source["Cohort"] = cohort
        parts.append(source[["Cohort", "Subject_ID", "Task", "Accuracy",
                             "Completion_Time_sec"]])
    behavior = pd.concat(parts, ignore_index=True)
    if len(behavior) != 232 * 3:
        raise ValueError(f"Expected 696 participant-task observations, got {len(behavior)}")
    for column in ("Accuracy", "Completion_Time_sec"):
        behavior[column] = pd.to_numeric(behavior[column], errors="raise")
        if not np.isfinite(behavior[column]).all():
            raise ValueError(f"Non-finite {column} in behavioral table")
    if not behavior["Accuracy"].between(0, 1).all():
        raise ValueError("Accuracy must be a proportion in [0, 1]")
    if not behavior["Completion_Time_sec"].gt(0).all():
        raise ValueError("Completion time must be positive")
    scales = {}
    for source, target in (("Accuracy", "Accuracy_z"),
                           ("Completion_Time_sec", "Completion_z")):
        mean = float(behavior[source].mean())
        sd = float(behavior[source].std(ddof=0))
        if sd <= 0:
            raise ValueError(f"Zero SD: {source}")
        behavior[target] = (behavior[source] - mean) / sd
        scales[source] = {"mean": mean, "sd": sd}
    return behavior, scales


def add_behavior_to_acoustics(base, acoustic: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    behavior, scales = load_behavior(base)
    joined = acoustic.merge(
        behavior, on=["Cohort", "Subject_ID", "Task"], how="left",
        validate="many_to_one", indicator=True,
    )
    if len(joined) != len(acoustic) or not joined["_merge"].eq("both").all():
        raise ValueError("Behavior-acoustic participant-task join is incomplete")
    joined = joined.drop(columns="_merge")
    if joined[["Accuracy_z", "Completion_z"]].isna().any().any():
        raise ValueError("Missing behavioral covariates after join")
    return joined, scales


def fit_all(base, df: pd.DataFrame, context: dict, features: list[str]):
    rows = []
    diagnostics = []
    for index, feature in enumerate(features, 1):
        print(f"[{index}/{len(features)}] {feature}", flush=True)
        diag = {
            "Feature": feature,
            "N_rows_total": len(df),
            "Missing_cells": int(df[feature].isna().sum()),
            "Missing_rate": float(df[feature].isna().mean()),
            "Converged": False,
            "Inference_ok": False,
            "Error": "",
        }
        try:
            try:
                result, model_data, scale, method, structure, fit_warnings = (
                    base.fit_one(df, feature, context)
                )
            except RuntimeError as first_error:
                # The original hierarchy ends at intercept + STT slope. If all
                # those fits fail after behavioral adjustment, retain a clearly
                # labeled random-intercept LME as a last sensitivity fallback.
                model_data = df.loc[df[feature].notna()].copy()
                scale = float(model_data[feature].std(ddof=0))
                if len(model_data) < 500 or model_data["PersonKey"].nunique() < 100 or scale <= 0:
                    raise first_error
                model_data["Y_z"] = (
                    model_data[feature] - float(model_data[feature].mean())) / scale
                model = smf.mixedlm(
                    FORMULA, model_data, groups=model_data["PersonKey"], re_formula="1")
                last_error = first_error
                for candidate in ("lbfgs", "bfgs"):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        try:
                            fit = model.fit(
                                reml=False, method=candidate, maxiter=300, disp=False)
                        except Exception as exc:
                            last_error = exc
                            continue
                    messages = [str(w.message) for w in caught]
                    if (fit.converged and np.isfinite(fit.fe_params).all()
                            and not any("Hessian matrix" in msg for msg in messages)):
                        result = fit
                        method = candidate
                        structure = "random_intercept_only"
                        fit_warnings = messages
                        break
                    last_error = RuntimeError("random-intercept fit unstable")
                else:
                    raise RuntimeError(
                        f"Original hierarchy failed: {first_error}; "
                        f"random-intercept fallback failed: {last_error}")
            diag.update({
                "Converged": True,
                "Optimizer": method,
                "N_rows_used": len(model_data),
                "N_subjects_used": model_data["PersonKey"].nunique(),
                "Raw_feature_SD": scale,
                "Log_likelihood": float(result.llf),
                "Random_cov_min_eigenvalue": float(
                    np.linalg.eigvalsh(result.cov_re).min()),
                "Random_covariance": structure,
                "Warnings": " | ".join(fit_warnings),
            })
            diag["Inference_ok"] = not any(
                "Hessian matrix" in message for message in fit_warnings
            )
        except Exception as exc:
            result = None
            diag["Error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {diag['Error']}", flush=True)
        diagnostics.append(diag)
        for kind, contrasts in (
            ("single_task_age", base.TASKS),
            ("task_contrast", tuple(f"{a}-{b}" for a, b in base.PAIRS)),
            ("age_by_task", tuple(f"{a}-{b}" for a, b in base.PAIRS)),
        ):
            for contrast in contrasts:
                row = {
                    "Feature": feature,
                    "Effect_type": kind,
                    "Contrast": contrast,
                    "N_rows_used": diag.get("N_rows_used", np.nan),
                    "N_subjects_used": diag.get("N_subjects_used", np.nan),
                    "Random_covariance": diag.get("Random_covariance", "failed"),
                    "Inference_ok": bool(diag["Inference_ok"]),
                }
                if result is None:
                    row.update({name: np.nan for name in (
                        base.ESTIMATE_COL, "SE", "CI_low", "CI_high", "p_raw")})
                else:
                    row.update(base.estimate_contrast(
                        result, base.effect_coefficients(kind, contrast, context)))
                rows.append(row)
    return base.add_fdr(pd.DataFrame(rows)), pd.DataFrame(diagnostics)


def make_comparison(adjusted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for kind, name in (
        ("single_task_age", "joint_lme_age_within_task_51_features.csv"),
        ("task_contrast", "joint_lme_task_contrasts_51_features.csv"),
        ("age_by_task", "joint_lme_age_by_task_51_features.csv"),
    ):
        original = pd.read_csv(PARENT / name)
        if len(original) != 153 or original["Feature"].nunique() != 51:
            raise ValueError(f"Original result is incomplete: {name}")
        a = adjusted.loc[adjusted["Effect_type"].eq(kind)]
        merged = original.merge(
            a, on=["Feature", "Effect_type", "Contrast"],
            how="outer", validate="one_to_one", suffixes=("_original", "_adjusted"),
            indicator=True,
        )
        if len(merged) != 153 or not merged["_merge"].eq("both").all():
            raise ValueError(f"Original-adjusted result keys do not match: {kind}")
        for suffix in ("original", "adjusted"):
            merged[f"significant_q05_{suffix}"] = (
                merged[f"Inference_ok_{suffix}"] &
                merged[f"q_family_153_{suffix}"].lt(.05))
            merged[f"significant_q001_{suffix}"] = (
                merged[f"Inference_ok_{suffix}"] &
                merged[f"q_family_153_{suffix}"].lt(.001))
        columns = [
            "Feature", "Effect_type", "Contrast",
            "N_rows_used_original", "N_rows_used_adjusted",
            "Inference_ok_original", "Inference_ok_adjusted",
            "Random_covariance_original", "Random_covariance_adjusted",
            "Estimate_SD_per1y_or_at_mean_age_original",
            "Estimate_SD_per1y_or_at_mean_age_adjusted",
            "p_raw_original", "p_raw_adjusted",
            "q_family_153_original", "q_family_153_adjusted",
            "significant_q05_original", "significant_q05_adjusted",
            "significant_q001_original", "significant_q001_adjusted",
        ]
        parts.append(merged[columns])
    comparison = pd.concat(parts, ignore_index=True)
    summary = comparison.groupby(["Effect_type", "Contrast"], sort=False).agg(
        original_q05=("significant_q05_original", "sum"),
        adjusted_q05=("significant_q05_adjusted", "sum"),
        original_q001=("significant_q001_original", "sum"),
        adjusted_q001=("significant_q001_adjusted", "sum"),
        valid_original=("Inference_ok_original", "sum"),
        valid_adjusted=("Inference_ok_adjusted", "sum"),
    ).reset_index()
    summary["retained_q05"] = [
        int((group["significant_q05_original"] & group["significant_q05_adjusted"]).sum())
        for _, group in comparison.groupby(["Effect_type", "Contrast"], sort=False)
    ]
    summary["new_q05"] = summary["adjusted_q05"] - summary["retained_q05"]
    summary["lost_q05"] = summary["original_q05"] - summary["retained_q05"]
    return comparison, summary


def main():
    base = load_original_module()
    features = base.load_feature_names()
    acoustic, context = base.load_data(features)
    data, scales = add_behavior_to_acoustics(base, acoustic)
    if data["PersonKey"].nunique() != 232:
        raise ValueError("Expected 232 participants after behavioral join")
    base.FORMULA = FORMULA
    adjusted, diagnostics = fit_all(base, data, context, features)
    comparison, summary = make_comparison(adjusted)
    HERE.mkdir(parents=True, exist_ok=True)
    outputs = (
        (adjusted.loc[adjusted["Effect_type"].eq("single_task_age")],
         "age_within_task_traditional_adjusted.csv"),
        (adjusted.loc[adjusted["Effect_type"].eq("age_by_task")],
         "joint_lme_age_by_task_adjusted_for_traditional.csv"),
        (adjusted.loc[adjusted["Effect_type"].eq("task_contrast")],
         "task_effects_traditional_adjusted.csv"),
        (diagnostics, "joint_lme_fit_diagnostics_51_features.csv"),
        (comparison, "comparison.csv"),
        (summary, "significant_counts.csv"),
    )
    for table, name in outputs:
        table.to_csv(HERE / name, index=False, encoding="utf-8-sig")
    config = {
        "formula": FORMULA,
        "random_formula": base.RE_FORMULA,
        "n_subjects": context["n_subjects"],
        "n_acoustic_rows": context["n_rows"],
        "n_participant_task_rows": 696,
        "features": features,
        "behavior_standardization": scales,
        "multiple_testing": "Separate BH-FDR families of 153 prespecified tests per effect type; invalid inference assigned p=1 before FDR.",
        "interpretation": "Conditioning on concurrent task performance; not a total task/age effect or causal direct effect.",
    }
    (HERE / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Completed: {HERE}", flush=True)


if __name__ == "__main__":
    main()
