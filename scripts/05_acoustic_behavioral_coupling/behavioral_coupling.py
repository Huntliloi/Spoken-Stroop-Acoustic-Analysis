"""HC158+HC74 acoustic–behavioral Stroop coupling on 46 non-direct-time features.

Primary contrast: STT−Hanzi (CW−W). Completion-time cost is STT−Hanzi;
accuracy cost is Hanzi−STT, so larger means more errors under conflict.
No acoustic-to-behavior prediction claim is made. Individual word-level
acoustic measures are first averaged within task, then differenced within
person. HC3 regressions and a permutation omnibus quantify convergence.
"""

from __future__ import annotations

import importlib.util
import json
import mimetypes
from pathlib import Path

mimetypes.MimeTypes.read_windows_registry = lambda self, *args, **kwargs: None

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


PACKAGE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PACKAGE / "scripts"))
from release_paths import output_root
ROOT = output_root(PACKAGE)
HERE = ROOT / "05_acoustic_behavioral_coupling"
LME_SCRIPT = PACKAGE / "scripts" / "04_age_task_lme" / "joint_lme.py"
META = {"Subject_ID", "Task", "Item_Index", "Absolute_Word_Start",
        "Absolute_Word_End", "Auto_Word"}
DIRECT_TIMING = {
    "Duration", "Gap_Before", "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec", "MeanVoicedSegmentLengthSec",
}
CONTRASTS = (("STT_minus_Hanzi", "STT", "Hanzi"),
             ("STT_minus_Dots", "STT", "Dots"),
             ("Dots_minus_Hanzi", "Dots", "Hanzi"))
SEED = 20261023
N_PERM = 4999


def module_at(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_hc_lme", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(data: pd.DataFrame, name: str) -> None:
    data.to_csv(HERE / name, index=False, encoding="utf-8-sig")


def load() -> tuple[pd.DataFrame, list[str]]:
    lme = module_at(LME_SCRIPT)
    features = lme.load_feature_names()
    if len(features) != 49:
        raise ValueError("Frozen 49-feature list or direct timing set changed")
    acoustics, traditional = [], []
    for cohort, expected in (("HC158", 158), ("HC74", 74)):
        a = pd.read_csv(PACKAGE / "data" / "public" / f"{cohort.lower()}_item_retained_transformed.csv",
                        dtype={"Subject_ID": str}, low_memory=False)
        if set(a.columns) - META != set(features):
            raise ValueError(f"49-feature table differs from LME selection: {cohort}")
        a["Subject_ID"] = a.Subject_ID.map(lme.normalize_id)
        a["Cohort"] = cohort
        if a.Subject_ID.nunique() != expected or a.duplicated(
                ["Subject_ID", "Task", "Item_Index"]).any():
            raise ValueError(f"Acoustic keys invalid: {cohort}")
        acoustics.append(a)
        t = pd.read_csv(PACKAGE / "data" / "public" / "healthy_traditional.csv",
                        dtype={"Subject_ID": str})
        t = t[t.Subject_ID.isin(a.Subject_ID)].copy()
        t["Subject_ID"] = t.Subject_ID.map(lme.normalize_id)
        t["Cohort"] = cohort
        if t.duplicated(["Subject_ID", "Task"]).any() or t.Subject_ID.nunique() != expected:
            raise ValueError(f"Traditional keys invalid: {cohort}")
        traditional.append(t)
    a = pd.concat(acoustics, ignore_index=True)
    t = pd.concat(traditional, ignore_index=True)
    info = lme.load_demographics()
    info = info.loc[info.set_index(["Cohort", "Subject_ID"]).index.isin(
        a.set_index(["Cohort", "Subject_ID"]).index)].copy()
    if len(info) != 232 or info.duplicated(["Cohort", "Subject_ID"]).any():
        raise ValueError("Expected 232 unique matching demographic records")
    if not info.Sex.isin(["M", "F"]).all() or info[["Age", "Education"]].isna().any().any():
        raise ValueError("Missing or invalid demographic covariate")
    for col in features:
        a[col] = pd.to_numeric(a[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    a["Item_Index"] = pd.to_numeric(a.Item_Index, errors="raise").astype(int)
    if not set(a.Task).issubset({"Hanzi", "Dots", "STT"}) or not set(t.Task).issubset(
            {"Hanzi", "Dots", "STT"}):
        raise ValueError("Unknown task")
    group = ["Cohort", "Subject_ID", "Task"]
    means = a.groupby(group, observed=True)[features].mean().reset_index()
    acoustic_wide = means.set_index(group)[features].unstack("Task")
    trad_wide = t.set_index(group)[["Accuracy", "Completion_Time_sec",
                                    "Raw_Color_Token_Count"]].unstack("Task")
    keys = ["Cohort", "Subject_ID"]
    subject = info.set_index(keys).copy()
    subject["Female"] = subject.Sex.eq("F").astype(int)
    for task in ("Hanzi", "Dots", "STT"):
        subject[f"Accuracy_{task}"] = pd.to_numeric(trad_wide[("Accuracy", task)])
        subject[f"Time_{task}"] = pd.to_numeric(trad_wide[("Completion_Time_sec", task)])
        subject[f"Tokens_{task}"] = pd.to_numeric(trad_wide[("Raw_Color_Token_Count", task)])
        subject[f"Acoustic_items_{task}"] = a.loc[a.Task.eq(task)].groupby(keys).Item_Index.nunique()
    delta_data = {}
    for name, hi, lo in CONTRASTS:
        subject[f"Time_cost_{name}"] = subject[f"Time_{hi}"] - subject[f"Time_{lo}"]
        subject[f"Accuracy_cost_{name}"] = subject[f"Accuracy_{lo}"] - subject[f"Accuracy_{hi}"]
        subject[f"Both12_{name}"] = subject[[f"Tokens_{hi}", f"Tokens_{lo}"]].eq(12).all(axis=1)
        for feature in features:
            delta_data[f"Delta_{name}__{feature}"] = (acoustic_wide[(feature, hi)]
                                                       - acoustic_wide[(feature, lo)])
    subject = pd.concat([subject, pd.DataFrame(delta_data, index=subject.index)], axis=1)
    subject = subject.reset_index()
    if len(subject) != 232 or subject[[f"Time_cost_{name}" for name, _, _ in CONTRASTS]
                                     + [f"Accuracy_cost_{name}" for name, _, _ in CONTRASTS]].isna().any().any():
        raise ValueError("Missing behavioral task cost")
    return subject, features


def covariates(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"const": 1.,
                         "Age_c": frame.Age.to_numpy(float) - frame.Age.mean(),
                         "Female": frame.Female.to_numpy(float),
                         "Education_c": frame.Education.to_numpy(float) - frame.Education.mean()},
                        index=frame.index)


def fit_feature(frame: pd.DataFrame, feature: str, contrast: str, outcome: str,
                sample: str) -> dict:
    ycol = f"Delta_{contrast}__{feature}"
    xcol = f"{outcome}_cost_{contrast}"
    data = frame[["Age", "Female", "Education", xcol, ycol]].dropna().copy()
    n = len(data)
    if n < 30 or data[ycol].std(ddof=1) == 0 or data[xcol].std(ddof=1) == 0:
        return {"Sample": sample, "Contrast": contrast, "Behavior": outcome,
                "Feature": feature, "N": n, "Status": "insufficient_variation"}
    xb = (data[xcol] - data[xcol].mean()) / data[xcol].std(ddof=1)
    ya = (data[ycol] - data[ycol].mean()) / data[ycol].std(ddof=1)
    c = covariates(data)
    x = c.assign(Behavior_cost_z=xb)
    fit = sm.OLS(ya, x).fit(cov_type="HC3")
    ci = fit.conf_int().loc["Behavior_cost_z"]
    bx = sm.OLS(xb, c).fit().resid
    ay = sm.OLS(ya, c).fit().resid
    rho, rho_p = spearmanr(bx, ay)
    return {"Sample": sample, "Contrast": contrast, "Behavior": outcome,
            "Feature": feature, "Category": "direct_timing" if feature in DIRECT_TIMING else "non_direct_timing",
            "N": n, "N_nonzero_behavior_cost": int(data[xcol].ne(0).sum()),
            "Beta_SD_per1SD_cost": float(fit.params["Behavior_cost_z"]),
            "CI_low": float(ci.iloc[0]), "CI_high": float(ci.iloc[1]),
            "P_HC3": float(fit.pvalues["Behavior_cost_z"]),
            "Partial_Spearman_rho": float(rho), "Partial_Spearman_p": float(rho_p),
            "Status": "ok"}


def fit_set(frame: pd.DataFrame, features: list[str], sample: str,
            contrasts: tuple[str, ...]) -> pd.DataFrame:
    if not features or DIRECT_TIMING.intersection(features):
        raise ValueError("Individual coupling tests require the retained non-direct-time features")
    rows = [fit_feature(frame, f, name, behavior, sample)
            for name in contrasts for behavior in ("Time", "Accuracy") for f in features]
    out = pd.DataFrame(rows)
    out["Q_BH_features_within_behavior_contrast"] = np.nan
    for _, idx in out.groupby(["Sample", "Contrast", "Behavior"]).groups.items():
        ix = list(idx)
        p = out.loc[ix, "P_HC3"].fillna(1).to_numpy(float)
        out.loc[ix, "Q_BH_features_within_behavior_contrast"] = multipletests(p, method="fdr_bh")[1]
    return out


def omnibus(frame: pd.DataFrame, features: list[str], contrast: str,
            behavior: str, rng: np.random.Generator) -> dict:
    subset = [f for f in features if f not in DIRECT_TIMING]
    cols = [f"Delta_{contrast}__{f}" for f in subset]
    xcol = f"{behavior}_cost_{contrast}"
    data = frame[["Age", "Female", "Education", xcol, *cols]].dropna()
    c = covariates(data).to_numpy(float)
    c_pinv = np.linalg.pinv(c)
    m = np.eye(len(data)) - c @ c_pinv
    y = data[xcol].to_numpy(float)
    a = data[cols].to_numpy(float)
    y_fitted = c @ (c_pinv @ y)
    y_residual = y - y_fitted
    if not np.allclose(y_fitted + y_residual, y):
        raise AssertionError("Reduced behavioral model did not reconstruct the observed cost")
    yr, ar = m @ y, m @ a
    ar /= np.linalg.norm(ar, axis=0)
    yr /= np.linalg.norm(yr)
    observed = float(np.mean((yr @ ar) ** 2))
    exceed = 0
    for _ in range(N_PERM):
        # Freedman–Lane: retain the age/sex/education fitted component and
        # permute only the residuals of the reduced behavioral model.
        y_star = y_fitted + rng.permutation(y_residual)
        yp = m @ y_star
        yp /= np.linalg.norm(yp)
        stat = float(np.mean((yp @ ar) ** 2))
        exceed += stat >= observed - 1e-15
    return {"Contrast": contrast, "Behavior": behavior,
            "N_complete": len(data), "N_features": len(subset),
            "Mean_partial_r2": observed,
            "P_permutation": (exceed + 1) / (N_PERM + 1),
            "N_permutations": N_PERM,
            "Note": "Mean squared partial Pearson correlation; Freedman-Lane permutation of behavioral residuals after age/sex/education adjustment"}




def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    subject, features = load()
    selected = [f for f in features if f not in DIRECT_TIMING]
    if len(selected) != len(features) - len(DIRECT_TIMING & set(features)):
        raise ValueError("Non-direct-time feature selection is inconsistent")
    contrast = "STT_minus_Hanzi"
    estimates = fit_set(subject, selected, "all_232", (contrast,))
    estimates.to_csv(HERE / "feature_coupling_recomputed.csv", index=False)
    rng = np.random.default_rng(SEED)
    global_rows = [omnibus(subject, selected, contrast, outcome, rng)
                   for outcome in ("Time", "Accuracy")]
    globals_ = pd.DataFrame(global_rows)
    globals_["Q_BH_two_behavior_tests"] = multipletests(
        globals_.P_permutation, method="fdr_bh")[1]
    globals_.to_csv(HERE / "overall_permutation_recomputed.csv", index=False)
    print(globals_.to_string(index=False))


if __name__ == "__main__":
    main()
