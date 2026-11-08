"""Balanced split-half reliability of task differences and reference anomaly scores."""

from __future__ import annotations

import importlib.util
import itertools
import json
import mimetypes
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, yeojohnson
import statsmodels.formula.api as smf

mimetypes.MimeTypes.read_windows_registry = lambda self, *args, **kwargs: None

PACKAGE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PACKAGE / "scripts"))
from release_paths import output_root
ROOT = output_root(PACKAGE)
HERE = ROOT / "07_within_session_reliability"
LME_SCRIPT = PACKAGE / "scripts" / "04_age_task_lme" / "joint_lme.py"
RAW = PACKAGE / "data" / "public"
RULES = ROOT / "02_acoustic_measurement_qc" / "frozen_transformation_rules_60.csv"
BASE = PACKAGE / "data" / "public" / "cancer_model_input.csv"
SEQ = {"Hanzi": "YRGBGYBRBGRY", "Dots": "RYBGGBYRYRGB", "STT": "YRGBGBYRRBGY"}
CONDITIONS = {"W": ("Hanzi",), "C": ("Dots",), "CW": ("STT",),
              "C-W": ("Dots", "Hanzi"), "CW-C": ("STT", "Dots"),
              "CW-W": ("STT", "Hanzi")}
FLOOR = {"F1amplitudeLogRelF0_sma3nz_amean", "F2amplitudeLogRelF0_sma3nz_amean",
         "F3amplitudeLogRelF0_sma3nz_amean"}


def import_lme():
    spec = importlib.util.spec_from_file_location("original_joint_lme", LME_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def partitions(tasks):
    pool = []
    for a in itertools.combinations(range(12), 6):
        if 0 not in a:  # count complementary halves only once
            continue
        counts = [Counter(SEQ[task][i] for i in a) for task in tasks]
        if sorted(counts[0].values()) != [1, 1, 2, 2] or len(counts[0]) != 4:
            continue
        if any(count != counts[0] for count in counts[1:]):
            continue
        b = tuple(i for i in range(12) if i not in a)
        pool.append((a, b))
    if not pool:
        raise ValueError(f"No matched partitions for {tasks}")
    return pool


def cube(frame, persons, features):
    index = pd.MultiIndex.from_product([persons, SEQ, range(1, 13)],
                                       names=["PersonKey", "Task", "Item_Index"])
    data = frame.set_index(["PersonKey", "Task", "Item_Index"])[features].reindex(index)
    return data.to_numpy(float).reshape(len(persons), 3, 12, len(features))


def half_means(values, positions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(values[:, :, positions, :], axis=2)
    count = np.isfinite(values[:, :, positions, :]).sum(axis=2)
    out[count < 5] = np.nan
    return out


def contrast(values, tasks):
    task_index = {task: index for index, task in enumerate(SEQ)}
    out = values[:, task_index[tasks[0]], :]
    if len(tasks) == 2:
        out = out - values[:, task_index[tasks[1]], :]
    return out


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.ptp(x[ok]) < 1e-12 or np.ptp(y[ok]) < 1e-12:
        return np.nan, int(ok.sum())
    return float(spearmanr(x[ok], y[ok]).statistic), int(ok.sum())


def sb(r):
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def save_summary(rows, prefix):
    detail = pd.DataFrame(rows)
    if prefix == "cancer_meanabsz":
        detail = detail.drop(columns="Spearman_Brown")
        detail.to_csv(HERE / f"{prefix}_by_split.csv", index=False, encoding="utf-8-sig")
        grouped = detail.groupby(["Condition", "Feature"], as_index=False).agg(
            median_r=("Spearman_r", "median"),
            r_split_p025=("Spearman_r", lambda x: x.quantile(.025)),
            r_split_p975=("Spearman_r", lambda x: x.quantile(.975)),
            median_n=("N_participants", "median"), n_splits=("Split", "count"))
        grouped.to_csv(HERE / f"{prefix}_by_feature.csv", index=False,
                       encoding="utf-8-sig")
        summary = grouped.drop(columns="Feature")
        summary.to_csv(HERE / f"{prefix}_summary.csv", index=False,
                       encoding="utf-8-sig")
        return summary
    detail.to_csv(HERE / f"{prefix}_by_split.csv", index=False, encoding="utf-8-sig")
    grouped = detail.groupby(["Condition", "Feature"], as_index=False).agg(
        median_r=("Spearman_r", "median"), median_SB=("Spearman_Brown", "median"),
        p025_r=("Spearman_r", lambda x: x.quantile(.025)),
        p975_r=("Spearman_r", lambda x: x.quantile(.975)),
        p025_SB=("Spearman_Brown", lambda x: x.quantile(.025)),
        p975_SB=("Spearman_Brown", lambda x: x.quantile(.975)),
        median_n=("N_participants", "median"), n_splits=("Split", "count"))
    grouped.to_csv(HERE / f"{prefix}_by_feature.csv", index=False, encoding="utf-8-sig")
    summary = grouped.groupby("Condition", as_index=False).agg(
        N_features=("Feature", "size"), median_feature_r=("median_r", "median"),
        median_feature_SB=("median_SB", "median"),
        Q1_feature_SB=("median_SB", lambda x: x.quantile(.25)),
        Q3_feature_SB=("median_SB", lambda x: x.quantile(.75)),
        n_feature_SB_ge_070=("median_SB", lambda x: int((x >= .70).sum())),
        n_feature_SB_ge_050=("median_SB", lambda x: int((x >= .50).sum())),
        n_splits=("n_splits", "median"), median_n=("median_n", "median"))
    summary.to_csv(HERE / f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_healthy(lme):
    features = lme.load_feature_names()
    frames = []
    for cohort, n in (("HC158", 158), ("HC74", 74)):
        path = PACKAGE / "data" / "public" / f"{cohort.lower()}_item_retained_transformed.csv"
        frame = pd.read_csv(path, dtype={"Subject_ID": str}, low_memory=False)
        if frame.Subject_ID.nunique() != n:
            raise ValueError(f"Wrong HC size: {cohort}")
        frame["Subject_ID"] = frame.Subject_ID.map(lme.normalize_id)
        frame["PersonKey"] = cohort + "::" + frame.Subject_ID
        frame["Item_Index"] = frame.Item_Index.astype(int)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    people = sorted(data.PersonKey.unique())
    return data, people, features


def analyze_differences(healthy, people, features):
    values = cube(healthy, people, features)
    rows = []
    for condition, tasks in CONDITIONS.items():
        splits = partitions(tasks)
        for k, (a, b) in enumerate(splits, 1):
            left = contrast(half_means(values, a), tasks)
            right = contrast(half_means(values, b), tasks)
            for j, feature in enumerate(features):
                r, n = corr(left[:, j], right[:, j])
                rows.append({"Condition": condition, "Feature": feature, "Split": k,
                             "Spearman_r": r, "Spearman_Brown": sb(r),
                             "N_participants": n})
    return save_summary(rows, "healthy_feature_difference")


def transform_cancer(raw, features):
    rules = pd.read_csv(RULES).set_index("openSMILE variable")
    data = raw[["Subject_ID", "Task", "Item_Index"]].copy()
    for feature in features:
        x = pd.to_numeric(raw[feature], errors="coerce").to_numpy(float, copy=True)
        x[~np.isfinite(x)] = np.nan
        if feature in FLOOR:
            x[np.isclose(x, -201.0, rtol=1e-5, atol=1e-8)] = np.nan
        method = rules.loc[feature, "Transformation"]
        ok = np.isfinite(x)
        if method == "log1p":
            if np.any(x[ok] < 0):
                raise ValueError(f"Negative input for {feature}")
            x[ok] = np.log1p(x[ok])
        elif method == "Yeo-Johnson":
            x[ok] = yeojohnson(x[ok], lmbda=float(rules.loc[feature, "Yeo-Johnson lambda"]))
        elif method != "Raw":
            raise ValueError(f"Unknown transformation: {method}")
        data[feature] = x
    return data


def analyze_anomaly(lme, healthy, features):
    base = pd.read_csv(BASE, dtype={"Subject_ID": str})
    raw = pd.read_csv(RAW / "cancer_item_raw.csv", dtype={"Subject_ID": str},
                      low_memory=False)
    raw = raw.loc[raw.Subject_ID.isin(base.Subject_ID)].copy()
    if len(base) != 92 or raw.Subject_ID.nunique() != 92:
        raise ValueError("Expected 92 original cancer subjects")
    clinical = transform_cancer(raw, features)
    clinical = clinical.merge(base[["Subject_ID", "Age", "Female", "Education_years_approx"]],
                              on="Subject_ID", how="left", validate="many_to_one")
    _, context = lme.load_data(features)
    clinical["Age_c"] = clinical.Age - context["age_mean"]
    clinical["Education_c"] = clinical.Education_years_approx - context["education_mean"]
    clinical["SexFemale"] = clinical.Female
    clinical["TaskH"] = clinical.Task.eq("Hanzi").astype(float)
    clinical["TaskS"] = clinical.Task.eq("STT").astype(float)
    clinical["PersonKey"] = "Cancer::" + clinical.Subject_ID
    clinical["Item_Index"] = clinical.Item_Index.astype(int)
    healthy_model, _ = lme.load_data(features)
    hres = healthy_model[["PersonKey", "Task", "Item_Index"]].copy()
    cres = clinical[["PersonKey", "Task", "Item_Index"]].copy()
    methods = []
    for j, feature in enumerate(features, 1):
        print(f"Reference model {j}/{len(features)}: {feature}", flush=True)
        try:
            fit, hdata, sd, optimizer, structure, warn = lme.fit_one(healthy_model, feature, context)
            if not fit.converged or any("Hessian matrix" in w for w in warn):
                raise RuntimeError("Unstable mixed model")
            mean = float(hdata[feature].mean())
            method = f"LME {optimizer} {structure}"
        except Exception as exc:
            hdata = healthy_model.loc[healthy_model[feature].notna()].copy()
            mean = float(hdata[feature].mean())
            sd = float(hdata[feature].std(ddof=0))
            hdata["Y_z"] = (hdata[feature] - mean) / sd
            fit = smf.ols(lme.FORMULA, hdata).fit()
            method = f"OLS fixed-effect fallback ({type(exc).__name__})"
        for source, target in ((hdata, hres), (clinical, cres)):
            valid = source.loc[source[feature].notna()].copy()
            residual = (valid[feature] - mean) / sd - np.asarray(fit.predict(valid), float)
            target[feature] = pd.Series(residual, index=valid.index).reindex(target.index)
        methods.append({"Feature": feature, "Method": method})
    pd.DataFrame(methods).to_csv(HERE / "reference_fit_methods.csv", index=False,
                                 encoding="utf-8-sig")
    hpeople = sorted(hres.PersonKey.unique())
    cpeople = ["Cancer::" + sid for sid in base.Subject_ID]
    hv = cube(hres, hpeople, features)
    cv = cube(cres, cpeople, features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hfull = np.nanmean(hv, axis=2)
    refs = {}
    for condition, tasks in CONDITIONS.items():
        z = contrast(hfull, tasks)
        refs[condition] = (np.nanmean(z, axis=0), np.nanstd(z, axis=0, ddof=1))
    rows = []
    scores = []
    for condition, tasks in CONDITIONS.items():
        splits = partitions(tasks)
        ref_mean, ref_sd = refs[condition]
        if not np.isfinite(ref_sd).all() or (ref_sd <= 0).any():
            raise ValueError(f"Invalid reference SD: {condition}")
        for k, (a, b) in enumerate(splits, 1):
            ma = contrast(half_means(cv, a), tasks)
            mb = contrast(half_means(cv, b), tasks)
            za = (ma - ref_mean) / ref_sd
            zb = (mb - ref_mean) / ref_sd
            na = np.isfinite(za).sum(axis=1)
            nb = np.isfinite(zb).sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sa = np.nanmean(np.abs(za), axis=1)
                sb_score = np.nanmean(np.abs(zb), axis=1)
            sa[na != len(features)] = np.nan
            sb_score[nb != len(features)] = np.nan
            r, n = corr(sa, sb_score)
            rows.append({"Condition": condition, "Feature": "MeanAbsZ_51", "Split": k,
                         "Spearman_r": r, "Spearman_Brown": np.nan,
                         "N_participants": n})
            scores.extend({"Subject_ID": sid, "Condition": condition, "Split": k,
                           "Half_A": va, "Half_B": vb}
                          for sid, va, vb in zip(base.Subject_ID, sa, sb_score))
    pd.DataFrame(scores).to_csv(HERE / "cancer_meanabsz_half_scores.csv", index=False,
                                encoding="utf-8-sig")
    return save_summary(rows, "cancer_meanabsz")


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    lme = import_lme()
    healthy, people, features = load_healthy(lme)
    if len(features) != 49 or len(people) != 232:
        raise ValueError("Original HC232/49 setup changed")
    difference = analyze_differences(healthy, people, features)
    print("Feature difference reliability:\n", difference.to_string(index=False), flush=True)
    anomaly = analyze_anomaly(lme, healthy, features)
    print("Cancer MeanAbsZ reliability:\n", anomaly.to_string(index=False), flush=True)
    (HERE / "method.json").write_text(json.dumps({
        "feature_source": "data/public/cancer_item_raw.csv", "transformed_HC_source": "data/public/hc*_item_retained_transformed.csv",
        "features": 49, "healthy_n": 232, "clinical_n": 92,
        "balancing": "same item positions and expected color-word counts across conditions; each half has 6 items",
        "minimum_valid_items_per_task_half": 5,
        "association": "across-person Spearman correlation, then Spearman-Brown correction",
        "reference": "original HC232 joint LME with same fixed-effect OLS fallback for unstable fits",
        "score": "frozen full-sample HC mean/SD of person-level residuals; MeanAbsZ across all 49 features",
        "interpretation": "within-session item-sampling precision, not test-retest reliability",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
