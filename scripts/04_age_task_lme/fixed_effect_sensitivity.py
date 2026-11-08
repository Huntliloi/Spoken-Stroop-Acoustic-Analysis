"""Supplement to the 49-feature joint LME: task gradient and baseline-speech sensitivity.

Run: python run_supplement.py

All acoustic inputs are the already QC-transformed, non-imputed 49-feature CSVs.
No original LME result is overwritten.  The subject-fixed-effect and baseline-
speech models are sensitivity analyses, not a cognitive-versus-motor causal
decomposition.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests


PACKAGE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PACKAGE / "scripts"))
from release_paths import output_root
ROOT = output_root(PACKAGE)
HERE = ROOT / "sensitivity"
LME_DIR = ROOT / "04_age_task_lme"
META = {
    "Subject_ID", "Task", "Item_Index", "Absolute_Word_Start",
    "Absolute_Word_End", "Auto_Word",
}
TASKS = ("Hanzi", "Dots", "STT")  # W, C, CW
CONTRASTS = ("STT-Hanzi", "STT-Dots", "Dots-Hanzi")
EXPECTED_WORDS = {
    "Hanzi": "yellow red green blue green yellow blue red blue green red yellow".split(),
    "Dots": "red yellow blue green green blue yellow red yellow red green blue".split(),
    "STT": "yellow red green blue green blue yellow red red blue green yellow".split(),
}
VOICE_FEATURES = (
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",
)
TIME_FEATURES = {
    "Duration", "Gap_Before", "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec", "MeanVoicedSegmentLengthSec",
}
RNG_SEED = 20261023
N_BOOT = 2000


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return value[:-2] if re.fullmatch(r"\d+\.0", value) else value


def load_demographics() -> pd.DataFrame:
    info = pd.read_csv(PACKAGE / "data" / "public" / "healthy_participants.csv",
                       dtype={"Subject_ID": str})
    return pd.DataFrame({"Subject_ID": info.Subject_ID, "Age": info.Age,
                         "Sex": np.where(info.SexFemale.eq(1), "F", "M"),
                         "Education": info.EducationYears,
                         "Cohort": info.SourceCohort})


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames = []
    for cohort, expected in (("HC158", 158), ("HC74", 74)):
        frame = pd.read_csv(
            PACKAGE / "data" / "public" / f"{cohort.lower()}_item_retained_transformed.csv",
            dtype={"Subject_ID": "string"}, low_memory=False,
        )
        frame["Subject_ID"] = frame["Subject_ID"].map(normalize_id)
        if frame["Subject_ID"].nunique() != expected:
            raise ValueError(f"Unexpected subject count in {cohort}")
        frame["Cohort"] = cohort
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    features = [name for name in frames[0].columns if name not in META | {"Cohort"}]
    if len(features) != 49 or any(name not in frames[1].columns for name in features):
        raise ValueError("Expected the same 49 features in both cohorts")
    if data.duplicated(["Cohort", "Subject_ID", "Task", "Item_Index"]).any():
        raise ValueError("Duplicate word rows")
    if not set(data["Task"]).issubset(TASKS):
        raise ValueError("Unknown task")
    data = data.merge(load_demographics(), on=["Cohort", "Subject_ID"],
                      how="left", validate="many_to_one")
    if data[["Age", "Sex", "Education"]].isna().any().any():
        raise ValueError("Missing demographic data")
    if not data["Sex"].isin(["M", "F"]).all():
        raise ValueError("Unexpected sex code")
    data["Item_Index"] = pd.to_numeric(data["Item_Index"], errors="raise").astype(int)
    if not data["Item_Index"].between(1, 12).all():
        raise ValueError("Item index outside 1..12")
    data["TargetWord"] = [EXPECTED_WORDS[task][item - 1]
                          for task, item in zip(data["Task"], data["Item_Index"])]
    data["PersonKey"] = data["Cohort"] + "::" + data["Subject_ID"]
    people = data[["PersonKey", "Age", "Sex", "Education"]].drop_duplicates()
    if len(people) != 232:
        raise ValueError("Expected 232 healthy subjects")
    age_mean = float(people["Age"].mean())
    edu_mean = float(people["Education"].mean())
    data["Age_c"] = data["Age"] - age_mean
    data["Education_c"] = data["Education"] - edu_mean
    data["SexFemale"] = data["Sex"].eq("F").astype(float)
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="raise")
        data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
    people = data[["PersonKey", "Cohort", "Subject_ID", "Age", "Age_c",
                   "SexFemale", "Education_c"]].drop_duplicates().set_index("PersonKey")
    return data, people, features


def load_traditional(people: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for cohort in ("HC158", "HC74"):
        frame = pd.read_csv(PACKAGE / "data" / "public" / "healthy_traditional.csv",
                            dtype={"Subject_ID": "string"})
        frame = frame[frame.Subject_ID.isin(
            people.loc[people.Cohort.eq(cohort), "Subject_ID"])].copy()
        frame["Subject_ID"] = frame["Subject_ID"].map(normalize_id)
        frame["PersonKey"] = cohort + "::" + frame["Subject_ID"]
        frames.append(frame)
    traditional = pd.concat(frames, ignore_index=True)
    if traditional.duplicated(["PersonKey", "Task"]).any():
        raise ValueError("Duplicate traditional subject-task rows")
    if set(traditional["PersonKey"]) != set(people.index):
        raise ValueError("Traditional and acoustic subjects do not match")
    for column in ("Accuracy", "Completion_Time_sec", "Paired_Item_Count"):
        traditional[column] = pd.to_numeric(traditional[column], errors="coerce")
    if len(traditional) != 232 * 3:
        raise ValueError("Expected three traditional task rows per subject")
    return traditional


def standardize_acoustic(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = data.copy()
    for feature in features:
        scale = float(out[feature].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid feature scale: {feature}")
        out[feature] = (out[feature] - float(out[feature].mean())) / scale
    return out


def subject_task_means(data: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    means = data.groupby(["PersonKey", "Task"], sort=False)[features].mean()
    result = {}
    for task in TASKS:
        result[task] = means.xs(task, level="Task").sort_index()
        if len(result[task]) != 232 or result[task].isna().any().any():
            raise ValueError(f"Incomplete person-task feature means for {task}")
    return result


def bootstrap_gradient(means: dict[str, pd.DataFrame], people: pd.DataFrame,
                       features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = sorted(people.index)
    p = people.loc[keys]
    x = np.column_stack([
        np.ones(len(keys)), p["Age_c"].to_numpy(float),
        p["SexFemale"].to_numpy(float) - p["SexFemale"].mean(),
        p["Education_c"].to_numpy(float),
    ])
    ys = {}
    for contrast in CONTRASTS:
        left, right = contrast.split("-")
        ys[contrast] = (means[left].loc[keys, features] -
                        means[right].loc[keys, features]).to_numpy(float)

    def coefficients(indices: np.ndarray) -> dict[str, np.ndarray]:
        xx = x[indices]
        return {name: np.linalg.lstsq(xx, values[indices], rcond=None)[0]
                for name, values in ys.items()}

    all_idx = np.arange(len(keys))
    point = coefficients(all_idx)
    rng = np.random.default_rng(RNG_SEED)
    columns = [(kind, name) for kind in ("Task_shift", "Age_modulation")
               for name in CONTRASTS]
    boot = np.empty((N_BOOT, len(columns)), dtype=float)
    for b in range(N_BOOT):
        estimates = coefficients(rng.choice(all_idx, size=len(all_idx), replace=True))
        for j, (kind, name) in enumerate(columns):
            beta = estimates[name][0 if kind == "Task_shift" else 1]
            boot[b, j] = float(np.median(np.abs(beta)))

    summary = []
    for j, (kind, name) in enumerate(columns):
        beta = point[name][0 if kind == "Task_shift" else 1]
        summary.append({
            "Metric": kind, "Contrast": name,
            "Median_abs_beta_SD_per_year_or_task": float(np.median(np.abs(beta))),
            "Bootstrap_CI_low": float(np.quantile(boot[:, j], .025)),
            "Bootstrap_CI_high": float(np.quantile(boot[:, j], .975)),
            "N_features": len(features), "N_subjects": len(keys),
            "Bootstrap_repetitions": N_BOOT,
        })
    comparisons = []
    for kind in ("Task_shift", "Age_modulation"):
        for high, low in (("STT-Hanzi", "STT-Dots"),
                          ("STT-Dots", "Dots-Hanzi"),
                          ("STT-Hanzi", "Dots-Hanzi")):
            jh, jl = columns.index((kind, high)), columns.index((kind, low))
            diff = boot[:, jh] - boot[:, jl]
            comparisons.append({
                "Metric": kind, "Higher_contrast": high, "Lower_contrast": low,
                "Difference_median_abs_beta":
                    summary[jh]["Median_abs_beta_SD_per_year_or_task"] -
                    summary[jl]["Median_abs_beta_SD_per_year_or_task"],
                "Bootstrap_CI_low": float(np.quantile(diff, .025)),
                "Bootstrap_CI_high": float(np.quantile(diff, .975)),
                "Bootstrap_fraction_difference_gt_0": float(np.mean(diff > 0)),
            })
    return pd.DataFrame(summary), pd.DataFrame(comparisons)


def baseline_proxies(means: dict[str, pd.DataFrame], traditional: pd.DataFrame,
                     people: pd.DataFrame) -> pd.DataFrame:
    proxies = pd.DataFrame(index=people.index)
    for feature in VOICE_FEATURES:
        proxies[f"W_{feature}"] = means["Hanzi"][feature].reindex(proxies.index)
    times = traditional.pivot(index="PersonKey", columns="Task", values="Completion_Time_sec")
    if times.isna().any().any() or (times <= 0).any().any():
        raise ValueError("Missing/nonpositive completion time")
    proxies["W_log_completion"] = np.log(times["Hanzi"].reindex(proxies.index))
    for column in proxies:
        scale = float(proxies[column].std(ddof=0))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError(f"Invalid baseline proxy: {column}")
        proxies[column] = (proxies[column] - proxies[column].mean()) / scale
    if proxies.isna().any().any():
        raise ValueError("Missing baseline proxy; no imputation permitted")
    return proxies


def proxy_names_for_feature(feature: str) -> list[str]:
    names = [f"W_{f}" for f in VOICE_FEATURES if f != feature]
    if feature not in TIME_FEATURES:
        names.append("W_log_completion")
    return names


def fixed_effects_matrix(data: pd.DataFrame, proxies: pd.DataFrame) -> pd.DataFrame:
    h = data["Task"].eq("Hanzi").astype(float)
    s = data["Task"].eq("STT").astype(float)
    x = pd.DataFrame({
        "TaskH": h, "TaskS": s,
        "AgeH": data["Age_c"] * h, "AgeS": data["Age_c"] * s,
        "SexH": data["SexFemale"] * h, "SexS": data["SexFemale"] * s,
        "EduH": data["Education_c"] * h, "EduS": data["Education_c"] * s,
    }, index=data.index)
    for item in range(2, 13):
        x[f"Item_{item}"] = data["Item_Index"].eq(item).astype(float)
    for word in ("red", "green", "blue"):
        x[f"Word_{word}"] = data["TargetWord"].eq(word).astype(float)
    p = proxies.reindex(data["PersonKey"]).set_axis(data.index)
    for name in proxies:
        x[f"{name}_H"] = p[name] * h
        x[f"{name}_S"] = p[name] * s
    return x.astype(float)


def subject_demean(values: pd.DataFrame | pd.Series, groups: pd.Series):
    return values - values.groupby(groups, sort=False).transform("mean")


def contrast_stats(result, left: str, right: str, n_subjects: int) -> dict:
    vector = np.zeros(len(result.params), dtype=float)
    names = list(result.params.index)
    term = {"Dots": None, "Hanzi": "AgeH", "STT": "AgeS"}
    for task, sign in ((left, 1), (right, -1)):
        if term[task] is not None:
            vector[names.index(term[task])] += sign
    estimate = float(vector @ result.params.to_numpy(float))
    covariance = result.cov_params().to_numpy(float)
    variance = float(vector @ covariance @ vector)
    if not np.isfinite(variance) or variance <= 0:
        return {"Beta_SD_per_year": estimate, "SE": np.nan, "CI_low": np.nan,
                "CI_high": np.nan, "p_raw": np.nan}
    se = math.sqrt(variance)
    crit = float(student_t.ppf(.975, n_subjects - 1))
    p = float(2 * student_t.sf(abs(estimate / se), n_subjects - 1))
    return {"Beta_SD_per_year": estimate, "SE": se,
            "CI_low": estimate - crit * se, "CI_high": estimate + crit * se,
            "p_raw": p}


def fixed_effects_models(data: pd.DataFrame, features: list[str],
                         proxies: pd.DataFrame) -> pd.DataFrame:
    x_all = fixed_effects_matrix(data, proxies)
    rows = []
    for index, feature in enumerate(features, 1):
        valid = data[feature].notna()
        person = data.loc[valid, "PersonKey"]
        y = subject_demean(data.loc[valid, feature], person)
        proxy_names = proxy_names_for_feature(feature)
        base_names = [name for name in x_all if not name.startswith("W_")]
        models = {
            "FE_demographics_word": base_names,
            "FE_plus_W_speech_proxy": base_names +
                [f"{name}_{suffix}" for name in proxy_names for suffix in ("H", "S")],
        }
        for label, names in models.items():
            x = subject_demean(x_all.loc[valid, names], person)
            if np.linalg.matrix_rank(x.to_numpy()) != len(names):
                raise ValueError(f"Rank deficient {feature}, {label}")
            fit = sm.OLS(y, x).fit(cov_type="cluster",
                                   cov_kwds={"groups": person}, use_t=False)
            n_subjects = person.nunique()
            for contrast in CONTRASTS:
                left, right = contrast.split("-")
                rows.append({
                    "Feature": feature, "Model": label, "Contrast": contrast,
                    "N_rows": int(valid.sum()), "N_subjects": n_subjects,
                    "N_baseline_proxies": len(proxy_names) if label.endswith("proxy") else 0,
                    **contrast_stats(fit, left, right, n_subjects),
                })
        if index % 10 == 0 or index == len(features):
            print(f"Fixed-effect models: {index}/{len(features)} features", flush=True)
    result = pd.DataFrame(rows)
    result["q_FDR_family"] = np.nan
    for model, ids in result.groupby("Model").groups.items():
        p = result.loc[ids, "p_raw"].fillna(1.0).to_numpy(float)
        result.loc[ids, "q_FDR_family"] = multipletests(p, method="fdr_bh")[1]
    return result




def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    data, people, features = load_data()
    traditional = load_traditional(people)
    standardized = standardize_acoustic(data, features)
    means = subject_task_means(standardized, features)
    proxies = baseline_proxies(means, traditional, people)
    fixed = fixed_effects_models(standardized, features, proxies)
    fixed.to_csv(HERE / "participant_fixed_effect_sensitivity.csv", index=False)
    print("Saved", len(fixed), "sensitivity contrasts")


if __name__ == "__main__":
    main()
