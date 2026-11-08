"""Original HC158 manual/automatic item-feature agreement by manual word length."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, yeojohnson
from statsmodels.stats.multitest import multipletests


PACKAGE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PACKAGE / "scripts"))
from release_paths import output_root
ROOT = output_root(PACKAGE)
HERE = ROOT / "01_automated_measurement_validation" / "short_word_precision"
INPUT = PACKAGE / "data" / "public"
RULES = ROOT / "02_acoustic_measurement_qc" / "frozen_transformation_rules_60.csv"
LME_DIR = ROOT / "04_age_task_lme"
KEY = ["Subject_ID", "Task", "Item_Index"]
META = {"Subject_ID", "Task", "Item_Index", "Absolute_Word_Start",
        "Absolute_Word_End", "Auto_Word"}
FLOOR = {"F1amplitudeLogRelF0_sma3nz_amean", "F2amplitudeLogRelF0_sma3nz_amean",
         "F3amplitudeLogRelF0_sma3nz_amean"}
STRATA = ["<200 ms", "200–300 ms", "300–500 ms", "≥500 ms"]


def transform(values, feature, rule):
    x = pd.to_numeric(values, errors="coerce").to_numpy(float, copy=True)
    x[~np.isfinite(x)] = np.nan
    if feature in FLOOR:
        x[np.isclose(x, -201.0, rtol=1e-5, atol=1e-8)] = np.nan
    method = rule["Transformation"]
    ok = np.isfinite(x)
    if method == "log1p":
        if np.any(x[ok] < 0):
            raise ValueError(f"Negative log1p input: {feature}")
        x[ok] = np.log1p(x[ok])
    elif method == "Yeo-Johnson":
        x[ok] = yeojohnson(x[ok], lmbda=float(rule["Yeo-Johnson lambda"]))
    elif method != "Raw":
        raise ValueError(f"Unknown transform: {feature}")
    return x


def icc_absolute(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 20:
        return np.nan
    a, b = x[ok], y[ok]
    s, d = a + b, a - b
    msr = np.var(s, ddof=1) / 2
    mse = np.var(d, ddof=1) / 2
    msc = n * np.mean(d) ** 2 / 2
    den = msr + mse + 2 * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def correlation(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 30 or np.ptp(x[good]) < 1e-12 or np.ptp(y[good]) < 1e-12:
        return np.nan
    return float(spearmanr(x[good], y[good]).statistic)


def load():
    manual = pd.read_csv(INPUT / "hc158_item_manual_raw.csv",
                         dtype={"Subject_ID": str}, low_memory=False)
    auto = pd.read_csv(INPUT / "hc158_item_raw.csv",
                       dtype={"Subject_ID": str}, low_memory=False)
    header = pd.read_csv(INPUT / "hc158_item_retained_transformed.csv", nrows=0).columns
    features = [feature for feature in header if feature not in META]
    if len(features) != 49:
        raise ValueError("Expected original 49 features")
    pair = manual.merge(auto, on=KEY, how="inner", suffixes=("_manual", "_auto"),
                        validate="one_to_one")
    if len(pair) < 5600 or pair.Subject_ID.nunique() != 158:
        raise ValueError("HC158 pairing incomplete")
    pair["Manual_duration_s"] = pair.Absolute_Word_End_manual - pair.Absolute_Word_Start_manual
    pair["Auto_duration_s"] = pair.Absolute_Word_End_auto - pair.Absolute_Word_Start_auto
    pair["Start_error_s"] = pair.Absolute_Word_Start_auto - pair.Absolute_Word_Start_manual
    pair["End_error_s"] = pair.Absolute_Word_End_auto - pair.Absolute_Word_End_manual
    pair["Max_boundary_error_s"] = np.maximum(pair.Start_error_s.abs(), pair.End_error_s.abs())
    pair["Relative_max_boundary_error"] = pair.Max_boundary_error_s / pair.Manual_duration_s
    pair["Interval_IoU"] = (
        np.maximum(0, np.minimum(pair.Absolute_Word_End_manual, pair.Absolute_Word_End_auto)
                   - np.maximum(pair.Absolute_Word_Start_manual, pair.Absolute_Word_Start_auto)) /
        (np.maximum(pair.Absolute_Word_End_manual, pair.Absolute_Word_End_auto)
         - np.minimum(pair.Absolute_Word_Start_manual, pair.Absolute_Word_Start_auto))
    )
    pair["Duration_stratum"] = pd.cut(pair.Manual_duration_s,
                                       bins=[0, .2, .3, .5, np.inf], labels=STRATA,
                                       right=False)
    rules = pd.read_csv(RULES).set_index("openSMILE variable")
    for feature in features:
        for source in ("manual", "auto"):
            pair[f"{feature}_{source}"] = transform(pair[f"{feature}_{source}"],
                                                      feature, rules.loc[feature])
    return pair, features


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    pair, features = load()
    pair[KEY + ["Absolute_Word_Start_manual", "Absolute_Word_End_manual",
                "Absolute_Word_Start_auto", "Absolute_Word_End_auto",
                "Manual_duration_s", "Auto_duration_s", "Start_error_s", "End_error_s",
                "Max_boundary_error_s", "Relative_max_boundary_error", "Interval_IoU",
                "Duration_stratum"]].to_csv(HERE / "paired_boundaries_and_strata.csv",
                                             index=False, encoding="utf-8-sig")
    counts = pair.groupby(["Task", "Duration_stratum"], observed=True).agg(
        N_items=("Subject_ID", "size"), N_people=("Subject_ID", "nunique"),
        median_manual_duration=("Manual_duration_s", "median"),
        median_abs_start_error=("Start_error_s", lambda x: x.abs().median()),
        median_abs_end_error=("End_error_s", lambda x: x.abs().median()),
        median_relative_max_error=("Relative_max_boundary_error", "median"),
        median_interval_IoU=("Interval_IoU", "median")).reset_index()
    counts.to_csv(HERE / "boundary_error_by_duration_stratum.csv", index=False,
                  encoding="utf-8-sig")
    rows = []
    for feature in features:
        x = pair[f"{feature}_manual"].to_numpy(float)
        y = pair[f"{feature}_auto"].to_numpy(float)
        full_manual = x[np.isfinite(x)]
        iqr = float(np.quantile(full_manual, .75) - np.quantile(full_manual, .25))
        scale = iqr if iqr > 1e-12 else float(np.std(full_manual, ddof=1))
        err = np.abs(x - y) / scale
        for task in ("ALL", "Hanzi", "Dots", "STT"):
            task_ok = np.ones(len(pair), dtype=bool) if task == "ALL" else pair.Task.eq(task).to_numpy()
            for stratum in ["ALL", *STRATA, "<300 ms"]:
                mask = task_ok.copy()
                if stratum == "<300 ms":
                    mask &= pair.Manual_duration_s.to_numpy() < .3
                elif stratum != "ALL":
                    mask &= pair.Duration_stratum.eq(stratum).to_numpy()
                valid = mask & np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 20:
                    continue
                rows.append({"Feature": feature, "Task": task, "Duration_stratum": stratum,
                             "N_items": int(valid.sum()),
                             "N_people": int(pair.loc[valid, "Subject_ID"].nunique()),
                             "ICC_A1": icc_absolute(x[valid], y[valid]),
                             "Median_abs_difference_IQR_units": float(np.median(err[valid])),
                             "P90_abs_difference_IQR_units": float(np.quantile(err[valid], .9)),
                             "Median_abs_difference_analysis_scale": float(np.median(np.abs(x[valid] - y[valid]))),
                             "rho_abs_error_vs_manual_duration": correlation(err[valid], pair.loc[valid, "Manual_duration_s"].to_numpy(float)),
                             "rho_abs_error_vs_relative_boundary_error": correlation(err[valid], pair.loc[valid, "Relative_max_boundary_error"].to_numpy(float))})
    detail = pd.DataFrame(rows)
    detail.to_csv(HERE / "feature_agreement_by_duration.csv", index=False,
                  encoding="utf-8-sig")
    pooled = detail.loc[detail.Task.eq("ALL")].pivot(index="Feature", columns="Duration_stratum")
    selection = pd.DataFrame({"Feature": features})
    selection["ICC_lt300"] = selection.Feature.map(pooled.ICC_A1["<300 ms"])
    selection["ICC_ge500"] = selection.Feature.map(pooled.ICC_A1["≥500 ms"])
    selection["ICC_drop_long_minus_short"] = selection.ICC_ge500 - selection.ICC_lt300
    selection["Poor_short_word_agreement"] = selection.ICC_lt300 < .50
    selection["Short_specific_decline"] = selection.ICC_drop_long_minus_short >= .20
    selection["Short_window_instability"] = selection.Poor_short_word_agreement
    selection.to_csv(HERE / "short_window_feature_flags.csv", index=False,
                     encoding="utf-8-sig")
    retained = set(selection.loc[~selection.Short_window_instability, "Feature"])
    retained_decline = set(selection.loc[~selection.Short_specific_decline, "Feature"])
    if not retained or not retained_decline:
        raise ValueError("No retained features")
    comparison = []
    for kind, file in (("single_task_age", "joint_lme_age_within_task_retained_features.csv"),
                       ("task_contrast", "joint_lme_task_contrasts_retained_features.csv"),
                       ("age_by_task", "joint_lme_age_by_task_retained_features.csv")):
        table = pd.read_csv(LME_DIR / file)
        for label, sub in (("Original 49", table),
                           ("Excluding poor short-word agreement", table[table.Feature.isin(retained)]),
                           ("Excluding short-specific decline", table[table.Feature.isin(retained_decline)])):
            sub = sub.copy()
            sub["p_for_FDR"] = sub.p_raw.where(sub.Inference_ok, 1.0).fillna(1.0)
            sub["q_recomputed"] = multipletests(sub.p_for_FDR, method="fdr_bh")[1]
            if kind == "age_by_task":
                sub.to_csv(HERE / ("LME_age_by_task_feature_detail_" +
                         label.lower().replace(" ", "_").replace("-", "_") + ".csv"),
                         index=False, encoding="utf-8-sig")
            for contrast, g in sub.groupby("Contrast"):
                comparison.append({"Effect_type": kind, "Set": label,
                                   "Contrast": contrast, "N_features": sub.Feature.nunique(),
                                   "N_significant_q05": int(g.q_recomputed.lt(.05).sum()),
                                   "N_significant_q001": int(g.q_recomputed.lt(.001).sum())})
    pd.DataFrame(comparison).to_csv(HERE / "LME_age_task_after_short_window_filter.csv",
                                    index=False, encoding="utf-8-sig")
    (HERE / "analysis_config.json").write_text(json.dumps({
        "source": "data/public", "n_paired_items": len(pair),
        "feature_set": "original QC-transformed 49",
        "strata_seconds": [0, .2, .3, .5, "infinity"],
        "agreement": "item-level ICC(A,1) absolute agreement and absolute difference / full-manual IQR",
        "short_window_flag": "ICC(A,1) < .50 at <300 ms; descriptive quality screen, not a diagnostic cutoff",
        "short_specific_decline": "ICC(A,1) at >=500 ms minus ICC(A,1) at <300 ms >= .20; separately flagged",
        "LME_recheck": "unchanged per-feature model p-values; BH-FDR recomputed across retained planned contrasts",
        "voiced_frames": "F0-positive low-level frame counts extracted from source audio within manual boundaries; these are a proxy, not valid-frame counts for all functionals",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(counts.to_string(index=False), flush=True)
    print(selection.Short_window_instability.value_counts().to_string(), flush=True)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
