"""Relate auto/manual feature discrepancies to word duration, voicing, and boundary error."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from run_short_word_measurement_audit import PACKAGE, ROOT, HERE, KEY, load


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.ptp(x[ok]) < 1e-12 or np.ptp(y[ok]) < 1e-12:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        pair, features = load()
    frames = pd.read_csv(PACKAGE / "data" / "public" / "hc158_manual_f0_frame_counts.csv",
                         dtype={"Subject_ID": str})
    pair = pair.merge(frames, on=KEY, how="inner", validate="one_to_one")
    if len(pair) < 5600:
        raise ValueError("Voiced frame coverage incomplete")
    frame_summary = pair.groupby("Duration_stratum", observed=True).agg(
        N_items=("Subject_ID", "size"),
        median_LLD_frames=("N_LLD_frames", "median"),
        median_F0_positive_frames=("N_F0_positive_frames", "median"),
        Q1_F0_positive_frames=("N_F0_positive_frames", lambda s: s.quantile(.25)),
        fraction_with_lt_10_F0_positive_frames=("N_F0_positive_frames", lambda s: (s < 10).mean()),
        median_F0_positive_fraction=("F0_positive_frame_fraction", "median")
    ).reset_index()
    frame_summary.to_csv(HERE / "voiced_frames_by_duration_stratum.csv",
                         index=False, encoding="utf-8-sig")

    predictors = pd.DataFrame({
        "log_word_duration": np.log(pair.Manual_duration_s.to_numpy(float)),
        "log1p_relative_boundary_error": np.log1p(pair.Relative_max_boundary_error.to_numpy(float)),
        "Task_C": pair.Task.eq("Dots").to_numpy(float),
        "Task_CW": pair.Task.eq("STT").to_numpy(float),
    })
    scaled = predictors.copy()
    for key in ("log_word_duration", "log1p_relative_boundary_error"):
        scaled[key] = (scaled[key] - scaled[key].mean()) / scaled[key].std(ddof=0)
    rows = []
    for feature in features:
        a = pair[f"{feature}_manual"].to_numpy(float)
        b = pair[f"{feature}_auto"].to_numpy(float)
        full = a[np.isfinite(a)]
        iqr = np.quantile(full, .75) - np.quantile(full, .25)
        scale = iqr if iqr > 1e-12 else np.std(full, ddof=1)
        err = np.abs(a - b) / scale
        ok = np.isfinite(err) & np.isfinite(scaled.to_numpy()).all(axis=1)
        if ok.sum() < 100:
            continue
        X = sm.add_constant(scaled.loc[ok], has_constant="add")
        y = np.log1p(err[ok])
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={
            "groups": pair.loc[ok, "Subject_ID"].to_numpy()})
        row = {"Feature": feature, "N_items": int(ok.sum()),
               "Median_abs_error_IQR_units": float(np.median(err[ok])),
               "rho_error_vs_duration": rho(err[ok], pair.loc[ok, "Manual_duration_s"].to_numpy(float)),
               "rho_error_vs_F0_positive_frames": rho(err[ok], pair.loc[ok, "N_F0_positive_frames"].to_numpy(float)),
               "rho_error_vs_relative_boundary_error": rho(err[ok], pair.loc[ok, "Relative_max_boundary_error"].to_numpy(float))}
        for key in ("log_word_duration", "log1p_relative_boundary_error"):
            row[f"beta_{key}"] = float(fit.params[key])
            row[f"p_{key}"] = float(fit.pvalues[key])
        rows.append(row)
    result = pd.DataFrame(rows)
    for key in ("log_word_duration", "log1p_relative_boundary_error"):
        result[f"q_{key}"] = multipletests(result[f"p_{key}"], method="fdr_bh")[1]
    result.to_csv(HERE / "feature_error_sensitivity_to_duration_frames_boundary.csv",
                  index=False, encoding="utf-8-sig")
    print(frame_summary.to_string(index=False))
    print("Number of 49 features with FDR q<.05:")
    print("Correlation of log word duration with log1p F0-positive frames:",
          round(float(np.corrcoef(np.log(pair.Manual_duration_s),
                                  np.log1p(pair.N_F0_positive_frames))[0, 1]), 3))
    for key in ("log_word_duration", "log1p_relative_boundary_error"):
        print(key, int(result[f"q_{key}"].lt(.05).sum()),
              "median beta", round(result[f"beta_{key}"].median(), 4))


if __name__ == "__main__":
    main()
