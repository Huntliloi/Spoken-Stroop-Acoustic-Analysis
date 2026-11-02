"""Four FACT-Cog domains: all-51 acoustic associations and nested-CV increment.

Predictors are complete and fixed before cross-validation. For each outer fold,
standardization, ridge-alpha tuning, and the final fit use training data only.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results" / "recomputed" / "06_cancer_fact_cog"
HERE.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / "data" / "public" / "cancer_model_input.csv"
DOMAINS = ("PCI", "Others", "Abilities", "QOL_impact")
CONDITIONS = ("W", "C", "CW", "C-W", "CW-C", "CW-W")
DEM = ("Age", "Female", "Education_years_approx")
TRAD = tuple(f"{kind}_{task}" for kind in ("Accuracy", "Time") for task in ("W", "C", "CW"))
ACOUSTIC = tuple(f"Acoustic_{condition}" for condition in CONDITIONS)
ALPHAS = np.array((0.01, 0.1, 1, 10, 100, 1000, 10000, 100000), float)
SEED = 20261030
N_BOOT = 5000
N_REPEATS = 200


def read_input():
    data = pd.read_csv(INPUT, dtype={"Subject_ID": str})
    expected = ["Subject_ID", *DOMAINS, *DEM, *TRAD, *ACOUSTIC]
    if data.columns.tolist() != expected or len(data) != 92 or data.Subject_ID.nunique() != 92:
        raise ValueError("Expected 92 people, four domains and fixed predictor columns")
    if data[expected[1:]].isna().any().any() or not np.isfinite(data[expected[1:]].to_numpy(float)).all():
        raise ValueError("Missing or nonfinite inputs; no imputation is planned")
    return data


def adjusted_associations(data):
    rows = []
    for adjustment, covariates in (("D", DEM), ("D+T", (*DEM, *TRAD))):
        for domain in DOMAINS:
            y = data[domain].to_numpy(float)
            y = (y - y.mean()) / y.std(ddof=0)
            for condition in CONDITIONS:
                score = f"Acoustic_{condition}"
                x = data[[score, *covariates]].astype(float)
                scales = x.std(ddof=0)
                if scales.eq(0).any():
                    raise ValueError(f"Constant predictor for {domain}/{condition}")
                x = (x - x.mean()) / scales
                fit = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit(cov_type="HC3")
                ci = fit.conf_int().loc[score]
                rows.append({"Adjustment": adjustment, "Domain": domain,
                             "Condition": condition, "N": len(data),
                             "Standardized_beta": float(fit.params[score]),
                             "HC3_CI_low": float(ci.iloc[0]), "HC3_CI_high": float(ci.iloc[1]),
                             "HC3_p": float(fit.pvalues[score])})
    result = pd.DataFrame(rows)
    result["HC3_q6"] = np.nan
    for adjustment in ("D", "D+T"):
        for domain in DOMAINS:
            mask = result.Adjustment.eq(adjustment) & result.Domain.eq(domain)
            result.loc[mask, "HC3_q6"] = multipletests(result.loc[mask, "HC3_p"], method="fdr_bh")[1]
    result.to_csv(HERE / "adjusted_single_score_associations.csv", index=False, encoding="utf-8-sig")
    return result


def ridge_predictions(x_train, y_train, x_test, alphas):
    """Return test predictions for all alphas via an SVD ridge solution."""
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0, ddof=0)
    scales[scales == 0] = 1
    a = (x_train - means) / scales
    b = (x_test - means) / scales
    y_mean = y_train.mean()
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    proj = u.T @ (y_train - y_mean)
    weights = vt.T @ ((s[:, None] / (s[:, None]**2 + alphas[None, :])) * proj[:, None])
    return y_mean + b @ weights


def predict_fold(x, y, train, test, inner_folds, seed):
    inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    fold_mse = np.zeros(len(ALPHAS))
    for in_train, in_val in inner.split(train):
        actual_train = train[in_train]
        actual_val = train[in_val]
        predictions = ridge_predictions(x[actual_train], y[actual_train], x[actual_val], ALPHAS)
        fold_mse += ((predictions - y[actual_val, None])**2).mean(axis=0)
    best = int(np.argmin(fold_mse / inner_folds))
    pred = ridge_predictions(x[train], y[train], x[test], ALPHAS[[best]])[:, 0]
    return pred, float(ALPHAS[best])


def evaluate(data, domain, splits, inner_folds, seed_offset):
    y = data[domain].to_numpy(float)
    xb = data[[*DEM, *TRAD]].to_numpy(float)
    xf = data[[*DEM, *TRAD, *ACOUSTIC]].to_numpy(float)
    pred_b = np.full(len(y), np.nan)
    pred_f = np.full(len(y), np.nan)
    tuning = []
    for fold, (train, test) in enumerate(splits, 1):
        inner_seed = SEED + seed_offset * 10 + fold
        pred_b[test], alpha_b = predict_fold(xb, y, train, test, inner_folds, inner_seed)
        pred_f[test], alpha_f = predict_fold(xf, y, train, test, inner_folds, inner_seed)
        tuning.append({"Domain": domain, "Fold": fold,
                       "Alpha_DT": alpha_b, "Alpha_DTA": alpha_f})
    if not np.isfinite(pred_b).all() or not np.isfinite(pred_f).all():
        raise ValueError("Incomplete out-of-fold predictions")
    return y, pred_b, pred_f, pd.DataFrame(tuning)


def score(y, pred_b, pred_f):
    r2_b = float(r2_score(y, pred_b))
    r2_f = float(r2_score(y, pred_f))
    rmse_b = float(np.sqrt(mean_squared_error(y, pred_b)))
    rmse_f = float(np.sqrt(mean_squared_error(y, pred_f)))
    mae_b = float(mean_absolute_error(y, pred_b))
    mae_f = float(mean_absolute_error(y, pred_f))
    return {"R2_DT": r2_b, "R2_DTA": r2_f, "Delta_R2": r2_f - r2_b,
            "RMSE_DT": rmse_b, "RMSE_DTA": rmse_f, "Delta_RMSE": rmse_f - rmse_b,
            "MAE_DT": mae_b, "MAE_DTA": mae_f, "Delta_MAE": mae_f - mae_b}


def paired_bootstrap(domain, y, pred_b, pred_f):
    rng = np.random.default_rng(SEED + DOMAINS.index(domain))
    rows = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if np.var(y[idx]) > 0:
            rows.append(score(y[idx], pred_b[idx], pred_f[idx]))
    samples = pd.DataFrame(rows)
    return [{"Domain": domain, "Metric": metric, "N_bootstrap": len(samples),
             "CI_2.5": float(samples[metric].quantile(.025)),
             "CI_97.5": float(samples[metric].quantile(.975)),
             "Fraction_favorable": float((samples[metric] > 0).mean() if metric == "Delta_R2"
                                         else (samples[metric] < 0).mean())}
            for metric in ("Delta_R2", "Delta_RMSE", "Delta_MAE")]


def main():
    data = read_input()
    adjusted = adjusted_associations(data)
    print("Adjusted D+T q6<.05:")
    print(adjusted.loc[adjusted.Adjustment.eq("D+T") & adjusted.HC3_q6.lt(.05),
                       ["Domain", "Condition", "Standardized_beta", "HC3_p", "HC3_q6"]].to_string(index=False), flush=True)

    loo_rows, prediction_rows, tuning_rows, bootstrap_rows = [], [], [], []
    loo_splits = list(LeaveOneOut().split(np.arange(len(data))))
    for domain in DOMAINS:
        y, b, f, tuning = evaluate(data, domain, loo_splits, 5, 0)
        loo_rows.append({"Domain": domain, "N": len(data), **score(y, b, f)})
        tuning_rows.append(tuning)
        bootstrap_rows.extend(paired_bootstrap(domain, y, b, f))
        prediction_rows.append(pd.DataFrame({"Subject_ID": data.Subject_ID, "Domain": domain,
                                             "Observed": y, "Pred_DT": b, "Pred_DTA": f}))
        print(f"LOOCV {domain}: {loo_rows[-1]}", flush=True)
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(HERE / "LOOCV_four_domains_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(HERE / "LOOCV_subject_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(tuning_rows, ignore_index=True).to_csv(HERE / "LOOCV_inner_alpha.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bootstrap_rows).to_csv(HERE / "LOOCV_paired_bootstrap_CI.csv", index=False, encoding="utf-8-sig")

    repeated_rows = []
    for repeat in range(N_REPEATS):
        outer = list(KFold(n_splits=5, shuffle=True, random_state=SEED + repeat).split(np.arange(len(data))))
        for domain in DOMAINS:
            y, b, f, _ = evaluate(data, domain, outer, 3, repeat + 1)
            repeated_rows.append({"Repeat": repeat + 1, "Domain": domain, **score(y, b, f)})
        if (repeat + 1) % 25 == 0:
            print(f"Repeated nested CV {repeat+1}/{N_REPEATS}", flush=True)
    repeated = pd.DataFrame(repeated_rows)
    repeated.to_csv(HERE / "200_repeated_nested_CV_detail.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for domain in DOMAINS:
        rows = repeated.loc[repeated.Domain.eq(domain)]
        summary_rows.append({"Domain": domain, "Repeats": N_REPEATS,
                             **{f"{key}_mean": float(rows[key].mean()) for key in loo.columns if key not in ("Domain", "N")},
                             **{f"{key}_SD": float(rows[key].std(ddof=1)) for key in loo.columns if key not in ("Domain", "N")},
                             "Positive_Delta_R2": int(rows.Delta_R2.gt(0).sum()),
                             "Lower_RMSE": int(rows.Delta_RMSE.lt(0).sum())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(HERE / "200_repeated_nested_CV_summary.csv", index=False, encoding="utf-8-sig")
    (HERE / "analysis_config.json").write_text(json.dumps({
        "N": len(data), "outcomes": DOMAINS,
        "demographics": DEM, "traditional": TRAD, "acoustic": ACOUSTIC,
        "acoustic_scoring": "Six HC232-frozen MeanAbsZ scores, each using all 51 measurement-qualified acoustic features",
        "adjusted": "Separate standardized HC3 OLS for each acoustic score; D and D+T covariates; BH-FDR over six conditions within each domain and adjustment; no q24 global correction",
        "prediction": "Training-fold standardization + ridge; alpha selected in inner K-fold CV by mean validation MSE",
        "alpha_grid": ALPHAS.tolist(), "LOOCV_inner_folds": 5,
        "repeated_nested_CV": "200 paired repetitions, five outer folds and three inner folds",
        "bootstrap": "5000 participant-level resamples of fixed LOOCV paired OOF predictions, without model refitting",
        "imputation": "None; complete fixed analysis input",
        "source": str(INPUT),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Repeated nested CV summary:")
    print(summary[["Domain", "R2_DT_mean", "R2_DTA_mean", "Delta_R2_mean", "RMSE_DT_mean", "RMSE_DTA_mean", "Positive_Delta_R2"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
