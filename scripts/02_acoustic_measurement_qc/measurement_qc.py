from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, rankdata, skew, yeojohnson, yeojohnson_normmax

SEED = 20261019
MAX_MISSING_STRATUM = 0.05
MAX_ZERO_TASK = 0.20
MAX_ABS_RAW_SKEW_EXPLOSION = 10.0
MAX_RAW_EXCESS_KURT_EXPLOSION = 100.0
SHORT_SEGMENT_SEC = 0.20
SHORT_ZERO_MIN = 0.20
SHORT_ZERO_EXCESS = 0.10
TRANSFORM_ABS_SKEW = 2.0
REDUNDANCY_RHO = 0.90

ID_COLS = [
    "Subject_ID", "Task", "Item_Index", "Absolute_Word_Start",
    "Absolute_Word_End", "Auto_Word"
]
TASK_LABEL = {"Hanzi": "W", "Dots": "C", "STT": "CW"}
TASK_ORDER = ["Hanzi", "Dots", "STT"]


def fnum(x: str) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def safe_skew(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(skew(x, bias=False)) if len(x) >= 3 and np.unique(x).size > 1 else np.nan


def safe_kurt(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(kurtosis(x, fisher=True, bias=False)) if len(x) >= 4 and np.unique(x).size > 1 else np.nan


def rank_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 30:
        return np.nan, n
    aa, bb = a[mask], b[mask]
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        return np.nan, n
    ra = rankdata(aa, method="average")
    rb = rankdata(bb, method="average")
    return float(np.corrcoef(ra, rb)[0, 1]), n


def sb_correct(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    if r <= -0.999999:
        return -1.0
    return float(2.0 * r / (1.0 + r))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fields:
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
        return
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_files(hc158: Path, hc74: Path):
    all_rows = []
    header = None
    for cohort, path in [("HC158", hc158), ("HC74", hc74)]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            if header is None:
                header = list(r.fieldnames or [])
            elif header != list(r.fieldnames or []):
                raise RuntimeError("Input CSV headers are not identical")
            for row in r:
                row["Cohort"] = cohort
                all_rows.append(row)
    if header is None:
        raise RuntimeError("No input data")
    return header, all_rows


def canonical_words(rows, task: str) -> dict[int, str]:
    out = {}
    for item in range(1, 13):
        cnt = Counter(
            r["Auto_Word"] for r in rows
            if r["Task"] == task and int(float(r["Item_Index"])) == item and r.get("Auto_Word")
        )
        out[item] = cnt.most_common(1)[0][0] if cnt else str(item)
    return out


def balanced_split_pool(rows, task: str):
    words = canonical_words(rows, task)
    items = list(range(1, 13))
    pool = []
    for a in itertools.combinations(items, 6):
        if 1 not in a:  # complementary A/B split counted once
            continue
        b = tuple(i for i in items if i not in a)
        ca, cb = Counter(words[i] for i in a), Counter(words[i] for i in b)
        if all(abs(ca[k] - cb[k]) <= 1 for k in set(ca) | set(cb)):
            pool.append((tuple(a), b))
    if not pool:
        raise RuntimeError(f"No balanced splits for {task}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc158", required=True)
    ap.add_argument("--hc74", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    header, rows = load_files(Path(args.hc158), Path(args.hc74))
    features = ["Duration", "Gap_Before"] + [
        c for c in header if c not in ID_COLS + ["Duration", "Gap_Before"]
    ]
    if len(features) != 90:
        raise RuntimeError(f"Expected 90 features, found {len(features)}")

    n = len(rows)
    tasks = [t for t in TASK_ORDER if any(r["Task"] == t for r in rows)]
    cohorts = ["HC158", "HC74"]
    duration = np.array([fnum(r["Duration"]) for r in rows], dtype=float)
    item_index = np.array([int(float(r["Item_Index"])) for r in rows], dtype=int)
    subject_key = [f"{r['Cohort']}::{r['Subject_ID']}" for r in rows]
    subjects = sorted(set(subject_key))
    subject_index = {s: i for i, s in enumerate(subjects)}

    raw = {f: np.array([fnum(r.get(f, "")) for r in rows], dtype=float) for f in features}
    # Exact -201 is treated as extraction-floor missing only for formant relative-amplitude means.
    for f in [x for x in features if "amplitudeLogRelF0_sma3nz_amean" in x]:
        x = raw[f].copy()
        x[np.isclose(x, -201.0, atol=1e-8, equal_nan=False)] = np.nan
        raw[f] = x

    audit_rows = []
    hard_pass = []
    transformed = {}
    transform_specs = []

    for feature in features:
        x = raw[feature]
        valid = x[np.isfinite(x)]
        strata_missing = []
        task_zero, task_iqr = [], []
        for cohort in cohorts:
            for task in tasks:
                mask = np.array([r["Cohort"] == cohort and r["Task"] == task for r in rows])
                strata_missing.append(float(np.mean(~np.isfinite(x[mask]))))
        for task in tasks:
            mask = np.array([r["Task"] == task for r in rows])
            z = x[mask & np.isfinite(x)]
            task_zero.append(float(np.mean(z == 0)) if len(z) else np.nan)
            task_iqr.append(float(np.quantile(z, .75) - np.quantile(z, .25)) if len(z) else np.nan)
        short = (duration < SHORT_SEGMENT_SEC) & np.isfinite(duration)
        long = (duration >= SHORT_SEGMENT_SEC) & np.isfinite(duration)
        zs = x[short & np.isfinite(x)]
        zl = x[long & np.isfinite(x)]
        zero_short = float(np.mean(zs == 0)) if len(zs) else np.nan
        zero_long = float(np.mean(zl == 0)) if len(zl) else np.nan
        raw_sk = safe_skew(valid)
        raw_ku = safe_kurt(valid)
        max_missing = float(np.nanmax(strata_missing))
        max_zero = float(np.nanmax(task_zero))
        min_iqr = float(np.nanmin(task_iqr))

        reasons = []
        if max_missing > MAX_MISSING_STRATUM:
            reasons.append("missingness >5% in a cohort×task stratum")
        if max_zero >= MAX_ZERO_TASK:
            reasons.append("structural zero >=20% in at least one task")
        if np.isfinite(min_iqr) and min_iqr <= 1e-12:
            reasons.append("zero IQR in at least one task")
        if np.isfinite(raw_sk) and abs(raw_sk) > MAX_ABS_RAW_SKEW_EXPLOSION:
            reasons.append("explosive raw skewness |skew|>10")
        if np.isfinite(raw_ku) and raw_ku > MAX_RAW_EXCESS_KURT_EXPLOSION:
            reasons.append("explosive raw excess kurtosis >100")
        if (
            np.isfinite(zero_short) and np.isfinite(zero_long)
            and zero_short >= SHORT_ZERO_MIN
            and zero_short - zero_long >= SHORT_ZERO_EXCESS
        ):
            reasons.append("short-segment-dependent structural zeros")

        hard_ok = len(reasons) == 0
        if hard_ok:
            hard_pass.append(feature)

        # Distribution treatment is defined only for measurement-QC pass features.
        transform = "—"
        yj_lambda = np.nan
        max_abs_raw_task_skew = np.nan
        max_abs_post_task_skew = np.nan
        max_post_task_kurt = np.nan
        dist_flag = "Not evaluated"
        if hard_ok:
            task_sks = []
            for task in tasks:
                mask = np.array([r["Task"] == task for r in rows])
                task_sks.append(safe_skew(x[mask]))
            max_abs_raw_task_skew = float(np.nanmax(np.abs(task_sks)))
            if max_abs_raw_task_skew <= TRANSFORM_ABS_SKEW:
                transform = "Raw"
                y = x.copy()
            elif len(valid) and np.nanmin(valid) >= 0:
                transform = "log1p"
                y = np.full(n, np.nan)
                m = np.isfinite(x)
                y[m] = np.log1p(x[m])
            else:
                transform = "Yeo-Johnson"
                yj_lambda = float(yeojohnson_normmax(valid))
                y = np.full(n, np.nan)
                m = np.isfinite(x)
                y[m] = yeojohnson(x[m], lmbda=yj_lambda)
            transformed[feature] = y
            post_sk, post_ku = [], []
            for task in tasks:
                mask = np.array([r["Task"] == task for r in rows])
                post_sk.append(abs(safe_skew(y[mask])))
                post_ku.append(safe_kurt(y[mask]))
            max_abs_post_task_skew = float(np.nanmax(post_sk))
            max_post_task_kurt = float(np.nanmax(post_ku))
            dist_flag = "Flag" if (max_abs_post_task_skew > 2 or max_post_task_kurt > 10) else "Acceptable"
            transform_specs.append({
                "Feature": feature,
                "transform": transform,
                "yeo_johnson_lambda": yj_lambda,
                "max_abs_raw_task_skew": max_abs_raw_task_skew,
                "max_abs_post_task_skew": max_abs_post_task_skew,
                "max_post_task_excess_kurtosis": max_post_task_kurt,
                "distribution_flag_descriptive_only": dist_flag,
            })

        audit_rows.append({
            "Feature": feature,
            "measurement_qc_pass": hard_ok,
            "final_measurement_qualified": hard_ok,
            "hard_qc_reasons": "; ".join(reasons),
            "missing_pct_overall": float(np.mean(~np.isfinite(x))),
            "max_missing_pct_cohort_task": max_missing,
            "zero_pct_overall": float(np.mean(valid == 0)) if len(valid) else np.nan,
            "max_zero_pct_task": max_zero,
            "zero_pct_duration_lt_0_20": zero_short,
            "zero_pct_duration_ge_0_20": zero_long,
            "raw_skewness": raw_sk,
            "raw_excess_kurtosis": raw_ku,
            "min_task_iqr": min_iqr,
            "raw_min": float(np.nanmin(x)) if np.isfinite(x).any() else np.nan,
            "raw_max": float(np.nanmax(x)) if np.isfinite(x).any() else np.nan,
            "transform": transform,
            "yeo_johnson_lambda": yj_lambda,
            "max_abs_post_task_skew": max_abs_post_task_skew,
            "max_post_task_excess_kurtosis": max_post_task_kurt,
            "distribution_flag_descriptive_only": dist_flag,
        })

    # Split-half precision: exhaustive 243 balanced splits/task, descriptive only.
    split_pools = {task: balanced_split_pool(rows, task) for task in tasks}
    split_rows = []
    wide_rel = defaultdict(dict)
    for feature in hard_pass:
        vals = transformed[feature]
        for task in tasks:
            mat = np.full((len(subjects), 12), np.nan)
            sums = np.zeros((len(subjects), 12), dtype=float)
            counts = np.zeros((len(subjects), 12), dtype=int)
            for i, r in enumerate(rows):
                if r["Task"] != task or not np.isfinite(vals[i]):
                    continue
                si = subject_index[subject_key[i]]
                it = item_index[i] - 1
                sums[si, it] += vals[i]
                counts[si, it] += 1
            ok = counts > 0
            mat[ok] = sums[ok] / counts[ok]
            sbs, ns = [], []
            for a, b in split_pools[task]:
                ia = [i - 1 for i in a]
                ib = [i - 1 for i in b]
                aa, bb = mat[:, ia], mat[:, ib]
                ca, cb = np.sum(np.isfinite(aa), axis=1), np.sum(np.isfinite(bb), axis=1)
                ma = np.divide(np.nansum(aa, axis=1), ca, out=np.full(len(subjects), np.nan), where=ca > 0)
                mb = np.divide(np.nansum(bb, axis=1), cb, out=np.full(len(subjects), np.nan), where=cb > 0)
                # allow one missing item per half
                ma[ca < 5] = np.nan
                mb[cb < 5] = np.nan
                rr, nn = rank_corr(ma, mb)
                sbs.append(sb_correct(rr))
                ns.append(nn)
            arr = np.asarray(sbs, dtype=float)
            result = {
                "Feature": feature,
                "Task": task,
                "Condition": TASK_LABEL.get(task, task),
                "n_unique_balanced_splits": len(split_pools[task]),
                "median_n_subjects": float(np.nanmedian(ns)),
                "sb_median": float(np.nanmedian(arr)),
                "sb_p025": float(np.nanquantile(arr, .025)),
                "sb_p975": float(np.nanquantile(arr, .975)),
            }
            split_rows.append(result)
            wide_rel[feature][task] = result

    # Subject-condition means and redundancy flags, descriptive only.
    means = {task: {} for task in tasks}
    for task in tasks:
        for feature in hard_pass:
            vals = transformed[feature]
            sums = np.zeros(len(subjects), dtype=float)
            counts = np.zeros(len(subjects), dtype=int)
            for i, r in enumerate(rows):
                if r["Task"] != task or not np.isfinite(vals[i]):
                    continue
                si = subject_index[subject_key[i]]
                sums[si] += vals[i]
                counts[si] += 1
            v = np.full(len(subjects), np.nan)
            ok = counts > 0
            v[ok] = sums[ok] / counts[ok]
            means[task][feature] = v

    red_rows = []
    red_partners = defaultdict(list)
    for a, b in itertools.combinations(hard_pass, 2):
        rs = []
        for task in tasks:
            rr, _ = rank_corr(means[task][a], means[task][b])
            rs.append(rr)
        same_sign = all(np.isfinite(rs)) and all(np.sign(r) == np.sign(rs[0]) and np.sign(r) != 0 for r in rs)
        min_abs = min(abs(r) for r in rs) if all(np.isfinite(rs)) else np.nan
        if same_sign and min_abs >= REDUNDANCY_RHO:
            red_rows.append({
                "Feature_A": a, "Feature_B": b,
                "rho_W": rs[tasks.index("Hanzi")] if "Hanzi" in tasks else np.nan,
                "rho_C": rs[tasks.index("Dots")] if "Dots" in tasks else np.nan,
                "rho_CW": rs[tasks.index("STT")] if "STT" in tasks else np.nan,
                "min_abs_rho": min_abs,
                "action": "Flag only; do not remove from measurement-qualified set",
            })
            red_partners[a].append(b)
            red_partners[b].append(a)

    # Merge precision/redundancy flags into audit.
    audit_lookup = {r["Feature"]: r for r in audit_rows}
    summary_rows = []
    for feature in hard_pass:
        meds = {TASK_LABEL.get(t, t): wide_rel[feature][t]["sb_median"] for t in tasks}
        mn = min(meds.values())
        med3 = float(np.median(list(meds.values())))
        if mn >= .80:
            grade = "Good (all tasks >=.80)"
        elif mn >= .70:
            grade = "Acceptable (all tasks >=.70)"
        elif mn >= .60:
            grade = "Marginal (one/more tasks .60-.69)"
        else:
            grade = "Low (one/more tasks <.60)"
        audit_lookup[feature].update({
            "split_half_W_median": meds.get("W", np.nan),
            "split_half_C_median": meds.get("C", np.nan),
            "split_half_CW_median": meds.get("CW", np.nan),
            "split_half_min_condition_median": mn,
            "split_half_median_across_conditions": med3,
            "split_half_precision_grade_descriptive_only": grade,
            "redundancy_flag": "Yes" if red_partners[feature] else "No",
            "redundant_partner_count": len(red_partners[feature]),
            "redundant_partners": "; ".join(red_partners[feature]),
        })
        summary_rows.append({
            "Feature": feature,
            "W": meds.get("W", np.nan), "C": meds.get("C", np.nan), "CW": meds.get("CW", np.nan),
            "min_condition": mn, "median_across_conditions": med3,
            "precision_grade_descriptive_only": grade,
            "redundancy_flag": "Yes" if red_partners[feature] else "No",
            "redundant_partner_count": len(red_partners[feature]),
        })

    # Output tables.
    audit_fields = list(audit_rows[0].keys()) + [
        "split_half_W_median", "split_half_C_median", "split_half_CW_median",
        "split_half_min_condition_median", "split_half_median_across_conditions",
        "split_half_precision_grade_descriptive_only", "redundancy_flag",
        "redundant_partner_count", "redundant_partners"
    ]
    write_csv(outdir / "01_all_90_measurement_qc.csv", audit_rows, audit_fields)
    write_csv(outdir / "02_retained_measurement_qualified.csv", [audit_lookup[f] for f in hard_pass], audit_fields)
    write_csv(outdir / "03_excluded_measurement_qc.csv", [r for r in audit_rows if not r["measurement_qc_pass"]], audit_fields)
    write_csv(outdir / "04_transform_spec.csv", transform_specs)
    write_csv(outdir / "05_split_half_detail.csv", split_rows)
    write_csv(outdir / "06_split_half_summary.csv", summary_rows)
    write_csv(outdir / "07_redundancy_pairs.csv", red_rows, ["Feature_A","Feature_B","rho_W","rho_C","rho_CW","min_abs_rho","action"])

    reason_counts = Counter()
    for r in audit_rows:
        if r["hard_qc_reasons"]:
            for z in r["hard_qc_reasons"].split("; "):
                reason_counts[z] += 1
    tr_counts = Counter(s["transform"] for s in transform_specs)
    min_rel = [r["min_condition"] for r in summary_rows]
    summary = {
        "subjects": len(subjects),
        "rows": n,
        "candidate_features": len(features),
        "measurement_qualified_features": len(hard_pass),
        "excluded_features": len(features) - len(hard_pass),
        "transform_counts": dict(tr_counts),
        "split_half_all_tasks_ge_0_60": int(sum(x >= .60 for x in min_rel)),
        "split_half_all_tasks_ge_0_70": int(sum(x >= .70 for x in min_rel)),
        "split_half_all_tasks_ge_0_80": int(sum(x >= .80 for x in min_rel)),
        "redundancy_pairs_flagged": len(red_rows),
        "hard_qc_reason_counts": dict(reason_counts),
    }
    (outdir / "08_analysis_manifest.json").write_text(json.dumps({
        "method": "Improved measurement-QC: hard measurement feasibility only determines retention; distribution, split-half and redundancy are descriptive/analytic annotations, not hard selection criteria.",
        "gap_before": "Analyzed as provided across all 12 items; no special Item 1 redefinition per current analysis decision.",
        "thresholds": {
            "max_missing_cohort_task": MAX_MISSING_STRATUM,
            "max_zero_task": MAX_ZERO_TASK,
            "explosive_abs_skew": MAX_ABS_RAW_SKEW_EXPLOSION,
            "explosive_excess_kurtosis": MAX_RAW_EXCESS_KURT_EXPLOSION,
            "short_segment_sec": SHORT_SEGMENT_SEC,
            "short_zero_min": SHORT_ZERO_MIN,
            "short_zero_excess_vs_long": SHORT_ZERO_EXCESS,
            "transform_abs_task_skew": TRANSFORM_ABS_SKEW,
            "redundancy_abs_rho_all_tasks": REDUNDANCY_RHO,
        },
        "transform_logic": "If max absolute task-level skew <=2: Raw; if >2 and nonnegative: log1p; if >2 and signed: pooled Yeo-Johnson. No winsorization. Post-transform skew/kurtosis is flagged but not used to exclude.",
        "split_half": "Exhaustive 243 unique color-balanced 6-vs-6 splits per task; Spearman correlation + Spearman-Brown correction; allow one missing item per half; descriptive only.",
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
