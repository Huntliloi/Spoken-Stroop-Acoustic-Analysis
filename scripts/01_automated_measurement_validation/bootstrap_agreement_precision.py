"""Participant-cluster bootstrap CIs for median ICC across the fixed 49 features."""

import warnings

import numpy as np
import pandas as pd

from run_short_word_measurement_audit import HERE, STRATA, load


def iccs(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    n = valid.sum(axis=0)
    s = np.where(valid, a + b, np.nan)
    d = np.where(valid, a - b, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        msr = np.nanvar(s, axis=0, ddof=1) / 2
        mse = np.nanvar(d, axis=0, ddof=1) / 2
        msc = n * np.nanmean(d, axis=0) ** 2 / 2
    den = msr + mse + 2 * (msc - mse) / n
    return np.where((n >= 20) & (den > 0), (msr - mse) / den, np.nan)


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        pair, features = load()
    a = pair[[f"{f}_manual" for f in features]].to_numpy(float)
    b = pair[[f"{f}_auto" for f in features]].to_numpy(float)
    people = pair.Subject_ID.to_numpy()
    unique = np.unique(people)
    rng = np.random.default_rng(20261108)
    rows = []
    for stratum in [*STRATA, "ALL"]:
        where = np.arange(len(pair)) if stratum == "ALL" else np.flatnonzero(pair.Duration_stratum.eq(stratum))
        by_person = {person: where[people[where] == person] for person in unique}
        original = float(np.nanmedian(iccs(a[where], b[where])))
        boots = []
        for _ in range(499):
            draws = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([by_person[person] for person in draws])
            boots.append(float(np.nanmedian(iccs(a[idx], b[idx]))))
        lo, hi = np.quantile(boots, [.025, .975])
        rows.append({"Duration_stratum": stratum, "N_items": len(where),
                     "Median_ICC_A1_across_retained_features": original,
                     "Participant_bootstrap_95CI_low": float(lo),
                     "Participant_bootstrap_95CI_high": float(hi)})
    result = pd.DataFrame(rows)
    result.to_csv(HERE / "median_ICC_by_duration_bootstrap_CI.csv",
                  index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
