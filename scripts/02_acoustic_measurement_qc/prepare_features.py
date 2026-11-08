from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT / "scripts"))
from release_paths import output_root
DATA=ROOT / "data/public"
ROOT=output_root(ROOT)
QC=ROOT / "02_acoustic_measurement_qc"
META=["Subject_ID","Task","Item_Index","Absolute_Word_Start","Absolute_Word_End"]
def main():
    qualified = pd.read_csv(QC / "02_retained_measurement_qualified.csv").Feature.tolist()
    dictionary = pd.read_csv(DATA / "feature_dictionary.csv").set_index("openSMILE variable")
    assert set(qualified) <= set(dictionary.index)
    pairs = pd.read_csv(QC / "07_redundancy_pairs.csv")
    precision = pd.read_csv(QC / "05_split_half_detail.csv")
    print("Precision columns", precision.columns.tolist(), flush=True)
    rel_col = next(c for c in precision if "median" in c.lower() and ("sb" in c.lower() or "brown" in c.lower()))
    rel = precision.groupby("Feature")[rel_col].min().to_dict()
    adjacency = {f: set() for f in qualified}
    for row in pairs.itertuples():
        a, b = row.Feature_A, row.Feature_B
        if dictionary.loc[a, "Category"] == dictionary.loc[b, "Category"]:
            adjacency[a].add(b); adjacency[b].add(a)
    def directness(f):
        name, s = dictionary.loc[f, "Name used in article"].lower(), f.lower()
        rank = 2
        if f in {"Duration", "Gap_Before"} or s.endswith("_amean") or name.startswith("mean "): rank = 0
        elif "percentile50" in s or "stddevnorm" in s: rank = 1
        if "voiced" in name or ("mfcc" in s and "v_sma" in s) or "spectralfluxv_" in s: rank += 1
        return rank
    seen, removed, decisions = set(), set(), []
    for f in qualified:
        if f in seen: continue
        stack, component = [f], []
        seen.add(f)
        while stack:
            u = stack.pop(); component.append(u)
            for v in sorted(adjacency[u]):
                if v not in seen: seen.add(v); stack.append(v)
        representative = sorted(component, key=lambda x: (directness(x), -rel.get(x, -9), x))[0]
        for u in component:
            keep = u == representative
            if not keep: removed.add(u)
            decisions.append({"Feature": u, "Category": dictionary.loc[u, "Category"],
                              "Representative": representative, "Keep": keep,
                              "Minimum_split_half": rel.get(u, np.nan)})
    features = [f for f in qualified if f not in removed]
    n = len(features)
    structure = ROOT / "03_acoustic_measurement_structure"
    structure.mkdir(exist_ok=True)
    pd.DataFrame(decisions).to_csv(structure / "redundancy_selection.csv", index=False)
    assert features == json.loads((DATA / "selected_features.json").read_text(encoding="utf-8"))
    rules = pd.read_csv(QC / "04_transform_spec.csv").set_index("Feature")
    rules_out = dictionary.loc[qualified].reset_index()[["openSMILE variable", "Category", "Name used in article"]]
    rules_out["Transformation"] = rules.loc[qualified, "transform"].to_numpy()
    rules_out["Yeo-Johnson lambda"] = rules.loc[qualified, "yeo_johnson_lambda"].to_numpy()
    rules_out.to_csv(QC / "frozen_transformation_rules_60.csv", index=False)
    for cohort in ("HC158", "HC74"):
        raw = pd.read_csv(DATA / f"{cohort.lower()}_item_raw.csv", dtype={"Subject_ID": str})
        pass
        out = raw[META + qualified].copy()
        for f in qualified:
            x = out[f].to_numpy(float, copy=True); x[~np.isfinite(x)] = np.nan
            if f in {f"F{i}amplitudeLogRelF0_sma3nz_amean" for i in (1,2,3)}:
                x[np.isclose(x, -201, rtol=1e-5, atol=1e-8)] = np.nan
            ok = np.isfinite(x); method = rules.loc[f, "transform"]
            if method == "log1p": x[ok] = np.log1p(x[ok])
            elif method == "Yeo-Johnson": x[ok] = yeojohnson(x[ok], lmbda=rules.loc[f, "yeo_johnson_lambda"])
            out[f] = x
        out.to_csv(ROOT / f"{cohort.lower()}_item_qualified_transformed.csv", index=False)
        out[META + features].to_csv(ROOT / f"{cohort.lower()}_item_retained_transformed.csv", index=False)

if __name__=="__main__":main()
