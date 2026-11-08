"""Recalculate boundary MAEs from the currently supplied de-identified times."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "public"
OUTPUT = ROOT / "results" / "recomputed" / "01_automated_measurement_validation"
KEY = ["Subject_ID", "Task", "Item_Index"]


def main() -> None:
    manual = pd.read_csv(DATA / "hc158_item_manual_raw.csv", dtype={"Subject_ID": str})
    auto = pd.read_csv(DATA / "hc158_alignment_timestamps.csv", dtype={"Subject_ID": str})
    joined = manual[KEY + ["Absolute_Word_Start", "Absolute_Word_End"]].merge(
        auto, on=KEY, how="inner", validate="one_to_one")
    if joined.Subject_ID.nunique() != 158:
        raise ValueError("Expected 158 paired healthy participants")
    rows = []
    for task in ("ALL", "Hanzi", "Dots", "STT"):
        d = joined if task == "ALL" else joined[joined.Task.eq(task)]
        for method, start, end in (("FA_only", "FA_Start", "FA_End"),
                                   ("FA_plus_refinement", "Refined_Start", "Refined_End")):
            reference_duration = d.Absolute_Word_End - d.Absolute_Word_Start
            duration = d[end] - d[start]
            rows.append({"Task": task, "Method": method, "N_items": len(d),
                         "Start_MAE_s": (d[start] - d.Absolute_Word_Start).abs().mean(),
                         "End_MAE_s": (d[end] - d.Absolute_Word_End).abs().mean(),
                         "Duration_MAE_s": (duration - reference_duration).abs().mean()})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "table_1_alignment_boundary_mae.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
