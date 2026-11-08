"""Audit the public analysis package without accessing private source files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "public"
DIRECT_IDENTIFIERS = {"name", "birth", "phone", "address", "hospital_id",
                      "audio_path", "recording_path", "qq", "patient_id"}


def table(name: str, n_people: int, prefix: str) -> pd.DataFrame:
    data = pd.read_csv(DATA / name, dtype={"Subject_ID": str}, low_memory=False)
    ids = data.Subject_ID.dropna()
    assert ids.nunique() == n_people, (name, ids.nunique(), n_people)
    assert ids.str.fullmatch(fr"{prefix}\d{{4}}").all(), name
    assert ids.is_monotonic_increasing, (name, "source row order retained")
    assert not (DIRECT_IDENTIFIERS & {str(c).lower() for c in data.columns}), name
    return data


def main() -> None:
    h = table("healthy_participants.csv", 232, "H")
    c = table("cancer_participants.csv", 92, "C")
    assert len(h) == 232 and len(c) == 92
    assert "ModelGroupCode" not in h.columns
    for cohort, expected_people, expected_rows in (("hc158", 158, 5677),
                                                   ("hc74", 74, 2631)):
        for suffix in ("raw", "62_transformed", "51_transformed"):
            d = table(f"{cohort}_item_{suffix}.csv", expected_people, "H")
            assert len(d) == expected_rows
            assert set(d.Subject_ID).issubset(h.Subject_ID)
    manual = table("hc158_item_manual_raw.csv", 158, "H")
    assert len(manual) == 5688
    aligned = table("hc158_alignment_timestamps.csv", 158, "H")
    assert len(aligned) == 5682
    frames = table("hc158_manual_f0_frame_counts.csv", 158, "H")
    assert len(frames) == 5677
    assert set(frames.Subject_ID) == set(manual.Subject_ID)
    raw_cancer = table("cancer_item_raw.csv", 92, "C")
    assert len(raw_cancer) == 3270
    model = table("cancer_model_input.csv", 92, "C")
    assert len(model) == 92 and not model.isna().any().any()
    assert set(model.Subject_ID) == set(c.Subject_ID)
    for name, people, prefix in (("healthy_traditional.csv", 232, "H"),
                                 ("cancer_traditional.csv", 92, "C")):
        d = table(name, people, prefix)
        assert len(d) == people * 3
    assert len(list((ROOT / "data" / "example" / "audio").rglob("*.wav"))) == 6
    assert len(list((ROOT / "data" / "example" / "manual_textgrid").glob("*.TextGrid"))) == 6
    assert len(list((ROOT / "data" / "example" / "asr_json").rglob("*.json"))) == 6
    supplemental = {
        "results/07_within_session_reliability/table_s3_task_contrast_split_half.csv": 6,
        "results/01_automated_measurement_validation/short_word_precision/table_s7_word_duration_agreement.csv": 4,
        "results/01_automated_measurement_validation/short_word_precision/table_s13_short_window_exclusion.csv": 3,
        "results/07_within_session_reliability/table_s17_anomaly_score_split_half.csv": 6,
    }
    for relative, n_rows in supplemental.items():
        assert len(pd.read_csv(ROOT / relative)) == n_rows, relative
    for path in ROOT.rglob("*"):
        if not path.is_file() or "recomputed" in path.parts:
            continue
        assert path.suffix.lower() not in {".xlsx", ".docx", ".xls", ".tif", ".mp3"}, path
        if path.suffix.lower() in {".py", ".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8-sig")
            assert not re.search(r"[A-Z]:[\\/]", text), path
            if path.parent != ROOT / "data" / "example" and "asr_json" not in path.parts:
                assert not re.search(r"(?<![\w.])20\d{9}(?![\w.])", text), path
            if "results" in path.parts and path.suffix.lower() == ".csv":
                assert not re.search(r"[\u4e00-\u9fff]", text), path

    manifest = ROOT / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        source = ROOT / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative
    print("Release audit passed: 232 healthy, 92 cancer, six example WAV files,")
    print("analysis outputs, consistent pseudonyms and checksums.")


if __name__ == "__main__":
    main()
