# Data dictionary

All participant IDs are release-specific pseudonyms. Tables use one row per participant, participant-task, or participant-task-item as indicated by their filenames. No source ID, name, recording path, date, or diagnosis text is included.

| Variable | Meaning |
|---|---|
| `Subject_ID` | Stable pseudonym across this release (`H####` or `C####`). |
| `SourceCohort` | Enrollment batch within the complete HC232 healthy sample (`HC158` or `HC74`); not a regression covariate. |
| `Task` | `Hanzi` = W, `Dots` = C, `STT` = CW. |
| `Item_Index` | Position of a spoken response within its 12-item task (1–12). |
| `Absolute_Word_Start`, `Absolute_Word_End` | Refined item boundaries in seconds from recording start. |
| `FA_Start`, `FA_End` | Initial forced-alignment boundaries in seconds. |
| `Refined_Start`, `Refined_End` | Boundaries after adaptive refinement in seconds. |
| `Duration`, `Gap_Before` | Item duration and preceding gap in seconds. |
| `N_LLD_frames`, `N_F0_positive_frames`, `F0_positive_frame_fraction` | In the HC158 manual-window frame-count table: total openSMILE eGeMAPS low-level frames, frames with positive smoothed F0, and their fraction. Counts are derived from the unavailable full audio and are not valid-frame counts for every eGeMAPS functional. |
| `Age` | Age in years. |
| `SexFemale`, `Female` | 1 = female, 0 = male. |
| `EducationYears` | Healthy-participant education in years. |
| `Education_years_approx` | Cancer education category mapped to approximate years: levels 1–6 to 0, 6, 9, 12, 16, and 19 years; secondary technical education to 12 years. |
| `CMMS` | Chinese Mini-Mental State Examination score, where available. |
| `MoCA` | Raw Montreal Cognitive Assessment score for participants without CMMS. |
| `PCI`, `Others`, `Abilities`, `QOL_impact` | FACT-Cog domain scores on the standard 0–4 item scale, with ranges 0–80, 0–16, 0–36, and 0–16. Higher scores indicate better self-reported cognitive functioning or less impact. |
| `Accuracy` | Proportion of correct color-word responses for a task. |
| `Completion_Time_sec` | Seconds from first item onset to last item offset in a task. |
| `Raw_Color_Token_Count`, `Paired_Item_Count` | Token count and matched item count used in reported completeness checks. |
| `Accuracy_W`, `Accuracy_C`, `Accuracy_CW` | Cancer participant's three task accuracies. |
| `Time_W`, `Time_C`, `Time_CW` | Cancer participant's three task completion times, in seconds. |
| `Acoustic_W`, `Acoustic_C`, `Acoustic_CW`, `Acoustic_C-W`, `Acoustic_CW-C`, `Acoustic_CW-W` | Six healthy-reference Mean |zdev| scores, each averaging 51 feature-level standardized deviations. |

The remaining acoustic column names are the original eGeMAPS variable names. Descriptions, categories, selection decisions, and frozen transformations are in `results/02_acoustic_measurement_qc/feature_dictionary_62.csv`, `results/02_acoustic_measurement_qc/frozen_transformation_rules_62.csv`, and `results/03_acoustic_measurement_structure/redundancy_selection_62_to_51.csv`. The raw item tables include all 90 candidates; missing cells are left missing. Only the transformed 62- and 51-feature healthy tables contain frozen log1p or Yeo–Johnson values.
