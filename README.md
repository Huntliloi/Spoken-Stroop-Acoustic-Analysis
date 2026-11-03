# Spoken Stroop acoustic measurement: analysis release

This package supports the analyses reported in the accompanying **current** manuscript and supplementary material. HC158 and HC74 are two enrollment batches from the same single-center healthy dataset; together they constitute the complete 232-participant healthy reference sample (HC232) and are analyzed as one cohort. The package also contains the 92-person cancer sample. Raw recordings are withheld except for two masked demonstration participants. The task images supplied by the study team remain in `task_materials/`.

## Contents

| Folder | Contents |
|---|---|
| `data/public/` | De-identified participant tables, un-imputed item-level acoustic features, frozen 62- and 51-feature healthy tables, HC158 manual and automated alignment times, and traditional Stroop measures. |
| `data/example/` | Two masked recordings per condition (six WAV files), matching ASR JSON files, manual TextGrids, and a 72-row token-time table. These examples are separate from the de-identified analysis tables. |
| `scripts/` | Analysis scripts grouped by experiment. |
| `results/` | Numeric analysis outputs and figure source data. |
| `task_materials/` | Supplied task images. |

The original source data and scripts were **copied** into this release after selecting the published analysis sample. All participant IDs in `data/public/` were replaced by internally consistent random labels (`H0001`–`H0232` and `C0001`–`C0092`). The private crosswalk, names, birth dates, hospital identifiers, collection dates, diagnoses, contact fields, and source audio paths are absent. Exact age, sex, education, cognitive scores, and item-level measurements remain because the reported models require them; therefore this is a **pseudonymized research dataset**, not a guarantee that re-identification is impossible. Do not attempt re-identification.

Column definitions and score ranges are in `DATA_DICTIONARY.md`.

## Task and variable conventions

The historical extraction scripts encode W as `Hanzi`, C as `Dots`, and CW as `STT`. These are English or romanized task labels retained to make the frozen tables and model contrasts reproducible. `STT-Hanzi` is CW−W, `STT-Dots` is CW−C, and `Dots-Hanzi` is C−W. The 90 candidate features comprise `Duration`, `Gap_Before`, and 88 eGeMAPS v02 functionals. The raw feature tables are **not imputed**. The 62-feature QC rule and fixed transformation parameters are in `results/02_acoustic_measurement_qc/`; the 51-feature selection is in `results/03_acoustic_measurement_structure/redundancy_selection_62_to_51.csv`.

`healthy_participants.csv` contains only variables needed to describe the sample and fit the models: age, female indicator, education years, CMMS, MoCA, and source batch (`HC158` or `HC74`). Source batch is a data provenance field and is **not** included as a covariate in the joint LME. Participant rows in the public tables are ordered by release pseudonym rather than source order. `cancer_participants.csv` contains age, female indicator, approximate education years, and four separately scored FACT-Cog domains. `cancer_model_input.csv` adds the six traditional measures and six frozen 51-feature Mean |zdev| scores used in the reported models. The source questionnaire's 1–5 response coding was converted to the standard 0–4 scale by subtracting the item count for each domain (20, 4, 9, and 4, respectively). The original questionnaire workbook is not released.

## Reproduction order

Python 3.11 was used. Install the versions in `requirements.txt` for the table-based analyses. All commands below are run from the package root and write new output to `results/recomputed/`, leaving the frozen `results/` tables intact.

```bash
python scripts/01_automated_measurement_validation/alignment_validation.py
python scripts/01_automated_measurement_validation/manual_auto_icc.py
python scripts/01_automated_measurement_validation/final51_icc.py
python scripts/01_automated_measurement_validation/within_subject_correlations.py
python scripts/02_acoustic_measurement_qc/measurement_qc.py --hc158 data/public/hc158_item_raw.csv --hc74 data/public/hc74_item_raw.csv --outdir results/recomputed/02_acoustic_measurement_qc
python scripts/02_acoustic_measurement_qc/prepare_features.py
python scripts/03_acoustic_measurement_structure/acoustic_structure.py --permutations 5000
python scripts/04_age_task_lme/joint_lme.py
python scripts/04_age_task_lme/fixed_effect_sensitivity.py
python scripts/04_age_task_lme/traditional_adjusted_lme.py
python scripts/05_acoustic_behavioral_coupling/behavioral_coupling.py
python scripts/06_cancer_fact_cog/cancer_scores.py
python scripts/06_cancer_fact_cog/cancer_models.py
python scripts/00_release_audit/verify_release.py
```

The full mixed-model and 200-repeat cancer cross-validation runs take longer than the initial data checks. `joint_lme.py --max-features 1` and `cancer_scores.py --max-features 1` provide smoke tests only; their resulting tables are **not** inferential results. The cancer score calculation refits the frozen HC232 joint model for each feature and uses a fixed-effect OLS fallback for the feature flagged as unstable in `results/04_age_task_lme/joint_lme_fit_diagnostics_51_features.csv`. It prints maximum differences from the archived six-score input when all 51 features are used.

`example_feature_extraction.py` can reproduce un-imputed eGeMAPS item features from `data/example/token_results.csv` and the six demonstration WAV files. `example_asr_alignment.py` also documents the ASR/forced-alignment pipeline but requires separately installed PyTorch and `qwen-asr`, plus local Qwen3-ASR and forced-aligner model directories supplied with `--asr-model` and `--aligner-model`. The models and full audio corpus are not bundled. Mandarin words in the demonstration ASR JSON and token table are task stimuli, not personal information.

The example alignment pipeline uses these exact model checkpoints. Download each model from its official page and pass the downloaded directories to `--asr-model` and `--aligner-model`:

| Model | ModelScope | Hugging Face |
|---|---|---|
| Qwen3-ASR-0.6B | [Qwen/Qwen3-ASR-0.6B](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B) | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| Qwen3-ForcedAligner-0.6B | [Qwen/Qwen3-ForcedAligner-0.6B](https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B) | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) |

## Experiment-to-file map

| Experiment | Script folder | Corresponding result folder |
|---|---|---|
| Automated measurement validation | `scripts/01_automated_measurement_validation/` | `results/01_automated_measurement_validation/` |
| Acoustic measurement QC (90→62) | `scripts/02_acoustic_measurement_qc/` | `results/02_acoustic_measurement_qc/` |
| Acoustic structure and redundancy reduction (62→51) | `scripts/03_acoustic_measurement_structure/` | `results/03_acoustic_measurement_structure/` |
| Age × task mixed models and sensitivity analyses | `scripts/04_age_task_lme/` | `results/04_age_task_lme/` |
| Acoustic–behavioral Stroop coupling | `scripts/05_acoustic_behavioral_coupling/` | `results/05_acoustic_behavioral_coupling/` |
| Cancer FACT-Cog association and prediction | `scripts/06_cancer_fact_cog/` | `results/06_cancer_fact_cog/` |

`results/01_automated_measurement_validation/table_1_alignment_boundary_mae.csv` contains the numerical estimates underlying manuscript Table 1. `results/figure_source_data/` contains numeric data for the acoustic-structure and cancer-association figures. The traditional-performance figure uses `data/public/healthy_traditional.csv`. `scripts/00_release_audit/verify_release.py` checks the release; `MANIFEST.sha256` lists file checksums.
