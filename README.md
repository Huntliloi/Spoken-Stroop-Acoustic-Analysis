# Card-based spoken Stroop task: acoustic measurement analysis release

This package contains the data and numerical analyses of a card-based spoken Stroop task reported in the accompanying manuscript. HC158 and HC74 are enrollment subsets of one single-center healthy reference dataset (HC232, N=232), analyzed together. The external cancer sample comprises 92 participants. The final measurement set contains 49 features after quality control of 90 candidates (60 qualified) and within-category redundancy reduction. No missing-value imputation is used.

## Contents

- `data/public/`: pseudonymized participant information, 90-feature item tables, 60-feature and 49-feature transformed healthy tables, manual features/timestamps, F0-positive frame counts, traditional measures, response mapping, and six-score cancer model inputs.
- `data/example/`: the two previously supplied demonstration participants, with six WAV recordings, ASR JSON, manual TextGrids, and token times.
- `scripts/`: experiment-specific analysis code. No manuscript-editing code or plotting scripts is included.
- `results/`: complete numerical outputs for the paper analyses and figure source tables. `results/table_source_index.csv` maps Table 1 and Tables S1-S18 to these outputs; these are analysis outputs rather than formatted publication tables.
- `task_materials/`: study-provided task images.

All participant tables use consistent pseudonyms (H0001-H0232, C0001-C0092). The private crosswalk, original identifiers, names, dates, and local audio paths are withheld. Age, sex, education, cognitive scores and item measurements remain for reproduction; this is pseudonymization rather than a guarantee against re-identification. The HC158/HC74 distinction is provenance only and is not a model covariate.

## Reproduction

Python 3.11 was used; install `requirements.txt`. Run commands from the package root. Frozen outputs remain unchanged; new calculations go to `results/recomputed/`. Frozen QC metadata seeds that output directory so downstream analyses can run without repeating the full QC audit. Randomized ID ordering may change participant-bootstrap confidence interval endpoints slightly, even with the same seed.

```bash
python scripts/00_release_audit/verify_release.py
python scripts/01_automated_measurement_validation/alignment_validation.py
python scripts/02_acoustic_measurement_qc/measurement_qc.py --hc158 data/public/hc158_item_raw.csv --hc74 data/public/hc74_item_raw.csv --outdir results/recomputed/02_acoustic_measurement_qc
python scripts/02_acoustic_measurement_qc/prepare_features.py
python scripts/03_acoustic_measurement_structure/acoustic_structure.py
python scripts/04_age_task_lme/joint_lme.py
python scripts/04_age_task_lme/fixed_effect_sensitivity.py
python scripts/04_age_task_lme/traditional_adjusted_lme.py
python scripts/08_traditional_performance/traditional_and_task_shift.py
python scripts/01_automated_measurement_validation/manual_auto_icc.py
python scripts/01_automated_measurement_validation/retained_features_icc.py
python scripts/01_automated_measurement_validation/median_icc_summary.py
python scripts/01_automated_measurement_validation/within_subject_correlations.py
python scripts/01_automated_measurement_validation/run_short_word_measurement_audit.py
python scripts/01_automated_measurement_validation/bootstrap_agreement_precision.py
python scripts/01_automated_measurement_validation/analyze_frame_sensitivity.py
python scripts/05_acoustic_behavioral_coupling/behavioral_coupling.py
python scripts/06_cancer_fact_cog/cancer_scores.py
python scripts/06_cancer_fact_cog/cancer_models.py
python scripts/07_within_session_reliability/run_split_half_reliability.py
```

The reference score calculation uses item-level fixed-effect residuals, aggregates them within condition, then standardizes each feature/condition or contrast against HC232. Mean absolute deviations aggregate all 49 frozen features. Unstable joint fits use the documented fixed-effect OLS fallback. Models adjust for age centered in years, sex, education, task and item position, without a cohort covariate. The supplied cancer model input contains the six scores used for the reported predictions; `cancer_scores.py` independently recalculates them. The prediction script reads the frozen input. To evaluate newly recalculated scores, merge its `cancer_six_scores.csv` by Subject_ID and Condition into the corresponding Acoustic_* columns of a working copy of the model input.

Full mixed-model, permutation, bootstrap and 200-repeat CV analyses take substantial time. `joint_lme.py --max-features 1` is a smoke test, not a complete inferential run. Short-window analyses use released manual-window F0-positive frame counts because the complete audio corpus is withheld. These are voicing proxies, not valid-frame counts for every eGeMAPS parameter.

## Response mapping and audio example

W=Hanzi (word reading), C=Dots (color naming), CW=STT (color-word interference). The task vocabulary comprises red, yellow, blue and green. Derived mapping/token tables use English color labels; Mandarin words in demonstration inputs and matching code are intentional UTF-8 task content.

Of 696 healthy recordings, 622 had exactly 12 recognized color tokens, 65 more and 9 fewer. The corresponding cancer counts (276 recordings) were 253, 10 and 13. Surplus sequences are matched left-to-right to the 12 targets, with at most two additional tokens searched and lookahead limited by the remaining surplus. The first response determines acoustic windows and accuracy; self-correction does not restore accuracy. Completion time includes the complete recognized response sequence. The 65 healthy and 10 cancer surplus recordings were realigned and re-extracted; other acoustic rows retain the original segmentation. `alignment_rules.py` implements the rule. `example_asr_alignment.py` retains full recognized sequences before matching; `example_feature_extraction.py` applies matching before acoustic extraction and calculates traditional scores.

```bash
python scripts/01_automated_measurement_validation/example_feature_extraction.py
```

ASR additionally requires PyTorch, qwen-asr and separately downloaded model checkpoints; pass local directories via --asr-model and --aligner-model. Full audio is not included.

| Checkpoint | ModelScope | Hugging Face |
|---|---|---|
| Qwen3-ASR-0.6B | https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B | https://huggingface.co/Qwen/Qwen3-ASR-0.6B |
| Qwen3-ForcedAligner-0.6B | https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B | https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B |

## Figure and table numbering

Figure 4 uses healthy traditional performance. Figure 5 uses overall/CW-W correlations, parallel analysis and rotated loadings. Figure 6 uses the supplied six-feature model predictions and contrasts. Figure 7 contains only perceived cognitive abilities: A adjusts for demographics, B additionally adjusts for traditional Stroop performance. Figure S1 uses overall-profile loadings. Their numerical sources are in `results/figure_source_data/`. S18 combines leave-one-out prediction intervals and repeated-CV increments; there is no separate S19 in the current supplement. FACT-Cog domains are analyzed separately, without a total score. Consult DATA_DICTIONARY.md for columns and MANIFEST.sha256 for file integrity.
