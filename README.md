# Frailty Classification from Digital Mobility Outcomes

Clean GitHub-ready code for generating Fried frailty status, selecting a low-redundancy Digital Mobility Outcome (DMO) feature set, and running machine-learning models for frailty classification.

## Authors

- Valerio Antonio Arcobelli
- Jose Albites Sanabria

## Repository structure

```text
frailty_dmo_ml_github/
├── README.md
├── AUTHORS.md
├── requirements.txt
├── .gitignore
└── scripts/
    ├── generate_fried_status.py
    ├── supportFrailty.py
    ├── generate_orthogonal_features.py
    └── run_frailty_ml.py
```

## Workflow overview

The analysis is organized in three steps.

### Step 1 — Generate Fried frailty status

`scripts/generate_fried_status.py` reads the InCHIANTI SAS files, computes the five Fried phenotype components, and generates `fried_status`.

Computed Fried components:

1. weight loss
2. exhaustion
3. low physical activity
4. walking speed / slowness
5. grip strength / weakness

Component coding:

```text
0 = component absent
1 = component present
8 = not applicable, age < 65 years
9 = undetermined / missing information
```

Final Fried status:

```text
0 components = Robust
1–2 components = Pre-frail
≥3 components = Frail
```

Run:

```bash
python scripts/generate_fried_status.py \
  --base-dir "/path/to/InCHIANTI/InCHIANTI 2023" \
  --baseline-pef-raw "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_rawe.sas7bdat" \
  --baseline-pef-ana "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_ana.sas7bdat" \
  --dmo-file "/path/to/dmos_python.csv" \
  --output-dir fried_outputs
```

Main output for the downstream scripts:

```text
fried_outputs/FU4_FU5_merged.csv
```

This file contains `CODE98`, `Wave`, `AGE`, the Fried components, `frailty_count`, `fried_status`, and the DMO columns when `--dmo-file` is provided.

Other outputs include:

```text
fried_phenotypes_fu4_fu5.csv
fried_phenotypes_fu4_fu5_complete_cases.csv
fried_phenotypes_fu4_fu5_side_by_side.csv
fried_phenotypes_fu4_fu5_with_dmos.csv
fried_phenotypes_fu4_fu5_complete_cases_with_dmos.csv
```

### Step 2 — Generate the orthogonal DMO feature set

`scripts/generate_orthogonal_features.py` evaluates redundancy among DMO features and proposes a low-redundancy ORTHO feature set.

Run:

```bash
python scripts/generate_orthogonal_features.py \
  --input-csv fried_outputs/FU4_FU5_merged.csv \
  --outdir orthogonality_report
```

Main output:

```text
orthogonality_report/orthogonal_feature_set.txt
```

The script also saves correlation matrices, heatmaps, VIF table, Belsley diagnostics, unique variance, and mRMR ranking.

### Step 3 — Run machine-learning models

`scripts/run_frailty_ml.py` evaluates two binary classification tasks:

1. Robust vs Pre-frail
2. Robust vs Non-robust, where Non-robust = Pre-frail + Frail

Default feature sets:

1. `ORTHO`
2. `ORTHO_AGE`

Run:

```bash
python scripts/run_frailty_ml.py \
  --frailty-csv fried_outputs/FU4_FU5_merged.csv \
  --orthogonal-features orthogonality_report/orthogonal_feature_set.txt \
  --outdir ml_results \
  --n-jobs -1
```

Main outputs:

```text
ml_results/frailty_ml_results.xlsx
ml_results/tables/
ml_results/predictions_oof/
ml_results/plots/
ml_results/config/
```

## Shared helper module

`scripts/supportFrailty.py` contains:

- DMO metadata: feature code, readable label, unit, and domain
- list of DMO columns
- frailty-status normalization helpers
- dataframe cleaning helpers
- optional descriptive/statistical helpers for DMO comparisons across Fried groups

This avoids duplicating DMO labels and domains across scripts.

## Installation

```bash
pip install -r requirements.txt
```

## Notes

- Raw InCHIANTI SAS files and output datasets are not included in this repository.
- The `.gitignore` is configured to avoid committing local raw data and generated results.
- Run Step 1 first if your working dataset does not already contain `fried_status`.
- If you already have a clean CSV with `CODE98`, `Wave`, `AGE`, `fried_status`, and DMO columns, you may start directly from Step 2.
