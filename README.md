# InCHIANTI Fried Frailty Phenotype Pipeline

Compute Fried frailty phenotype components for InCHIANTI Follow-up 4 and Follow-up 5.

## Authors

- Valerio Antonio Arcobelli
- Jose Albites Sanabria

## What the script does

The script computes the five Fried frailty components:

1. Weight loss
2. Exhaustion
3. Low physical activity
4. Walking speed / slowness
5. Grip strength / weakness

It then derives:

- `frailty_count`
- `fried_status`
- complete-case outputs
- optional DMO-merged outputs
- optional exploratory plots by Fried status

Phenotype coding:

| Code | Meaning |
|---:|---|
| 0 | Component absent |
| 1 | Component present |
| 8 | Not applicable, age < 65 years |
| 9 | Undetermined / missing information |

## Installation

```bash
pip install -r requirements.txt
```

## Basic usage

```bash
python fried_frailty_inchianti.py \
  --base-dir "/path/to/InCHIANTI/InCHIANTI 2023" \
  --baseline-pef-raw "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_rawe.sas7bdat" \
  --baseline-pef-ana "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_ana.sas7bdat" \
  --output-dir outputs
```

## With DMO merge

```bash
python fried_frailty_inchianti.py \
  --base-dir "/path/to/InCHIANTI/InCHIANTI 2023" \
  --baseline-pef-raw "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_rawe.sas7bdat" \
  --baseline-pef-ana "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_ana.sas7bdat" \
  --dmo-file "/path/to/FU4_FU5_dmos_w_clinical.csv" \
  --output-dir outputs
```

The DMO file is expected to include:

- `ID`
- `Wave`, with values `FU4` or `FU5`

## Optional plotting

```bash
python fried_frailty_inchianti.py \
  --base-dir "/path/to/InCHIANTI/InCHIANTI 2023" \
  --baseline-pef-raw "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_rawe.sas7bdat" \
  --baseline-pef-ana "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_ana.sas7bdat" \
  --dmo-file "/path/to/FU4_FU5_dmos_w_clinical.csv" \
  --plot-features walkdur_all_sum_w steps_all_sum_w ws_30_avg_w \
  --plot-wave Follow-up4_v2 \
  --output-dir outputs
```

## Optional validation

Follow-up 4 can be compared with `adjf4ana.sas7bdat` when the file is available:

```bash
python fried_frailty_inchianti.py \
  --base-dir "/path/to/InCHIANTI/InCHIANTI 2023" \
  --baseline-pef-raw "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_rawe.sas7bdat" \
  --baseline-pef-ana "/path/to/Baseline_V6/English/4.Data/SAS_Datasets/Physical_Exam/per_ana.sas7bdat" \
  --validate \
  --output-dir outputs
```

## Output files

The script saves:

- `fried_phenotypes_fu4_fu5.csv`
- `fried_phenotypes_fu4_fu5_complete_cases.csv`
- `fried_phenotypes_fu4_fu5_side_by_side.csv`
- `fried_phenotypes_fu4_fu5_with_dmos.csv`, if a DMO file is provided
- `fried_phenotypes_fu4_fu5_complete_cases_with_dmos.csv`, if a DMO file is provided
- feature plots and statistical test tables, if plotting is requested

## Notes for GitHub

Data files are intentionally excluded from version control through `.gitignore`.
Do not commit raw InCHIANTI SAS files or local output files.
