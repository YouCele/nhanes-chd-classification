# Coronary heart disease classification with NHANES

A portfolio project: can routinely collected demographic, examination,
laboratory and questionnaire data separate adults who report a
physician-diagnosed coronary heart disease (CHD) from adults who do not?

Four NHANES cycles (2011-2012 through 2017-2018), self-reported outcome,
binary classification, logistic regression against random forest and
gradient boosting.

**This is not a diagnostic tool.** The outcome is a self-report, not a
clinical or angiography-confirmed diagnosis, and the project makes no causal
or prognostic claims. See `reports/final_report.md` for the full write-up and
`reports/research_decision_log.md` for the reasoning behind every
methodological choice.

## Headline result

21,570 adults, 881 with the outcome (4.08%). On a held-out quarter of the
cohort, never touched before the final evaluation:

| metric | value | 95% CI |
|---|---|---|
| ROC-AUC | 0.875 | 0.854 - 0.895 |
| PR-AUC | 0.239 | 0.205 - 0.291 |

Age alone gets to a ROC-AUC of 0.83; the full 34-variable model reaches
0.87. The rest of the discrimination comes mostly from blood pressure,
lipids and body size - the variables clinical guidance already uses, not
anything new. Full numbers, including the leakage check, the calibration
curve and the subgroup breakdown, are in `outputs/` and in the report.

## What this project does not have

Smoking, diabetes and blood-pressure-medication data are not part of the
local file set (no SMQ, DIQ, BPQ, GHB or GLU), so three of the
best-established CHD risk factors are missing from the model. This is
stated as a limitation, not worked around with a proxy variable.

## Project structure

```
.
├── data/
│   ├── raw/            NHANES XPT files (not included, see data/raw/README.md)
│   └── processed/      intermediate cohort and split files, built by scripts/run_01
├── src/                the actual analysis code, one module per concern
├── scripts/            eight numbered scripts, one per group of phases, plus run_all.py
├── notebooks/          thin notebooks that run the scripts and display the results
├── outputs/            every table the analysis produces, as CSV or JSON
├── figures/            every plot, saved as PNG
├── models/             fitted pipelines (not tracked in git, rebuilt by script 4)
├── reports/            the final report, the decision log, and the audit report
├── HOW_TO_RUN.md        execution order and a map of every output file
└── requirements.txt
```

## Running it

```
pip install -r requirements.txt
# put the NHANES XPT files under data/raw/, see data/raw/README.md
python scripts/run_all.py
```

See `HOW_TO_RUN.md` for the full execution order, what each step depends on,
and a complete map of which file holds which result.

## Data source

NHANES is public. Download it from the CDC
(<https://wwwn.cdc.gov/nchs/nhanes/>); `data/raw/README.md` lists the exact
files for each cycle. Raw XPT files are not committed to this repository.

## License

Code is MIT licensed (see `LICENSE`). NHANES data is a public-domain product
of the U.S. National Center for Health Statistics and is not covered by that
license - it is not redistributed here in any case, see above.
