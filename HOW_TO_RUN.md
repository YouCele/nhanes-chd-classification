# How to run this, and where each result lands

## Execution order

Run them in this order. Each script depends on the files written by the ones
before it, and each one can also be re-run on its own.

| order | command | phases | roughly how long on one CPU |
|---|---|---|---|
| 1 | `python scripts/run_01_prepare.py` | 1-6: outcome, harmonisation, leakage, cohort, missingness, imputation strategy | 3 s |
| 2 | `python scripts/run_02_eda.py` | 7: exploratory analysis | 2 s |
| 3 | `python scripts/run_03_statistics.py` | 8 and 19: group comparisons, survey-weighted prevalence | 2 s |
| 4 | `python scripts/run_04_modeling.py` | 9-12: split design, baselines, class imbalance, tuning, comparison | 150 s |
| 5 | `python scripts/run_05_evaluation.py` | 13-15: test evaluation, calibration, interpretation | 60 s |
| 6 | `python scripts/run_06_robustness.py` | 16-18: error analysis, robustness, subgroups | 20 s |
| 7 | `python scripts/run_07_audit.py` | 20: automated audit of the saved outputs | 2 s |
| 8 | `python scripts/run_08_report.py` | 21: report skeleton filled with the computed numbers | 2 s |

Or all of them:

    python scripts/run_all.py            # everything
    python scripts/run_all.py 5 6        # only steps 5 and 6

The notebooks in `notebooks/` run the same scripts with `%run` and then display
the tables and figures. They contain no analysis logic, so they cannot drift
away from the scripts.

Before anything else, put the XPT files where the loader can find them
(see `data/raw/README.md`), then:

    pip install -r requirements.txt

## Where the results are

### Data and setup

| file | what it holds |
|---|---|
| `outputs/file_inventory.csv` | every raw file, row count, whether SEQN is unique, which registry variables it has |
| `outputs/variable_coverage_by_cycle.csv` | how complete each variable is in each cycle |
| `outputs/merge_audit.csv` | row counts and match rates after each merge step |
| `data/processed/cohort.csv` | the analytical cohort (parquet if pyarrow is installed) |
| `data/processed/feature_spec.json` | the frozen predictor list, indicators and exclusions |
| `data/processed/split.csv` | which participant is in development and which in test |

### Target and cohort

| file | what it holds |
|---|---|
| `outputs/target_definition.json` | the exact question, codes and mapping |
| `outputs/target_audit.csv` | how many yes, no, refused, don't know and blank answers per cycle |
| `outputs/target_prevalence_*.csv` | prevalence overall, by cycle, sex, age group and cycle by sex |
| `outputs/cohort_flow.csv` | the sample flow with a reason for every reduction |
| `outputs/cohort_summary.csv` | final N, cases, prevalence, median age, share female |

### Leakage and predictors

| file | what it holds |
|---|---|
| `outputs/leakage_single_variable_screen.csv` | what each candidate achieves on its own: AUC, odds ratio, missingness, whether missingness tracks the outcome |
| `outputs/leakage_decisions.csv` | the written decision and reason for each candidate, next to its numbers |
| `outputs/predictor_table.csv` | variable, source file, meaning, type, missingness, included, reason |
| `outputs/robustness_leakage_sensitivity.csv` | how much easier the task becomes with the excluded cardiovascular block |

### Missing data

| file | what it holds |
|---|---|
| `outputs/missingness.csv` | missing counts and percentages, by outcome, by cycle, with tests |
| `outputs/missingness_by_{survey_cycle,sex,age_group}.csv` | the same by group |
| `outputs/missing_indicator_decisions.csv` | where an indicator was added and why |
| `outputs/robustness_imputation.csv` | median against mean imputation |

### Description and statistics

| file | what it holds |
|---|---|
| `outputs/table_one.csv` | descriptive table split by outcome |
| `outputs/describe_numeric.csv`, `outputs/describe_categorical.csv` | distributions, skewness, level shares |
| `outputs/statistical_tests_continuous.csv` | Welch and Mann-Whitney, mean differences with intervals, Cohen's d, rank-biserial |
| `outputs/statistical_tests_categorical.csv` | chi-square or Fisher, odds ratios with intervals, Cramer's V |
| `outputs/effect_size_summary.csv` | every variable with its effect size and a plain magnitude label |
| `outputs/survey_weighted_prevalence.csv` | design-based population prevalence with linearised standard errors |

### Modelling and evaluation

| file | what it holds |
|---|---|
| `outputs/split_design.json` | the rule, the numbers that triggered it, and the chosen design |
| `outputs/cv_results.csv` | cross-validated baselines including the dummy |
| `outputs/tuning_results.csv` | every grid point with its CV mean and standard deviation |
| `outputs/model_comparison.csv` | the tuned models on the same folds |
| `outputs/paired_model_comparison.csv` | fold-by-fold differences between models |
| `outputs/model_selection.json` | which model was selected and on what basis |
| `outputs/test_metrics.csv` | test-set metrics with bootstrap intervals and the confusion counts |
| `outputs/evaluation_summary.json` | CV against test, threshold, whether they agree |
| `outputs/threshold_sweep.csv` | sensitivity, specificity and PPV across thresholds |
| `outputs/calibration.csv`, `outputs/calibration_by_risk_band.csv` | before and after recalibration |
| `outputs/calibration_summary.json` | slope, intercept, observed over expected, Brier |

### Interpretation, errors, robustness

| file | what it holds |
|---|---|
| `outputs/logistic_coefficients.csv` | standardised coefficients, odds ratios, Wald intervals |
| `outputs/feature_importance.csv` | permutation importance for each model |
| `outputs/feature_importance_agreement.csv` | rankings side by side, with the spread between models |
| `outputs/error_analysis.csv` | the four error groups profiled |
| `outputs/error_group_comparison.csv` | formal comparison of missed against detected cases |
| `outputs/error_missingness.csv` | whether errors have more missing predictors |
| `outputs/error_cases.csv` | one row per test participant with predicted risk and error type |
| `outputs/robustness_repeated_splits.csv` | ten random splits |
| `outputs/robustness_cycle_holdout.csv` | earlier cycles to later cycles |
| `outputs/robustness_predictor_sets.csv` | what each block of predictors contributes |
| `outputs/robustness_class_weight.csv` | weighted against unweighted |
| `outputs/subgroup_results.csv` | performance by sex, age group, race and ethnicity, cycle, with a reliability flag |
| `outputs/robustness_summary.json` | the headline robustness numbers in one place |

### Audit and write-up

| file | what it holds |
|---|---|
| `outputs/final_audit.csv`, `reports/audit_report.md` | 51 automated checks with pass, fail or manual |
| `outputs/research_decisions.json` | the machine-readable decision record |
| `reports/research_decision_log.md` | the same decisions in prose, with reasons |
| `reports/final_report.md` | the report with every number filled in and `[WRITE]` where your own words go |
| `outputs/eda_notes.md` | short EDA summary generated from the numbers |

### Figures

`figures/01_chd_prevalence.png`, `02_distributions_by_chd.png`,
`03_missingness.png`, `04_age_by_status.png`, `05_roc_pr.png`,
`06_calibration.png`, `06b_calibration_recalibrated.png`,
`07_confusion_matrix.png`, `08_permutation_importance.png`,
`09_subgroup_performance.png`, `10_threshold_sensitivity.png`.
