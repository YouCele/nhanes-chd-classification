# Analysis audit

51 automated checks were run over the saved outputs. 0 failed and 2 are judgement calls that a person has to confirm.

## data

| item                                    | status   | detail                                     |
|:----------------------------------------|:---------|:-------------------------------------------|
| all required raw files found            | pass     | 28 of 28 files                             |
| four cycles present                     | pass     | 2011-2012, 2013-2014, 2015-2016, 2017-2018 |
| SEQN unique in the cohort               | pass     | 21570 rows                                 |
| SEQN never missing                      | pass     |                                            |
| merge retention recorded for every file | pass     | 28 merge steps logged                      |
| cycle differences documented            | pass     |                                            |

## target

| item                                          | status   | detail                                            |
|:----------------------------------------------|:---------|:--------------------------------------------------|
| definition documented                         | pass     | MCQ160C                                           |
| refused and don't know excluded, not set to 0 | pass     | valid target count equals yes + no in every cycle |
| target is strictly binary                     | pass     |                                                   |
| no missing target in the cohort               | pass     |                                                   |
| prevalence checked across cycles              | pass     |                                                   |

## leakage

| item                                                     | status   | detail                                                    |
|:---------------------------------------------------------|:---------|:----------------------------------------------------------|
| every decision has a written reason                      | pass     | 18 decisions                                              |
| the outcome variable is not a predictor                  | pass     |                                                           |
| cardiovascular questionnaire block kept out of the model | pass     | excluded: age_told_chd, angina, chf, heart_attack, stroke |
| every excluded variable has a documented reason          | pass     | 14 excluded variables                                     |
| effect of the excluded block quantified                  | pass     |                                                           |

## missingness

| item                                         | status   | detail                                                          |
|:---------------------------------------------|:---------|:----------------------------------------------------------------|
| missingness measured for every predictor     | pass     | 35 variables                                                    |
| missingness compared by outcome and by cycle | pass     |                                                                 |
| indicator decisions documented               | pass     | indicators: ['ldl', 'triglycerides']                            |
| imputation happens inside the pipelines      | pass     | SimpleImputer sits inside every model pipeline, fitted per fold |
| alternative imputation tested                | pass     |                                                                 |

## modelling

| item                                            | status   | detail                                                                                                 |
|:------------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------|
| split design justified before modelling         | pass     | stratified_random: cycles treated as exchangeable when the prevalence difference is small or not si... |
| test set is a quarter of the cohort, stratified | pass     | 0.250 of rows                                                                                          |
| prevalence similar in both parts                | pass     | development 0.0409 vs test 0.0408                                                                      |
| a dummy baseline was run                        | pass     | dummy PR-AUC 0.041                                                                                     |
| at least three real models compared             | pass     |                                                                                                        |
| class imbalance handled                         | pass     | class weights inside the estimators, no oversampling of any test data                                  |
| tuning recorded                                 | pass     |                                                                                                        |
| models compared on the same folds               | pass     |                                                                                                        |

## evaluation

| item                                                  | status   | detail                                                           |
|:------------------------------------------------------|:---------|:-----------------------------------------------------------------|
| all required metrics reported                         | pass     | roc_auc, pr_auc, brier, sensitivity, specificity, ppv, npv       |
| confidence intervals reported                         | pass     | 1000 bootstrap resamples, stratified                             |
| calibration evaluated                                 | pass     |                                                                  |
| recalibrated probabilities are close to observed risk | pass     | calibration slope 1.034                                          |
| confusion matrix reported                             | pass     |                                                                  |
| test set used once, after the decisions were frozen   | manual   | the threshold comes from cross-validated development predictions |

## interpretation

| item                                    | status   | detail                                |
|:----------------------------------------|:---------|:--------------------------------------|
| coefficients with intervals reported    | pass     |                                       |
| permutation importance reported         | pass     |                                       |
| agreement between models examined       | pass     |                                       |
| prediction kept separate from causation | manual   | wording checked by hand in the report |

## robustness

| item                                            | status   | detail                           |
|:------------------------------------------------|:---------|:---------------------------------|
| repeated splits                                 | pass     | robustness_repeated_splits.csv   |
| cycle based evaluation                          | pass     | robustness_cycle_holdout.csv     |
| predictor reduction                             | pass     | robustness_predictor_sets.csv    |
| class weight sensitivity                        | pass     | robustness_class_weight.csv      |
| subgroup analysis                               | pass     | subgroup_results.csv             |
| error analysis                                  | pass     | error_analysis.csv               |
| performance stable across splits                | pass     | ROC-AUC sd 0.0080 over 10 splits |
| small subgroups flagged rather than interpreted | pass     | 3 subgroups marked as too small  |

## reproducibility

| item                      | status   | detail            |
|:--------------------------|:---------|:------------------|
| random seed fixed         | pass     | seed 2026         |
| relative paths only       | pass     | raw dir: data/raw |
| requirements file present | pass     |                   |
| decision log present      | pass     |                   |
