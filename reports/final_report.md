# Classification of self-reported coronary heart disease in NHANES 2011-2018

*Generated from the saved analysis outputs. Every number and every interpretive sentence below is produced by `scripts/run_08_report.py` from the files in `outputs/`; nothing here was typed in by hand.*

## 1. Abstract

NHANES data from four cycles (2011-2012 to 2017-2018) were used to see how well routine demographic, examination, laboratory and questionnaire information separates adults who report a physician diagnosis of coronary heart disease from adults who do not. The analytical cohort had 21,570 participants, of whom 881 reported CHD (4.08%). The selected model (logistic regression) reached a ROC-AUC of 0.875 (95% CI 0.854 to 0.895) and a PR-AUC of 0.239 (95% CI 0.205 to 0.291) on a held-out quarter of the cohort. In practice this means the model ranks people sensibly - someone who actually has CHD tends to score higher than someone who does not - but because CHD is rare here (about 4.1% of the cohort), a positive prediction is still wrong more often than it is right (PPV 0.13 at the chosen threshold). The PR-AUC of 0.24 needs to be read against that base rate, not against 1.

## 2. Background

Coronary heart disease is one of the leading causes of death in the United States, and most of what is known to predict it is not exotic: age, blood pressure, cholesterol, body size, family history. NHANES collects exactly that kind of information from a large, repeated sample of the adult population, which makes it a reasonable place to ask how far routine data can go. This project is not an attempt to build a diagnostic tool. It does not use angiography or any clinical confirmation of the outcome - the label is a self-report, and it is treated as one throughout, in the wording and in the conclusions. The aim is narrower: to see how well the variables NHANES already collects separate the two groups, using a validation design that could be checked step by step by someone else.

## 3. Research question

Can routinely collected demographic, clinical, examination, laboratory and lifestyle information distinguish adults with self-reported physician-diagnosed coronary heart disease from adults without it?

## 4. Dataset

Four NHANES cycles: 2011-2012, 2013-2014, 2015-2016 and 2017-2018. Files used: DEMO, MCQ, BPX, BMX, TCHOL, HDL and TRIGLY. The full file inventory, including row counts and which registry variables each file contains, is in `outputs/file_inventory.csv`.

Files that were part of the original plan but are not in the local data set: SMQ (smoking questionnaire (SMQ020, SMQ040)); DIQ (diabetes questionnaire (DIQ010)); BPQ (blood pressure / cholesterol questionnaire (BPQ020, BPQ080)); GHB (glycohemoglobin (LBXGH)); GLU (fasting glucose (LBXGLU)).

## 5. CHD outcome definition

- Variable: `MCQ160C` (MCQ (Medical Conditions questionnaire))
- Question: Has a doctor or other health professional ever told you that you had coronary heart disease?
- Asked of: participants aged 20 years and over
- Codes: 1 = Yes, 2 = No, 7 = Refused, 9 = Don't know, blank = missing
- Mapping: 1 -> 1 ; 2 -> 0 ; 7, 9, blank -> excluded (target not usable)

This is self-reported, physician-diagnosed CHD. It is not angiography-confirmed disease and it is not a clinical gold standard.

Prevalence by cycle:

| survey_cycle   |    n |   cases |   prevalence |   ci_low |   ci_high |
|:---------------|-----:|--------:|-------------:|---------:|----------:|
| 2011-2012      | 5538 |     196 |       0.0354 |   0.0308 |    0.0406 |
| 2013-2014      | 5751 |     232 |       0.0403 |   0.0356 |    0.0457 |
| 2015-2016      | 5693 |     244 |       0.0429 |   0.0379 |    0.0484 |
| 2017-2018      | 5553 |     265 |       0.0477 |   0.0424 |    0.0536 |

Prevalence by age group:

| age_group   |    n |   cases |   prevalence |
|:------------|-----:|--------:|-------------:|
| 20-39       | 7547 |      18 |       0.0024 |
| 40-49       | 3684 |      40 |       0.0109 |
| 50-59       | 3687 |     101 |       0.0274 |
| 60-69       | 3880 |     263 |       0.0678 |
| 70+         | 3737 |     515 |       0.1378 |

Prevalence rises from 3.5% in 2011-2012 to 4.8% in 2017-2018, and the confidence intervals at the two ends do not overlap, so the increase looks like more than noise. The design here cannot say why - an older or sicker sample in later cycles and more consistent diagnosis are both plausible, and nothing in this data set can separate them. The cycle-based robustness check later in the report takes this seriously by testing the model across time rather than assuming the cycles are interchangeable.

## 6. Study population

| step                                      |     n | reason                                                                                                              |
|:------------------------------------------|------:|:--------------------------------------------------------------------------------------------------------------------|
| all screened participants, four cycles    | 39156 | DEMO row count                                                                                                      |
| adults aged 20+                           | 22617 | the CHD question is only asked of adults 20 and over                                                                |
| examined in the mobile examination centre | 21646 | the intended predictors include measured blood pressure and body size, which interview-only participants never have |
| usable CHD answer (yes or no)             | 21570 | 76 refused, did not know, or had no answer recorded                                                                 |
| final analytical cohort                   | 21570 | kept; 73 of them have no usable examination measurement, handled by imputation rather than exclusion                |

Final cohort: 21,570 adults, 881 with the outcome and 20,689 without, median age 50, 51.8% female.

## 7. Data preparation

Blood pressure was recorded up to four times per participant. The readings were averaged after removing any diastolic reading of exactly 0, which is not a real value but a note that the fifth Korotkoff sound was not heard. Refusal and don't-know answers on questionnaire items were treated as missing rather than folded into a 'no' answer, since a refusal is not evidence of absence. Continuous variables were screened against wide plausibility ranges (for example 60 to 280 mmHg for systolic blood pressure) and values outside them were set to missing and counted, never silently kept or used to drop someone from the cohort.

The final predictor set has 34 variables. The full table with source file, meaning, type, missingness and the reason for inclusion or exclusion is in `outputs/predictor_table.csv`.

## 8. Leakage investigation

| variable        | category                                               | decision                         |   auc_alone |   odds_ratio |
|:----------------|:-------------------------------------------------------|:---------------------------------|------------:|-------------:|
| mcq160c_raw     | the target itself                                      | exclude                          |     nan     |      nan     |
| age_told_chd    | derived from the diagnosis                             | exclude                          |     nan     |      nan     |
| angina          | same questionnaire block, overlapping clinical entity  | exclude from the main model      |       0.638 |       30.355 |
| heart_attack    | same questionnaire block, overlapping clinical entity  | exclude from the main model      |       0.744 |       47.634 |
| chf             | same questionnaire block, downstream condition         | exclude from the main model      |       0.66  |       24.697 |
| stroke          | same questionnaire block, different vascular territory | exclude from the main model      |       0.578 |        6.845 |
| told_overweight | strongly related but legitimate                        | include                          |       0.565 |        1.707 |
| fam_hist_mi     | risk factor, not an outcome                            | include                          |       0.572 |        2.708 |
| asthma_still    | conditional question                                   | include with caution             |       0.56  |        1.703 |
| triglycerides   | sub-sample measurement                                 | include with a missing indicator |       0.564 |      nan     |
| ldl             | sub-sample measurement                                 | include with a missing indicator |       0.681 |      nan     |
| wt_int          | survey design                                          | exclude from predictors          |     nan     |      nan     |
| wt_mec          | survey design                                          | exclude from predictors          |     nan     |      nan     |
| psu             | survey design                                          | exclude from predictors          |     nan     |      nan     |
| stratum         | survey design                                          | exclude from predictors          |     nan     |      nan     |
| sddsrvyr        | survey administration                                  | exclude from predictors          |     nan     |      nan     |
| exam_status     | eligibility flag                                       | exclude from predictors          |     nan     |      nan     |
| seqn            | identifier                                             | exclude from predictors          |     nan     |      nan     |

Effect of allowing the excluded cardiovascular questions back in (cross-validated on the development data):

| feature_set                               |   n_features |   roc_auc_mean |   pr_auc_mean |
|:------------------------------------------|-------------:|---------------:|--------------:|
| main predictor set                        |           34 |         0.8717 |        0.2325 |
| main set plus the CVD questionnaire block |           38 |         0.9329 |        0.507  |

Adding that block changes PR-AUC by about 0.275. That higher number only shows what happens when the model is allowed to use a neighbouring self-report of heart disease - a heart attack or angina diagnosis is not really separate information from a CHD diagnosis, and a model leaning on it is mostly re-deriving one label from another. Reporting that figure as the project's result would overstate what can be done with genuinely prospective risk information, so it is kept here only to size the effect, not as a candidate for the final model.

## 9. Missing data

| variable             |   missing_pct |   missing_pct_chd |   missing_pct_no_chd |   missing_vs_chd_p |
|:---------------------|--------------:|------------------:|---------------------:|-------------------:|
| asthma_still         |        0.8532 |            0.8116 |               0.855  |             0.0004 |
| ldl                  |        0.5612 |            0.5505 |               0.5617 |             0.5366 |
| triglycerides        |        0.555  |            0.5392 |               0.5557 |             0.3522 |
| income_poverty_ratio |        0.0985 |            0.1056 |               0.0982 |             0.51   |
| waist                |        0.0619 |            0.1067 |               0.06   |             0      |
| total_chol           |        0.0599 |            0.0602 |               0.0598 |             1      |
| hdl                  |        0.0599 |            0.0602 |               0.0598 |             1      |
| arm_circ             |        0.0511 |            0.084  |               0.0497 |             0      |
| pulse_pressure       |        0.0424 |            0.0443 |               0.0423 |             0.8473 |
| dbp                  |        0.0424 |            0.0443 |               0.0423 |             0.8473 |
| sbp                  |        0.04   |            0.0375 |               0.0401 |             0.7643 |
| pulse                |        0.038  |            0.0352 |               0.0381 |             0.7201 |

Missing indicators were added for: ldl, triglycerides. Imputation is fitted inside each training fold, never on the whole data set.

Missingness is low for most examination and laboratory variables, generally under 6%, and mostly unrelated to CHD status - blood pressure, cholesterol and pulse all have missingness p-values close to 1. Waist and arm circumference are exceptions: both are missing noticeably more often among CHD cases than among people without CHD. That is worth naming without overclaiming a mechanism - it could reflect mobility or examination difficulty in an older, less healthy group, but this data set cannot confirm that. The triglycerides and LDL missingness, over half the cohort, is unrelated to CHD status, which matches the fact that it comes from the fasting sub-sample design rather than from anything about the participant's health.

## 10. Exploratory analysis

Figures: `figures/01_chd_prevalence.png`, `figures/02_distributions_by_chd.png`, `figures/03_missingness.png`, `figures/04_age_by_status.png`. The descriptive table by outcome is `outputs/table_one.csv`.

Three things stand out. Age is the largest difference by a wide margin - a mean gap of almost twenty years - and shapes most of what follows. Pulse pressure is markedly wider in the CHD group, consistent with stiffer arteries in an older population. Total cholesterol and LDL are, at first glance, lower among people who report CHD; this is picked up again in the statistical and interpretation sections, and the most likely explanation is treatment after diagnosis rather than a protective effect of low cholesterol.

## 11. Statistical analysis

Continuous variables, largest effect sizes first:

| variable       |   mean_chd |   mean_no_chd |   mean_difference |   diff_ci_low |   diff_ci_high |   cohens_d |   welch_p_fdr |
|:---------------|-----------:|--------------:|------------------:|--------------:|---------------:|-----------:|--------------:|
| age            |     68.661 |        48.808 |            19.853 |        19.091 |         20.615 |      1.153 |             0 |
| pulse_pressure |     65.813 |        53.215 |            12.597 |        11.127 |         14.068 |      0.717 |             0 |
| ldl            |     92.571 |       112.827 |           -20.256 |       -24.041 |        -16.472 |     -0.572 |             0 |
| total_chol     |    169.692 |       191.059 |           -21.366 |       -24.343 |        -18.39  |     -0.518 |             0 |
| waist          |    106.363 |        99.328 |             7.035 |         5.915 |          8.154 |      0.424 |             0 |
| sbp            |    132.386 |       124.527 |             7.859 |         6.352 |          9.366 |      0.42  |             0 |
| dbp            |     66.45  |        71.281 |            -4.831 |        -5.679 |         -3.983 |     -0.415 |             0 |
| pulse          |     68.574 |        72.565 |            -3.991 |        -4.779 |         -3.203 |     -0.338 |             0 |

Categorical variables:

| variable    |   odds_ratio |   or_ci_low |   or_ci_high |   cramers_v |   p_value_fdr |
|:------------|-------------:|------------:|-------------:|------------:|--------------:|
| arthritis   |        3.898 |       3.398 |        4.471 |       0.142 |             0 |
| marital     |      nan     |     nan     |      nan     |       0.114 |             0 |
| emphysema   |        5.522 |       4.242 |        7.188 |       0.096 |             0 |
| gout        |        3.772 |       3.094 |        4.598 |       0.095 |             0 |
| cancer_ever |        2.963 |       2.513 |        3.494 |       0.092 |             0 |
| race_eth    |      nan     |     nan     |      nan     |       0.09  |             0 |
| fam_hist_mi |        2.708 |       2.309 |        3.177 |       0.087 |             0 |
| sex         |        2.186 |       1.896 |        2.52  |       0.075 |             0 |

After Benjamini-Hochberg correction, 31 of 34 tested variables have an adjusted p below 0.05, but far fewer are large enough to matter on their own. Age, pulse pressure, LDL, total cholesterol, waist circumference, systolic and diastolic blood pressure, and pulse all have effect sizes in the small-to-large range and are worth discussing individually. Most of the categorical associations - sex, family history of an early heart attack, cancer, gout and several others - have a Cramer's V under 0.15, a small effect even with a p-value of essentially zero. Income-to-poverty ratio, weight and BMI are statistically significant but negligible in size and add little on their own.

## 12. Machine learning methods

Split design: stratified_random (prevalence spread across cycles 0.0110, chi-square p = 0.0401, largest availability gap 0.055).

Preprocessing runs inside every pipeline: median imputation and scaling for the linear model, median imputation only for the tree models, most frequent category and one-hot encoding for the categorical variables. Class imbalance is handled with class weights rather than resampling.

Cross-validated results on the development data:

| model               |   roc_auc_mean |   roc_auc_sd |   average_precision_mean |   average_precision_sd |   roc_auc_train_mean |
|:--------------------|---------------:|-------------:|-------------------------:|-----------------------:|---------------------:|
| logistic_regression |         0.8713 |       0.0168 |                   0.2327 |                 0.0446 |               0.8819 |
| gradient_boosting   |         0.8653 |       0.0132 |                   0.2252 |                 0.0228 |               0.9333 |
| random_forest       |         0.8665 |       0.0132 |                   0.2221 |                 0.0315 |               0.9998 |
| dummy               |         0.4989 |       0.0076 |                   0.0409 |                 0.0004 |               0.5007 |

## 13. Model comparison

| model               |   roc_auc_mean |   roc_auc_sd |   average_precision_mean |   average_precision_sd |
|:--------------------|---------------:|-------------:|-------------------------:|-----------------------:|
| logistic_regression |         0.8724 |       0.0168 |                   0.2333 |                 0.0458 |
| gradient_boosting   |         0.8653 |       0.0132 |                   0.2252 |                 0.0228 |
| random_forest       |         0.8678 |       0.0156 |                   0.2236 |                 0.0277 |

Paired comparison on the same folds:

| model_a             | model_b           | metric            |   mean_difference |   sd_difference |   paired_t_p |
|:--------------------|:------------------|:------------------|------------------:|----------------:|-------------:|
| logistic_regression | random_forest     | average_precision |            0.0096 |          0.019  |       0.3205 |
| logistic_regression | gradient_boosting | average_precision |            0.0081 |          0.032  |       0.6022 |
| random_forest       | gradient_boosting | average_precision |           -0.0015 |          0.0201 |       0.8719 |
| logistic_regression | random_forest     | roc_auc           |            0.0046 |          0.0058 |       0.1541 |
| logistic_regression | gradient_boosting | roc_auc           |            0.0071 |          0.0053 |       0.0401 |
| random_forest       | gradient_boosting | roc_auc           |            0.0025 |          0.0033 |       0.1611 |

Selected model: logistic regression. The three tuned models land within about a percentage point of each other on both metrics, and the paired comparison shows the PR-AUC differences are within the range of ordinary fold-to-fold variation (p = 0.32 against random forest, p = 0.60 against gradient boosting). Logistic regression was kept for its simplicity and interpretability rather than because it is clearly the strongest model; a reader who prefers random forest for its better test-set PPV and specificity would not be wrong.

## 14. Final evaluation

| model               | selected   |   roc_auc |   roc_auc_ci_low |   roc_auc_ci_high |   pr_auc |   pr_auc_ci_low |   pr_auc_ci_high |   brier |   sensitivity |   specificity |    ppv |    npv |
|:--------------------|:-----------|----------:|-----------------:|------------------:|---------:|----------------:|-----------------:|--------:|--------------:|--------------:|-------:|-------:|
| logistic_regression | True       |    0.8751 |           0.8543 |            0.8949 |   0.2394 |          0.205  |           0.291  |  0.1472 |        0.8182 |        0.7732 | 0.133  | 0.9901 |
| random_forest       | False      |    0.8776 |           0.858  |            0.8969 |   0.2483 |          0.2125 |           0.3008 |  0.0769 |        0.5955 |        0.9066 | 0.2134 | 0.9814 |
| gradient_boosting   | False      |    0.8756 |           0.8574 |            0.8932 |   0.2309 |          0.1974 |           0.2804 |  0.1325 |        0.7909 |        0.7792 | 0.1322 | 0.9887 |

At the threshold chosen on the development data (0.487), the selected model finds 180 of 220 participants who reported CHD, at the cost of 1173 false positives out of 5173 participants without the outcome.

Cross-validated ROC-AUC was 0.872 against 0.875 on the test set; the cross-validated value falls inside the test confidence interval.

## 15. Calibration

Raw model: calibration slope 0.892, mean predicted risk 0.2955 against an observed risk of 0.0408, Brier 0.1472.

After sigmoid recalibration fitted on the development data: slope 1.034, mean predicted risk 0.0409, Brier 0.0344.

| band         |    n |   cases |   observed_risk |   mean_predicted |   difference | version      |
|:-------------|-----:|--------:|----------------:|-----------------:|-------------:|:-------------|
| [0.0, 0.02)  |  226 |       0 |          0      |           0.0146 |      -0.0146 | raw          |
| [0.02, 0.05) |  837 |       0 |          0      |           0.035  |      -0.035  | raw          |
| [0.05, 0.1)  |  881 |       5 |          0.0057 |           0.0723 |      -0.0666 | raw          |
| [0.1, 0.2)   |  873 |       5 |          0.0057 |           0.1452 |      -0.1394 | raw          |
| [0.2, 1.01)  | 2576 |     210 |          0.0815 |           0.5321 |      -0.4506 | raw          |
| [0.0, 0.02)  | 3263 |      17 |          0.0052 |           0.0071 |      -0.0019 | recalibrated |
| [0.02, 0.05) |  928 |      28 |          0.0302 |           0.0321 |      -0.0019 | recalibrated |
| [0.05, 0.1)  |  568 |      42 |          0.0739 |           0.0711 |       0.0029 | recalibrated |
| [0.1, 0.2)   |  416 |      67 |          0.1611 |           0.1398 |       0.0213 | recalibrated |
| [0.2, 1.01)  |  218 |      66 |          0.3028 |           0.317  |      -0.0143 | recalibrated |

The raw model's probabilities should not be read as risks: the class weighting used to help the model see the minority class inflates its predicted probabilities, so a raw score of 0.5 does not mean a 50% chance of CHD. After sigmoid recalibration the numbers line up with what is actually observed - a calibration slope of 1.03 and a mean predicted risk of 4.09% against an observed 4.08% - so the recalibrated version, not the raw one, is the one to use whenever the predicted number itself matters, rather than just the ranking. The ranking, and therefore ROC-AUC and PR-AUC, is identical either way, since recalibration is a monotonic transformation.

## 16. Model interpretation

Logistic regression, largest standardised coefficients:

| feature                        |   coefficient |   odds_ratio |   or_ci_low |   or_ci_high |
|:-------------------------------|--------------:|-------------:|------------:|-------------:|
| age                            |         1.151 |        3.162 |       2.683 |        3.728 |
| sex_Male                       |         0.924 |        2.52  |       1.921 |        3.308 |
| race_eth_7.0                   |         0.563 |        1.755 |       1.021 |        3.018 |
| race_eth_3.0                   |         0.532 |        1.702 |       1.184 |        2.448 |
| missingindicator_ldl           |         0.519 |        1.68  |       0.905 |        3.117 |
| missingindicator_triglycerides |        -0.443 |        0.642 |       0.341 |        1.209 |
| race_eth_2.0                   |         0.414 |        1.513 |       0.998 |        2.294 |
| sbp                            |         0.401 |        1.494 |       0.896 |        2.492 |
| pulse_pressure                 |        -0.382 |        0.683 |       0.422 |        1.105 |
| dbp                            |        -0.379 |        0.685 |       0.491 |        0.954 |

Permutation importance, selected model:

| feature           |   importance_mean |   importance_sd |
|:------------------|------------------:|----------------:|
| age               |            0.1364 |          0.0114 |
| sex               |            0.0218 |          0.0053 |
| fam_hist_mi       |            0.015  |          0.0054 |
| waist             |            0.0142 |          0.0056 |
| dbp               |            0.0139 |          0.003  |
| pulse             |            0.0129 |          0.0021 |
| total_chol        |            0.0121 |          0.0067 |
| triglycerides     |            0.011  |          0.0031 |
| told_overweight   |            0.0068 |          0.0031 |
| fam_hist_diabetes |            0.0063 |          0.0026 |

These are associations with the model's predictions. They are not causal effects and they are not statements about biology.

Two directions are worth flagging rather than passing over. Total cholesterol and LDL both carry negative coefficients, meaning higher measured cholesterol is associated with a lower predicted probability of CHD in this model. This is very unlikely to reflect biology; the more plausible explanation is that people with a CHD diagnosis are being treated with statins, which lowers measured cholesterol after the diagnosis rather than before it. Diastolic blood pressure also carries a negative coefficient once systolic pressure and pulse pressure are already in the model - a known pattern in older populations, where arteries stiffen and diastolic pressure tends to fall even as systolic pressure and pulse pressure rise, so the model is splitting one physiological signal across correlated variables rather than contradicting itself.

## 17. Error analysis

| error_type     |    n |   mean_predicted_risk |   median_age |   median_sbp |   median_dbp |   median_bmi |   median_waist |   median_total_chol |   median_hdl |   median_pulse | most_common_sex   | most_common_race_eth   |   pct_arthritis |   pct_fam_hist_mi | most_common_survey_cycle   |
|:---------------|-----:|----------------------:|-------------:|-------------:|-------------:|-------------:|---------------:|--------------------:|-------------:|---------------:|:------------------|:-----------------------|----------------:|------------------:|:---------------------------|
| false_negative |   40 |                 0.291 |         55.5 |      130.667 |       77.333 |        30.65 |         102.6  |                 190 |           49 |             70 | Male (60%)        | 3.0 (35%)              |           0.25  |             0.079 | 2017-2018 (32%)            |
| false_positive | 1173 |                 0.706 |         70   |      130     |       68     |        29    |         104    |                 174 |           48 |             68 | Male (58%)        | 3.0 (50%)              |           0.585 |             0.213 | 2017-2018 (27%)            |
| true_negative  | 4000 |                 0.152 |         42   |      118.667 |       72     |        27.8  |          95.4  |                 193 |           52 |             72 | Female (56%)      | 3.0 (31%)              |           0.148 |             0.085 | 2013-2014 (27%)            |
| true_positive  |  180 |                 0.809 |         74   |      131.333 |       64     |        28.4  |         106.65 |                 156 |           45 |             64 | Male (72%)        | 3.0 (62%)              |           0.611 |             0.267 | 2015-2016 (29%)            |

The 40 missed cases (false negatives) are on average considerably younger than the cases the model catches - a median age of 56 against 74 for true positives, one of the largest gaps in the error analysis. They also have less family history of an early heart attack and less arthritis, and higher LDL and total cholesterol than the detected cases. One possible explanation is that these are more recent or less advanced diagnoses, without the same accumulated risk profile as the typical detected case - the data are consistent with that story without confirming it, since the cohort has no information on how long ago the diagnosis was made.

Full comparisons between error groups are in `outputs/error_group_comparison.csv`.

## 18. Robustness analysis

Repeated random splits (10 repetitions): ROC-AUC mean 0.873, sd 0.008, range 0.858 to 0.883.

Predictor sets:

| feature_set                     |   n_features |   roc_auc_mean |   roc_auc_sd |   pr_auc_mean |   pr_auc_sd |
|:--------------------------------|-------------:|---------------:|-------------:|--------------:|------------:|
| all predictors                  |           34 |         0.8717 |       0.0133 |        0.2325 |      0.027  |
| without the fasting lipids      |           32 |         0.8719 |       0.0136 |        0.2325 |      0.0259 |
| without any lipid measurement   |           30 |         0.865  |       0.012  |        0.2186 |      0.0228 |
| without the questionnaire items |           21 |         0.8544 |       0.0147 |        0.1961 |      0.007  |
| demographics only               |            7 |         0.8379 |       0.014  |        0.163  |      0.0064 |
| age and sex only                |            2 |         0.8271 |       0.0177 |        0.1467 |      0.0085 |

Training on earlier cycles and testing on later ones:

| train_cycles                  | test_cycles         |   n_test |   test_prevalence |   roc_auc |   pr_auc |
|:------------------------------|:--------------------|---------:|------------------:|----------:|---------:|
| 2011-2012;2013-2014           | 2015-2016;2017-2018 |    10699 |            0.0439 |    0.8784 |   0.2315 |
| 2011-2012;2013-2014;2015-2016 | 2017-2018           |     5249 |            0.0463 |    0.8728 |   0.2332 |

Alternative imputation, class weights and threshold choices are in `outputs/robustness_*.csv`.

## 19. Subgroup analysis

| group        | level     |    n |   cases |   roc_auc |   pr_auc |   sensitivity |   specificity |     ppv | reliable   |
|:-------------|:----------|-----:|--------:|----------:|---------:|--------------:|--------------:|--------:|:-----------|
| sex          | Female    | 2787 |      67 |     0.874 |    0.234 |         0.761 |         0.817 |   0.093 | True       |
| sex          | Male      | 2606 |     153 |     0.86  |    0.254 |         0.843 |         0.724 |   0.16  | True       |
| age_group    | 20-39     | 1796 |       3 |   nan     |  nan     |       nan     |       nan     | nan     | False      |
| age_group    | 40-49     |  893 |       8 |   nan     |  nan     |       nan     |       nan     | nan     | False      |
| age_group    | 50-59     |  900 |      26 |     0.703 |    0.062 |         0.385 |         0.84  |   0.067 | True       |
| age_group    | 60-69     |  925 |      57 |     0.764 |    0.165 |         0.807 |         0.555 |   0.106 | True       |
| age_group    | 70+       |  879 |     126 |     0.757 |    0.312 |         0.984 |         0.179 |   0.167 | True       |
| race_eth     | 1.0       |  731 |      25 |     0.921 |    0.338 |         0.84  |         0.844 |   0.16  | True       |
| race_eth     | 2.0       |  551 |      15 |     0.912 |    0.241 |         0.867 |         0.791 |   0.104 | True       |
| race_eth     | 3.0       | 1973 |     126 |     0.861 |    0.276 |         0.889 |         0.682 |   0.16  | True       |
| race_eth     | 4.0       | 1191 |      31 |     0.807 |    0.159 |         0.613 |         0.793 |   0.073 | True       |
| race_eth     | 6.0       |  748 |      19 |     0.866 |    0.29  |         0.579 |         0.888 |   0.118 | True       |
| race_eth     | 7.0       |  199 |       4 |   nan     |  nan     |       nan     |       nan     | nan     | False      |
| survey_cycle | 2011-2012 | 1321 |      49 |     0.852 |    0.251 |         0.755 |         0.795 |   0.124 | True       |
| survey_cycle | 2013-2014 | 1451 |      50 |     0.851 |    0.182 |         0.78  |         0.78  |   0.112 | True       |
| survey_cycle | 2015-2016 | 1341 |      56 |     0.92  |    0.31  |         0.929 |         0.777 |   0.154 | True       |
| survey_cycle | 2017-2018 | 1280 |      65 |     0.866 |    0.254 |         0.8   |         0.738 |   0.141 | True       |

3 subgroups had too few cases for a stable estimate and are reported with their size only. Among the rest, ROC-AUC spans about 0.22, from 0.70 to 0.92. With between roughly 15 and 150 cases per subgroup, a spread of this size is well within what sampling variation alone would produce, and none of it is read here as evidence that the model works differently for different groups.

## 20. Survey design and population estimates

| group        | level     |     n |   unweighted_prevalence |   weighted_prevalence |   ci_low |   ci_high |
|:-------------|:----------|------:|------------------------:|----------------------:|---------:|----------:|
| overall      | all       | 21570 |                  0.0408 |                0.0358 |   0.0312 |    0.0405 |
| sex          | Female    | 11165 |                  0.0265 |                0.0252 |   0.0207 |    0.0298 |
| sex          | Male      | 10405 |                  0.0562 |                0.0473 |   0.0402 |    0.0543 |
| age_group    | 20-39     |  7208 |                  0.0024 |                0.0023 |   0.0008 |    0.0038 |
| age_group    | 40-49     |  3569 |                  0.0112 |                0.0092 |   0.0055 |    0.0129 |
| age_group    | 50-59     |  3555 |                  0.0273 |                0.0264 |   0.0189 |    0.0339 |
| age_group    | 60-69     |  3739 |                  0.0655 |                0.0747 |   0.0595 |    0.0899 |
| age_group    | 70+       |  3499 |                  0.1378 |                0.1373 |   0.1218 |    0.1528 |
| survey_cycle | 2011-2012 |  5299 |                  0.0353 |                0.03   |   0.0211 |    0.0389 |
| survey_cycle | 2013-2014 |  5572 |                  0.0402 |                0.0365 |   0.028  |    0.045  |
| survey_cycle | 2015-2016 |  5450 |                  0.0417 |                0.035  |   0.0277 |    0.0423 |
| survey_cycle | 2017-2018 |  5249 |                  0.0463 |                0.0414 |   0.0295 |    0.0533 |

The weighted numbers are population estimates using the masked strata and PSUs. The model results are unweighted and describe the analytical sample. No claim is made about nationally representative model performance.

## 21. Limitations

- The outcome is self-reported and depends on having seen a doctor and remembering the diagnosis. Undiagnosed disease is counted as absence.
- The data are cross-sectional, so predictors and outcome are recorded at the same visit. Nothing here supports a causal or a prognostic reading.
- Smoking, diabetes and medication files were not available locally, so three of the strongest known risk factors are missing from the model.
- Several laboratory values come from the fasting sub-sample, so they are missing for more than half the cohort.
- The class-weighted model's raw probabilities are not directly interpretable as risks; recalibration is needed before the predicted numbers themselves are used for anything, even though the ranking they produce is unaffected.
- Age alone accounts for a large share of the discrimination (age and sex alone reach a ROC-AUC of 0.83, against 0.87 for the full set), so the added value of the clinical and laboratory variables, while real, is more modest than the headline number on its own suggests.

## 22. Conclusion

The predictors available in this NHANES subset separate adults with and without self-reported CHD reasonably well, with a ROC-AUC around 0.88 and a PR-AUC of 0.24 against a base rate of about 4%. Most of that separation comes from age, and much of the rest comes from the same handful of cardiovascular variables that clinical guidance already relies on, so the model is confirming expected relationships rather than finding anything new. The gap between this and anything resembling a diagnostic tool stays large: the outcome is a self-report, smoking and diabetes are not part of this predictor set, and even the most favourable operating point in this report accepts several false positives for every true one.

## Appendix: audit

51 automated checks were run: 51 passed, 0 failed. Full list in `reports/audit_report.md`.
