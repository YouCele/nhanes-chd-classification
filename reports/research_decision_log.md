# Research decision log

Every methodological choice that could have gone another way, with the reason.
Numbers are not repeated here; they live in `outputs/` and in the final report,
which is generated from those files.

## Target

| item | decision |
|---|---|
| variable | `MCQ160C`, from the Medical Conditions questionnaire |
| wording | "Has a doctor or other health professional ever told you that you had coronary heart disease?" |
| asked of | participants aged 20 and over |
| mapping | 1 (yes) -> 1, 2 (no) -> 0 |
| refused (7), don't know (9), blank | excluded from the analysis, not mapped to 0 |
| why | a refusal is not evidence that the person has no diagnosis; mapping it to 0 would invent information and would push prevalence down |
| what the label is | self-reported, physician-diagnosed CHD. Not angiography, not a chart review, not a clinical gold standard |

## Cohort

| item | decision | reason |
|---|---|---|
| age | 20 and over | the question is not asked below 20, so younger participants have no outcome |
| examination status | MEC-examined participants only (`RIDSTATR == 2`) | blood pressure and body measurements are part of the intended predictor set and interview-only participants never have them |
| outcome | must be usable (yes or no) | see above |
| predictors | no complete-case restriction | requiring the fasting lipids would remove more than half of the sample and would change the population being studied |
| people with no examination data at all | kept, counted in the flow table | they are few, and dropping them would be a silent extra exclusion |

## Leakage

| variable | decision | reason |
|---|---|---|
| `MCQ160C` | excluded | the outcome itself |
| `MCQ180C` (age told had CHD) | excluded | only recorded for people who answered yes, so it restates the label; also missing from the 2017-2018 file |
| `MCQ160D` angina | excluded from the main model | same questionnaire block; angina is a presentation of coronary disease rather than a separate risk factor |
| `MCQ160E` heart attack | excluded from the main model | myocardial infarction is part of coronary heart disease |
| `MCQ160B` heart failure | excluded from the main model | frequently a consequence of coronary disease, asked in the same block |
| `MCQ160F` stroke | excluded from the main model | a different event, so a weaker case, but the whole block is treated the same way for consistency |
| the same block | quantified, not just removed | `outputs/robustness_leakage_sensitivity.csv` shows how much easier the task becomes when these questions are allowed back in |
| survey weights, PSU, stratum, cycle number, SEQN | excluded from predictors | they describe the sampling design or the administration of the survey, not the participant |
| `MCQ080` told overweight, `MCQ300A` family history | kept | contact with the health system and family history are legitimate information; neither is a statement about the participant's own heart disease |

## Missing data

| item | decision | reason |
|---|---|---|
| special codes | 7/9 and 77/99 turned into missing before anything else | they are not numeric values |
| implausible values | set to missing and counted, per variable ranges in `config.PLAUSIBLE_RANGES` | wide ranges, so genuine biological extremes are kept |
| diastolic readings of 0 | treated as missing | 0 means the fifth Korotkoff sound was not heard |
| numeric imputation | median, fitted inside each training fold | simple, robust to the skewed lipid variables |
| categorical imputation | most frequent category, also inside the fold | |
| missing indicators | only for variables whose missingness is structural (the fasting sub-sample) or high and related to the outcome | adding indicators everywhere would add near-empty columns for variables missing in under 1% of rows |
| variables missing for more than 60% of the cohort | dropped, with the number recorded in the predictor table | too sparse to support a model |
| leakage control | no imputation parameter is ever computed on the full data set | the whole preprocessing chain sits inside the pipeline |
| sensitivity | median compared with mean imputation | `outputs/robustness_imputation.csv` |

## Design and evaluation

| item | decision | reason |
|---|---|---|
| split | decided by a rule written before modelling: cycles treated as exchangeable unless prevalence differs materially or a predictor is missing from a whole cycle | recorded in `outputs/split_design.json` together with the numbers that triggered it |
| test size | 25%, stratified on the outcome | |
| cycle-based evaluation | kept as a robustness check | the survey period is still a plausible source of shift even when the split is random |
| class imbalance | class weights inside the estimators | no synthetic participants, and nothing to leak across folds; SMOTE was not used because it adds a resampling step whose only benefit here would be cosmetic |
| primary metric | PR-AUC, with ROC-AUC alongside | prevalence is low, so precision and recall carry more information than the ROC alone |
| accuracy | not used as a headline number | predicting "no CHD" for everyone would already look accurate |
| threshold | Youden's J on cross-validated development predictions | chosen before the test set is touched |
| confidence intervals | stratified percentile bootstrap on the test set | |
| calibration | reported for the raw model and after a sigmoid recalibration fitted on the development data | a class-weighted model is trained to separate, not to produce risks |

## Models

| item | decision | reason |
|---|---|---|
| models | dummy, logistic regression, random forest, histogram gradient boosting | enough to answer the question; a longer list would be a leaderboard |
| tuning | small grids, cross-validated on the development data only | the hardware target is a modest laptop |
| selection | cross-validated PR-AUC, with a paired fold comparison against the runner-up | a small difference between models is reported as small |
| interpretation model | a separate, almost unpenalised logistic regression without class weights | Wald intervals from a penalised, weighted fit would be hard to read |

## Survey design

| item | decision | reason |
|---|---|---|
| weighted estimates | used for descriptive prevalence only, with design-based standard errors from the masked strata and PSUs | that is a population question |
| model evaluation | unweighted | the classifier is evaluated inside the analytical sample; weighting a classifier is a separate methodological question and is not claimed here |
| generalisation | no claim of nationally representative model performance | |

## Things that stayed unresolved

* Smoking, diabetes, glycohaemoglobin and the blood-pressure questionnaire are
  not in the local file set, so the model has no smoking or diabetes variable.
  This is a real limitation and is not worked around by a proxy.
* The outcome is self-reported. Participants who have coronary disease but have
  never been told so are counted as negatives.
* Cholesterol and pulse are lower in the CHD group. Treatment is the obvious
  candidate explanation, but the data cannot confirm it because no medication
  file was downloaded.
