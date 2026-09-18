# EDA notes

The cohort has 21570 participants from four NHANES cycles, 881 of them reporting a CHD diagnosis (4.08%).

Prevalence by cycle:

| survey_cycle   |    n |   cases |   prevalence |
|:---------------|-----:|--------:|-------------:|
| 2011-2012      | 5299 |     187 |    0.0352897 |
| 2013-2014      | 5572 |     224 |    0.040201  |
| 2015-2016      | 5450 |     227 |    0.0416514 |
| 2017-2018      | 5249 |     243 |    0.0462945 |

Prevalence by age group:

| age_group   |    n |   cases |   prevalence |
|:------------|-----:|--------:|-------------:|
| 20-39       | 7208 |      17 |   0.00235849 |
| 40-49       | 3569 |      40 |   0.0112076  |
| 50-59       | 3555 |      97 |   0.0272855  |
| 60-69       | 3739 |     245 |   0.0655255  |
| 70+         | 3499 |     482 |   0.137754   |

Variables with |skewness| above 1: bmi, weight, hdl, triglycerides, sbp, pulse_pressure.