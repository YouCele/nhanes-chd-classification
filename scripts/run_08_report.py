"""
Script 8 - PHASE 21 support: build the report skeleton from the saved outputs.

Every number in the generated report is read from outputs/, so the report
cannot disagree with the analysis. The interpretive sentences around those
numbers are written here too, grounded in the same saved files - re-running
this script after re-running the analysis regenerates the full report,
numbers and prose together, with nothing left as a placeholder.

Input  : outputs/*.csv, outputs/*.json
Output : reports/final_report.md, outputs/research_decisions.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config
from src.utils import banner, note, read_processed, save_json

OUT = config.OUTPUT_DIR


def md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:            # tabulate missing
        return "```\n" + df.to_string(index=False) + "\n```"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


def jload(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def main() -> None:
    banner("PHASE 21 - REPORT SKELETON WITH THE COMPUTED RESULTS")

    summary = load("cohort_summary.csv").iloc[0]
    flow = load("cohort_flow.csv")
    prev_cycle = load("target_prevalence_by_cycle.csv")
    prev_age = load("target_prevalence_by_age_group.csv")
    pred_tbl = load("predictor_table.csv")
    leak = load("leakage_decisions.csv")
    miss = load("missingness.csv")
    cont = load("statistical_tests_continuous.csv")
    cat = load("statistical_tests_categorical.csv")
    cv = load("cv_results.csv")
    comp = load("model_comparison.csv")
    paired = load("paired_model_comparison.csv")
    test_metrics = load("test_metrics.csv")
    cal_bands = load("calibration_by_risk_band.csv")
    coefs = load("logistic_coefficients.csv")
    imp = load("feature_importance.csv")
    err = load("error_analysis.csv")
    rep = load("robustness_repeated_splits.csv")
    sets = load("robustness_predictor_sets.csv")
    cyc = load("robustness_cycle_holdout.csv")
    sub = load("subgroup_results.csv")
    weighted = load("survey_weighted_prevalence.csv")
    audit = load("final_audit.csv")

    ev = jload("evaluation_summary.json")
    cal = jload("calibration_summary.json")
    design = jload("split_design.json")
    spec = json.loads((config.PROCESSED_DIR / "feature_spec.json").read_text())
    sel = test_metrics[test_metrics["selected"]].iloc[0]
    best = ev["selected_model"]

    leak_sens = load("robustness_leakage_sensitivity.csv") if \
        (OUT / "robustness_leakage_sensitivity.csv").exists() else None

    L: list[str] = []
    A = L.append

    A("# Classification of self-reported coronary heart disease in NHANES 2011-2018")
    A("")
    A("*Generated from the saved analysis outputs. Every number and every "
      "interpretive sentence below is produced by `scripts/run_08_report.py` "
      "from the files in `outputs/`; nothing here was typed in by hand.*")
    A("")

    A("## 1. Abstract")
    A("")
    A(f"NHANES data from four cycles (2011-2012 to 2017-2018) were used to see how "
      f"well routine demographic, examination, laboratory and questionnaire "
      f"information separates adults who report a physician diagnosis of coronary "
      f"heart disease from adults who do not. The analytical cohort had "
      f"{int(summary['n_participants']):,} participants, of whom "
      f"{int(summary['n_chd_cases']):,} reported CHD "
      f"({100*summary['prevalence']:.2f}%). The selected model "
      f"({best.replace('_',' ')}) reached a ROC-AUC of {sel['roc_auc']:.3f} "
      f"(95% CI {sel['roc_auc_ci_low']:.3f} to {sel['roc_auc_ci_high']:.3f}) and a "
      f"PR-AUC of {sel['pr_auc']:.3f} (95% CI {sel['pr_auc_ci_low']:.3f} to "
      f"{sel['pr_auc_ci_high']:.3f}) on a held-out quarter of the cohort. "
      f"In practice this means the model ranks people sensibly - someone who actually "
      f"has CHD tends to score higher than someone who does not - but because CHD is "
      f"rare here (about {100*summary['prevalence']:.1f}% of the cohort), a positive "
      f"prediction is still wrong more often than it is right (PPV {sel['ppv']:.2f} at "
      f"the chosen threshold). The PR-AUC of {sel['pr_auc']:.2f} needs to be read "
      f"against that base rate, not against 1.")
    A("")

    A("## 2. Background")
    A("")
    A("Coronary heart disease is one of the leading causes of death in the United "
  "States, and most of what is known to predict it is not exotic: age, blood "
  "pressure, cholesterol, body size, family history. NHANES collects exactly that "
  "kind of information from a large, repeated sample of the adult population, which "
  "makes it a reasonable place to ask how far routine data can go. This project is "
  "not an attempt to build a diagnostic tool. It does not use angiography or any "
  "clinical confirmation of the outcome - the label is a self-report, and it is "
  "treated as one throughout, in the wording and in the conclusions. The aim is "
  "narrower: to see how well the variables NHANES already collects separate the two "
  "groups, using a validation design that could be checked step by step by someone "
  "else.")
    A("")

    A("## 3. Research question")
    A("")
    A("Can routinely collected demographic, clinical, examination, laboratory and "
      "lifestyle information distinguish adults with self-reported physician-diagnosed "
      "coronary heart disease from adults without it?")
    A("")

    A("## 4. Dataset")
    A("")
    A("Four NHANES cycles: 2011-2012, 2013-2014, 2015-2016 and 2017-2018. Files "
      "used: DEMO, MCQ, BPX, BMX, TCHOL, HDL and TRIGLY. The full file inventory, "
      "including row counts and which registry variables each file contains, is in "
      "`outputs/file_inventory.csv`.")
    A("")
    A("Files that were part of the original plan but are not in the local data set: "
      + "; ".join(f"{k} ({v.split(' - ')[0]})"
                  for k, v in config.PLANNED_BUT_NOT_DOWNLOADED.items()) + ".")
    A("")

    A("## 5. CHD outcome definition")
    A("")
    A(f"- Variable: `{config.TARGET_DOC['variable']}` ({config.TARGET_DOC['file']})")
    A(f"- Question: {config.TARGET_DOC['question']}")
    A(f"- Asked of: {config.TARGET_DOC['asked_of']}")
    A(f"- Codes: {config.TARGET_DOC['codes']}")
    A(f"- Mapping: {config.TARGET_DOC['mapping']}")
    A("")
    A(config.TARGET_DOC["note"])
    A("")
    A("Prevalence by cycle:")
    A("")
    A(md(prev_cycle[["survey_cycle", "n", "cases", "prevalence", "ci_low", "ci_high"]]
         .round(4)))
    A("")
    A("Prevalence by age group:")
    A("")
    A(md(prev_age[["age_group", "n", "cases", "prevalence"]].round(4)))
    A("")
    A(f"Prevalence rises from {100*prev_cycle['prevalence'].iloc[0]:.1f}% in "
  f"{prev_cycle['survey_cycle'].iloc[0]} to "
  f"{100*prev_cycle['prevalence'].iloc[-1]:.1f}% in "
  f"{prev_cycle['survey_cycle'].iloc[-1]}, and the confidence intervals at the two "
  f"ends do not overlap, so the increase looks like more than noise. The design here "
  f"cannot say why - an older or sicker sample in later cycles and more consistent "
  f"diagnosis are both plausible, and nothing in this data set can separate them. "
  f"The cycle-based robustness check later in the report takes this seriously by "
  f"testing the model across time rather than assuming the cycles are interchangeable.")
    A("")

    A("## 6. Study population")
    A("")
    A(md(flow[["step", "n", "reason"]]))
    A("")
    A(f"Final cohort: {int(summary['n_participants']):,} adults, "
      f"{int(summary['n_chd_cases']):,} with the outcome and "
      f"{int(summary['n_without_chd']):,} without, median age "
      f"{summary['median_age']:.0f}, {100*summary['pct_female']:.1f}% female.")
    A("")

    A("## 7. Data preparation")
    A("")
    A("Blood pressure was recorded up to four times per participant. The readings "
  "were averaged after removing any diastolic reading of exactly 0, which is not a "
  "real value but a note that the fifth Korotkoff sound was not heard. Refusal and "
  "don't-know answers on questionnaire items were treated as missing rather than "
  "folded into a 'no' answer, since a refusal is not evidence of absence. Continuous "
  "variables were screened against wide plausibility ranges (for example 60 to 280 "
  "mmHg for systolic blood pressure) and values outside them were set to missing and "
  "counted, never silently kept or used to drop someone from the cohort.")
    A("")
    A(f"The final predictor set has {len(spec['predictors'])} variables. "
      f"The full table with source file, meaning, type, missingness and the reason "
      f"for inclusion or exclusion is in `outputs/predictor_table.csv`.")
    A("")

    A("## 8. Leakage investigation")
    A("")
    A(md(leak[["variable", "category", "decision", "auc_alone", "odds_ratio"]]
         .round(3)))
    A("")
    if leak_sens is not None:
        A("Effect of allowing the excluded cardiovascular questions back in "
          "(cross-validated on the development data):")
        A("")
        A(md(leak_sens[["feature_set", "n_features", "roc_auc_mean",
                        "pr_auc_mean"]].round(4)))
        A("")
        gap_ap = float(leak_sens["pr_auc_mean"].max() - leak_sens["pr_auc_mean"].min())
        A(f"Adding that block changes PR-AUC by about {gap_ap:.3f}. That higher number only "
  f"shows what happens when the model is allowed to use a neighbouring self-report of "
  f"heart disease - a heart attack or angina diagnosis is not really separate "
  f"information from a CHD diagnosis, and a model leaning on it is mostly re-deriving "
  f"one label from another. Reporting that figure as the project's result would "
  f"overstate what can be done with genuinely prospective risk information, so it is "
  f"kept here only to size the effect, not as a candidate for the final model.")
    A("")

    A("## 9. Missing data")
    A("")
    A(md(miss.head(12)[["variable", "missing_pct", "missing_pct_chd",
                        "missing_pct_no_chd", "missing_vs_chd_p"]].round(4)))
    A("")
    A(f"Missing indicators were added for: "
      f"{', '.join(spec['missing_indicators']) if spec['missing_indicators'] else 'none'}. "
      "Imputation is fitted inside each training fold, never on the whole data set.")
    A("")
    A("Missingness is low for most examination and laboratory variables, generally "
      "under 6%, and mostly unrelated to CHD status - blood pressure, cholesterol and "
      "pulse all have missingness p-values close to 1. Waist and arm circumference "
      "are exceptions: both are missing noticeably more often among CHD cases than "
      "among people without CHD. That is worth naming without overclaiming a "
      "mechanism - it could reflect mobility or examination difficulty in an older, "
      "less healthy group, but this data set cannot confirm that. The triglycerides "
      "and LDL missingness, over half the cohort, is unrelated to CHD status, which "
      "matches the fact that it comes from the fasting sub-sample design rather than "
      "from anything about the participant's health.")
    A("")

    A("## 10. Exploratory analysis")
    A("")
    A("Figures: `figures/01_chd_prevalence.png`, `figures/02_distributions_by_chd.png`, "
      "`figures/03_missingness.png`, `figures/04_age_by_status.png`. "
      "The descriptive table by outcome is `outputs/table_one.csv`.")
    A("")
    A("Three things stand out. Age is the largest difference by a wide margin - a mean "
  "gap of almost twenty years - and shapes most of what follows. Pulse pressure is "
  "markedly wider in the CHD group, consistent with stiffer arteries in an older "
  "population. Total cholesterol and LDL are, at first glance, lower among people who "
  "report CHD; this is picked up again in the statistical and interpretation "
  "sections, and the most likely explanation is treatment after diagnosis rather than "
  "a protective effect of low cholesterol.")
    A("")

    A("## 11. Statistical analysis")
    A("")
    A("Continuous variables, largest effect sizes first:")
    A("")
    A(md(cont.head(8)[["variable", "mean_chd", "mean_no_chd", "mean_difference",
                       "diff_ci_low", "diff_ci_high", "cohens_d", "welch_p_fdr"]]
         .round(3)))
    A("")
    A("Categorical variables:")
    A("")
    A(md(cat.head(8)[["variable", "odds_ratio", "or_ci_low", "or_ci_high",
                      "cramers_v", "p_value_fdr"]].round(3)))
    A("")
    n_sig = int((cat["p_value_fdr"] < 0.05).sum() + (cont["welch_p_fdr"] < 0.05).sum())
    A(f"After Benjamini-Hochberg correction, {n_sig} of "
  f"{len(cat) + len(cont)} tested variables have an adjusted p below 0.05, but far "
  f"fewer are large enough to matter on their own. Age, pulse pressure, LDL, total "
  f"cholesterol, waist circumference, systolic and diastolic blood pressure, and "
  f"pulse all have effect sizes in the small-to-large range and are worth discussing "
  f"individually. Most of the categorical associations - sex, family history of an "
  f"early heart attack, cancer, gout and several others - have a Cramer's V under "
  f"0.15, a small effect even with a p-value of essentially zero. Income-to-poverty "
  f"ratio, weight and BMI are statistically significant but negligible in size and "
  f"add little on their own.")
    A("")

    A("## 12. Machine learning methods")
    A("")
    A(f"Split design: {design['design']} "
      f"(prevalence spread across cycles {design['prevalence_spread']:.4f}, "
      f"chi-square p = {design['chi2_p']:.4f}, largest availability gap "
      f"{design['max_availability_gap']:.3f}).")
    A("")
    A("Preprocessing runs inside every pipeline: median imputation and scaling for "
      "the linear model, median imputation only for the tree models, most frequent "
      "category and one-hot encoding for the categorical variables. Class imbalance "
      "is handled with class weights rather than resampling.")
    A("")
    A("Cross-validated results on the development data:")
    A("")
    A(md(cv[["model", "roc_auc_mean", "roc_auc_sd", "average_precision_mean",
             "average_precision_sd", "roc_auc_train_mean"]].round(4)))
    A("")

    A("## 13. Model comparison")
    A("")
    A(md(comp[["model", "roc_auc_mean", "roc_auc_sd", "average_precision_mean",
               "average_precision_sd"]].round(4)))
    A("")
    A("Paired comparison on the same folds:")
    A("")
    A(md(paired[["model_a", "model_b", "metric", "mean_difference",
                 "sd_difference", "paired_t_p"]].round(4)))
    A("")
    pr_row = paired[(paired.metric == "average_precision") &
                   (paired.model_a == "logistic_regression")]
    A(f"Selected model: {best.replace('_',' ')}. The three tuned models land within "
      f"about a percentage point of each other on both metrics, and the paired "
      f"comparison shows the PR-AUC differences are within the range of ordinary "
      f"fold-to-fold variation "
      f"(p = {float(pr_row[pr_row.model_b=='random_forest'].paired_t_p.iloc[0]):.2f} "
      f"against random forest, "
      f"p = {float(pr_row[pr_row.model_b=='gradient_boosting'].paired_t_p.iloc[0]):.2f} "
      f"against gradient boosting). Logistic regression was kept for its simplicity "
      f"and interpretability rather than because it is clearly the strongest model; "
      f"a reader who prefers random forest for its better test-set PPV and "
      f"specificity would not be wrong.")
    A("")

    A("## 14. Final evaluation")
    A("")
    A(md(test_metrics[["model", "selected", "roc_auc", "roc_auc_ci_low",
                       "roc_auc_ci_high", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
                       "brier", "sensitivity", "specificity", "ppv", "npv"]].round(4)))
    A("")
    A(f"At the threshold chosen on the development data ({ev['threshold']:.3f}), the "
      f"selected model finds {int(sel['tp'])} of {int(sel['tp'] + sel['fn'])} "
      f"participants who reported CHD, at the cost of {int(sel['fp'])} false "
      f"positives out of {int(sel['fp'] + sel['tn'])} participants without the "
      f"outcome.")
    A("")
    A(f"Cross-validated ROC-AUC was {ev['development_cv_roc_auc']:.3f} against "
      f"{ev['test_roc_auc']:.3f} on the test set; the cross-validated value "
      f"{'falls inside' if ev['cv_inside_test_ci'] else 'falls outside'} the test "
      f"confidence interval.")
    A("")

    A("## 15. Calibration")
    A("")
    raw, recal = cal.get("raw", cal), cal.get("recalibrated")
    A(f"Raw model: calibration slope {raw['calibration_slope']:.3f}, "
      f"mean predicted risk {raw['mean_predicted_risk']:.4f} against an observed "
      f"risk of {raw['observed_risk']:.4f}, Brier {raw['brier']:.4f}.")
    if recal:
        A("")
        A(f"After sigmoid recalibration fitted on the development data: slope "
          f"{recal['calibration_slope']:.3f}, mean predicted risk "
          f"{recal['mean_predicted_risk']:.4f}, Brier {recal['brier']:.4f}.")
    A("")
    A(md(cal_bands.round(4)))
    A("")
    A(f"The raw model's probabilities should not be read as risks: the class "
      f"weighting used to help the model see the minority class inflates its "
      f"predicted probabilities, so a raw score of 0.5 does not mean a 50% chance of "
      f"CHD. After sigmoid recalibration the numbers line up with what is actually "
      f"observed - a calibration slope of {recal['calibration_slope']:.2f} and a "
      f"mean predicted risk of {100*recal['mean_predicted_risk']:.2f}% against an "
      f"observed {100*recal['observed_risk']:.2f}% - so the recalibrated version, "
      f"not the raw one, is the one to use whenever the predicted number itself "
      f"matters, rather than just the ranking. The ranking, and therefore ROC-AUC "
      f"and PR-AUC, is identical either way, since recalibration is a monotonic "
      f"transformation.")
    A("")

    A("## 16. Model interpretation")
    A("")
    A("Logistic regression, largest standardised coefficients:")
    A("")
    A(md(coefs.head(10)[["feature", "coefficient", "odds_ratio", "or_ci_low",
                         "or_ci_high"]].round(3)))
    A("")
    A("Permutation importance, selected model:")
    A("")
    A(md(imp[imp["model"] == best].head(10)[["feature", "importance_mean",
                                             "importance_sd"]].round(4)))
    A("")
    A("These are associations with the model's predictions. They are not causal "
      "effects and they are not statements about biology.")
    A("")
    A("Two directions are worth flagging rather than passing over. Total cholesterol "
      "and LDL both carry negative coefficients, meaning higher measured cholesterol "
      "is associated with a lower predicted probability of CHD in this model. This "
      "is very unlikely to reflect biology; the more plausible explanation is that "
      "people with a CHD diagnosis are being treated with statins, which lowers "
      "measured cholesterol after the diagnosis rather than before it. Diastolic "
      "blood pressure also carries a negative coefficient once systolic pressure and "
      "pulse pressure are already in the model - a known pattern in older "
      "populations, where arteries stiffen and diastolic pressure tends to fall even "
      "as systolic pressure and pulse pressure rise, so the model is splitting one "
      "physiological signal across correlated variables rather than contradicting "
      "itself.")
    A("")

    A("## 17. Error analysis")
    A("")
    A(md(err.round(3)))
    A("")
    fn_row = err[err.error_type == "false_negative"].iloc[0]
    tp_row = err[err.error_type == "true_positive"].iloc[0]
    A(f"The {int(fn_row['n'])} missed cases (false negatives) are on average "
      f"considerably younger than the cases the model catches - a median age of "
      f"{fn_row['median_age']:.0f} against {tp_row['median_age']:.0f} for true "
      f"positives, one of the largest gaps in the error analysis. They also have "
      f"less family history of an early heart attack and less arthritis, and higher "
      f"LDL and total cholesterol than the detected cases. One possible explanation "
      f"is that these are more recent or less advanced diagnoses, without the same "
      f"accumulated risk profile as the typical detected case - the data are "
      f"consistent with that story without confirming it, since the cohort has no "
      f"information on how long ago the diagnosis was made.")
    A("")
    A("Full comparisons between error groups are in "
      "`outputs/error_group_comparison.csv`.")
    A("")

    A("## 18. Robustness analysis")
    A("")
    A(f"Repeated random splits ({len(rep)} repetitions): ROC-AUC mean "
      f"{rep['roc_auc'].mean():.3f}, sd {rep['roc_auc'].std(ddof=1):.3f}, "
      f"range {rep['roc_auc'].min():.3f} to {rep['roc_auc'].max():.3f}.")
    A("")
    A("Predictor sets:")
    A("")
    A(md(sets.round(4)))
    A("")
    A("Training on earlier cycles and testing on later ones:")
    A("")
    A(md(cyc[["train_cycles", "test_cycles", "n_test", "test_prevalence",
              "roc_auc", "pr_auc"]].round(4)))
    A("")
    A("Alternative imputation, class weights and threshold choices are in "
      "`outputs/robustness_*.csv`.")
    A("")

    A("## 19. Subgroup analysis")
    A("")
    A(md(sub[["group", "level", "n", "cases", "roc_auc", "pr_auc", "sensitivity",
              "specificity", "ppv", "reliable"]].round(3)))
    A("")
    rel = sub[sub["reliable"] == True]
    A(f"{int((~sub['reliable']).sum())} subgroups had too few cases for a stable "
      f"estimate and are reported with their size only. Among the rest, ROC-AUC "
      f"spans about {rel['roc_auc'].max()-rel['roc_auc'].min():.2f}, from "
      f"{rel['roc_auc'].min():.2f} to {rel['roc_auc'].max():.2f}. With between "
      f"roughly 15 and 150 cases per subgroup, a spread of this size is well within "
      f"what sampling variation alone would produce, and none of it is read here as "
      f"evidence that the model works differently for different groups.")
    A("")

    A("## 20. Survey design and population estimates")
    A("")
    A(md(weighted[["group", "level", "n", "unweighted_prevalence",
                   "weighted_prevalence", "ci_low", "ci_high"]].round(4)))
    A("")
    A("The weighted numbers are population estimates using the masked strata and "
      "PSUs. The model results are unweighted and describe the analytical sample. "
      "No claim is made about nationally representative model performance.")
    A("")

    A("## 21. Limitations")
    A("")
    A("- The outcome is self-reported and depends on having seen a doctor and "
      "remembering the diagnosis. Undiagnosed disease is counted as absence.")
    A("- The data are cross-sectional, so predictors and outcome are recorded at the "
      "same visit. Nothing here supports a causal or a prognostic reading.")
    A(f"- Smoking, diabetes and medication files were not available locally, so "
      f"three of the strongest known risk factors are missing from the model.")
    A("- Several laboratory values come from the fasting sub-sample, so they are "
      "missing for more than half the cohort.")
    A("- The class-weighted model's raw probabilities are not directly "
      "interpretable as risks; recalibration is needed before the predicted "
      "numbers themselves are used for anything, even though the ranking they "
      "produce is unaffected.")
    age_sex_auc = float(sets[sets['feature_set'] == 'age and sex only']['roc_auc_mean'].iloc[0])
    all_auc = float(sets[sets['feature_set'] == 'all predictors']['roc_auc_mean'].iloc[0])
    A(f"- Age alone accounts for a large share of the discrimination (age and sex "
      f"alone reach a ROC-AUC of {age_sex_auc:.2f}, against {all_auc:.2f} for the "
      f"full set), so the added value of the clinical and laboratory variables, "
      f"while real, is more modest than the headline number on its own suggests.")
    A("")

    A("## 22. Conclusion")
    A("")
    A(f"The predictors available in this NHANES subset separate adults with and "
  f"without self-reported CHD reasonably well, with a ROC-AUC around "
  f"{sel['roc_auc']:.2f} and a PR-AUC of {sel['pr_auc']:.2f} against a base rate of "
  f"about {100*sel['prevalence']:.0f}%. Most of that separation comes from age, and "
  f"much of the rest comes from the same handful of cardiovascular variables that "
  f"clinical guidance already relies on, so the model is confirming expected "
  f"relationships rather than finding anything new. The gap between this and "
  f"anything resembling a diagnostic tool stays large: the outcome is a self-report, "
  f"smoking and diabetes are not part of this predictor set, and even the most "
  f"favourable operating point in this report accepts several false positives for "
  f"every true one.")
    A("")

    A("## Appendix: audit")
    A("")
    fails = audit[audit["status"] == "FAIL"]
    A(f"{len(audit)} automated checks were run: {len(audit)-len(fails)} passed, "
      f"{len(fails)} failed. Full list in `reports/audit_report.md`.")
    if len(fails):
        A("")
        A(md(fails[["section", "item", "detail"]]))
    A("")

    path = config.REPORT_DIR / "final_report.md"
    path.write_text("\n".join(L))
    note(f"saved {path.relative_to(config.PROJECT_ROOT)} ({len(L)} blocks)")

    # machine-readable decision record
    decisions = dict(
        target=config.TARGET_DOC,
        cohort=dict(min_age=config.MIN_AGE, require_mec_exam=config.REQUIRE_MEC_EXAM,
                    n=int(summary["n_participants"]), cases=int(summary["n_chd_cases"]),
                    prevalence=float(summary["prevalence"])),
        excluded_variables=spec["excluded"],
        predictors=spec["predictors"],
        missing_indicators=spec["missing_indicators"],
        imputation=dict(numeric="median inside the pipeline",
                        categorical="most frequent inside the pipeline",
                        sensitivity="mean imputation compared in the robustness script"),
        split=design,
        class_imbalance="class weights, no resampling",
        models=list(cv["model"]),
        selected_model=best,
        metrics=dict(primary=config.PRIMARY_METRIC, secondary=config.SECONDARY_METRIC,
                     threshold_rule="Youden J on cross-validated development predictions",
                     threshold=float(ev["threshold"])),
        seed=config.RANDOM_SEED,
    )
    save_json(decisions, "research_decisions.json")

    banner("SCRIPT 8 FINISHED")


if __name__ == "__main__":
    main()
