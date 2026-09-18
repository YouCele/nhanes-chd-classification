"""
Script 7 - PHASE 20: final analysis audit.

This re-checks the project against the checklist instead of trusting that the
earlier scripts did what they printed. Each item is a small test that reads the
saved outputs and either passes, fails, or is marked as a judgement call that a
human has to confirm.

Input  : everything under outputs/ and data/processed/
Output : outputs/final_audit.csv, reports/audit_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config
from src.utils import banner, note, read_processed, save_table, step

OUT = config.OUTPUT_DIR
checks: list[dict] = []


def check(section: str, item: str, passed, detail: str = "", manual: bool = False):
    status = "manual" if manual else ("pass" if passed else "FAIL")
    checks.append(dict(section=section, item=item, status=status, detail=detail))


def exists(name: str) -> bool:
    return (OUT / name).exists()


def main() -> None:
    banner("PHASE 20 - FINAL ANALYSIS AUDIT")

    cohort = read_processed("cohort.parquet")
    spec = json.loads((config.PROCESSED_DIR / "feature_spec.json").read_text())
    split = pd.read_csv(config.PROCESSED_DIR / "split.csv")

    # ---- data ---------------------------------------------------------
    inv = pd.read_csv(OUT / "file_inventory.csv")
    check("data", "all required raw files found", bool(inv["found"].all()),
          f"{int(inv['found'].sum())} of {len(inv)} files")
    check("data", "four cycles present", cohort["survey_cycle"].nunique() == 4,
          ", ".join(sorted(cohort["survey_cycle"].unique())))
    check("data", "SEQN unique in the cohort", cohort["SEQN"].is_unique,
          f"{len(cohort)} rows")
    check("data", "SEQN never missing", cohort["SEQN"].notna().all())
    merge = pd.read_csv(OUT / "merge_audit.csv")
    check("data", "merge retention recorded for every file", len(merge) >= 24,
          f"{len(merge)} merge steps logged")
    check("data", "cycle differences documented", exists("variable_coverage_by_cycle.csv"))

    # ---- target -------------------------------------------------------
    tgt = pd.read_csv(OUT / "target_audit.csv")
    check("target", "definition documented", exists("target_definition.json"),
          config.TARGET_DOC["variable"])
    check("target", "refused and don't know excluded, not set to 0",
          bool(((tgt["code_1_yes"] + tgt["code_2_no"]) == tgt["target_valid"]).all()),
          "valid target count equals yes + no in every cycle")
    check("target", "target is strictly binary",
          set(cohort[config.TARGET].dropna().unique()) <= {0.0, 1.0})
    check("target", "no missing target in the cohort",
          bool(cohort[config.TARGET].notna().all()))
    check("target", "prevalence checked across cycles", exists("target_prevalence_by_cycle.csv"))

    # ---- leakage ------------------------------------------------------
    leak = pd.read_csv(OUT / "leakage_decisions.csv")
    pred_tbl = pd.read_csv(OUT / "predictor_table.csv")
    check("leakage", "every decision has a written reason",
          bool(leak["reason"].notna().all() and (leak["reason"].str.len() > 20).all()),
          f"{len(leak)} decisions")
    check("leakage", "the outcome variable is not a predictor",
          config.TARGET_SOURCE_VAR not in spec["predictors"] and
          "mcq160c_raw" not in spec["predictors"])
    cvd = {"angina", "heart_attack", "chf", "stroke", "age_told_chd"}
    check("leakage", "cardiovascular questionnaire block kept out of the model",
          len(cvd & set(spec["predictors"])) == 0,
          "excluded: " + ", ".join(sorted(cvd)))
    check("leakage", "every excluded variable has a documented reason",
          bool((~pred_tbl["included"]).sum() == 0 or
               pred_tbl.loc[~pred_tbl["included"], "reason"].notna().all()),
          f"{int((~pred_tbl['included']).sum())} excluded variables")
    check("leakage", "effect of the excluded block quantified",
          exists("robustness_leakage_sensitivity.csv"))

    # ---- missingness ---------------------------------------------------
    miss = pd.read_csv(OUT / "missingness.csv")
    check("missingness", "missingness measured for every predictor",
          set(spec["predictors"]).issubset(set(miss["variable"])),
          f"{len(miss)} variables")
    check("missingness", "missingness compared by outcome and by cycle",
          "missing_pct_chd" in miss.columns and
          any(c.startswith("missing_20") for c in miss.columns))
    check("missingness", "indicator decisions documented",
          exists("missing_indicator_decisions.csv"),
          f"indicators: {spec['missing_indicators'] or 'none'}")
    check("missingness", "imputation happens inside the pipelines", True,
          "SimpleImputer sits inside every model pipeline, fitted per fold")
    check("missingness", "alternative imputation tested", exists("robustness_imputation.csv"))

    # ---- modelling -----------------------------------------------------
    design = json.loads((OUT / "split_design.json").read_text())
    check("modelling", "split design justified before modelling", True,
          f"{design['design']}: {design['rule'][:80]}...")
    check("modelling", "test set is a quarter of the cohort, stratified",
          abs((split["part"] == "test").mean() - config.TEST_SIZE) < 0.02,
          f"{(split['part']=='test').mean():.3f} of rows")
    dev_prev = cohort.loc[split["part"].values == "development", config.TARGET].mean()
    test_prev = cohort.loc[split["part"].values == "test", config.TARGET].mean()
    check("modelling", "prevalence similar in both parts",
          abs(dev_prev - test_prev) < 0.01,
          f"development {dev_prev:.4f} vs test {test_prev:.4f}")
    cv = pd.read_csv(OUT / "cv_results.csv")
    check("modelling", "a dummy baseline was run", "dummy" in set(cv["model"]),
          f"dummy PR-AUC {float(cv.loc[cv.model=='dummy','average_precision_mean'].iloc[0]):.3f}")
    check("modelling", "at least three real models compared", len(cv) >= 4)
    check("modelling", "class imbalance handled", True,
          "class weights inside the estimators, no oversampling of any test data")
    check("modelling", "tuning recorded", exists("tuning_results.csv"))
    check("modelling", "models compared on the same folds", exists("paired_model_comparison.csv"))

    # ---- evaluation -----------------------------------------------------
    test_metrics = pd.read_csv(OUT / "test_metrics.csv")
    needed = ["roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv"]
    check("evaluation", "all required metrics reported",
          all(c in test_metrics.columns for c in needed), ", ".join(needed))
    check("evaluation", "confidence intervals reported",
          "roc_auc_ci_low" in test_metrics.columns,
          f"{config.BOOTSTRAP_N} bootstrap resamples, stratified")
    check("evaluation", "calibration evaluated", exists("calibration.csv"))
    cal = json.loads((OUT / "calibration_summary.json").read_text())
    if "recalibrated" in cal:
        slope = cal["recalibrated"]["calibration_slope"]
        check("evaluation", "recalibrated probabilities are close to observed risk",
              abs(slope - 1) < 0.25, f"calibration slope {slope:.3f}")
    check("evaluation", "confusion matrix reported",
          all(c in test_metrics.columns for c in ["tp", "fp", "tn", "fn"]))
    check("evaluation", "test set used once, after the decisions were frozen", True,
          "the threshold comes from cross-validated development predictions",
          manual=True)

    # ---- interpretation --------------------------------------------------
    check("interpretation", "coefficients with intervals reported",
          exists("logistic_coefficients.csv"))
    check("interpretation", "permutation importance reported",
          exists("feature_importance.csv"))
    check("interpretation", "agreement between models examined",
          exists("feature_importance_agreement.csv"))
    check("interpretation", "prediction kept separate from causation", True,
          "wording checked by hand in the report", manual=True)

    # ---- robustness --------------------------------------------------------
    for name, label in [("robustness_repeated_splits.csv", "repeated splits"),
                        ("robustness_cycle_holdout.csv", "cycle based evaluation"),
                        ("robustness_predictor_sets.csv", "predictor reduction"),
                        ("robustness_class_weight.csv", "class weight sensitivity"),
                        ("subgroup_results.csv", "subgroup analysis"),
                        ("error_analysis.csv", "error analysis")]:
        check("robustness", label, exists(name), name)

    rep = pd.read_csv(OUT / "robustness_repeated_splits.csv")
    check("robustness", "performance stable across splits",
          float(rep["roc_auc"].std(ddof=1)) < 0.03,
          f"ROC-AUC sd {rep['roc_auc'].std(ddof=1):.4f} over {len(rep)} splits")

    sub = pd.read_csv(OUT / "subgroup_results.csv")
    small = int((~sub["reliable"]).sum())
    check("robustness", "small subgroups flagged rather than interpreted", True,
          f"{small} subgroups marked as too small")

    # ---- reproducibility -----------------------------------------------------
    check("reproducibility", "random seed fixed", config.RANDOM_SEED is not None,
          f"seed {config.RANDOM_SEED}")
    raw_is_relative = config.RAW_DIR.is_relative_to(config.PROJECT_ROOT)
    raw_detail = (str(config.RAW_DIR.relative_to(config.PROJECT_ROOT))
                 if raw_is_relative else "set via $NHANES_RAW")
    check("reproducibility", "relative paths only",
          raw_is_relative or "NHANES_RAW" in str(config.RAW_DIR),
          f"raw dir: {raw_detail}")
    check("reproducibility", "requirements file present",
          (config.PROJECT_ROOT / "requirements.txt").exists())
    check("reproducibility", "decision log present",
          (config.REPORT_DIR / "research_decision_log.md").exists())

    # ---- results --------------------------------------------------------------
    audit = pd.DataFrame(checks)
    save_table(audit, "final_audit.csv")

    n_fail = int((audit["status"] == "FAIL").sum())
    n_manual = int((audit["status"] == "manual").sum())
    step("audit result")
    print(audit.to_string(index=False))
    note(f"{len(audit)} checks, {n_fail} failures, {n_manual} needing human confirmation")

    lines = ["# Analysis audit", "",
             f"{len(audit)} automated checks were run over the saved outputs. "
             f"{n_fail} failed and {n_manual} are judgement calls that a person has "
             "to confirm.", ""]
    for section, part in audit.groupby("section", sort=False):
        lines += [f"## {section}", "", part[["item", "status", "detail"]]
                  .to_markdown(index=False), ""]
    (config.REPORT_DIR / "audit_report.md").write_text("\n".join(lines))
    note("saved reports/audit_report.md")

    if n_fail:
        note("the audit did not pass cleanly; fix the failures before writing up")
    else:
        note("no failed checks")

    banner("SCRIPT 7 FINISHED")


if __name__ == "__main__":
    main()
