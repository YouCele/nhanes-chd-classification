"""
Script 6 - PHASES 16 to 18.

    Phase 16  error analysis
    Phase 17  robustness and sensitivity checks
    Phase 18  subgroup performance

Each check here answers a specific worry, not a wish for more tables:

    imputation         does the median rule drive the result?
    repeated splits    is the single test estimate a lucky draw?
    cycle holdout      does the model transfer to a later survey period?
    predictor sets     what do the fasting lipids and the questionnaire add?
    class weights      is the comparison an artefact of the weighting?
    thresholds         how does the operating point trade sensitivity for PPV?
    CVD block          how much easier does the task become if the neighbouring
                       cardiovascular questions are allowed back in?

Input  : data/processed/cohort.parquet, data/processed/split.csv, models/*.joblib
Output : outputs/error_analysis.csv, outputs/error_group_comparison.csv,
         outputs/robustness_*.csv, outputs/subgroup_results.csv
         figure 09
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config, eda, evaluation as ev, modeling, robustness as rb
from src.preprocessing import build_preprocessor, split_feature_types
from src.utils import banner, note, read_processed, save_json, save_table, step

MODEL_DIR = config.PROJECT_ROOT / "models"


def main() -> None:
    df = read_processed("cohort.parquet")
    spec = json.loads((config.PROCESSED_DIR / "feature_spec.json").read_text())
    split = pd.read_csv(config.PROCESSED_DIR / "split.csv")
    df = df.merge(split, on="SEQN", how="left", validate="one_to_one")
    features, indicators = spec["predictors"], spec["missing_indicators"]

    dev = df[df["part"] == "development"].reset_index(drop=True)
    test = df[df["part"] == "test"].reset_index(drop=True)
    y_dev = dev[config.TARGET].astype(int)
    y_test = test[config.TARGET].astype(int).values
    y_all = df[config.TARGET].astype(int)

    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    best_name, best_model = bundle["name"], bundle["model"]
    ev_summary = json.loads((config.OUTPUT_DIR / "evaluation_summary.json").read_text())
    threshold = ev_summary["threshold"]
    proba = best_model.predict_proba(test[features])[:, 1]

    numeric, categorical = split_feature_types(test, features)
    binary = [f for f in numeric
              if set(pd.Series(test[f]).dropna().unique().tolist()) <= {0.0, 1.0}]
    continuous = [f for f in numeric if f not in binary]

    # ------------------------------------------------------------------
    banner("PHASE 16 - ERROR ANALYSIS")
    err = rb.error_frame(test, y_test, proba, threshold)
    counts = err["error_type"].value_counts()
    note(f"at threshold {threshold:.3f}: " +
         ", ".join(f"{k} {v}" for k, v in counts.items()))

    profile = rb.error_profile(err, ["age", "sbp", "dbp", "bmi", "waist",
                                     "total_chol", "hdl", "pulse"],
                               ["sex", "race_eth", "arthritis", "fam_hist_mi",
                                "survey_cycle"])
    save_table(profile, "error_analysis.csv")
    print(profile.round(3).to_string(index=False))

    comparisons = []
    for a, b in [("false_negative", "true_positive"),
                 ("false_positive", "true_negative")]:
        part = rb.compare_error_groups(err, a, b, continuous, binary + categorical)
        if len(part):
            comparisons.append(part)
    if comparisons:
        comp = pd.concat(comparisons, ignore_index=True)
        save_table(comp, "error_group_comparison.csv")
        step("what separates missed cases from detected cases")
        fn = comp[comp["comparison"] == "false_negative vs true_positive"]
        show = ["comparison", "variable", "n_group_a", "n_group_b", "mean_group_a",
                "mean_group_b", "cohens_d", "welch_p", "odds_ratio", "cramers_v"]
        print(fn.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
              .head(8)[[c for c in show if c in fn.columns]].round(3).to_string(index=False))
        step("what pushes the model towards a false positive")
        fp = comp[comp["comparison"] == "false_positive vs true_negative"]
        print(fp.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
              .head(8)[[c for c in show if c in fp.columns]].round(3).to_string(index=False))

    # missingness among the errors: a plausible explanation to check
    err["n_missing_predictors"] = test[features].isna().sum(axis=1).values
    miss_by_error = err.groupby("error_type")["n_missing_predictors"].agg(
        ["mean", "median", "size"]).reset_index()
    save_table(miss_by_error, "error_missingness.csv")
    print(miss_by_error.round(2).to_string(index=False))

    high_conf = err[(err["error_type"].isin(["false_positive", "false_negative"])) &
                    (err["confidence_gap"] > 0.3)]
    note(f"{len(high_conf)} errors were made with a predicted probability more than "
         f"0.3 away from the threshold")
    save_table(err[["SEQN", "y_true", "predicted_risk", "y_pred", "error_type",
                    "confidence_gap", "age", "sex", "survey_cycle"]],
               "error_cases.csv")

    # ------------------------------------------------------------------
    banner("PHASE 17 - ROBUSTNESS")

    step("1. alternative imputation")
    def lr_factory(strategy: str) -> Pipeline:
        return Pipeline([
            ("pre", build_preprocessor(dev, features, scale=True,
                                       numeric_strategy=strategy,
                                       indicator_for=indicators)),
            ("clf", LogisticRegression(max_iter=2000, C=0.03, class_weight="balanced",
                                       random_state=config.RANDOM_SEED)),
        ])
    imp = rb.imputation_sensitivity(lr_factory, dev, features, y_dev, indicators)
    save_table(imp, "robustness_imputation.csv")
    print(imp.round(4).to_string(index=False))

    step("2. repeated random splits")
    rep = rb.repeated_split_stability(best_model, df, features, y_all, n_repeats=10)
    save_table(rep, "robustness_repeated_splits.csv")
    note(f"ROC-AUC over 10 splits: mean {rep['roc_auc'].mean():.3f}, "
         f"sd {rep['roc_auc'].std(ddof=1):.3f}, "
         f"range {rep['roc_auc'].min():.3f} to {rep['roc_auc'].max():.3f}")
    note(f"PR-AUC over 10 splits: mean {rep['pr_auc'].mean():.3f}, "
         f"sd {rep['pr_auc'].std(ddof=1):.3f}")

    step("3. cycle based evaluation")
    cycles = sorted(df["survey_cycle"].unique())
    cyc_rows = [rb.cycle_holdout(best_model, df, features, y_all,
                                 train_cycles=cycles[:2], test_cycles=cycles[2:]),
                rb.cycle_holdout(best_model, df, features, y_all,
                                 train_cycles=cycles[:3], test_cycles=cycles[3:])]
    cyc = pd.DataFrame(cyc_rows)
    save_table(cyc, "robustness_cycle_holdout.csv")
    print(cyc[["train_cycles", "test_cycles", "n_train", "n_test",
               "train_prevalence", "test_prevalence", "roc_auc", "pr_auc",
               "cal_calibration_slope"]].round(4).to_string(index=False))

    step("4. predictor sets")
    lipids = [f for f in ["triglycerides", "ldl", "total_chol", "hdl"] if f in features]
    questionnaire = [f for f in features if f in
                     ["asthma_ever", "anemia_treated", "told_overweight", "arthritis",
                      "emphysema", "bronchitis", "liver_condition", "thyroid_problem",
                      "gout", "cancer_ever", "fam_hist_mi", "fam_hist_asthma",
                      "fam_hist_diabetes"]]
    demo_only = [f for f in ["age", "sex", "race_eth", "education", "marital",
                             "income_poverty_ratio", "born_us"] if f in features]
    sets = {
        "all predictors": features,
        "without the fasting lipids": [f for f in features
                                       if f not in ("triglycerides", "ldl")],
        "without any lipid measurement": [f for f in features if f not in lipids],
        "without the questionnaire items": [f for f in features if f not in questionnaire],
        "demographics only": demo_only,
        "age and sex only": [f for f in ["age", "sex"] if f in features],
    }
    red = rb.predictor_reduction(best_model, dev, sets, y_dev, indicators)
    save_table(red, "robustness_predictor_sets.csv")
    print(red.round(4).to_string(index=False))

    step("5. class weight sensitivity")
    def weight_factory(weight):
        return Pipeline([
            ("pre", build_preprocessor(dev, features, scale=True,
                                       indicator_for=indicators)),
            ("clf", LogisticRegression(max_iter=2000, C=0.03, class_weight=weight,
                                       random_state=config.RANDOM_SEED)),
        ])
    cw = rb.class_weight_sensitivity(weight_factory, dev, features, y_dev)
    save_table(cw, "robustness_class_weight.csv")
    print(cw.round(4).to_string(index=False))
    note("class weighting mainly moves the probability scale and therefore the "
         "useful threshold; discrimination barely changes")

    step("6. threshold sensitivity")
    sweep = pd.read_csv(config.OUTPUT_DIR / "threshold_sweep.csv")
    picks = sweep.iloc[(sweep["sensitivity"] - 0.9).abs().argsort()[:1]]
    note(f"to reach about 90% sensitivity the threshold would be "
         f"{picks['threshold'].iloc[0]:.3f}, with specificity "
         f"{picks['specificity'].iloc[0]:.3f} and PPV {picks['ppv'].iloc[0]:.3f}")

    step("7. what the excluded cardiovascular questions would add")
    cvd = [c for c in spec["sensitivity_cvd_block"] if c in df.columns]
    if cvd:
        sets2 = {"main predictor set": features,
                 "main set plus the CVD questionnaire block": features + cvd}
        leak_test = rb.predictor_reduction(best_model, dev, sets2, y_dev, indicators)
        save_table(leak_test, "robustness_leakage_sensitivity.csv")
        print(leak_test.round(4).to_string(index=False))
        note("this is reported to show the size of the effect, not as an "
             "alternative model: those questions are part of the same self-report "
             "block as the outcome")

    # ------------------------------------------------------------------
    banner("PHASE 18 - SUBGROUP ANALYSIS")
    test["age_group"] = pd.cut(test["age"], bins=config.AGE_BINS,
                               labels=config.AGE_LABELS, right=False)
    sub = rb.subgroup_performance(test, y_test, proba, threshold,
                                  groups=["sex", "age_group", "race_eth",
                                          "survey_cycle"])
    save_table(sub, "subgroup_results.csv")
    cols = ["group", "level", "n", "cases", "prevalence", "roc_auc", "pr_auc",
            "sensitivity", "specificity", "ppv", "npv", "reliable"]
    print(sub[[c for c in cols if c in sub.columns]].round(3).to_string(index=False))

    unreliable = sub[~sub["reliable"]]
    if len(unreliable):
        note(f"{len(unreliable)} subgroups had too few cases for a stable estimate "
             "and are reported with their size only")
    eda.fig_subgroups(sub)

    reliable = sub[sub["reliable"] == True]
    if len(reliable) > 1:
        spread = reliable["roc_auc"].max() - reliable["roc_auc"].min()
        note(f"ROC-AUC across the subgroups with enough cases spans {spread:.3f}")
        note("differences of this size with a few dozen cases per group are not "
             "strong evidence of unequal performance")

    save_json(dict(
        threshold=threshold,
        repeated_split_roc_auc_mean=float(rep["roc_auc"].mean()),
        repeated_split_roc_auc_sd=float(rep["roc_auc"].std(ddof=1)),
        cycle_holdout=cyc_rows,
        imputation_difference=float(imp["roc_auc_mean"].max() - imp["roc_auc_mean"].min()),
    ), "robustness_summary.json")

    banner("SCRIPT 6 FINISHED")


if __name__ == "__main__":
    main()
