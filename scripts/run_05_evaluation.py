"""
Script 5 - PHASES 13 to 15.

    Phase 13  final evaluation on the untouched test set
    Phase 14  calibration
    Phase 15  interpretation

The decision threshold is chosen inside the development data with cross
validated predictions, so the test set is only used once, for reporting.

Input  : data/processed/cohort.parquet, data/processed/split.csv, models/*.joblib
Output : outputs/test_metrics.csv, outputs/calibration.csv,
         outputs/feature_importance.csv, outputs/logistic_coefficients.csv
         figures 05 to 08
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

from src import config, eda, evaluation as ev, interpretation as interp, modeling
from src.utils import banner, note, read_processed, save_json, save_table, step

MODEL_DIR = config.PROJECT_ROOT / "models"


def main() -> None:
    banner("PHASE 13 - FINAL TEST EVALUATION")
    df = read_processed("cohort.parquet")
    spec = json.loads((config.PROCESSED_DIR / "feature_spec.json").read_text())
    split = pd.read_csv(config.PROCESSED_DIR / "split.csv")
    df = df.merge(split, on="SEQN", how="left", validate="one_to_one")

    features, indicators = spec["predictors"], spec["missing_indicators"]
    dev = df[df["part"] == "development"].reset_index(drop=True)
    test = df[df["part"] == "test"].reset_index(drop=True)
    y_dev = dev[config.TARGET].astype(int)
    y_test = test[config.TARGET].astype(int).values

    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    best_name, best_model = bundle["name"], bundle["model"]
    note(f"selected model: {best_name}")

    # ---- threshold picked on development data only --------------------
    step("choosing the operating point inside the development data")
    oof = cross_val_predict(best_model, dev[features], y_dev,
                            cv=modeling.cv_splitter(), method="predict_proba",
                            n_jobs=config.N_JOBS)[:, 1]
    threshold = ev.choose_threshold(y_dev.values, oof, rule="youden")
    note(f"threshold from cross-validated development predictions: {threshold:.4f}")
    dev_metrics = ev.classification_metrics(y_dev.values, oof, threshold)
    note(f"development (cross-validated): ROC-AUC {dev_metrics['roc_auc']:.3f}, "
         f"PR-AUC {dev_metrics['pr_auc']:.3f}")

    # ---- the test set, used once --------------------------------------
    rows, curves = [], {}
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        proba = model.predict_proba(test[features])[:, 1]
        m = ev.classification_metrics(y_test, proba, threshold)
        auc, auc_lo, auc_hi = ev.bootstrap_metric_ci(y_test, proba, "roc_auc")
        ap, ap_lo, ap_hi = ev.bootstrap_metric_ci(y_test, proba, "average_precision")
        m.update(model=name, roc_auc_ci_low=auc_lo, roc_auc_ci_high=auc_hi,
                 pr_auc_ci_low=ap_lo, pr_auc_ci_high=ap_hi, selected=(name == best_name))
        rows.append(m)
        curves[name] = (ev.roc_points(y_test, proba), ev.pr_points(y_test, proba))
        if name == best_name:
            best_proba = proba

    test_metrics = pd.DataFrame(rows)
    cols = ["model", "selected", "n", "prevalence", "threshold", "roc_auc",
            "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "pr_auc_ci_low",
            "pr_auc_ci_high", "brier", "sensitivity", "specificity", "ppv", "npv",
            "f1", "tp", "fp", "tn", "fn"]
    test_metrics = test_metrics[cols]
    save_table(test_metrics, "test_metrics.csv")
    print(test_metrics.round(4).to_string(index=False))

    sel = test_metrics[test_metrics["selected"]].iloc[0]
    step("selected model on the test set")
    note(f"ROC-AUC {sel['roc_auc']:.3f} (95% CI {sel['roc_auc_ci_low']:.3f} to "
         f"{sel['roc_auc_ci_high']:.3f})")
    note(f"PR-AUC {sel['pr_auc']:.3f} (95% CI {sel['pr_auc_ci_low']:.3f} to "
         f"{sel['pr_auc_ci_high']:.3f}), prevalence {sel['prevalence']:.4f}")
    note(f"at threshold {threshold:.3f}: sensitivity {sel['sensitivity']:.3f}, "
         f"specificity {sel['specificity']:.3f}, PPV {sel['ppv']:.3f}, "
         f"NPV {sel['npv']:.3f}")

    cv_table = pd.read_csv(config.OUTPUT_DIR / "model_comparison.csv")
    cv_auc = float(cv_table.loc[cv_table.model == best_name, "roc_auc_mean"].iloc[0])
    cv_ap = float(cv_table.loc[cv_table.model == best_name,
                               "average_precision_mean"].iloc[0])
    step("cross-validation compared with the test set")
    note(f"ROC-AUC: CV {cv_auc:.3f} vs test {sel['roc_auc']:.3f}")
    note(f"PR-AUC: CV {cv_ap:.3f} vs test {sel['pr_auc']:.3f}")
    inside = sel["roc_auc_ci_low"] <= cv_auc <= sel["roc_auc_ci_high"]
    note("the cross-validated value falls inside the test confidence interval"
         if inside else
         "the cross-validated value falls outside the test interval, which is "
         "investigated in the robustness script (sampling variation, prevalence "
         "shift and cycle differences are the candidates)")

    ev_summary = dict(selected_model=best_name, threshold=float(threshold),
                      development_cv_roc_auc=cv_auc, development_cv_pr_auc=cv_ap,
                      test_roc_auc=float(sel["roc_auc"]), test_pr_auc=float(sel["pr_auc"]),
                      cv_inside_test_ci=bool(inside))
    save_json(ev_summary, "evaluation_summary.json")

    eda.fig_roc_pr(curves, prevalence=float(np.mean(y_test)))
    eda.fig_confusion(ev.classification_metrics(y_test, best_proba, threshold))

    sweep = ev.threshold_sweep(y_test, best_proba)
    save_table(sweep, "threshold_sweep.csv")
    eda.fig_threshold(sweep, threshold)

    # ------------------------------------------------------------------
    banner("PHASE 14 - CALIBRATION")
    cal = ev.calibration_table(y_test, best_proba, n_bins=10)
    save_table(cal, "calibration.csv")
    print(cal[["bin", "n", "cases", "predicted_mean", "observed",
               "obs_ci_low", "obs_ci_high"]].round(4).to_string(index=False))

    stats = ev.calibration_slope_intercept(y_test, best_proba)
    save_json(stats, "calibration_summary.json")
    note(f"calibration slope {stats['calibration_slope']:.3f} "
         f"(1 would be ideal), intercept {stats['calibration_intercept']:.3f}")
    note(f"mean predicted risk {stats['mean_predicted_risk']:.4f} against an "
         f"observed risk of {stats['observed_risk']:.4f} "
         f"(observed/expected {stats['ratio_observed_expected']:.2f})")
    note(f"Brier score {stats['brier']:.4f}")
    if abs(stats["ratio_observed_expected"] - 1) > 0.2:
        note("the class-weighted model produces probabilities on a shifted scale, "
             "which is expected: weighting changes the intercept. The ranking is "
             "unaffected, but the raw numbers should not be read as risks without "
             "recalibration.")

    bands = ev.calibration_by_risk_band(y_test, best_proba)
    save_table(bands, "calibration_by_risk_band.csv")
    print(bands.round(4).to_string(index=False))
    eda.fig_calibration(cal)

    # ---- recalibrated version ----------------------------------------
    step("recalibrating the probabilities")
    note("the class-weighted model is trained to separate the classes, not to "
         "produce risks. A sigmoid recalibration is fitted on the development "
         "data with its own internal cross-validation, so the test set stays "
         "untouched. A monotone transformation does not change the ranking, so "
         "ROC-AUC and PR-AUC are unchanged.")
    calibrated = CalibratedClassifierCV(clone(best_model), method="sigmoid",
                                        cv=modeling.cv_splitter())
    calibrated.fit(dev[features], y_dev)
    proba_cal = calibrated.predict_proba(test[features])[:, 1]

    cal_after = ev.calibration_table(y_test, proba_cal, n_bins=10)
    cal_after["version"] = "recalibrated"
    cal["version"] = "raw"
    save_table(pd.concat([cal, cal_after], ignore_index=True), "calibration.csv")
    eda.fig_calibration(cal_after, name="06b_calibration_recalibrated.png")

    stats_after = ev.calibration_slope_intercept(y_test, proba_cal)
    save_json(dict(raw=stats, recalibrated=stats_after), "calibration_summary.json")
    note(f"after recalibration: slope {stats_after['calibration_slope']:.3f}, "
         f"intercept {stats_after['calibration_intercept']:.3f}, "
         f"mean predicted {stats_after['mean_predicted_risk']:.4f} against "
         f"observed {stats_after['observed_risk']:.4f}, "
         f"Brier {stats_after['brier']:.4f} (was {stats['brier']:.4f})")

    bands_after = ev.calibration_by_risk_band(y_test, proba_cal)
    bands["version"], bands_after["version"] = "raw", "recalibrated"
    save_table(pd.concat([bands, bands_after], ignore_index=True),
               "calibration_by_risk_band.csv")
    print(bands_after.round(4).to_string(index=False))
    joblib.dump(calibrated, MODEL_DIR / "calibrated_model.joblib")

    # ------------------------------------------------------------------
    banner("PHASE 15 - INTERPRETATION")
    coefs = interp.logistic_coefficients(dev, features, y_dev, indicators)
    save_table(coefs, "logistic_coefficients.csv")
    step("logistic regression, largest coefficients (interpretation model)")
    print(coefs.head(15)[["feature", "coefficient", "odds_ratio", "or_ci_low",
                          "or_ci_high"]].round(3).to_string(index=False))
    note("these odds ratios describe associations inside the model, adjusted for "
         "the other variables. They are not causal effects.")

    perm_tables = {}
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        step(f"permutation importance for {name}")
        perm = interp.permutation_scores(model, test[features], y_test, n_repeats=8)
        perm["model"] = name
        perm_tables[name] = perm
        print(perm.head(8)[["feature", "importance_mean", "importance_sd"]]
              .round(4).to_string(index=False))

    all_perm = pd.concat(perm_tables.values(), ignore_index=True)
    save_table(all_perm, "feature_importance.csv")
    comparison = interp.compare_importance(perm_tables)
    save_table(comparison, "feature_importance_agreement.csv")
    step("where the models disagree most about what matters")
    print(comparison.sort_values("rank_spread", ascending=False)
          .head(6)[["feature", "mean_rank", "rank_spread"]].round(2).to_string(index=False))

    eda.fig_importance(perm_tables[best_name])

    banner("SCRIPT 5 FINISHED")


if __name__ == "__main__":
    main()
