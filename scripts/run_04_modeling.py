"""
Script 4 - PHASES 9 to 12.

    Phase 9   prediction design and the development / test split
    Phase 10  baseline models inside preprocessing pipelines
    Phase 11  class imbalance handling
    Phase 12  restrained tuning and model comparison

The test set is created here and then left alone. Nothing in this script
looks at test performance.

Input  : data/processed/cohort.parquet, data/processed/feature_spec.json
Output : data/processed/split.csv, models/best_model.joblib,
         outputs/cv_results.csv, outputs/model_comparison.csv,
         outputs/tuning_results.csv, outputs/split_design.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from src import config, modeling
from src.utils import banner, note, read_processed, save_json, save_table, step

MODEL_DIR = config.PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def decide_split(df: pd.DataFrame, features: list[str]) -> dict:
    """
    Choose between a temporal (cycle-based) design and a stratified random one.

    The rule is written down before looking at any model result:

    * if prevalence differs materially across cycles, or a predictor is
      available in some cycles and not others, the cycles are not exchangeable
      and a temporal test set is the honest design;
    * otherwise a stratified random split is used, because it keeps more CHD
      cases in both parts, and the cycle-based evaluation is still run later
      as a robustness check.
    """
    tab = pd.crosstab(df["survey_cycle"], df[config.TARGET])
    chi2, p, _, _ = chi2_contingency(tab)
    prev = df.groupby("survey_cycle")[config.TARGET].mean()
    spread = float(prev.max() - prev.min())

    coverage = df.groupby("survey_cycle")[features].apply(
        lambda g: g.notna().mean()).T
    max_gap = float((coverage.max(axis=1) - coverage.min(axis=1)).max())
    worst = coverage.assign(gap=coverage.max(axis=1) - coverage.min(axis=1)) \
                    .sort_values("gap", ascending=False).head(3)

    exchangeable = (p >= 0.05 or spread < 0.02) and max_gap < 0.25
    decision = "stratified_random" if exchangeable else "temporal_by_cycle"

    note(f"prevalence by cycle: {prev.round(4).to_dict()}")
    note(f"chi-square across cycles p = {p:.4f}, spread = {spread:.4f}")
    note(f"largest availability gap between cycles: {max_gap:.3f}")
    note("variables with the biggest availability gap:\n" + worst.round(3).to_string())
    note(f"decision: {decision}")

    return dict(design=decision, chi2_p=float(p), prevalence_spread=spread,
                max_availability_gap=max_gap,
                prevalence_by_cycle=prev.round(5).to_dict(),
                rule=("cycles treated as exchangeable when the prevalence difference "
                      "is small or not significant and no predictor is missing in a "
                      "whole cycle; the cycle-based evaluation is then kept as a "
                      "robustness check instead of the main design"))


def main() -> None:
    banner("PHASE 9 - PREDICTION DESIGN AND SPLIT")
    df = read_processed("cohort.parquet")
    spec = json.loads((config.PROCESSED_DIR / "feature_spec.json").read_text())
    features = spec["predictors"]
    indicators = spec["missing_indicators"]
    y = df[config.TARGET].astype(int)

    design = decide_split(df, features)
    save_json(design, "split_design.json")

    if design["design"] == "stratified_random":
        idx_dev, idx_test = train_test_split(
            df.index, test_size=config.TEST_SIZE, stratify=y,
            random_state=config.RANDOM_SEED)
    else:
        cycles = sorted(df["survey_cycle"].unique())
        test_cycles = cycles[-1:]
        idx_test = df.index[df["survey_cycle"].isin(test_cycles)]
        idx_dev = df.index.difference(idx_test)

    split = pd.DataFrame(dict(SEQN=df["SEQN"],
                              part=np.where(df.index.isin(idx_test), "test", "development")))
    split.to_csv(config.PROCESSED_DIR / "split.csv", index=False)

    dev, test = df.loc[idx_dev], df.loc[idx_test]
    note(f"development: {len(dev)} rows, {int(dev[config.TARGET].sum())} cases "
         f"({100*dev[config.TARGET].mean():.2f}%)")
    note(f"test: {len(test)} rows, {int(test[config.TARGET].sum())} cases "
         f"({100*test[config.TARGET].mean():.2f}%)")
    note("the test set is not touched again until script 5")

    X_dev, y_dev = dev[features], dev[config.TARGET].astype(int)

    # ------------------------------------------------------------------
    banner("PHASE 10 and 11 - BASELINES WITH CLASS IMBALANCE HANDLING")
    note(f"prevalence in development data: {100*y_dev.mean():.2f}% - accuracy is "
         "not used as a headline metric")
    note("class weights are used instead of resampling: they change the loss "
         "without inventing synthetic participants, and they keep the pipeline "
         "simple enough to stay inside cross-validation")

    models = modeling.make_models(dev, features, indicators, class_weight="balanced")
    cv_table, fold_scores = modeling.cross_validate_models(models, X_dev, y_dev)
    save_table(cv_table, "cv_results.csv")
    print(cv_table[["model", "roc_auc_mean", "roc_auc_sd", "average_precision_mean",
                    "average_precision_sd", "roc_auc_train_mean",
                    "fit_time_s"]].to_string(index=False))

    gap = cv_table.set_index("model")["roc_auc_train_mean"] - \
        cv_table.set_index("model")["roc_auc_mean"]
    step("train minus validation ROC-AUC (a large gap means overfitting)")
    print(gap.round(3).to_string())

    # ------------------------------------------------------------------
    banner("PHASE 12 - TUNING AND MODEL COMPARISON")
    tuned, tuning = modeling.tune_models(
        {k: v for k, v in models.items() if k != "dummy"}, X_dev, y_dev)
    save_table(tuning, "tuning_results.csv")

    tuned_table, tuned_folds = modeling.cross_validate_models(tuned, X_dev, y_dev)
    save_table(tuned_table, "model_comparison.csv")
    print(tuned_table[["model", "roc_auc_mean", "roc_auc_sd",
                       "average_precision_mean", "average_precision_sd"]]
          .to_string(index=False))

    paired = modeling.paired_fold_comparison(tuned_folds, metric=config.PRIMARY_METRIC)
    paired_auc = modeling.paired_fold_comparison(tuned_folds, metric="roc_auc")
    both = pd.concat([paired, paired_auc], ignore_index=True)
    save_table(both, "paired_model_comparison.csv")
    step("paired comparison on the same folds")
    print(both.to_string(index=False))

    best_name = tuned_table.iloc[0]["model"]
    best_score = tuned_table.iloc[0][f"{config.PRIMARY_METRIC}_mean"]
    runner_up = tuned_table.iloc[1]["model"]
    diff_row = both[(both["metric"] == config.PRIMARY_METRIC) &
                    (((both.model_a == best_name) & (both.model_b == runner_up)) |
                     ((both.model_b == best_name) & (both.model_a == runner_up)))]
    step("model selection")
    note(f"highest cross-validated {config.PRIMARY_METRIC}: {best_name} "
         f"({best_score:.4f})")
    if len(diff_row):
        r = diff_row.iloc[0]
        note(f"difference against {runner_up}: {abs(r['mean_difference']):.4f} "
             f"(sd across folds {r['sd_difference']:.4f}, paired t p = {r['paired_t_p']:.3f})")
        if r["paired_t_p"] > 0.05:
            note("the difference between the top two models is not clearly larger "
                 "than fold to fold variation, so the simpler model is a reasonable "
                 "choice as well; the selected model is kept but this is recorded")

    best_model = tuned[best_name].fit(X_dev, y_dev)
    joblib.dump(dict(name=best_name, model=best_model, features=features,
                     indicators=indicators, seed=config.RANDOM_SEED),
                MODEL_DIR / "best_model.joblib")
    for name, m in tuned.items():
        joblib.dump(m.fit(X_dev, y_dev), MODEL_DIR / f"{name}.joblib")
    note(f"saved models/best_model.joblib ({best_name}) and one file per model")

    save_json(dict(selected_model=best_name,
                   primary_metric=config.PRIMARY_METRIC,
                   cv_score=float(best_score),
                   runner_up=runner_up,
                   selection_note=("selected on cross-validated PR-AUC in the "
                                   "development data only; the test set was not "
                                   "used at any point in this script")),
              "model_selection.json")

    banner("SCRIPT 4 FINISHED")


if __name__ == "__main__":
    main()
