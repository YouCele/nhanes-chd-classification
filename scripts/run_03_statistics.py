"""
Script 3 - PHASE 8 (statistical comparison) and PHASE 19 (survey design).

Input  : data/processed/cohort.parquet, data/processed/feature_spec.json
Output : outputs/statistical_tests_continuous.csv
         outputs/statistical_tests_categorical.csv
         outputs/effect_size_summary.csv
         outputs/survey_weighted_prevalence.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, robustness, statistics as st
from src.preprocessing import split_feature_types
from src.utils import banner, note, read_processed, save_table, step


def main() -> None:
    banner("PHASE 8 - STATISTICAL ANALYSIS")
    df = read_processed("cohort.parquet")
    spec = json.loads((config.PROCESSED_DIR / "feature_spec.json").read_text())
    features = spec["predictors"]

    numeric, categorical = split_feature_types(df, features)
    binary = [f for f in numeric
              if set(pd.Series(df[f]).dropna().unique().tolist()) <= {0.0, 1.0}]
    continuous = [f for f in numeric if f not in binary]

    cont = st.compare_continuous(df, continuous)
    save_table(cont, "statistical_tests_continuous.csv")
    step("continuous variables, ordered by effect size")
    print(cont[["variable", "mean_chd", "mean_no_chd", "mean_difference",
                "diff_ci_low", "diff_ci_high", "cohens_d", "welch_p_fdr"]]
          .to_string(index=False))

    cat = st.compare_categorical(df, categorical + binary)
    save_table(cat, "statistical_tests_categorical.csv")
    step("categorical variables, ordered by Cramer's V")
    print(cat[["variable", "test", "odds_ratio", "or_ci_low", "or_ci_high",
               "cramers_v", "p_value_fdr"]].to_string(index=False))

    summary = st.interpret_effect_sizes(cont, cat)
    save_table(summary, "effect_size_summary.csv")

    step("how to read this")
    n_sig = int((summary["p_value_fdr"] < 0.05).sum())
    n_neg = int((summary["magnitude"] == "negligible").sum())
    note(f"{n_sig} of {len(summary)} variables have an FDR-adjusted p below 0.05")
    note(f"{n_neg} of them still have a negligible effect size")
    note("with more than twenty thousand participants a very small difference can "
         "reach significance, so the effect size column is the one to read")

    # ------------------------------------------------------------------
    banner("PHASE 19 - SURVEY DESIGN")
    note("two different questions: how well the classifier separates people inside "
         "this sample, and what the sample says about the US adult population")

    overall = robustness.weighted_prevalence(df)
    rows = [dict(group="overall", level="all", **overall)]
    for g in ["sex", "age_group", "survey_cycle"]:
        if g in df.columns:
            part = robustness.weighted_prevalence_by_group(df, g)
            if len(part):
                rows.extend(part.to_dict("records"))
    weighted = pd.DataFrame(rows)
    save_table(weighted, "survey_weighted_prevalence.csv")
    print(weighted[["group", "level", "n", "unweighted_prevalence",
                    "weighted_prevalence", "ci_low", "ci_high"]].to_string(index=False))

    note("the weighted numbers describe the population; everything in the modelling "
         "scripts is unweighted and describes the analytical sample only")
    note("no survey weights are used inside the models: the aim is discrimination "
         "within the sample, and weighting a classifier would need a separate "
         "methodological justification")

    banner("SCRIPT 3 FINISHED")


if __name__ == "__main__":
    main()
