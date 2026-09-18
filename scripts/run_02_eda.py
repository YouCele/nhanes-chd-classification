"""
Script 2 - PHASE 7: exploratory analysis.

Input  : data/processed/cohort.parquet (or .csv), data/processed/feature_spec.json
Output : outputs/table_one.csv, outputs/describe_*.csv, outputs/eda_notes.md
         figures 01 to 04
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config, eda
from src.harmonization import target_prevalence_tables
from src.preprocessing import split_feature_types
from src.utils import banner, note, read_processed, save_table, step


def main() -> None:
    banner("PHASE 7 - EXPLORATORY ANALYSIS")
    df = read_processed("cohort.parquet")
    spec = json.loads((config.PROCESSED_DIR / "feature_spec.json").read_text())
    features = spec["predictors"]

    numeric, categorical = split_feature_types(df, features)
    binary = [f for f in numeric
              if set(pd.Series(df[f]).dropna().unique().tolist()) <= {0.0, 1.0}]
    continuous = [f for f in numeric if f not in binary]

    note(f"{len(continuous)} continuous, {len(binary)} binary, "
         f"{len(categorical)} multi-level categorical predictors")

    step("population structure")
    print(f"   n = {len(df)}, CHD cases = {int(df[config.TARGET].sum())} "
          f"({100*df[config.TARGET].mean():.2f}%)")
    print(f"   age: median {df['age'].median():.0f}, "
          f"IQR {df['age'].quantile(.25):.0f}-{df['age'].quantile(.75):.0f}, "
          f"{(df['age']>=80).mean()*100:.1f}% at the top-coded value of 80")

    desc_num = eda.describe_numeric(df, continuous)
    save_table(desc_num, "describe_numeric.csv")
    print(desc_num[["variable", "n", "mean", "sd", "median", "q95", "skewness"]]
          .to_string(index=False))

    desc_cat = eda.describe_categorical(df, categorical + binary)
    save_table(desc_cat, "describe_categorical.csv")

    t1 = eda.table_one(df, continuous, categorical + binary)
    save_table(t1, "table_one.csv")
    step("descriptive table by CHD status (first rows)")
    print(t1.head(20).to_string(index=False))

    # skewness is worth a look before deciding on transformations
    step("strongly skewed variables (|skew| > 1)")
    skewed = desc_num.loc[desc_num["skewness"].abs() > 1, ["variable", "skewness",
                                                           "median", "q95", "max"]]
    print(skewed.to_string(index=False) if len(skewed) else "   none")
    note("extreme values were checked against the plausibility ranges during "
         "harmonisation and were not removed when biologically possible")

    # figures
    prev = target_prevalence_tables(df)
    eda.fig_prevalence(prev["by_cycle"], prev["by_age_group"], prev["by_sex"])
    eda.fig_distributions(df, ["age", "sbp", "bmi", "hdl", "total_chol", "waist"])
    eda.fig_age_by_status(df)

    miss = pd.read_csv(config.OUTPUT_DIR / "missingness.csv")
    eda.fig_missingness(miss)

    # a short written summary, filled from the numbers just computed
    lines = [
        "# EDA notes",
        "",
        f"The cohort has {len(df)} participants from four NHANES cycles, "
        f"{int(df[config.TARGET].sum())} of them reporting a CHD diagnosis "
        f"({100*df[config.TARGET].mean():.2f}%).",
        "",
        "Prevalence by cycle:",
        "",
        prev["by_cycle"][["survey_cycle", "n", "cases", "prevalence"]]
        .to_markdown(index=False),
        "",
        "Prevalence by age group:",
        "",
        prev["by_age_group"][["age_group", "n", "cases", "prevalence"]]
        .to_markdown(index=False),
        "",
        "Variables with |skewness| above 1: " +
        (", ".join(skewed["variable"]) if len(skewed) else "none") + ".",
    ]
    (config.OUTPUT_DIR / "eda_notes.md").write_text("\n".join(lines))
    note("saved outputs/eda_notes.md")

    banner("SCRIPT 2 FINISHED")


if __name__ == "__main__":
    main()
