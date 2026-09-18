"""
Script 1 - PHASES 1 to 6

    Phase 1  CHD outcome definition
    Phase 2  cycle harmonisation and merging
    Phase 3  leakage investigation
    Phase 4  eligibility and final cohort
    Phase 5  data quality and missingness
    Phase 6  imputation strategy (decided here, applied inside the pipelines)

Input  : the raw NHANES XPT files under data/raw (or $NHANES_RAW)
Output : data/processed/cohort.parquet, data/processed/feature_spec.json
         plus the audit tables listed at the end of the run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import cohort as cohort_mod
from src import config, data_loading, harmonization, leakage
from src.utils import (banner, note, pct, save_json, save_table, step,
                       write_processed)


def main() -> None:
    banner("PHASE 0 (re-check) - RAW FILE INVENTORY")
    inv = data_loading.file_inventory()
    data_loading.check_inventory(inv)
    save_table(inv, "file_inventory.csv")

    # ------------------------------------------------------------------
    banner("PHASE 1 and 2 - OUTCOME DEFINITION, HARMONISATION, MERGE")
    note(f"target: {config.TARGET_DOC['variable']} - {config.TARGET_DOC['question']}")
    note(f"coding: {config.TARGET_DOC['codes']}")
    note(f"mapping: {config.TARGET_DOC['mapping']}")

    pooled, merge_audit, target_audit, coverage = harmonization.build_all_cycles()
    save_table(merge_audit, "merge_audit.csv")
    save_table(target_audit, "target_audit.csv")
    save_table(coverage, "variable_coverage_by_cycle.csv")
    save_json(config.TARGET_DOC, "target_definition.json")

    step("target coding across cycles")
    print(target_audit[["cycle", "code_1_yes", "code_2_no", "code_7_refused",
                        "code_9_dontknow", "code_missing", "target_valid"]]
          .to_string(index=False))

    # a coding difference between cycles would show up as a code that only
    # appears in some of them
    odd = target_audit.loc[target_audit["other_codes"] != "", ["cycle", "other_codes"]]
    if len(odd):
        note(f"unexpected target codes found: {odd.to_dict('records')}")
    else:
        note("no unexpected target codes in any cycle")

    prev = harmonization.target_prevalence_tables(pooled)
    for key, tbl in prev.items():
        save_table(tbl, f"target_prevalence_{key}.csv")
    step("CHD prevalence among adults with a usable answer")
    print(prev["by_cycle"].to_string(index=False))

    # ------------------------------------------------------------------
    banner("PHASE 4 - ELIGIBILITY AND FINAL COHORT")
    coh, flow = cohort_mod.build_cohort(pooled)
    save_table(flow, "cohort_flow.csv")
    print(flow.to_string(index=False))

    summary = pd.DataFrame([dict(
        n_participants=len(coh),
        n_chd_cases=int(coh[config.TARGET].sum()),
        n_without_chd=int((coh[config.TARGET] == 0).sum()),
        prevalence=float(coh[config.TARGET].mean()),
        n_cycles=coh["survey_cycle"].nunique(),
        median_age=float(coh["age"].median()),
        pct_female=float((coh["sex"] == "Female").mean()),
    )])
    save_table(summary, "cohort_summary.csv")

    # ------------------------------------------------------------------
    banner("PHASE 3 - LEAKAGE INVESTIGATION")
    candidate_names = [v["name"] for v in config.VARIABLES
                       if v["role"] in ("predictor", "leakage_candidate")]
    candidate_names += ["sbp", "dbp", "pulse_pressure", "n_bp_readings"]
    candidate_names = [c for c in dict.fromkeys(candidate_names) if c in coh.columns]

    screen = leakage.single_variable_screen(coh, candidate_names)
    save_table(screen, "leakage_single_variable_screen.csv")
    leakage.report(screen)

    step("the cardiovascular questionnaire block on its own")
    block = screen[screen["variable"].isin(["angina", "heart_attack", "chf", "stroke"])]
    print(block[["variable", "auc_alone", "odds_ratio", "or_ci_low",
                 "or_ci_high"]].to_string(index=False))

    leak_tbl = leakage.build_leakage_table(coh, screen)
    save_table(leak_tbl, "leakage_decisions.csv")

    excluded_reasons = {r["variable"]: r["reason"] for r in leakage.LEAKAGE_RULES
                        if r["decision"].startswith("exclude")}

    # ------------------------------------------------------------------
    banner("PHASE 5 - DATA QUALITY AND MISSINGNESS")
    predictors = [v["name"] for v in config.VARIABLES if v["role"] == "predictor"]
    predictors += ["sbp", "dbp", "pulse_pressure"]
    predictors = [p for p in dict.fromkeys(predictors)
                  if p in coh.columns and p not in excluded_reasons]

    miss = cohort_mod.missingness_table(coh, predictors)
    save_table(miss, "missingness.csv")
    print(miss[["variable", "missing_pct", "missing_pct_chd", "missing_pct_no_chd",
                "missing_vs_chd_p"]].head(15).to_string(index=False))

    coh["age_group"] = pd.cut(coh["age"], bins=config.AGE_BINS,
                              labels=config.AGE_LABELS, right=False)
    for g in ["survey_cycle", "sex", "age_group"]:
        save_table(cohort_mod.missingness_by_group(coh, predictors, g),
                   f"missingness_by_{g}.csv")

    indicators, indicator_decisions = cohort_mod.decide_missing_indicators(miss)
    save_table(indicator_decisions, "missing_indicator_decisions.csv")
    note(f"missing indicators will be added for: {indicators or 'none'}")

    # Variables that are almost never observed cannot support a model.
    too_sparse = miss.loc[miss["missing_pct"] > 0.60, "variable"].tolist()
    for v in too_sparse:
        if v not in config.FORCE_MISSING_INDICATOR:
            excluded_reasons[v] = (f"observed for only {100*(1-miss.set_index('variable').loc[v,'missing_pct']):.1f}% "
                                   "of the cohort, which is too sparse to be useful")
    final_predictors = [p for p in predictors if p not in excluded_reasons]
    indicators = [i for i in indicators if i in final_predictors]

    step("final predictor set")
    note(f"{len(final_predictors)} predictors kept, {len(excluded_reasons)} variables excluded")
    print(", ".join(final_predictors))

    pred_tbl = leakage.final_predictor_table(coh, screen, final_predictors, excluded_reasons)
    save_table(pred_tbl, "predictor_table.csv")

    # ------------------------------------------------------------------
    banner("PHASE 6 - IMPUTATION STRATEGY (decided, applied later)")
    note("numeric variables: median imputation, fitted inside each training fold")
    note("categorical variables: most frequent category, fitted inside each fold")
    note(f"missing indicators only for: {indicators or 'none'}")
    note("no global imputation is performed on the full data at any point")
    note("an alternative (mean) strategy is compared in the robustness script")

    # ------------------------------------------------------------------
    keep_cols = (["SEQN", config.TARGET, "survey_cycle", "age_group", "wt_mec",
                  "wt_int", "psu", "stratum"] + final_predictors +
                 ["angina", "heart_attack", "chf", "stroke"])
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in coh.columns]
    out = coh[keep_cols].copy()
    write_processed(out, "cohort.parquet")

    spec = dict(
        predictors=final_predictors,
        missing_indicators=indicators,
        excluded=excluded_reasons,
        sensitivity_cvd_block=[c for c in ["angina", "heart_attack", "chf", "stroke"]
                               if c in coh.columns],
        target=config.TARGET,
        n=len(out), cases=int(out[config.TARGET].sum()),
        prevalence=float(out[config.TARGET].mean()),
    )
    save_json(spec, "feature_spec.json")
    with open(config.PROCESSED_DIR / "feature_spec.json", "w") as fh:
        json.dump(spec, fh, indent=2)

    banner("SCRIPT 1 FINISHED")
    note(f"cohort: {len(out)} participants, {int(out[config.TARGET].sum())} CHD cases "
         f"({pct(out[config.TARGET].mean())})")


if __name__ == "__main__":
    main()
