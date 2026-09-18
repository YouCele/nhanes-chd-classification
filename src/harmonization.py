"""
Phase 1 (outcome) and Phase 2 (harmonisation + merging).

The two phases live in one module because they share the same recoding
helpers. Every transformation writes a row into an audit table, so the
path from raw codes to the analysis variables can be checked afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_loading import load_raw
from .utils import assert_unique_id, fail, note, pct, step

# ----------------------------------------------------------------------
# recoding helpers
# ----------------------------------------------------------------------


def recode_special(s: pd.Series, codes) -> pd.Series:
    """Turn refusal / don't-know codes into missing values."""
    if codes is None:
        return s
    return s.mask(s.isin(list(codes)))


def yesno_to_binary(s: pd.Series) -> pd.Series:
    """
    NHANES yes/no items: 1 = yes, 2 = no, 7 = refused, 9 = don't know.

    Only 1 and 2 are mapped. Everything else becomes missing; refusals and
    don't-know answers are not evidence of absence, so mapping them to 0
    would invent information.
    """
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[s == 1] = 1.0
    out[s == 2] = 0.0
    return out


def clip_to_plausible(s: pd.Series, key: str) -> tuple[pd.Series, int]:
    """Set values outside the plausible range to missing, and count them."""
    if key not in config.PLAUSIBLE_RANGES:
        return s, 0
    lo, hi = config.PLAUSIBLE_RANGES[key]
    bad = s.notna() & ((s < lo) | (s > hi))
    return s.mask(bad), int(bad.sum())


# ----------------------------------------------------------------------
# Phase 1 - target
# ----------------------------------------------------------------------


def build_target(mcq: pd.DataFrame, cycle: str) -> tuple[pd.DataFrame, dict]:
    """Map MCQ160C to the binary target and record what happened to each code."""
    var = config.TARGET_SOURCE_VAR
    if var not in mcq.columns:
        fail(f"{var} not found in the MCQ file for {cycle}")

    raw = mcq[var]
    counts = raw.value_counts(dropna=False).to_dict()
    target = yesno_to_binary(raw)

    audit = dict(
        cycle=cycle,
        n_rows_mcq=len(mcq),
        code_1_yes=int((raw == 1).sum()),
        code_2_no=int((raw == 2).sum()),
        code_7_refused=int((raw == 7).sum()),
        code_9_dontknow=int((raw == 9).sum()),
        code_missing=int(raw.isna().sum()),
        other_codes=";".join(
            f"{k}:{v}" for k, v in counts.items()
            if pd.notna(k) and k not in (1, 2, 7, 9)
        ),
        target_valid=int(target.notna().sum()),
        target_positive=int((target == 1).sum()),
    )
    out = mcq[[config.ID_COL]].copy()
    out[config.TARGET] = target
    return out, audit


# ----------------------------------------------------------------------
# Phase 2 - per-cycle assembly
# ----------------------------------------------------------------------


def _harmonise_bp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average the available blood-pressure readings.

    NHANES records up to four readings. A diastolic value of exactly 0 means
    the fifth Korotkoff phase was not heard, so it is treated as missing
    rather than as a real 0 mmHg reading.
    """
    sbp_cols = [c for c in ["sbp1", "sbp2", "sbp3", "sbp4"] if c in df.columns]
    dbp_cols = [c for c in ["dbp1", "dbp2", "dbp3", "dbp4"] if c in df.columns]

    for c in dbp_cols:
        df[c] = df[c].mask(df[c] == 0)
    for c in sbp_cols:
        df[c], _ = clip_to_plausible(df[c], "sbp")
    for c in dbp_cols:
        df[c], _ = clip_to_plausible(df[c], "dbp")

    df["sbp"] = df[sbp_cols].mean(axis=1) if sbp_cols else np.nan
    df["dbp"] = df[dbp_cols].mean(axis=1) if dbp_cols else np.nan
    df["n_bp_readings"] = df[sbp_cols].notna().sum(axis=1) if sbp_cols else 0
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    return df.drop(columns=sbp_cols + dbp_cols)


def build_cycle(letter: str) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    Build one harmonised cycle table and its merge audit.

    DEMO is the backbone: every screened participant appears there, so the
    left joins show exactly how many people each file covers.
    """
    cycle = config.CYCLES[letter]
    merge_audit = []

    demo = load_raw("DEMO", letter, config.raw_columns_for("DEMO"))
    if demo is None:
        fail(f"DEMO file missing for {cycle}")
    assert_unique_id(demo, where=f"DEMO {cycle}")
    demo = demo.rename(columns=config.rename_map("DEMO"))
    demo["survey_cycle"] = cycle
    demo["cycle_letter"] = letter

    if "sddsrvyr" in demo.columns:
        expected = config.EXPECTED_SDDSRVYR[letter]
        found = sorted(demo["sddsrvyr"].dropna().unique().tolist())
        if found != [expected]:
            note(f"warning: SDDSRVYR in {cycle} is {found}, expected [{expected}]")

    df = demo
    merge_audit.append(dict(cycle=cycle, step="demographics", n_rows=len(df),
                            matched=len(df), retention=1.0))

    for stem in ["MCQ", "BPX", "BMX", "TCHOL", "HDL", "TRIGLY"]:
        part = load_raw(stem, letter, config.raw_columns_for(stem))
        if part is None:
            note(f"{cycle}: {stem} not available, columns will stay missing")
            merge_audit.append(dict(cycle=cycle, step=f"after {stem}", n_rows=len(df),
                                    matched=0, retention=0.0))
            continue
        assert_unique_id(part, where=f"{stem} {cycle}")

        present = set(part.columns)
        wanted = [v["var"] for v in config.variables_for(stem) if v["var"] in present]
        part = part[[config.ID_COL] + [c for c in wanted if c != config.ID_COL]]
        part = part.rename(columns=config.rename_map(stem))

        n_before = len(df)
        df = df.merge(part, on=config.ID_COL, how="left", validate="one_to_one")
        if len(df) != n_before:
            fail(f"row count changed when merging {stem} for {cycle}")

        # how many DEMO participants the file actually covers
        value_cols = [c for c in part.columns if c != config.ID_COL]
        matched = int(df[value_cols].notna().any(axis=1).sum()) if value_cols else 0
        merge_audit.append(dict(cycle=cycle, step=f"after {stem}", n_rows=len(df),
                                matched=matched, retention=matched / len(df)))

    # ---- target -------------------------------------------------------
    mcq_raw = load_raw("MCQ", letter, [config.ID_COL, config.TARGET_SOURCE_VAR])
    tgt, tgt_audit = build_target(mcq_raw, cycle)
    df = df.merge(tgt, on=config.ID_COL, how="left", validate="one_to_one")

    # ---- recoding ------------------------------------------------------
    df = _harmonise_bp(df)

    for entry in config.VARIABLES:
        name = entry["name"]
        if name not in df.columns:
            continue
        if entry.get("special"):
            df[name] = recode_special(df[name], entry["special"])
        if entry["dtype"] == "cat" and entry["role"] in ("predictor", "leakage_candidate"):
            # yes/no questionnaire items become 0/1; multi-level items stay as codes
            vals = set(pd.Series(df[name]).dropna().unique().tolist())
            if vals.issubset({1.0, 2.0, 7.0, 9.0}) and name not in ("sex", "pulse_regular"):
                df[name] = yesno_to_binary(df[name])

    # sex and pulse_regular keep their own coding but are made explicit
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({1: "Male", 2: "Female"})
    if "pulse_regular" in df.columns:
        df["pulse_regular"] = df["pulse_regular"].map({1: 1.0, 2: 0.0})

    # plausibility screening for the continuous variables
    implausible = {}
    for key in ["age", "pulse", "bmi", "waist", "height", "weight", "arm_circ",
                "total_chol", "hdl", "triglycerides", "ldl", "income_poverty_ratio"]:
        if key in df.columns:
            df[key], n_bad = clip_to_plausible(df[key], key)
            if n_bad:
                implausible[key] = n_bad

    tgt_audit["implausible_values_set_to_missing"] = ";".join(
        f"{k}:{v}" for k, v in implausible.items()) or "none"

    return df, merge_audit, tgt_audit


def build_all_cycles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stack the four harmonised cycles and return the two audit tables."""
    frames, merges, targets = [], [], []
    for letter in config.CYCLES:
        step(f"building cycle {config.CYCLES[letter]}")
        df, merge_audit, tgt_audit = build_cycle(letter)
        note(f"{len(df)} participants, {df.shape[1]} columns, "
             f"valid target for {int(df[config.TARGET].notna().sum())}")
        frames.append(df)
        merges.extend(merge_audit)
        targets.append(tgt_audit)

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    assert_unique_id(all_df, where="pooled four cycles")

    # columns that only some cycles have
    coverage = []
    for col in all_df.columns:
        by_cycle = all_df.groupby("survey_cycle")[col].apply(lambda s: s.notna().mean())
        coverage.append(dict(variable=col, **{c: round(v, 4) for c, v in by_cycle.items()}))
    coverage = pd.DataFrame(coverage)

    note(f"pooled data: {all_df.shape[0]} rows x {all_df.shape[1]} columns")
    return all_df, pd.DataFrame(merges), pd.DataFrame(targets), coverage


def target_prevalence_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prevalence overall and by cycle, sex and age group, among valid targets."""
    d = df[df[config.TARGET].notna()].copy()
    d["age_group"] = pd.cut(d["age"], bins=config.AGE_BINS,
                            labels=config.AGE_LABELS, right=False)

    def _tab(by):
        g = d.groupby(by, observed=True)[config.TARGET]
        out = g.agg(n="size", cases="sum")
        out["cases"] = out["cases"].astype(int)
        out["prevalence"] = out["cases"] / out["n"]
        # Wilson interval, which behaves better than the normal one for small p
        z = 1.959964
        p, n = out["prevalence"], out["n"]
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
        out["ci_low"] = (centre - half).clip(lower=0)
        out["ci_high"] = (centre + half).clip(upper=1)
        return out.reset_index()

    overall = pd.DataFrame([dict(
        n=len(d), cases=int(d[config.TARGET].sum()),
        prevalence=float(d[config.TARGET].mean()))])

    return dict(
        overall=overall,
        by_cycle=_tab("survey_cycle"),
        by_sex=_tab("sex"),
        by_age_group=_tab("age_group"),
        by_cycle_sex=_tab(["survey_cycle", "sex"]),
    )
