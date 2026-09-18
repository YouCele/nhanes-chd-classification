"""
Phase 3 - leakage investigation.

The question is not "is this variable strongly associated with CHD?" but
"would this information plausibly be available, and would using it make the
prediction task circular?".

Two things happen here:

1. Every candidate variable gets an explicit decision with a written reason.
2. The decisions that rest on an empirical claim are checked against the
   data: single-variable discrimination (AUC of the variable on its own),
   the odds ratio, and whether the variable's missingness is related to the
   target.

Nothing is removed silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .utils import note, step

# ----------------------------------------------------------------------
# The decisions. Each one is a claim that the audit at the end re-checks.
# ----------------------------------------------------------------------

LEAKAGE_RULES = [
    dict(
        variable="mcq160c_raw",
        category="the target itself",
        decision="exclude",
        reason="This is the source of the outcome. Keeping it would be circular.",
    ),
    dict(
        variable="age_told_chd",
        category="derived from the diagnosis",
        decision="exclude",
        reason=(
            "MCQ180C asks at what age the participant was told they had CHD. "
            "It is only recorded for people who answered yes, so it is a direct "
            "restatement of the label. It is also absent from the 2017-2018 file."
        ),
    ),
    dict(
        variable="angina",
        category="same questionnaire block, overlapping clinical entity",
        decision="exclude from the main model",
        reason=(
            "Angina is asked in the same MCQ160 series, in the same interview, "
            "and in clinical terms angina is a presentation of coronary disease "
            "rather than a separate risk factor. A model using it would mostly "
            "be predicting one self-report from a neighbouring self-report."
        ),
    ),
    dict(
        variable="heart_attack",
        category="same questionnaire block, overlapping clinical entity",
        decision="exclude from the main model",
        reason=(
            "Myocardial infarction is part of coronary heart disease. Someone "
            "reporting a heart attack has, by definition, coronary disease, so "
            "this is close to a second label rather than a predictor."
        ),
    ),
    dict(
        variable="chf",
        category="same questionnaire block, downstream condition",
        decision="exclude from the main model",
        reason=(
            "Heart failure is frequently a consequence of coronary disease and "
            "is asked in the same block. It is kept for the sensitivity model "
            "but not used in the main one."
        ),
    ),
    dict(
        variable="stroke",
        category="same questionnaire block, different vascular territory",
        decision="exclude from the main model",
        reason=(
            "Stroke is a different event from CHD, so this is a weaker case than "
            "angina or MI. It is still excluded from the main model for "
            "consistency: the whole MCQ160 cardiovascular block is treated the "
            "same way, and its effect is quantified in the sensitivity model."
        ),
    ),
    dict(
        variable="told_overweight",
        category="strongly related but legitimate",
        decision="include",
        reason=(
            "Being told by a doctor that you are overweight reflects contact "
            "with the health system and body size. It is not a statement about "
            "heart disease, so it stays."
        ),
    ),
    dict(
        variable="fam_hist_mi",
        category="risk factor, not an outcome",
        decision="include",
        reason=(
            "Family history of early heart attack is a classic risk factor and "
            "says nothing about the participant's own diagnosis."
        ),
    ),
    dict(
        variable="asthma_still",
        category="conditional question",
        decision="include with caution",
        reason=(
            "MCQ035 is only asked of participants who reported asthma, so it is "
            "missing by design for everyone else. It is kept but its missingness "
            "is structural, which the missingness analysis records."
        ),
    ),
    dict(
        variable="triglycerides",
        category="sub-sample measurement",
        decision="include with a missing indicator",
        reason=(
            "Measured only in the fasting sub-sample. Fasting status is set by "
            "the survey session, not by the participant's heart disease, so the "
            "missingness is a survey feature. The indicator makes that explicit."
        ),
    ),
    dict(
        variable="ldl",
        category="sub-sample measurement",
        decision="include with a missing indicator",
        reason="Same fasting sub-sample as triglycerides.",
    ),
    dict(
        variable="wt_int",
        category="survey design",
        decision="exclude from predictors",
        reason=(
            "Sample weights describe the sampling design, not the participant. "
            "They are kept aside for the weighted descriptive estimates."
        ),
    ),
    dict(
        variable="wt_mec",
        category="survey design",
        decision="exclude from predictors",
        reason="Same as the interview weight.",
    ),
    dict(
        variable="psu",
        category="survey design",
        decision="exclude from predictors",
        reason="Masked variance unit, meaningless as a predictor.",
    ),
    dict(
        variable="stratum",
        category="survey design",
        decision="exclude from predictors",
        reason="Masked variance stratum, meaningless as a predictor.",
    ),
    dict(
        variable="sddsrvyr",
        category="survey administration",
        decision="exclude from predictors",
        reason=(
            "The cycle number is kept as a grouping variable for the temporal "
            "checks, but a model should not learn from which survey round a "
            "person happened to be sampled in."
        ),
    ),
    dict(
        variable="exam_status",
        category="eligibility flag",
        decision="exclude from predictors",
        reason="Used to define the cohort, so it is constant after selection.",
    ),
    dict(
        variable="seqn",
        category="identifier",
        decision="exclude from predictors",
        reason="Sequence number carries no information about the person.",
    ),
]


def single_variable_screen(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    For each variable: how well does it separate the classes on its own?

    For numeric variables the AUC is computed directly. For binary variables
    the AUC and the odds ratio are both reported. A variable that reaches an
    AUC close to 1 on its own deserves a hard look.
    """
    y = df[config.TARGET]
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        s = df[v]
        ok = s.notna() & y.notna()
        if ok.sum() < 50 or s[ok].nunique() < 2:
            rows.append(dict(variable=v, n_used=int(ok.sum()), auc_alone=np.nan,
                             odds_ratio=np.nan, or_ci_low=np.nan, or_ci_high=np.nan,
                             missing_pct=float(s.isna().mean()),
                             missing_vs_target_p=np.nan))
            continue

        if pd.api.types.is_numeric_dtype(s):
            auc = roc_auc_score(y[ok], s[ok])
            auc = max(auc, 1 - auc)   # direction does not matter for screening
        else:
            auc = np.nan

        odds, lo, hi = np.nan, np.nan, np.nan
        vals = set(pd.Series(s[ok]).unique().tolist())
        if vals.issubset({0.0, 1.0}):
            a = int(((s == 1) & (y == 1)).sum())
            b = int(((s == 1) & (y == 0)).sum())
            c = int(((s == 0) & (y == 1)).sum())
            d = int(((s == 0) & (y == 0)).sum())
            # Haldane correction so that a zero cell does not break the log
            a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            odds = (a_ * d_) / (b_ * c_)
            se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
            lo, hi = np.exp(np.log(odds) - 1.96 * se), np.exp(np.log(odds) + 1.96 * se)

        # is missingness itself related to the target?
        p_missing = np.nan
        if s.isna().any() and s.notna().any():
            from scipy.stats import chi2_contingency
            tab = pd.crosstab(s.isna(), y)
            if tab.shape == (2, 2) and tab.values.min() >= 0:
                try:
                    p_missing = float(chi2_contingency(tab)[1])
                except ValueError:
                    p_missing = np.nan

        rows.append(dict(variable=v, n_used=int(ok.sum()), auc_alone=auc,
                         odds_ratio=odds, or_ci_low=lo, or_ci_high=hi,
                         missing_pct=float(s.isna().mean()),
                         missing_vs_target_p=p_missing))
    return pd.DataFrame(rows).sort_values("auc_alone", ascending=False, na_position="last")


def build_leakage_table(df: pd.DataFrame, screen: pd.DataFrame) -> pd.DataFrame:
    """Join the written decisions with the empirical screen."""
    meta = {v["name"]: v for v in config.VARIABLES}
    rules = pd.DataFrame(LEAKAGE_RULES)
    rules["meaning"] = rules["variable"].map(
        lambda n: meta.get(n, {}).get("meaning", "derived variable"))
    out = rules.merge(screen, on="variable", how="left")
    cols = ["variable", "meaning", "category", "decision", "reason",
            "n_used", "auc_alone", "odds_ratio", "or_ci_low", "or_ci_high",
            "missing_pct", "missing_vs_target_p"]
    return out[[c for c in cols if c in out.columns]]


def final_predictor_table(df: pd.DataFrame, screen: pd.DataFrame,
                          included: list[str], excluded: dict[str, str]) -> pd.DataFrame:
    """The predictor table asked for in the brief: what is in, what is out, why."""
    meta = {v["name"]: v for v in config.VARIABLES}
    rows = []
    for name in included + list(excluded):
        entry = meta.get(name, {})
        s = df[name] if name in df.columns else pd.Series(dtype=float)
        rows.append(dict(
            variable=name,
            source_file=entry.get("file", "derived"),
            meaning=entry.get("meaning", "derived from raw NHANES variables"),
            data_type=("numeric" if pd.api.types.is_numeric_dtype(s) and
                       (s.dropna().nunique() > 2) else
                       ("binary" if s.dropna().nunique() == 2 else "categorical")),
            missingness=float(s.isna().mean()) if len(s) else np.nan,
            included=name in included,
            reason=excluded.get(name, "retained as a legitimate predictor"),
        ))
    return pd.DataFrame(rows).sort_values(["included", "variable"], ascending=[False, True])


def report(screen: pd.DataFrame, auc_threshold: float = 0.90) -> None:
    step("variables that separate the classes unusually well on their own")
    suspicious = screen[screen["auc_alone"] >= auc_threshold]
    if len(suspicious) == 0:
        note(f"none reach AUC >= {auc_threshold} on their own")
    else:
        for _, r in suspicious.iterrows():
            note(f"{r['variable']}: AUC {r['auc_alone']:.3f} - check the decision table")
