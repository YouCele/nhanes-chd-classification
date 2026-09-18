"""
Phase 4 (eligibility and cohort) and Phase 5 (data quality, missingness).

The cohort is built in visible steps so that every drop in sample size can be
pointed at. No complete-case rule is applied to the predictors: dropping
people because a laboratory value is missing would quietly change the
population, and the fasting lipids alone would remove most of the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from . import config
from .utils import note, pct, step


def build_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the eligibility rules and return the cohort plus the flow table."""
    flow = []

    def record(label, frame, reason=""):
        flow.append(dict(step=label, n=len(frame),
                         cases=int(frame[config.TARGET].sum()) if
                         frame[config.TARGET].notna().all() else np.nan,
                         reason=reason))

    step("building the analytical cohort")
    d = df.copy()
    flow.append(dict(step="all screened participants, four cycles", n=len(d),
                     cases=np.nan, reason="DEMO row count"))

    d = d[d["age"] >= config.MIN_AGE]
    flow.append(dict(step=f"adults aged {config.MIN_AGE}+", n=len(d), cases=np.nan,
                     reason="the CHD question is only asked of adults 20 and over"))

    if config.REQUIRE_MEC_EXAM:
        d = d[d["exam_status"] == 2]
        flow.append(dict(step="examined in the mobile examination centre", n=len(d),
                         cases=np.nan,
                         reason=("the intended predictors include measured blood "
                                 "pressure and body size, which interview-only "
                                 "participants never have")))

    valid = d[config.TARGET].notna()
    n_invalid = int((~valid).sum())
    d = d[valid]
    flow.append(dict(step="usable CHD answer (yes or no)", n=len(d),
                     cases=int(d[config.TARGET].sum()),
                     reason=f"{n_invalid} refused, did not know, or had no answer recorded"))

    # people with no measured data at all would contribute nothing but their
    # demographics; they are kept, but counted, so the choice is visible
    exam_cols = [c for c in ["sbp", "dbp", "bmi", "waist"] if c in d.columns]
    no_exam = d[exam_cols].isna().all(axis=1).sum() if exam_cols else 0
    flow.append(dict(step="final analytical cohort", n=len(d),
                     cases=int(d[config.TARGET].sum()),
                     reason=(f"kept; {int(no_exam)} of them have no usable examination "
                             "measurement, handled by imputation rather than exclusion")))

    flow_df = pd.DataFrame(flow)
    flow_df["pct_of_previous"] = (flow_df["n"] / flow_df["n"].shift(1)).round(4)

    note(f"final cohort: {len(d)} participants, {int(d[config.TARGET].sum())} CHD cases "
         f"({pct(d[config.TARGET].mean())})")
    return d.reset_index(drop=True), flow_df


def missingness_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Missingness overall, by cycle, by CHD status, with a test of the difference."""
    rows = []
    y = df[config.TARGET]
    for v in variables:
        if v not in df.columns:
            continue
        miss = df[v].isna()
        row = dict(variable=v,
                   n_missing=int(miss.sum()),
                   missing_pct=float(miss.mean()),
                   missing_pct_chd=float(miss[y == 1].mean()),
                   missing_pct_no_chd=float(miss[y == 0].mean()))
        for cycle, sub in df.groupby("survey_cycle"):
            row[f"missing_{cycle}"] = float(sub[v].isna().mean())

        p = np.nan
        if miss.any() and (~miss).any():
            tab = pd.crosstab(miss, y)
            if tab.shape == (2, 2):
                try:
                    p = float(chi2_contingency(tab)[1])
                except ValueError:
                    p = np.nan
        row["missing_vs_chd_p"] = p

        # is missingness related to age? a simple rank test is enough here
        if miss.any() and (~miss).any() and "age" in df.columns:
            try:
                row["missing_vs_age_p"] = float(
                    mannwhitneyu(df.loc[miss, "age"].dropna(),
                                 df.loc[~miss, "age"].dropna()).pvalue)
                row["median_age_missing"] = float(df.loc[miss, "age"].median())
                row["median_age_observed"] = float(df.loc[~miss, "age"].median())
            except ValueError:
                row["missing_vs_age_p"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    return out


def missingness_by_group(df: pd.DataFrame, variables: list[str],
                         group: str) -> pd.DataFrame:
    """Missingness of each variable inside demographic groups."""
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        g = df.groupby(group, observed=True)[v].apply(lambda s: float(s.isna().mean()))
        rows.append(dict(variable=v, **{str(k): round(val, 4) for k, val in g.items()}))
    return pd.DataFrame(rows)


def decide_missing_indicators(miss_tbl: pd.DataFrame,
                              threshold: float = 0.10,
                              p_threshold: float = 0.01) -> tuple[list[str], pd.DataFrame]:
    """
    Decide where a missing indicator is worth adding.

    The rule: add an indicator when a lot of values are missing and the
    missingness is related to the outcome, or when the variable is measured
    only in a sub-sample (those are listed in the config). Adding indicators
    everywhere would add noise columns for variables missing in under 1% of
    rows, which is not useful.
    """
    decisions = []
    keep = []
    for _, r in miss_tbl.iterrows():
        v = r["variable"]
        forced = v in config.FORCE_MISSING_INDICATOR
        high = r["missing_pct"] >= threshold
        related = pd.notna(r.get("missing_vs_chd_p")) and r["missing_vs_chd_p"] < p_threshold
        use = bool(forced or (high and related))
        if use:
            keep.append(v)
        decisions.append(dict(
            variable=v, missing_pct=r["missing_pct"],
            missing_vs_chd_p=r.get("missing_vs_chd_p"),
            sub_sample_by_design=forced,
            indicator_added=use,
            reason=("measured only in a survey sub-sample" if forced else
                    ("high missingness and missingness differs by CHD status" if use else
                     "missingness is low or unrelated to the outcome, imputation alone"))))
    return keep, pd.DataFrame(decisions)
