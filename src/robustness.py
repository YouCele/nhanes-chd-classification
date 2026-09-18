"""
Phases 16 to 19 - error analysis, sensitivity checks, subgroup performance
and the survey-design question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config
from .evaluation import (calibration_slope_intercept, classification_metrics,
                         threshold_sweep)
from .statistics import compare_categorical, compare_continuous
from .utils import note, step


# ----------------------------------------------------------------------
# Phase 16 - error analysis
# ----------------------------------------------------------------------

def error_frame(test_df: pd.DataFrame, y_true: np.ndarray, proba: np.ndarray,
                threshold: float) -> pd.DataFrame:
    d = test_df.copy()
    d["y_true"] = y_true
    d["predicted_risk"] = proba
    d["y_pred"] = (proba >= threshold).astype(int)
    d["error_type"] = np.select(
        [(d.y_true == 1) & (d.y_pred == 1), (d.y_true == 0) & (d.y_pred == 0),
         (d.y_true == 0) & (d.y_pred == 1), (d.y_true == 1) & (d.y_pred == 0)],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unclassified")
    # distance from the threshold: how confident the model was when it was wrong
    d["confidence_gap"] = (d["predicted_risk"] - threshold).abs()
    return d


def error_profile(err: pd.DataFrame, numeric_vars: list[str],
                  categorical_vars: list[str]) -> pd.DataFrame:
    """Compare the four error groups on the variables that matter clinically."""
    rows = []
    for grp, sub in err.groupby("error_type"):
        row = dict(error_type=grp, n=len(sub),
                   mean_predicted_risk=float(sub["predicted_risk"].mean()))
        for v in numeric_vars:
            if v in sub.columns:
                row[f"median_{v}"] = float(sub[v].median())
        for v in categorical_vars:
            if v in sub.columns:
                s = sub[v]
                if s.dropna().nunique() <= 2 and pd.api.types.is_numeric_dtype(s):
                    row[f"pct_{v}"] = float(s.mean(skipna=True))
                else:
                    top = s.value_counts(normalize=True)
                    row[f"most_common_{v}"] = (f"{top.index[0]} ({top.iloc[0]:.0%})"
                                               if len(top) else "")
        rows.append(row)
    return pd.DataFrame(rows)


def compare_error_groups(err: pd.DataFrame, group_a: str, group_b: str,
                         numeric_vars: list[str],
                         categorical_vars: list[str]) -> pd.DataFrame:
    """
    Formal comparison between two error groups, for example false negatives
    against true positives: what distinguishes the cases the model misses?
    """
    sub = err[err["error_type"].isin([group_a, group_b])].copy()
    if sub["error_type"].nunique() < 2 or len(sub) < 40:
        return pd.DataFrame()
    sub = sub.rename(columns={config.TARGET: "_orig_target"})
    sub[config.TARGET] = (sub["error_type"] == group_a).astype(int)
    cont = compare_continuous(sub, [v for v in numeric_vars if v in sub.columns])
    cat = compare_categorical(sub, [v for v in categorical_vars if v in sub.columns])
    cont["comparison"] = f"{group_a} vs {group_b}"
    if len(cat):
        cat["comparison"] = f"{group_a} vs {group_b}"
    keep_cont = ["comparison", "variable", "n_chd", "n_no_chd", "mean_chd",
                 "mean_no_chd", "cohens_d", "welch_p"]
    keep_cat = ["comparison", "variable", "odds_ratio", "or_ci_low", "or_ci_high",
                "cramers_v", "p_value"]
    parts = [cont[[c for c in keep_cont if c in cont.columns]].rename(
        columns={"n_chd": "n_group_a", "n_no_chd": "n_group_b",
                 "mean_chd": "mean_group_a", "mean_no_chd": "mean_group_b"})]
    if len(cat):
        parts.append(cat[[c for c in keep_cat if c in cat.columns]])
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------
# Phase 17 - robustness
# ----------------------------------------------------------------------

def repeated_split_stability(model, df: pd.DataFrame, features: list[str],
                             y: pd.Series, n_repeats: int = 10) -> pd.DataFrame:
    """Refit on several random splits to see how much the metrics move."""
    rows = []
    for i in range(n_repeats):
        Xtr, Xte, ytr, yte = train_test_split(
            df[features], y, test_size=config.TEST_SIZE, stratify=y,
            random_state=config.RANDOM_SEED + i)
        m = clone(model).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        rows.append(dict(repeat=i, roc_auc=roc_auc_score(yte, p),
                         pr_auc=average_precision_score(yte, p),
                         test_prevalence=float(yte.mean())))
    return pd.DataFrame(rows)


def imputation_sensitivity(model_factory, df: pd.DataFrame, features: list[str],
                           y: pd.Series, indicator_for: list[str],
                           strategies=("median", "mean")) -> pd.DataFrame:
    """Does the choice of numeric imputation change anything material?"""
    from sklearn.model_selection import cross_val_score
    from .modeling import cv_splitter
    rows = []
    for strategy in strategies:
        pipe = model_factory(strategy)
        cv = cv_splitter()
        auc = cross_val_score(pipe, df[features], y, cv=cv, scoring="roc_auc",
                              n_jobs=config.N_JOBS)
        ap = cross_val_score(pipe, df[features], y, cv=cv, scoring="average_precision",
                             n_jobs=config.N_JOBS)
        rows.append(dict(imputation=strategy, roc_auc_mean=float(auc.mean()),
                         roc_auc_sd=float(auc.std(ddof=1)),
                         pr_auc_mean=float(ap.mean()), pr_auc_sd=float(ap.std(ddof=1))))
    return pd.DataFrame(rows)


def predictor_reduction(model, df: pd.DataFrame, feature_sets: dict[str, list[str]],
                        y: pd.Series, indicator_for: list[str] | None = None) -> pd.DataFrame:
    """
    Compare feature sets, for example with and without the fasting lipids.

    The pipeline is rebuilt for every set, because a fitted ColumnTransformer
    is tied to the exact columns it saw.
    """
    from sklearn.model_selection import cross_val_score
    from .modeling import cv_splitter
    from .preprocessing import rebuild_for_features
    rows = []
    for label, feats in feature_sets.items():
        feats = [f for f in feats if f in df.columns]
        pipe = rebuild_for_features(model, df, feats, indicator_for)
        cv = cv_splitter()
        auc = cross_val_score(pipe, df[feats], y, cv=cv, scoring="roc_auc",
                              n_jobs=config.N_JOBS, error_score="raise")
        ap = cross_val_score(clone(pipe), df[feats], y, cv=cv,
                             scoring="average_precision", n_jobs=config.N_JOBS,
                             error_score="raise")
        rows.append(dict(feature_set=label, n_features=len(feats),
                         roc_auc_mean=float(auc.mean()), roc_auc_sd=float(auc.std(ddof=1)),
                         pr_auc_mean=float(ap.mean()), pr_auc_sd=float(ap.std(ddof=1))))
    return pd.DataFrame(rows)


def cycle_holdout(model, df: pd.DataFrame, features: list[str], y: pd.Series,
                  train_cycles: list[str], test_cycles: list[str]) -> dict:
    """Train on earlier cycles, test on later ones."""
    tr = df["survey_cycle"].isin(train_cycles)
    te = df["survey_cycle"].isin(test_cycles)
    m = clone(model).fit(df.loc[tr, features], y[tr])
    p = m.predict_proba(df.loc[te, features])[:, 1]
    out = dict(train_cycles=";".join(train_cycles), test_cycles=";".join(test_cycles),
               n_train=int(tr.sum()), n_test=int(te.sum()),
               train_prevalence=float(y[tr].mean()), test_prevalence=float(y[te].mean()),
               roc_auc=float(roc_auc_score(y[te], p)),
               pr_auc=float(average_precision_score(y[te], p)))
    out.update({f"cal_{k}": v for k, v in
                calibration_slope_intercept(y[te].values, p).items()})
    return out


def class_weight_sensitivity(model_factory, df: pd.DataFrame, features: list[str],
                             y: pd.Series) -> pd.DataFrame:
    from sklearn.model_selection import cross_val_score
    from .modeling import cv_splitter
    rows = []
    for weight in ["balanced", None]:
        pipe = model_factory(weight)
        cv = cv_splitter()
        auc = cross_val_score(pipe, df[features], y, cv=cv, scoring="roc_auc",
                              n_jobs=config.N_JOBS)
        ap = cross_val_score(pipe, df[features], y, cv=cv, scoring="average_precision",
                             n_jobs=config.N_JOBS)
        rows.append(dict(class_weight=str(weight), roc_auc_mean=float(auc.mean()),
                         pr_auc_mean=float(ap.mean()),
                         roc_auc_sd=float(auc.std(ddof=1)),
                         pr_auc_sd=float(ap.std(ddof=1))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Phase 18 - subgroups
# ----------------------------------------------------------------------

def subgroup_performance(test_df: pd.DataFrame, y_true: np.ndarray, proba: np.ndarray,
                         threshold: float, groups: list[str],
                         min_cases: int = 15) -> pd.DataFrame:
    """
    Performance inside demographic groups.

    Subgroups with very few cases are reported with their size but their
    metrics are left blank, because an AUC computed on a handful of cases is
    not informative.
    """
    d = test_df.copy()
    d["_y"] = y_true
    d["_p"] = proba
    rows = []
    for g in groups:
        if g not in d.columns:
            continue
        for level, sub in d.groupby(g, observed=True):
            n_cases = int(sub["_y"].sum())
            row = dict(group=g, level=str(level), n=len(sub), cases=n_cases,
                       prevalence=float(sub["_y"].mean()))
            if n_cases >= min_cases and sub["_y"].nunique() == 2:
                m = classification_metrics(sub["_y"].values, sub["_p"].values, threshold)
                cal = calibration_slope_intercept(sub["_y"].values, sub["_p"].values)
                row.update({k: m[k] for k in
                            ["roc_auc", "pr_auc", "sensitivity", "specificity",
                             "ppv", "npv", "brier"]})
                row["calibration_slope"] = cal["calibration_slope"]
                row["observed_over_expected"] = cal["ratio_observed_expected"]
                row["reliable"] = True
            else:
                row["reliable"] = False
                row["note"] = f"only {n_cases} cases, too few for a stable estimate"
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Phase 19 - survey design
# ----------------------------------------------------------------------

def weighted_prevalence(df: pd.DataFrame, weight_col: str = "wt_mec",
                        n_cycles: int = 4) -> dict:
    """
    Design-based prevalence with a linearised standard error.

    NHANES is a stratified, multistage probability sample. For a descriptive
    estimate the 2-year weights are divided by the number of cycles, and the
    variance uses the masked strata and PSUs with the usual ratio-estimator
    linearisation. This is a population estimate; it is kept separate from the
    model evaluation, which is about the analytical sample only.
    """
    d = df[[config.TARGET, weight_col, "stratum", "psu"]].dropna()
    w = d[weight_col].to_numpy(dtype=float) / n_cycles
    y = d[config.TARGET].to_numpy(dtype=float)
    total_w = w.sum()
    p_hat = float((w * y).sum() / total_w)

    u = w * (y - p_hat) / total_w
    d = d.assign(_u=u)
    var = 0.0
    for _, stratum in d.groupby("stratum"):
        psu_sums = stratum.groupby("psu")["_u"].sum().to_numpy()
        n_h = len(psu_sums)
        if n_h < 2:
            continue
        var += n_h / (n_h - 1) * ((psu_sums - psu_sums.mean()) ** 2).sum()
    se = float(np.sqrt(var))
    return dict(weighted_prevalence=p_hat, standard_error=se,
                ci_low=max(p_hat - 1.96 * se, 0.0), ci_high=p_hat + 1.96 * se,
                unweighted_prevalence=float(df[config.TARGET].mean()),
                n=int(len(d)), note=("design-based estimate using masked strata and "
                                     "PSUs, MEC weights divided by the number of cycles"))


def weighted_prevalence_by_group(df: pd.DataFrame, group: str,
                                 weight_col: str = "wt_mec") -> pd.DataFrame:
    rows = []
    for level, sub in df.groupby(group, observed=True):
        if sub[config.TARGET].notna().sum() < 50:
            continue
        res = weighted_prevalence(sub, weight_col=weight_col)
        rows.append(dict(group=group, level=str(level), **res))
    return pd.DataFrame(rows)
