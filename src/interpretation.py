"""
Phase 15 - what the models are using.

Two separate things are reported and they should not be confused:

* the coefficients of the logistic regression, which describe an
  association inside this model, given the other variables;
* permutation importance, which says how much a model's predictions get
  worse when one column is shuffled.

Neither of them is a statement about biology or causation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor, feature_names


def logistic_coefficients(df: pd.DataFrame, features: list[str], y: pd.Series,
                          indicator_for: list[str]) -> pd.DataFrame:
    """
    Refit a logistic regression for interpretation and add Wald intervals.

    The model used for prediction is class-weighted and penalised, which makes
    its standard errors awkward to interpret. For the association table a
    second, almost unpenalised and unweighted model is fitted on the same
    features. The odds ratios below therefore belong to this interpretation
    model, not to the tuned classifier.
    """
    pre = build_preprocessor(df, features, scale=True, indicator_for=indicator_for,
                             drop_first=True)
    X = pre.fit_transform(df[features])
    names = feature_names(pre)
    X = np.asarray(X, dtype=float)

    fit = LogisticRegression(max_iter=5000, C=1e4, random_state=config.RANDOM_SEED)
    fit.fit(X, y)
    beta = fit.coef_.ravel()

    # Wald standard errors from the observed information matrix
    p = fit.predict_proba(X)[:, 1]
    W = p * (1 - p)
    Xd = np.hstack([np.ones((X.shape[0], 1)), X])
    try:
        cov = np.linalg.pinv(Xd.T * W @ Xd)
        se = np.sqrt(np.diag(cov))[1:]
    except np.linalg.LinAlgError:
        se = np.full_like(beta, np.nan)

    out = pd.DataFrame(dict(
        feature=names,
        coefficient=beta,
        std_error=se,
        odds_ratio=np.exp(beta),
        or_ci_low=np.exp(beta - 1.96 * se),
        or_ci_high=np.exp(beta + 1.96 * se),
    ))
    out["abs_coef"] = out["coefficient"].abs()
    out["note"] = ("coefficients are on standardised inputs, so an odds ratio is "
                   "per one standard deviation for continuous variables")
    return out.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")


def permutation_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series,
                       n_repeats: int = 10, scoring: str | None = None) -> pd.DataFrame:
    """Permutation importance on held-out data, at the original column level."""
    scoring = scoring or config.PRIMARY_METRIC
    res = permutation_importance(model, X, y, n_repeats=n_repeats,
                                 random_state=config.RANDOM_SEED,
                                 scoring=scoring, n_jobs=config.N_JOBS)
    return pd.DataFrame(dict(
        feature=X.columns,
        importance_mean=res.importances_mean,
        importance_sd=res.importances_std,
        scoring=scoring,
    )).sort_values("importance_mean", ascending=False)


def compare_importance(perm_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Put the rankings of several models side by side and flag disagreement."""
    merged = None
    for name, tbl in perm_tables.items():
        t = tbl[["feature", "importance_mean"]].copy()
        t[f"rank_{name}"] = t["importance_mean"].rank(ascending=False)
        t = t.rename(columns={"importance_mean": f"importance_{name}"})
        merged = t if merged is None else merged.merge(t, on="feature", how="outer")
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    merged["rank_spread"] = merged[rank_cols].max(axis=1) - merged[rank_cols].min(axis=1)
    merged["mean_rank"] = merged[rank_cols].mean(axis=1)
    return merged.sort_values("mean_rank")
