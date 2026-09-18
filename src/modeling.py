"""
Phases 10 to 12 - baseline models, cross-validation, restrained tuning and
model comparison.

Four candidates: a dummy classifier as the floor, logistic regression,
random forest and histogram gradient boosting. That is enough to answer the
research question. The grids are deliberately small; the point is a fair
comparison on identical folds, not a leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor
from .utils import note, step


def cv_splitter(seed: int | None = None) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                           random_state=config.RANDOM_SEED if seed is None else seed)


def make_models(df: pd.DataFrame, features: list[str],
                indicator_for: list[str],
                class_weight: str | None = "balanced") -> dict[str, Pipeline]:
    """One pipeline per model, each with its own appropriate preprocessing."""
    pre_scaled = lambda: build_preprocessor(df, features, scale=True,
                                            indicator_for=indicator_for)
    pre_plain = lambda: build_preprocessor(df, features, scale=False,
                                           indicator_for=indicator_for)

    models = {
        "dummy": Pipeline([
            ("pre", pre_plain()),
            ("clf", DummyClassifier(strategy="stratified",
                                    random_state=config.RANDOM_SEED)),
        ]),
        "logistic_regression": Pipeline([
            ("pre", pre_scaled()),
            ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight,
                                       random_state=config.RANDOM_SEED)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre_plain()),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                class_weight=("balanced_subsample" if class_weight else None),
                n_jobs=config.N_JOBS, random_state=config.RANDOM_SEED)),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", pre_plain()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                class_weight=("balanced" if class_weight else None),
                random_state=config.RANDOM_SEED)),
        ]),
    }
    return models


def cross_validate_models(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
                          seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Same folds for every model, so the comparison is paired.
    Returns a summary table and the per-fold scores for the paired test.
    """
    cv = cv_splitter(seed)
    scoring = ["roc_auc", "average_precision", "balanced_accuracy", "neg_brier_score"]
    rows, fold_scores = [], {}

    for name, pipe in models.items():
        step(f"cross-validating {name}")
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                             n_jobs=config.N_JOBS, return_train_score=True,
                             error_score="raise")
        fold_scores[name] = {m: res[f"test_{m}"] for m in scoring}
        row = dict(model=name, fit_time_s=float(np.mean(res["fit_time"])))
        for m in scoring:
            row[f"{m}_mean"] = float(np.mean(res[f"test_{m}"]))
            row[f"{m}_sd"] = float(np.std(res[f"test_{m}"], ddof=1))
        row["roc_auc_train_mean"] = float(np.mean(res["train_roc_auc"]))
        rows.append(row)
        note(f"{name}: ROC-AUC {row['roc_auc_mean']:.3f} (sd {row['roc_auc_sd']:.3f}), "
             f"PR-AUC {row['average_precision_mean']:.3f}")

    return pd.DataFrame(rows).sort_values(f"{config.PRIMARY_METRIC}_mean",
                                          ascending=False), fold_scores


def paired_fold_comparison(fold_scores: dict, metric: str = "average_precision") -> pd.DataFrame:
    """
    Compare models on the folds they share.

    With five folds a formal test has very little power, so the mean
    difference and its spread are reported rather than a p-value alone.
    """
    from scipy.stats import ttest_rel
    names = [n for n in fold_scores if n != "dummy"]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = fold_scores[a][metric], fold_scores[b][metric]
            diff = da - db
            p = float(ttest_rel(da, db).pvalue) if len(diff) > 1 else np.nan
            rows.append(dict(model_a=a, model_b=b, metric=metric,
                             mean_a=float(np.mean(da)), mean_b=float(np.mean(db)),
                             mean_difference=float(np.mean(diff)),
                             sd_difference=float(np.std(diff, ddof=1)),
                             paired_t_p=p,
                             folds_a_wins=int((diff > 0).sum()), n_folds=len(diff)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# tuning
# ----------------------------------------------------------------------

PARAM_GRIDS = {
    # only the regularisation strength is searched; the penalty is left at the
    # library default (ridge), which avoids a deprecated argument
    "logistic_regression": {
        "clf__C": [0.03, 0.1, 0.3, 1.0],
    },
    # a deeper forest is the expensive part of this grid, so only two depths
    # are searched; the project targets a laptop, not a cluster
    "random_forest": {
        "clf__max_depth": [10, None],
        "clf__min_samples_leaf": [5, 20],
    },
    "gradient_boosting": {
        "clf__learning_rate": [0.03, 0.06],
        "clf__max_leaf_nodes": [7, 15],
        "clf__min_samples_leaf": [20, 50],
    },
}


def tune_models(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
                metric: str | None = None) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Small grid search per model, scored with the primary metric."""
    metric = metric or config.PRIMARY_METRIC
    cv = cv_splitter()
    best, rows = {}, []
    for name, pipe in models.items():
        if name not in PARAM_GRIDS:
            best[name] = pipe
            continue
        step(f"tuning {name}")
        gs = GridSearchCV(pipe, PARAM_GRIDS[name], scoring=metric, cv=cv,
                          n_jobs=config.N_JOBS, refit=True, error_score="raise")
        gs.fit(X, y)
        best[name] = gs.best_estimator_
        cvres = pd.DataFrame(gs.cv_results_)
        for _, r in cvres.iterrows():
            rows.append(dict(model=name, params=str(r["params"]),
                             cv_mean=float(r["mean_test_score"]),
                             cv_sd=float(r["std_test_score"]),
                             rank=int(r["rank_test_score"]),
                             selected=bool(r["params"] == gs.best_params_)))
        note(f"{name}: best {metric} {gs.best_score_:.4f} with {gs.best_params_}")
    return best, pd.DataFrame(rows)
