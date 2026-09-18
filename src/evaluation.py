"""
Phases 11, 13 and 14 - metrics for an imbalanced problem, confidence
intervals, and calibration.

Accuracy is deliberately not used as a headline number: with a prevalence
around a few percent, predicting "no CHD" for everybody already looks
accurate and is useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from . import config


def classification_metrics(y_true: np.ndarray, proba: np.ndarray,
                           threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens and (ppv + sens) > 0) else np.nan
    return dict(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, proba)),
        pr_auc=float(average_precision_score(y_true, proba)),
        brier=float(brier_score_loss(y_true, proba)),
        sensitivity=float(sens), specificity=float(spec),
        ppv=float(ppv), npv=float(npv), f1=float(f1) if f1 == f1 else np.nan,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        n=int(len(y_true)), prevalence=float(np.mean(y_true)),
    )


def bootstrap_metric_ci(y_true: np.ndarray, proba: np.ndarray,
                        metric: str = "roc_auc", n_boot: int | None = None,
                        seed: int | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap over participants. Stratified so both classes stay in."""
    n_boot = n_boot or config.BOOTSTRAP_N
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    fn = roc_auc_score if metric == "roc_auc" else average_precision_score
    point = float(fn(y_true, proba))

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        s = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                            rng.choice(neg_idx, len(neg_idx), replace=True)])
        stats[i] = fn(y_true[s], proba[s])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def choose_threshold(y_true: np.ndarray, proba: np.ndarray,
                     rule: str = "youden") -> float:
    """
    Pick an operating point on validation data, never on the test set.

    "youden" maximises sensitivity + specificity - 1.
    "prevalence" simply uses the observed prevalence as the cut-off, which is
    a reasonable default when the classes are very unbalanced.
    """
    if rule == "prevalence":
        return float(np.mean(y_true))
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray,
                    grid: np.ndarray | None = None) -> pd.DataFrame:
    grid = grid if grid is not None else np.quantile(proba, np.linspace(0.5, 0.995, 25))
    return pd.DataFrame([classification_metrics(y_true, proba, t) for t in np.unique(grid)])


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------

def calibration_table(y_true: np.ndarray, proba: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Observed versus predicted risk inside quantile bins."""
    q = pd.qcut(proba, q=n_bins, duplicates="drop")
    d = pd.DataFrame(dict(y=y_true, p=proba, bin=q))
    out = d.groupby("bin", observed=True).agg(
        n=("y", "size"), observed=("y", "mean"),
        predicted_mean=("p", "mean"), predicted_min=("p", "min"),
        predicted_max=("p", "max"), cases=("y", "sum")).reset_index()
    out["bin"] = out["bin"].astype(str)
    # Wilson interval for the observed proportion in each bin
    z = 1.959964
    p, n = out["observed"], out["n"]
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    out["obs_ci_low"] = (centre - half).clip(lower=0)
    out["obs_ci_high"] = (centre + half).clip(upper=1)
    return out


def calibration_slope_intercept(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """
    Regress the outcome on the logit of the predicted probability.

    Slope 1 and intercept 0 mean the probabilities are calibrated. A slope
    below 1 means the predictions are too extreme in both directions.
    """
    eps = 1e-6
    p = np.clip(proba, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y_true)
    slope = float(fit.coef_[0][0])
    intercept_full = float(fit.intercept_[0])

    fixed = LogisticRegression(C=1e6, max_iter=1000, fit_intercept=True)
    fixed.coef_ = np.array([[1.0]])
    # calibration-in-the-large: intercept with the slope forced to 1
    offset = logit.ravel()
    mean_obs, mean_pred = float(np.mean(y_true)), float(np.mean(p))
    return dict(
        calibration_slope=slope,
        calibration_intercept=intercept_full,
        mean_predicted_risk=mean_pred,
        observed_risk=mean_obs,
        ratio_observed_expected=mean_obs / mean_pred if mean_pred > 0 else np.nan,
        brier=float(brier_score_loss(y_true, proba)),
    )


def calibration_by_risk_band(y_true: np.ndarray, proba: np.ndarray,
                             edges=(0, 0.02, 0.05, 0.10, 0.20, 1.01)) -> pd.DataFrame:
    """Low, middle and high predicted risk looked at separately."""
    band = pd.cut(proba, bins=list(edges), right=False)
    d = pd.DataFrame(dict(y=y_true, p=proba, band=band))
    out = d.groupby("band", observed=True).agg(
        n=("y", "size"), cases=("y", "sum"),
        observed_risk=("y", "mean"), mean_predicted=("p", "mean")).reset_index()
    out["band"] = out["band"].astype(str)
    out["difference"] = out["observed_risk"] - out["mean_predicted"]
    return out


# ----------------------------------------------------------------------
# curves for plotting
# ----------------------------------------------------------------------

def roc_points(y_true, proba) -> pd.DataFrame:
    fpr, tpr, thr = roc_curve(y_true, proba)
    return pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thr))


def pr_points(y_true, proba) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    return pd.DataFrame(dict(precision=prec[:-1], recall=rec[:-1], threshold=thr))
