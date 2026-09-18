"""
Phase 7 - exploratory analysis, plus the plotting helpers used later for the
model figures. Every figure here answers one question and is written to
figures/ so it survives outside the notebook.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .utils import save_fig

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})

CHD_COLOURS = {0: "#5B8FF9", 1: "#D1495B"}


def describe_numeric(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        s = df[v].dropna()
        if s.empty:
            continue
        rows.append(dict(
            variable=v, n=len(s), missing_pct=float(df[v].isna().mean()),
            mean=s.mean(), sd=s.std(ddof=1), min=s.min(),
            q25=s.quantile(0.25), median=s.median(), q75=s.quantile(0.75),
            q95=s.quantile(0.95), max=s.max(),
            skewness=float(s.skew()),
        ))
    return pd.DataFrame(rows)


def describe_categorical(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        counts = df[v].value_counts(dropna=False)
        for level, n in counts.items():
            sub = df[df[v] == level] if pd.notna(level) else df[df[v].isna()]
            rows.append(dict(variable=v, level=str(level), n=int(n),
                             share=float(n / len(df)),
                             chd_prevalence=float(sub[config.TARGET].mean())
                             if len(sub) else np.nan))
    return pd.DataFrame(rows)


def table_one(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """The usual descriptive table split by outcome."""
    y = df[config.TARGET]
    rows = []
    for v in numeric:
        if v not in df.columns:
            continue
        a, b = df.loc[y == 1, v].dropna(), df.loc[y == 0, v].dropna()
        rows.append(dict(variable=v, type="numeric",
                         overall=f"{df[v].mean():.1f} ({df[v].std(ddof=1):.1f})",
                         chd=f"{a.mean():.1f} ({a.std(ddof=1):.1f})" if len(a) else "",
                         no_chd=f"{b.mean():.1f} ({b.std(ddof=1):.1f})" if len(b) else "",
                         missing_pct=round(float(df[v].isna().mean()), 4)))
    for v in categorical:
        if v not in df.columns:
            continue
        s = df[v]
        if pd.api.types.is_numeric_dtype(s) and set(s.dropna().unique()) <= {0.0, 1.0}:
            rows.append(dict(variable=f"{v} (yes)", type="binary",
                             overall=f"{100*s.mean():.1f}%",
                             chd=f"{100*s[y==1].mean():.1f}%",
                             no_chd=f"{100*s[y==0].mean():.1f}%",
                             missing_pct=round(float(s.isna().mean()), 4)))
        else:
            for level in sorted(s.dropna().unique(), key=str):
                m = s == level
                rows.append(dict(variable=f"{v} = {level}", type="categorical",
                                 overall=f"{100*m.mean():.1f}%",
                                 chd=f"{100*m[y==1].mean():.1f}%",
                                 no_chd=f"{100*m[y==0].mean():.1f}%",
                                 missing_pct=round(float(s.isna().mean()), 4)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

def fig_prevalence(prev_cycle: pd.DataFrame, prev_age: pd.DataFrame,
                   prev_sex: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, tbl, key, title in [
        (axes[0], prev_cycle, "survey_cycle", "By survey cycle"),
        (axes[1], prev_age, "age_group", "By age group"),
        (axes[2], prev_sex, "sex", "By sex"),
    ]:
        x = np.arange(len(tbl))
        ax.bar(x, 100 * tbl["prevalence"], color="#5B8FF9")
        ax.errorbar(x, 100 * tbl["prevalence"],
                    yerr=[100 * (tbl["prevalence"] - tbl["ci_low"]),
                          100 * (tbl["ci_high"] - tbl["prevalence"])],
                    fmt="none", ecolor="#333", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(tbl[key].astype(str), rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("CHD prevalence (%)")
    fig.suptitle("Self-reported physician-diagnosed CHD, 95% Wilson intervals", y=1.04)
    save_fig(fig, "01_chd_prevalence.png")


def fig_distributions(df: pd.DataFrame, variables: list[str]) -> None:
    variables = [v for v in variables if v in df.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    y = df[config.TARGET]
    for ax, v in zip(axes.ravel(), variables):
        for cls in (0, 1):
            s = df.loc[y == cls, v].dropna()
            if len(s) < 10:
                continue
            ax.hist(s, bins=40, density=True, alpha=0.5,
                    label=("CHD" if cls else "no CHD"), color=CHD_COLOURS[cls])
        ax.set_title(v)
        ax.set_ylabel("density")
    axes.ravel()[0].legend()
    fig.suptitle("Key variables by CHD status (densities, so the groups are comparable)",
                 y=1.02)
    fig.tight_layout()
    save_fig(fig, "02_distributions_by_chd.png")


def fig_missingness(miss: pd.DataFrame, top: int = 20) -> None:
    d = miss.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.5))
    ax.barh(d["variable"], 100 * d["missing_pct"], color="#8C8C8C")
    ax.set_xlabel("missing (%)")
    ax.set_title("Missingness in the analytical cohort")
    save_fig(fig, "03_missingness.png")


def fig_age_by_status(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    data = [df.loc[df[config.TARGET] == c, "age"].dropna() for c in (0, 1)]
    ax.boxplot(data, labels=["no CHD", "CHD"], showfliers=False)
    ax.set_ylabel("age (years)")
    ax.set_title("Age distribution by CHD status (80 is the top-coded value)")
    save_fig(fig, "04_age_by_status.png")


def fig_roc_pr(curves: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
               prevalence: float, name: str = "05_roc_pr.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, (roc, pr) in curves.items():
        axes[0].plot(roc["fpr"], roc["tpr"], lw=1.4, label=label)
        axes[1].plot(pr["recall"], pr["precision"], lw=1.4, label=label)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("sensitivity")
    axes[0].set_title("ROC")
    axes[1].axhline(prevalence, ls="--", color="grey", lw=1,
                    label=f"prevalence ({prevalence:.3f})")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, name)


def fig_calibration(cal: pd.DataFrame, name: str = "06_calibration.png") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, cal["predicted_mean"].max() * 1.1], [0, cal["predicted_mean"].max() * 1.1],
            "--", color="grey", lw=1, label="perfect calibration")
    ax.errorbar(cal["predicted_mean"], cal["observed"],
                yerr=[cal["observed"] - cal["obs_ci_low"],
                      cal["obs_ci_high"] - cal["observed"]],
                fmt="o", color="#D1495B", capsize=3, lw=1, label="deciles of risk")
    ax.set_xlabel("mean predicted risk")
    ax.set_ylabel("observed proportion with CHD")
    ax.set_title("Calibration on the test set")
    ax.legend(fontsize=8)
    save_fig(fig, name)


def fig_confusion(metrics: dict, name: str = "07_confusion_matrix.png") -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=("white" if cm[i, j] > cm.max() / 2 else "black"))
    ax.set_xticks([0, 1], ["predicted no CHD", "predicted CHD"])
    ax.set_yticks([0, 1], ["no CHD", "CHD"])
    ax.set_title(f"Test set, threshold {metrics['threshold']:.3f}")
    ax.grid(False)
    save_fig(fig, name)


def fig_importance(imp: pd.DataFrame, top: int = 15,
                   name: str = "08_permutation_importance.png") -> None:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(d) + 1.5))
    ax.barh(d["feature"], d["importance_mean"],
            xerr=d["importance_sd"], color="#5B8FF9", ecolor="#555")
    ax.set_xlabel(f"drop in {d['scoring'].iloc[0]} when the column is shuffled")
    ax.set_title("Permutation importance (association with the predictions, not causation)")
    save_fig(fig, name)


def fig_subgroups(sub: pd.DataFrame, metric: str = "roc_auc",
                  name: str = "09_subgroup_performance.png") -> None:
    d = sub[sub.get("reliable", False) == True].copy()
    if d.empty:
        return
    d["label"] = d["group"] + ": " + d["level"]
    d = d.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(d) + 1.5))
    ax.barh(d["label"], d[metric], color="#5B8FF9")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel(metric)
    ax.set_title("Test-set discrimination by subgroup (groups with enough cases only)")
    save_fig(fig, name)


def fig_threshold(sweep: pd.DataFrame, chosen: float,
                  name: str = "10_threshold_sensitivity.png") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sweep["threshold"], sweep["sensitivity"], label="sensitivity")
    ax.plot(sweep["threshold"], sweep["specificity"], label="specificity")
    ax.plot(sweep["threshold"], sweep["ppv"], label="PPV")
    ax.axvline(chosen, ls="--", color="grey", lw=1, label="chosen threshold")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("value")
    ax.set_title("How the operating point changes with the threshold")
    ax.legend(fontsize=8)
    save_fig(fig, name)
