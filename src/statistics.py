"""
Phase 8 - group comparisons between participants with and without the
self-reported CHD outcome.

Each continuous variable gets Welch's t-test (means, unequal variance) and
the Mann-Whitney U test (ranks, no distributional assumption), plus two
effect sizes. Each categorical variable gets a chi-square test, or Fisher's
exact test when a 2x2 table has small counts, plus Cramer's V and an odds
ratio where that makes sense.

p-values are corrected with Benjamini-Hochberg because many variables are
tested at once. With more than twenty thousand participants, very small
p-values are easy to get, so the effect sizes matter more than the stars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ----------------------------------------------------------------------
# effect sizes
# ----------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def rank_biserial(u_stat: float, na: int, nb: int) -> float:
    """Effect size for Mann-Whitney: 0 = no separation, 1 = complete."""
    return 2 * u_stat / (na * nb) - 1


def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return np.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else np.nan


def mean_diff_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Welch confidence interval for the difference in means."""
    na, nb = len(a), len(b)
    diff = np.mean(a) - np.mean(b)
    va, vb = np.var(a, ddof=1) / na, np.var(b, ddof=1) / nb
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    t = stats.t.ppf(1 - alpha / 2, dof)
    return diff, diff - t * se, diff + t * se


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """a = exposed cases, b = exposed non-cases, c, d the unexposed pair."""
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a_ * d_) / (b_ * c_)
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    return or_, float(np.exp(np.log(or_) - 1.96 * se)), float(np.exp(np.log(or_) + 1.96 * se))


def bh_fdr(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, ignoring missing entries."""
    p = pvals.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pvals
    adj = (p.values * m / (np.arange(1, m + 1)))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = pd.Series(np.nan, index=pvals.index)
    out.loc[p.index] = np.clip(adj, 0, 1)
    return out


# ----------------------------------------------------------------------
# the comparison tables
# ----------------------------------------------------------------------

def compare_continuous(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    y = df[config.TARGET]
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        a = df.loc[y == 1, v].dropna().values
        b = df.loc[y == 0, v].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        t_p = stats.ttest_ind(a, b, equal_var=False).pvalue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        diff, lo, hi = mean_diff_ci(a, b)
        rows.append(dict(
            variable=v, n_chd=len(a), n_no_chd=len(b),
            mean_chd=np.mean(a), mean_no_chd=np.mean(b),
            sd_chd=np.std(a, ddof=1), sd_no_chd=np.std(b, ddof=1),
            median_chd=np.median(a), median_no_chd=np.median(b),
            iqr_chd=np.subtract(*np.percentile(a, [75, 25])),
            iqr_no_chd=np.subtract(*np.percentile(b, [75, 25])),
            mean_difference=diff, diff_ci_low=lo, diff_ci_high=hi,
            welch_p=t_p, mannwhitney_p=float(u.pvalue),
            cohens_d=cohens_d(a, b),
            rank_biserial=rank_biserial(float(u.statistic), len(a), len(b)),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        out["welch_p_fdr"] = bh_fdr(out["welch_p"])
        out["abs_d"] = out["cohens_d"].abs()
        out = out.sort_values("abs_d", ascending=False).drop(columns="abs_d")
    return out


def compare_categorical(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    y = df[config.TARGET]
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        sub = df.loc[df[v].notna() & y.notna()]
        if sub[v].nunique() < 2:
            continue
        table = pd.crosstab(sub[v], sub[config.TARGET])
        if table.shape[1] < 2:
            continue
        counts = table.values
        expected = stats.chi2_contingency(counts, correction=False)[3]
        small = (expected < 5).any()
        if counts.shape == (2, 2) and small:
            p = float(stats.fisher_exact(counts)[1])
            test = "Fisher exact"
        else:
            p = float(stats.chi2_contingency(counts)[1])
            test = "chi-square"

        or_, lo, hi = (np.nan, np.nan, np.nan)
        if counts.shape == (2, 2):
            # rows are the predictor levels sorted ascending -> row 1 is "yes"
            b, a = counts[1][0], counts[1][1]
            d, c = counts[0][0], counts[0][1]
            or_, lo, hi = odds_ratio_ci(a, b, c, d)

        prev = sub.groupby(v, observed=True)[config.TARGET].mean()
        rows.append(dict(
            variable=v, test=test, p_value=p, cramers_v=cramers_v(counts),
            odds_ratio=or_, or_ci_low=lo, or_ci_high=hi,
            n_used=int(counts.sum()),
            levels=";".join(f"{k}:{v_:.4f}" for k, v_ in prev.items()),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_value_fdr"] = bh_fdr(out["p_value"])
        out = out.sort_values("cramers_v", ascending=False)
    return out


def interpret_effect_sizes(cont: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a plain reading of the effect size, so that a tiny p-value on a
    trivial difference does not get read as an important finding.
    """
    def label_d(d):
        if pd.isna(d):
            return ""
        a = abs(d)
        return ("negligible" if a < 0.2 else "small" if a < 0.5
                else "moderate" if a < 0.8 else "large")

    def label_v(v):
        if pd.isna(v):
            return ""
        return ("negligible" if v < 0.1 else "small" if v < 0.2
                else "moderate" if v < 0.3 else "large")

    rows = []
    for _, r in cont.iterrows():
        rows.append(dict(variable=r["variable"], kind="continuous",
                         effect_size=r["cohens_d"], effect_size_name="Cohen's d",
                         magnitude=label_d(r["cohens_d"]),
                         p_value=r["welch_p"], p_value_fdr=r.get("welch_p_fdr")))
    for _, r in cat.iterrows():
        rows.append(dict(variable=r["variable"], kind="categorical",
                         effect_size=r["cramers_v"], effect_size_name="Cramer's V",
                         magnitude=label_v(r["cramers_v"]),
                         p_value=r["p_value"], p_value_fdr=r.get("p_value_fdr")))
    out = pd.DataFrame(rows)
    return out.sort_values("effect_size", key=lambda s: s.abs(), ascending=False)
