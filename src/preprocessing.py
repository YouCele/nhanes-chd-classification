"""
Phase 6 / Phase 10 - preprocessing.

Everything that learns something from the data (imputation medians, scaling
parameters, one-hot categories) is wrapped in a ColumnTransformer that lives
inside the model pipeline. It is therefore fitted on the training part of each
cross-validation fold only. No global imputation is done anywhere before the
split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """
    Numeric vs categorical.

    Binary 0/1 questionnaire answers are treated as numeric: one-hot encoding
    them would only duplicate the column.
    """
    numeric, categorical = [], []
    for f in features:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            vals = set(pd.Series(s).dropna().unique().tolist())
            if vals.issubset({0.0, 1.0}) or s.dropna().nunique() > 12:
                numeric.append(f)
            elif s.dropna().nunique() <= 12 and f in ("race_eth", "education", "marital"):
                categorical.append(f)
            else:
                numeric.append(f)
        else:
            categorical.append(f)
    return numeric, categorical


def _onehot(drop_first: bool = False):
    """
    One-hot encoder.

    drop_first is used for the interpretation model only: keeping every level
    makes the dummy columns collinear, which splits one effect over two
    coefficients and inflates the standard errors.
    """
    kwargs = dict(handle_unknown="ignore", min_frequency=0.01)
    if drop_first:
        kwargs = dict(handle_unknown="infrequent_if_exist", drop="first",
                      min_frequency=0.01)
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:                                   # older scikit-learn
        kwargs.pop("min_frequency", None)
        return OneHotEncoder(sparse=False, **kwargs)


def build_preprocessor(df: pd.DataFrame, features: list[str],
                       scale: bool = True,
                       numeric_strategy: str = "median",
                       indicator_for: list[str] | None = None,
                       drop_first: bool = False) -> ColumnTransformer:
    """
    Build the column transformer.

    scale=True for models that need comparable scales (logistic regression).
    scale=False for tree based models, which do not care.
    indicator_for lists variables whose missingness should become its own
    column; those variables go through a separate branch with add_indicator.
    """
    numeric, categorical = split_feature_types(df, features)
    indicator_for = [c for c in (indicator_for or []) if c in numeric]
    plain_numeric = [c for c in numeric if c not in indicator_for]

    num_steps = [("impute", SimpleImputer(strategy=numeric_strategy))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    ind_steps = [("impute", SimpleImputer(strategy=numeric_strategy, add_indicator=True))]
    if scale:
        ind_steps.append(("scale", StandardScaler()))

    blocks = []
    if plain_numeric:
        blocks.append(("num", Pipeline(num_steps), plain_numeric))
    if indicator_for:
        blocks.append(("num_ind", Pipeline(ind_steps), indicator_for))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", _onehot(drop_first)),
        ]), categorical))

    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable names after fitting, used for coefficients and importances."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform_shape_[1])]


def uses_scaling(ct: ColumnTransformer) -> bool:
    """Was this transformer built with a scaling step? Used when rebuilding it."""
    for _, trans, _ in getattr(ct, "transformers", []):
        if isinstance(trans, Pipeline) and any(n == "scale" for n, _ in trans.steps):
            return True
    return False


def rebuild_for_features(model, df: pd.DataFrame, features: list[str],
                         indicator_for: list[str] | None = None):
    """
    Clone a fitted pipeline but give it a preprocessor for a different set of
    columns. Needed for the predictor-reduction checks: a ColumnTransformer
    remembers the exact column names it was fitted with.
    """
    from sklearn.base import clone
    fresh = clone(model)
    old = model.named_steps["pre"] if hasattr(model, "named_steps") else None
    scale = uses_scaling(old) if old is not None else True
    inds = [i for i in (indicator_for or []) if i in features]
    fresh.set_params(pre=build_preprocessor(df, features, scale=scale,
                                            indicator_for=inds))
    return fresh
