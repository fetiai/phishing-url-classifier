"""Standardisation over a named subset of columns.

Scales exactly the 30 numeric columns and passes the 19 binary indicators through
bit-identically.

WHY THE SUBSET IS DELIBERATE
============================

This started as an accident. In the original, the clipping helper and the standardisation
path both closed over a notebook-level ``numerical_columns`` global, so the 19 binary
columns were never in scope for either. Nobody decided this; it fell out of variable
scoping.

It is now a frozen decision, for two reasons:

1. The 19 columns are already 0/1 indicators. Z-scoring a Bernoulli indicator maps it to
   two points at -p/sqrt(p(1-p)) and (1-p)/sqrt(p(1-p)); it changes the scale of an
   already-comparable quantity, and for a rare indicator it inflates the minority level's
   magnitude substantially.
2. Changing it would change every recorded metric. Reporting as-recorded beside
   as-corrected is only meaningful if the two differ by one identified cause -- the
   fit/transform fix -- rather than by a bundle of simultaneous changes.

THE HONEST CAVEAT
=================

A Euclidean distance over 30 z-scored dimensions plus 19 raw 0/1 dimensions is not a
principled metric. The unit-variance dimensions and the at-most-0.25-variance dimensions
contribute on different scales, and the effective weighting is an accident of where the
fix stopped rather than a modelling choice. KNN is the model that cares; Gaussian NB
treats each dimension independently and its decision rule is invariant to per-feature
affine rescaling. An ablation that scales all 49 is reported as a comparison, not adopted
as the default -- changing it is a product decision with a metric-invalidation cost, not a
silent bugfix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phishguard.preprocess.stats import ScalerParams


class SubsetStandardScaler:
    """Fit and apply a per-column affine map over a fixed column subset.

    Note what this class does not have: a ``fit_transform``. Its absence is the point.
    Serving-time code receives :class:`ScalerParams` -- plain numbers -- and can only
    apply them.
    """

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = tuple(columns)

    def fit(self, X: pd.DataFrame) -> ScalerParams:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"cannot fit scaler, columns absent from frame: {sorted(missing)}")

        block = X[list(self.columns)].astype(np.float64)
        mean_ = block.mean(axis=0).to_numpy()

        # Population standard deviation (ddof=0), matching StandardScaler. Using the
        # sample standard deviation here would shift every scaled value by a factor of
        # sqrt(n/(n-1)) relative to the reference implementation.
        scale_ = block.std(axis=0, ddof=0).to_numpy()

        # A constant column has zero variance and would divide by zero. Mapping its scale
        # to 1.0 turns it into a pure mean-shift, which is what StandardScaler does too.
        scale_ = np.where(scale_ == 0.0, 1.0, scale_)

        return ScalerParams(
            columns=self.columns,
            mean_=tuple(float(v) for v in mean_),
            scale_=tuple(float(v) for v in scale_),
        )

    @staticmethod
    def apply(X: pd.DataFrame, params: ScalerParams) -> pd.DataFrame:
        """Apply a fitted affine map in place on a copy.

        Elementwise and row-local by construction: the value written for row i depends
        only on row i and on the persisted constants. That is what makes transforming one
        row and transforming a batch produce bitwise-identical output.
        """
        missing = [c for c in params.columns if c not in X.columns]
        if missing:
            raise KeyError(f"cannot scale, columns absent from frame: {sorted(missing)}")

        cols = list(params.columns)
        mean_ = np.asarray(params.mean_, dtype=np.float64)
        scale_ = np.asarray(params.scale_, dtype=np.float64)

        X = X.copy()
        X[cols] = (X[cols].astype(np.float64).to_numpy() - mean_) / scale_
        return X
