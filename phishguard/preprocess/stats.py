"""Everything learned at fit time, in one immutable record.

The governing rule of this package: **transform() may not compute any statistic.**

Operationally, inside ``transform`` there is no call to ``.mean()``, ``.median()``,
``.mode()``, ``.std()``, ``.skew()``, ``.quantile()``, ``.value_counts()``,
``.groupby(...).agg(...)``, ``fit()`` or ``fit_transform()`` on the input frame. Anything
depending on more than the current row is a fitted statistic: computed once in ``fit()``,
stored here, serialized, and read back at serving time.

This is not a style preference. The original recomputed every statistic from whatever
batch it was handed, and standardized the training and validation splits by their own
separate means -- so the scaling that produced its headline numbers does not exist at
serving time and cannot be reconstructed, because there is no batch to take a mean over
when a user pastes one URL. Making the statistics data rather than control flow is the
precondition for the service existing at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FillMethod = Literal["mean", "median"]
NbDropReason = Literal["empty_contingency_cell", "zero_std"]
FallbackSource = Literal["numeric_mean", "numeric_median", "categorical_mode"]


@dataclass(frozen=True)
class NumericFill:
    """Imputation value for one numeric column.

    ``skew`` is recorded even though ``transform`` never reads it, so a reader can audit
    which branch was taken without re-running training.
    """

    column: str
    method: FillMethod  # median iff abs(skew) > 3
    value: float
    skew: float


@dataclass(frozen=True)
class CategoricalFillStep:
    """One step of the ordered categorical cascade.

    ``groupby_cols`` is a snapshot of the group key *at this step*, persisted explicitly
    rather than reconstructed as "the first k-1 filled columns plus HasObfuscation". If
    the cascade is ever edited, the artifact still describes what actually ran.
    """

    column: str
    groupby_cols: tuple[str, ...]
    modes: dict[tuple[float, ...], float]
    global_mode: float


@dataclass(frozen=True)
class ClipBound:
    column: str
    lower: float  # 1st percentile of the training split
    upper: float  # 99th percentile of the training split


@dataclass(frozen=True)
class ScalerParams:
    """An affine map, stored as plain numbers.

    Deliberately not a pickled StandardScaler. A pickled estimator carries a
    ``fit_transform`` method, and the entire class of defect this package exists to
    eliminate is someone calling it at serving time. Numbers cannot be re-fitted.
    """

    columns: tuple[str, ...]  # exactly the 30 numeric columns, ordered
    mean_: tuple[float, ...]
    scale_: tuple[float, ...]  # zeros replaced by 1.0 so transform cannot divide by zero

    def __post_init__(self) -> None:
        if not (len(self.columns) == len(self.mean_) == len(self.scale_)):
            raise ValueError(
                f"scaler arity mismatch: {len(self.columns)} columns, "
                f"{len(self.mean_)} means, {len(self.scale_)} scales"
            )
        if any(s == 0.0 for s in self.scale_):
            raise ValueError("scale_ contains a zero; constant columns must be mapped to 1.0")


@dataclass(frozen=True)
class HtmlFallback:
    """The value a page-derived feature takes when the fetch failed or the feature was
    demoted.

    Derived, not separately computed: each entry is exactly the value this column's own
    fill rule already produces. It is lifted into its own record purely so the interface
    can *display* the fallback and where it came from, rather than silently applying it.
    A mismatch between this table and the fills it was derived from is a bug, and a test
    asserts they agree.
    """

    column: str
    value: float
    source: FallbackSource


@dataclass(frozen=True)
class FittedStats:
    """The complete serialized state of a fitted preprocessor."""

    # --- URL character model -------------------------------------------------
    char_prob: dict[str, float]

    # --- TLD legitimacy ------------------------------------------------------
    tld_prob_mean: dict[str, float]
    tld_prob_global_fill: float
    tld_skew: float
    tld_fill_method: FillMethod  # median iff abs(tld_skew) > 1 -- note: 1, not 3

    # --- numeric imputation --------------------------------------------------
    numeric_fill: tuple[NumericFill, ...]

    # --- categorical cascade. ORDER IS SEMANTIC ------------------------------
    categorical_cascade: tuple[CategoricalFillStep, ...]

    # --- outlier clipping ----------------------------------------------------
    clip_bounds: tuple[ClipBound, ...]

    # --- scaling -------------------------------------------------------------
    scaler: ScalerParams

    # --- Naive-Bayes column drop, fitted here for locality -------------------
    nb_drop: tuple[str, ...]
    nb_drop_reasons: dict[str, NbDropReason]

    # --- page-feature fallbacks ---------------------------------------------
    html_fallbacks: tuple[HtmlFallback, ...]

    # --- frozen column lists, so nothing closes over a module global ---------
    feature_order: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    discrete_columns: tuple[str, ...]
    categorical_columns_filtered: tuple[str, ...]

    # --- provenance ----------------------------------------------------------
    n_train_rows: int = 0
    demoted_features: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.feature_order) != 49:
            raise ValueError(f"feature_order must have 49 entries, got {len(self.feature_order)}")
        if len(self.numerical_columns) != 30:
            raise ValueError(
                f"numerical_columns must have 30 entries, got {len(self.numerical_columns)}"
            )
        if len(self.categorical_columns_filtered) != 19:
            raise ValueError(
                "categorical_columns_filtered must have 19 entries, got "
                f"{len(self.categorical_columns_filtered)}"
            )
        if len(self.categorical_cascade) != 18:
            raise ValueError(
                f"categorical_cascade must have 18 steps, got {len(self.categorical_cascade)}"
            )
        if tuple(self.scaler.columns) != tuple(self.numerical_columns):
            raise ValueError(
                "scaler.columns must be exactly numerical_columns, in the same order. "
                "A mismatch here means the matrix was scaled on a different column set "
                "than the one recorded, which silently changes every distance."
            )

    @property
    def numeric_fill_map(self) -> dict[str, float]:
        return {f.column: f.value for f in self.numeric_fill}

    @property
    def html_fallback_map(self) -> dict[str, float]:
        return {f.column: f.value for f in self.html_fallbacks}
