"""Tier 2 and Tier 3 -- the fit/transform contract.

The row/batch invariance test in this file is the single most important test in the suite.
It is the proposition that classifying one pasted URL is the same computation as training
on a batch. If it fails, then every number produced anywhere downstream -- every metric,
every prediction, every feature attribution -- is partly a function of what the input
happened to be batched with, and every test written against those numbers is measuring
that accident rather than the model.

Equality is asserted bitwise on float32, never with a tolerance. A tolerance-based
comparison would pass even while a statistic was being recomputed from a batch that
happens to resemble the training set, which is exactly the failure mode under test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from phishguard import schema
from phishguard.preprocess.scaler import SubsetStandardScaler
from phishguard.preprocess.transformer import (
    FeatureContractError,
    NotFittedError,
    Preprocessor,
)


@pytest.fixture(scope="module")
def fitted(raw_X: pd.DataFrame, raw_y: pd.Series) -> Preprocessor:
    pre = Preprocessor()
    pre.fit(raw_X.copy(), raw_y.copy())
    return pre


@pytest.fixture(scope="module")
def batch_matrix(fitted: Preprocessor, raw_X: pd.DataFrame) -> np.ndarray:
    return fitted.transform_matrix(raw_X.copy())


# --- Tier 2: row/batch invariance --------------------------------------------


def test_row_batch_invariance(fitted, raw_X, batch_matrix):
    """Transform each row alone; assert it equals that row of the batch, bitwise.

    Index selection is deliberate rather than purely random: the first and last rows catch
    off-by-one handling, and the rows with the most and fewest NaNs are where imputation
    does the most and least work.
    """
    nan_counts = raw_X.isna().sum(axis=1)
    indices = {0, len(raw_X) - 1, int(nan_counts.idxmax()), int(nan_counts.idxmin())}
    rng = np.random.default_rng(20240728)
    indices.update(int(i) for i in rng.choice(len(raw_X), size=120, replace=False))

    mismatches = []
    for i in sorted(indices):
        single = fitted.transform_matrix(raw_X.iloc[[i]].copy())
        if not np.array_equal(single[0], batch_matrix[i]):
            differing = np.nonzero(single[0] != batch_matrix[i])[0]
            mismatches.append(
                (i, [(schema.FEATURE_ORDER[j], batch_matrix[i][j], single[0][j]) for j in differing])
            )

    assert not mismatches, f"row/batch disagreement in {len(mismatches)} rows: {mismatches[:3]}"


def test_invariance_under_batch_composition(fitted, raw_X, batch_matrix):
    """Splitting the batch must not change any row's output.

    A statistic hiding in transform would make each half's rows depend on that half's
    contents, so the halves would disagree with the whole.
    """
    half = len(raw_X) // 2
    first = fitted.transform_matrix(raw_X.iloc[:half].copy())
    second = fitted.transform_matrix(raw_X.iloc[half:].copy())
    assert np.array_equal(np.vstack([first, second]), batch_matrix)


def test_invariance_under_shuffling(fitted, raw_X, batch_matrix):
    """Row order must not affect any row's value."""
    order = np.random.default_rng(7).permutation(len(raw_X))
    shuffled = fitted.transform_matrix(raw_X.iloc[order].copy())
    assert np.array_equal(shuffled, batch_matrix[order])


def test_all_nan_row_is_transformable(fitted, raw_X):
    """Every value missing is a legitimate input, not an error: it is what a fetch failure
    against an unparseable URL produces."""
    blank = raw_X.iloc[[0]].copy()
    for col in blank.columns:
        blank[col] = np.nan

    out = fitted.transform_matrix(blank)
    assert out.shape == (1, 49)
    assert np.isfinite(out).all(), "an all-NaN input must impute to finite values"


def test_a_row_needing_the_whole_cascade_transforms_cleanly(fitted, raw_X):
    """Every cascade column missing at once is a realistic input -- it is what a failed
    fetch produces -- and must impute to finite values without raising."""
    row = raw_X.iloc[[0]].copy()
    for col in schema.CATEGORICAL_COLUMNS_TO_FILL:
        row[col] = np.nan
    row["HasObfuscation"] = 1

    out = Preprocessor(fitted.stats).transform_matrix(row)
    assert np.isfinite(out).all()


def test_unseen_cascade_key_falls_back_to_the_global_mode(fitted, raw_X):
    """An unseen cascade key is expected, not exceptional.

    At the deep end of the cascade the key is 17 columns wide, so the training frame
    observed only a small fraction of the combinatorial space. The defined behaviour is to
    use the recorded global mode -- not to raise, and under no circumstances to recompute
    the mode from the incoming batch.

    The fallback is forced by emptying one step's mode table rather than by hunting for a
    rare row: filling each column from its own mode walks the *most common* key path,
    which is precisely the path that does exist in training.
    """
    step_index = 5
    original = fitted.stats.categorical_cascade[step_index]
    doctored_step = dataclasses.replace(original, modes={})
    doctored = dataclasses.replace(
        fitted.stats,
        categorical_cascade=(
            *fitted.stats.categorical_cascade[:step_index],
            doctored_step,
            *fitted.stats.categorical_cascade[step_index + 1 :],
        ),
    )

    row = raw_X.iloc[[0]].copy()
    row[original.column] = np.nan

    pre = Preprocessor(doctored)
    out = pre.transform(row)

    assert pre.fallback_counts[original.column] == 1
    assert np.isfinite(out.to_numpy(dtype=np.float64)).all()


def test_no_fallback_is_counted_when_the_key_is_known(fitted, raw_X):
    """The counter is a diagnostic, so it must not fire on the ordinary path."""
    row = raw_X.iloc[[0]].copy()
    pre = Preprocessor(fitted.stats)
    pre.transform(row)
    assert sum(pre.fallback_counts.values()) == 0


def test_unseen_tld_uses_the_recorded_global_fill(fitted, raw_X):
    row = raw_X.iloc[[0]].copy()
    row["URL"] = "https://example.invalidtldthatcannotexist/"
    row["Domain"] = np.nan
    row["TLD"] = np.nan
    row["TLDLegitimateProb"] = np.nan

    out = fitted.transform(row)
    assert np.isfinite(out["TLDLegitimateProb"].to_numpy()).all()


def test_column_order_of_the_input_does_not_matter(fitted, raw_X, batch_matrix):
    """transform reindexes to the frozen order, so a caller handing columns over in a
    different order must still get the same matrix."""
    reversed_cols = list(raw_X.columns)[::-1]
    out = fitted.transform_matrix(raw_X[reversed_cols].copy())
    assert np.array_equal(out, batch_matrix)


def test_missing_feature_column_raises_rather_than_filling(fitted, raw_X):
    """A missing *value* is unknown and gets imputed. A missing *column* means the
    extractor contract broke, and failing loudly is correct."""
    broken = raw_X.drop(columns=["NoOfImage"]).copy()
    with pytest.raises(FeatureContractError) as exc:
        fitted.transform(broken)
    assert "NoOfImage" in str(exc.value)


def test_transform_before_fit_raises(raw_X):
    with pytest.raises(NotFittedError):
        Preprocessor().transform(raw_X.iloc[[0]].copy())


# --- Tier 3: determinism, shape, scaler scope --------------------------------


def test_fit_is_deterministic(raw_X, raw_y):
    a = Preprocessor().fit(raw_X.copy(), raw_y.copy())
    b = Preprocessor().fit(raw_X.copy(), raw_y.copy())

    assert a.char_prob == b.char_prob
    assert a.tld_prob_global_fill == b.tld_prob_global_fill
    assert a.scaler.mean_ == b.scaler.mean_
    assert a.scaler.scale_ == b.scaler.scale_
    assert [f.value for f in a.numeric_fill] == [f.value for f in b.numeric_fill]
    assert [s.modes for s in a.categorical_cascade] == [s.modes for s in b.categorical_cascade]
    assert a.nb_drop == b.nb_drop


def test_transform_is_deterministic(fitted, raw_X):
    assert np.array_equal(
        fitted.transform_matrix(raw_X.copy()), fitted.transform_matrix(raw_X.copy())
    )


def test_matrix_shape_and_dtype(batch_matrix, raw_X):
    assert batch_matrix.shape == (len(raw_X), 49)
    assert batch_matrix.dtype == np.float32


def test_scaler_touches_exactly_the_thirty_numeric_columns(fitted):
    """The scope is the frozen decision. Widening it would rescale the 19 binary
    indicators and change every recorded metric, so it is asserted rather than trusted."""
    stats = fitted.stats
    assert tuple(stats.scaler.columns) == schema.NUMERICAL_COLUMNS
    assert len(stats.scaler.columns) == 30
    assert not (set(stats.scaler.columns) & set(schema.CATEGORICAL_COLUMNS_FILTERED))


def test_binary_columns_pass_through_the_scaler_unchanged(fitted, raw_X):
    """The 19 indicators must emerge bit-identical, so their values stay in {0, 1}."""
    transformed = fitted.transform(raw_X.copy())
    for col in schema.CATEGORICAL_COLUMNS_FILTERED:
        values = set(np.unique(transformed[col].to_numpy()))
        assert values <= {0.0, 1.0}, f"{col} was rescaled: {sorted(values)[:5]}"


def test_scaler_has_no_fit_transform():
    """The absence is the point: serving-time code receives numbers, not an estimator that
    could be re-fitted on the batch in front of it."""
    assert not hasattr(SubsetStandardScaler, "fit_transform")


def test_scaled_columns_are_standardised_on_the_training_frame(fitted, raw_X):
    transformed = fitted.transform(raw_X.copy())
    for col in schema.CONTINUOUS_COLUMNS:
        values = transformed[col].to_numpy(dtype=np.float64)
        assert abs(float(values.mean())) < 1e-6, col


def test_scale_never_contains_zero(fitted):
    """A constant column would otherwise divide by zero at serving time."""
    assert all(s != 0.0 for s in fitted.stats.scaler.scale_)


def test_html_fallbacks_agree_with_the_fills_they_came_from(fitted):
    """The fallback table is derived, not separately computed. A divergence would mean the
    interface displays one value while transform applies another."""
    stats = fitted.stats
    numeric = {f.column: f.value for f in stats.numeric_fill}
    cascade = {s.column: float(s.global_mode) for s in stats.categorical_cascade}

    assert len(stats.html_fallbacks) == 28  # 25 page features + 3 title-dependent hybrids
    for fb in stats.html_fallbacks:
        if fb.source.startswith("numeric"):
            assert fb.value == numeric[fb.column], fb.column
        else:
            assert fb.value == cascade[fb.column], fb.column


def test_cascade_order_is_preserved(fitted):
    """Permuting the cascade changes every mode table from step 2 onward, so the recorded
    order must match the frozen one exactly."""
    columns = tuple(s.column for s in fitted.stats.categorical_cascade)
    assert columns == schema.CATEGORICAL_COLUMNS_TO_FILL


def test_cascade_group_key_grows_by_one_column_per_step(fitted):
    steps = fitted.stats.categorical_cascade
    assert steps[0].groupby_cols == schema.CATEGORICAL_FILL_INITIAL_GROUP_BY
    for i in range(1, len(steps)):
        assert steps[i].groupby_cols == (*steps[i - 1].groupby_cols, steps[i - 1].column)


def test_nb_drop_records_a_reason_for_every_dropped_column(fitted):
    """The original computed this list and discarded it, so nothing downstream could say
    which columns a served model had actually been fitted on."""
    stats = fitted.stats
    assert set(stats.nb_drop) == set(stats.nb_drop_reasons)
    assert all(
        r in {"empty_contingency_cell", "zero_std"} for r in stats.nb_drop_reasons.values()
    )


def test_transform_computes_no_statistic_from_its_input(fitted, raw_X):
    """A behavioural version of the governing rule.

    Duplicating one row 200 times moves every batch-level mean, median, mode and quantile
    substantially. If transform consulted any of them, that row's output would shift.
    """
    row = raw_X.iloc[[3]].copy()
    alone = fitted.transform_matrix(row)

    skewed = pd.concat([row] * 200, ignore_index=True)
    in_crowd = fitted.transform_matrix(skewed)

    assert np.array_equal(alone[0], in_crowd[0])
