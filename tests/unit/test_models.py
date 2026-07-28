"""Tier 4 -- the four models.

Two properties carry the weight here. The vectorised KNN must agree with the naive loop it
replaced *including on ties*, and the log-space Naive Bayes must not collapse to zero
probability the way the raw product does at 49 features. Both are corrections that no
accuracy-only test would catch: the tie disagreement moves a small subset of rows, and the
underflow makes the model return a constant class while still looking like it is
predicting.
"""

from __future__ import annotations

import numpy as np
import pytest

from phishguard.models.knn_scratch import NaiveScratchKNN, ScratchKNN
from phishguard.models.nb_scratch import ProductGaussianNB, ScratchGaussianNB
from phishguard.models.sk import SklearnGaussianNB, SklearnKNN
from phishguard.schema import PHISHING_LABEL


@pytest.fixture
def synthetic():
    """Two separable-ish Gaussian blobs at the real feature count."""
    rng = np.random.default_rng(11)
    n, d = 600, 49
    X = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(n // 2, d)),
            rng.normal(0.8, 1.0, size=(n // 2, d)),
        ]
    ).astype(np.float32)
    y = np.concatenate([np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)])
    order = rng.permutation(n)
    return X[order], y[order]


# --- KNN ---------------------------------------------------------------------


def test_vectorised_knn_matches_the_naive_loop(synthetic):
    X, y = synthetic
    X_ref, y_ref, X_q = X[:400], y[:400], X[400:]

    fast = ScratchKNN(k=20).fit(X_ref, y_ref)
    slow = NaiveScratchKNN(k=20).fit(X_ref, y_ref)

    np.testing.assert_array_equal(fast.predict(X_q), slow.predict(X_q))


def test_vectorised_knn_matches_the_naive_loop_on_exact_ties():
    """The case the vectorisation is most likely to get wrong.

    With an even k and an engineered geometry, count ties are the norm rather than the
    exception. argpartition does not order the k it selects, so voting on its raw output
    breaks the nearest-neighbour tie-break the original relied on.
    """
    # Ten reference points at distances 1..10 from the origin, alternating class, so the
    # k=10 neighbourhood contains exactly five of each.
    X_ref = np.zeros((10, 3), dtype=np.float32)
    X_ref[:, 0] = np.arange(1, 11, dtype=np.float32)
    y_ref = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    X_q = np.zeros((1, 3), dtype=np.float32)

    fast = ScratchKNN(k=10).fit(X_ref, y_ref)
    slow = NaiveScratchKNN(k=10).fit(X_ref, y_ref)

    assert fast.predict(X_q)[0] == slow.predict(X_q)[0]
    # The nearest neighbour is class 0, so a tie must resolve to class 0.
    assert fast.predict(X_q)[0] == 0


def test_knn_tie_break_follows_the_nearest_neighbour_across_many_geometries():
    rng = np.random.default_rng(3)
    for _ in range(25):
        n = 12
        X_ref = rng.normal(size=(n, 4)).astype(np.float32)
        y_ref = np.array([0, 1] * (n // 2), dtype=np.int64)
        X_q = rng.normal(size=(3, 4)).astype(np.float32)

        fast = ScratchKNN(k=n).fit(X_ref, y_ref)
        slow = NaiveScratchKNN(k=n).fit(X_ref, y_ref)
        np.testing.assert_array_equal(fast.predict(X_q), slow.predict(X_q))


def test_knn_chunking_does_not_change_predictions(synthetic, monkeypatch):
    """Chunk size is a memory knob and must not be observable in the output."""
    X, y = synthetic
    model = ScratchKNN(k=7).fit(X[:400], y[:400])
    full = model.predict(X[400:])

    monkeypatch.setattr("phishguard.models.knn_scratch.CHUNK_SIZE", 3)
    assert np.array_equal(model.predict(X[400:]), full)


def test_knn_distance_kernel_matches_a_direct_computation(synthetic):
    """The expanded form of the squared distance is an optimisation, so it is checked
    against the obvious formulation rather than trusted."""
    X, y = synthetic
    model = ScratchKNN(k=3).fit(X[:100], y[:100])
    Q = X[100:110]

    fast = model._distances(Q)
    direct = np.sqrt(((Q[:, None, :] - X[:100][None, :, :]) ** 2).sum(axis=2))
    np.testing.assert_allclose(fast, direct, rtol=1e-4, atol=1e-4)


def test_knn_score_is_the_phishing_vote_fraction(synthetic):
    X, y = synthetic
    model = ScratchKNN(k=10).fit(X[:400], y[:400])
    scores = model.score_phishing(X[400:410])
    assert scores.shape == (10,)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    # A vote fraction over k=10 can only land on multiples of 0.1.
    np.testing.assert_allclose(scores * 10, np.round(scores * 10))


def test_knn_rejects_invalid_hyperparameters():
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            ScratchKNN(k=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ScratchKNN(metric="cosine")


def test_knn_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ScratchKNN().predict(np.zeros((1, 3), dtype=np.float32))


# --- Naive Bayes -------------------------------------------------------------


def test_raw_product_collapses_on_outlying_rows(synthetic):
    """Demonstrates the defect rather than asserting it.

    Being precise about when this bites: at 49 unit-variance features a typical row's
    product lands around 1e-30, which float64 represents comfortably. The collapse happens
    on rows that sit far from both class means -- each standard deviation of distance
    costs roughly another factor of e^-0.5 per feature, so a row several deviations out
    across 49 dimensions drives the product straight through the 1e-308 floor.

    Those rows are the ones that matter. A URL unlike anything in training is exactly when
    a classifier should be least confident, and instead both classes reach zero, argmax
    returns class 0 unconditionally, and the model reports phishing because the arithmetic
    collapsed rather than because of the evidence.
    """
    X, y = synthetic
    product = ProductGaussianNB().fit(X, y)

    outliers = np.full((5, X.shape[1]), 12.0, dtype=np.float32)
    scores = product.score_phishing(outliers)
    assert np.isnan(scores).all(), "expected the raw product to underflow on outlying rows"

    # Same rows, log space: still finite, still decided.
    log_model = ScratchGaussianNB().fit(X, y)
    assert np.isfinite(log_model.score_phishing(outliers)).all()


def test_raw_product_shrinks_toward_the_floor_as_features_are_added():
    """The mechanism, isolated: the product falls geometrically in the feature count,
    so 'more features' and 'underflow' are the same axis."""
    rng = np.random.default_rng(9)
    magnitudes = []
    for d in (10, 40, 160, 640):
        X = rng.normal(size=(80, d)).astype(np.float32)
        y = np.array([0, 1] * 40, dtype=np.int64)
        products = ProductGaussianNB().fit(X, y)._class_products(X)
        magnitudes.append(np.nanmax(products))

    assert magnitudes == sorted(magnitudes, reverse=True)
    assert magnitudes[-1] == 0.0, "at 640 features the product should have underflowed"


def test_log_space_nb_never_collapses(synthetic):
    X, y = synthetic
    model = ScratchGaussianNB().fit(X, y)
    scores = model.score_phishing(X)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    # Not every row can be the same class, which is what a collapsed model produces.
    assert len(np.unique(model.predict(X))) == 2


def test_log_space_and_product_agree_where_the_product_is_representable():
    """Log is monotone, so wherever the product did not underflow the argmax must match.

    Checked at low dimension, where the product survives.
    """
    rng = np.random.default_rng(5)
    X = np.vstack(
        [rng.normal(0.0, 1.0, size=(150, 4)), rng.normal(2.0, 1.0, size=(150, 4))]
    ).astype(np.float32)
    y = np.concatenate([np.zeros(150, dtype=np.int64), np.ones(150, dtype=np.int64)])

    log_model = ScratchGaussianNB().fit(X, y)
    product_model = ProductGaussianNB().fit(X, y)

    product_scores = product_model.score_phishing(X)
    representable = np.isfinite(product_scores)
    assert representable.sum() > 250, "low-dimensional case should mostly be representable"

    np.testing.assert_array_equal(
        log_model.predict(X)[representable], product_model.predict(X)[representable]
    )


def test_zero_variance_feature_does_not_produce_infinities():
    """A constant column has a degenerate Gaussian; without a floor its density is
    infinite at the mean and zero everywhere else."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    X[:, 3] = 1.0  # constant
    y = np.array([0, 1] * 60, dtype=np.int64)

    model = ScratchGaussianNB().fit(X, y)
    assert np.isfinite(model.score_phishing(X)).all()


def test_scratch_nb_tracks_sklearn(synthetic):
    """Not identical -- sklearn adds variance smoothing and uses ddof=0 -- but a large
    divergence would mean one of them is not computing Gaussian Naive Bayes."""
    X, y = synthetic
    scratch = ScratchGaussianNB().fit(X, y)
    reference = SklearnGaussianNB().fit(X, y)
    agreement = (scratch.predict(X) == reference.predict(X)).mean()
    assert agreement > 0.95, f"scratch and sklearn NB agree on only {agreement:.1%} of rows"


def test_scratch_knn_tracks_sklearn(synthetic):
    X, y = synthetic
    X_ref, y_ref, X_q = X[:400], y[:400], X[400:]
    scratch = ScratchKNN(k=15).fit(X_ref, y_ref)
    reference = SklearnKNN(k=15).fit(X_ref, y_ref)
    agreement = (scratch.predict(X_q) == reference.predict(X_q)).mean()
    assert agreement > 0.95, f"scratch and sklearn KNN agree on only {agreement:.1%} of rows"


# --- shared interface --------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [lambda: ScratchKNN(k=5), lambda: SklearnKNN(k=5), ScratchGaussianNB, SklearnGaussianNB],
    ids=["knn_scratch", "knn_sklearn", "nb_scratch", "nb_sklearn"],
)
def test_all_four_models_share_one_interface(synthetic, factory):
    X, y = synthetic
    model = factory().fit(X[:400], y[:400])

    predictions = model.predict(X[400:])
    scores = model.score_phishing(X[400:])

    assert predictions.shape == (200,)
    assert scores.shape == (200,)
    assert set(np.unique(predictions)) <= {0, 1}
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


@pytest.mark.parametrize(
    "factory",
    [lambda: ScratchKNN(k=5), lambda: SklearnKNN(k=5), ScratchGaussianNB, SklearnGaussianNB],
    ids=["knn_scratch", "knn_sklearn", "nb_scratch", "nb_sklearn"],
)
def test_score_is_the_phishing_probability_not_the_legitimate_one(synthetic, factory):
    """The orientation that silently inverts everything if it is wrong.

    A row drawn from the class-0 cluster must score above 0.5 for phishing. Getting this
    backwards passes every type check and reverses every threshold in the application.
    """
    X, y = synthetic
    model = factory().fit(X, y)
    phishing_rows = X[y == PHISHING_LABEL][:50]
    assert model.score_phishing(phishing_rows).mean() > 0.5


@pytest.mark.parametrize(
    "factory",
    [lambda: ScratchKNN(k=5), lambda: SklearnKNN(k=5), ScratchGaussianNB, SklearnGaussianNB],
    ids=["knn_scratch", "knn_sklearn", "nb_scratch", "nb_sklearn"],
)
def test_single_row_prediction_matches_the_batch(synthetic, factory):
    """The models must be as row/batch invariant as the preprocessing is; single-URL
    inference goes through them one row at a time."""
    X, y = synthetic
    model = factory().fit(X[:400], y[:400])
    batch = model.predict(X[400:420])
    for i in range(20):
        assert model.predict(X[400 + i : 401 + i])[0] == batch[i]


def test_threshold_prediction_uses_the_phishing_direction(synthetic):
    X, y = synthetic
    model = ScratchGaussianNB().fit(X, y)
    at_zero = model.predict_at_threshold(X, 0.0)
    at_one = model.predict_at_threshold(X, 1.01)
    assert (at_zero == PHISHING_LABEL).all()
    assert (at_one != PHISHING_LABEL).all()
