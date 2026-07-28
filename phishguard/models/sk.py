"""scikit-learn counterparts, behind the same interface.

They exist to answer one question: does the from-scratch implementation compute what it
claims to? A reimplementation with nothing to check it against is an assertion. Serving
both side by side turns the claim into a measured number -- the parity delta, which the
interface displays.

The pair is presented as two implementations of one algorithm, not as two independent
opinions. Four models are shown, but there are two algorithm families, and averaging a
scratch model with its own reference would be double-counting.
"""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from phishguard.models.base import Classifier, as_float32
from phishguard.schema import PHISHING_LABEL


class SklearnKNN(Classifier):
    name = "KNN (scikit-learn)"
    family = "knn"
    is_scratch = False

    def __init__(self, k: int = 20, n_jobs: int = 1) -> None:
        # n_jobs=1 deliberately: the container is single-replica on a small VPS, and
        # sklearn's process pool would multiply resident memory by the worker count for
        # no latency gain at this reference-set size.
        self.k = k
        self._model = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnKNN:
        self._model.fit(as_float32(X), np.asarray(y, dtype=np.int64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(as_float32(X)).astype(np.int64)

    def score_phishing(self, X: np.ndarray) -> np.ndarray:
        return _phishing_column(self._model, as_float32(X))


class SklearnGaussianNB(Classifier):
    name = "Gaussian Naive Bayes (scikit-learn)"
    family = "naive_bayes"
    is_scratch = False

    def __init__(self) -> None:
        self._model = GaussianNB()

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnGaussianNB:
        self._model.fit(as_float32(X), np.asarray(y, dtype=np.int64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(as_float32(X)).astype(np.int64)

    def score_phishing(self, X: np.ndarray) -> np.ndarray:
        return _phishing_column(self._model, as_float32(X))


def _phishing_column(model: object, X: np.ndarray) -> np.ndarray:
    """Pick P(class 0) out of predict_proba by locating the class, not by index.

    sklearn orders the probability columns by sorted class label, so column 0 happens to
    be class 0 here. Looking it up anyway means a future relabelling cannot silently
    invert every score in the application.
    """
    proba = model.predict_proba(X)  # type: ignore[attr-defined]
    classes = list(model.classes_)  # type: ignore[attr-defined]
    if PHISHING_LABEL not in classes:
        raise ValueError(f"model was never fitted on the phishing class; saw {classes}")
    return np.asarray(proba[:, classes.index(PHISHING_LABEL)], dtype=np.float64)
