"""The interface all four models present.

The application treats a from-scratch implementation and its scikit-learn counterpart
identically, so neither gets special-cased in the interface layer.

``score`` is always P(class 0) -- the probability of *phishing*. Class 0 is the positive
class throughout this codebase, and a model that returned P(class 1) here would invert
every threshold, every ranking and every confidence display without failing a single type
check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from phishguard.schema import PHISHING_LABEL


class Classifier(ABC):
    """Fit on a float32 matrix; predict labels and phishing scores."""

    #: Shown in the interface. The from-scratch pair is labelled as an educational
    #: reimplementation rather than presented as an independent fourth opinion.
    name: str = "classifier"
    family: str = "unknown"
    is_scratch: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def score_phishing(self, X: np.ndarray) -> np.ndarray:
        """P(class 0) per row, in [0, 1]."""

    def predict_at_threshold(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Label as phishing when the phishing score reaches the threshold."""
        scores = self.score_phishing(X)
        return np.where(scores >= threshold, PHISHING_LABEL, 1 - PHISHING_LABEL).astype(np.int64)


def as_float32(X: np.ndarray) -> np.ndarray:
    """Pin the dtype at every model boundary.

    The transform emits float32 and the distance kernel expects it. Letting a float64
    matrix through would silently change the arithmetic between training and serving,
    which is the same class of skew the preprocessing split exists to prevent.
    """
    array = np.asarray(X, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array
