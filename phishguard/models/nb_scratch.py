"""From-scratch Gaussian Naive Bayes, in log space.

WHY LOG SPACE IS NOT AN OPTIMISATION
====================================

The original multiplied one Gaussian density per feature into a running product:

    p = self.pv[k]
    for feature in X_test.columns:
        p *= self._calculate_probability(value, mean, std)

With 49 standardised features, the densities are individually below 0.4 and their product
lands somewhere around 1e-40 to 1e-300. Under float64 that underflows to exactly 0.0 for a
large share of rows, and once both classes underflow, ``np.argmax([0.0, 0.0])`` returns 0
unconditionally -- so the model reports "phishing" not because the evidence says so but
because the arithmetic collapsed.

Summing log-densities instead is the same decision rule with no underflow: log is
monotone, so the argmax is unchanged wherever the product was actually representable, and
it stays meaningful where it was not. This is a correctness fix, not a speed one.
"""

from __future__ import annotations

import numpy as np

from phishguard.models.base import Classifier, as_float32
from phishguard.schema import PHISHING_LABEL

#: Floor for a degenerate (zero-variance) feature, matching the original's guard. Without
#: it the Gaussian has infinite density at its mean and zero everywhere else.
MIN_STD = 1e-10

LOG_2PI = float(np.log(2.0 * np.pi))


class ScratchGaussianNB(Classifier):
    name = "Gaussian Naive Bayes (from scratch)"
    family = "naive_bayes"
    is_scratch = True

    def __init__(self) -> None:
        self.classes_: np.ndarray = np.array([0, 1], dtype=np.int64)
        self.log_prior_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None  # per-class means
        self.sigma_: np.ndarray | None = None  # per-class standard deviations

    def fit(self, X: np.ndarray, y: np.ndarray) -> ScratchGaussianNB:
        Xf = as_float32(X).astype(np.float64)
        yf = np.asarray(y, dtype=np.int64)

        n_features = Xf.shape[1]
        self.log_prior_ = np.empty(2, dtype=np.float64)
        self.theta_ = np.empty((2, n_features), dtype=np.float64)
        self.sigma_ = np.empty((2, n_features), dtype=np.float64)

        for k in (0, 1):
            mask = yf == k
            self.log_prior_[k] = np.log(mask.sum() / len(yf)) if mask.any() else -np.inf
            if not mask.any():
                self.theta_[k] = 0.0
                self.sigma_[k] = MIN_STD
                continue
            block = Xf[mask]
            self.theta_[k] = block.mean(axis=0)
            # ddof=1 matches the original, which used pandas' .std() default.
            std = block.std(axis=0, ddof=1) if len(block) > 1 else np.zeros(n_features)
            self.sigma_[k] = np.where(std == 0.0, MIN_STD, std)

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log P(class) + sum of log Gaussian densities. Shape (n, 2)."""
        if self.log_prior_ is None or self.theta_ is None or self.sigma_ is None:
            raise RuntimeError("ScratchGaussianNB used before fit()")

        Xf = as_float32(X).astype(np.float64)
        out = np.empty((len(Xf), 2), dtype=np.float64)
        for k in (0, 1):
            mean = self.theta_[k]
            std = self.sigma_[k]
            z = (Xf - mean) / std
            log_density = -0.5 * (LOG_2PI + 2.0 * np.log(std)) - 0.5 * z * z
            out[:, k] = self.log_prior_[k] + log_density.sum(axis=1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score_phishing(self, X: np.ndarray) -> np.ndarray:
        """Posterior P(class 0), normalised in log space.

        Subtracting the row max before exponentiating keeps the largest term at exp(0)=1,
        so the normalisation is stable no matter how far apart the two log-likelihoods
        are. This is the step whose absence in the original caused the collapse.
        """
        jll = self._joint_log_likelihood(X)
        shifted = jll - jll.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        posterior = exp / exp.sum(axis=1, keepdims=True)
        return posterior[:, PHISHING_LABEL]


class ProductGaussianNB(ScratchGaussianNB):
    """The original's raw-product formulation, kept as the reference for the log version.

    Retained so the underflow is demonstrable rather than merely asserted: a test fits
    both on the real 49-feature data and shows this one collapsing to zero probability on
    rows the log-space model handles.
    """

    name = "Gaussian Naive Bayes (raw product)"

    def _class_products(self, X: np.ndarray) -> np.ndarray:
        if self.log_prior_ is None or self.theta_ is None or self.sigma_ is None:
            raise RuntimeError("ProductGaussianNB used before fit()")

        Xf = as_float32(X).astype(np.float64)
        out = np.empty((len(Xf), 2), dtype=np.float64)
        for k in (0, 1):
            mean = self.theta_[k]
            std = self.sigma_[k]
            coefficient = 1.0 / (np.sqrt(2.0 * np.pi) * std)
            exponent = -((Xf - mean) ** 2) / (2.0 * std**2)
            densities = coefficient * np.exp(exponent)
            out[:, k] = np.exp(self.log_prior_[k]) * densities.prod(axis=1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._class_products(X), axis=1)]

    def score_phishing(self, X: np.ndarray) -> np.ndarray:
        products = self._class_products(X)
        total = products.sum(axis=1)
        # Where both classes underflowed there is no information left to normalise.
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, products[:, PHISHING_LABEL] / total, np.nan)
