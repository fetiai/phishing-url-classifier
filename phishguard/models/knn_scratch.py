"""From-scratch k-nearest neighbours, vectorised.

The original iterated ``X_test.iterrows()`` in pure Python and took 5m43s to classify the
validation split. Same algorithm, same answers, expressed as matrix arithmetic.

THE TIE-BREAKING HAZARD
=======================

This is the one subtlety, and it is easy to vectorise straight past.

The original selected neighbours with ``np.argsort(distances)[:k]`` and voted with
``Counter(...).most_common(1)``. When two classes tie on count, ``most_common`` returns
whichever was inserted into the counter first -- which, given argsort's output, is the
class of the nearest neighbour among the k.

``np.argpartition`` is the fast way to take the k smallest, and it does *not* order the k
it returns. Feeding its output straight into the vote silently changes which class wins
every tie, and with an even k ties are common rather than exotic. So the k selected
indices are re-sorted by distance before voting. Without that line the vectorised model
disagrees with the original on a subset of rows and no test that only checks accuracy
would notice.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from phishguard.models.base import Classifier, as_float32
from phishguard.schema import PHISHING_LABEL

#: Query rows per chunk. Bounds the distance matrix at chunk x n_reference floats, which
#: at 512 x 10,000 float32 is about 20 MB -- large enough to keep BLAS busy, small enough
#: that it never dominates the container's memory budget.
CHUNK_SIZE = 512


class ScratchKNN(Classifier):
    name = "KNN (from scratch)"
    family = "knn"
    is_scratch = True

    def __init__(self, k: int = 20, metric: str = "euclidean", p: int = 1) -> None:
        if not isinstance(k, int) or k < 1:
            raise ValueError("Invalid neighbor count. k must be an integer greater than 0.")
        if metric not in {"manhattan", "euclidean", "minkowski"}:
            raise ValueError(
                "Invalid distance metric. Valid metric: 'euclidean', 'manhattan', or 'minkowski'."
            )
        if not isinstance(p, int) or p < 1:
            raise ValueError("Invalid minkowski distance variable. p must be an integer > 0.")

        self.k = k
        self.metric = metric
        self.p = p
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ScratchKNN:
        self.X_train = as_float32(X)
        self.y_train = np.asarray(y, dtype=np.int64)
        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                f"reference set and labels disagree: {len(self.X_train)} rows, "
                f"{len(self.y_train)} labels"
            )
        return self

    def _distances(self, Q: np.ndarray) -> np.ndarray:
        assert self.X_train is not None
        R = self.X_train

        if self.metric == "euclidean":
            # d^2 = |q|^2 - 2 q.r + |r|^2, which turns the whole distance computation into
            # one matrix product. Small negatives appear from cancellation when a query
            # nearly coincides with a reference point, so the result is clipped at zero
            # before the square root.
            q_sq = np.einsum("ij,ij->i", Q, Q)[:, None]
            r_sq = np.einsum("ij,ij->i", R, R)[None, :]
            d2 = q_sq - 2.0 * (Q @ R.T) + r_sq
            np.maximum(d2, 0.0, out=d2)
            return np.sqrt(d2)

        if self.metric == "manhattan":
            return np.abs(Q[:, None, :] - R[None, :, :]).sum(axis=2)

        diff = np.abs(Q[:, None, :] - R[None, :, :]) ** self.p
        return diff.sum(axis=2) ** (1.0 / self.p)

    def _neighbours(self, Q: np.ndarray) -> np.ndarray:
        """Indices of the k nearest reference rows, ordered nearest-first."""
        assert self.X_train is not None
        n_ref = len(self.X_train)
        k = min(self.k, n_ref)

        distances = self._distances(Q)

        if k < n_ref:
            candidates = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        else:
            candidates = np.tile(np.arange(n_ref), (len(Q), 1))

        # Re-sort the selected k by distance. argpartition leaves them unordered, and the
        # vote's tie-break depends on nearest-first order.
        rows = np.arange(len(Q))[:, None]
        order = np.argsort(distances[rows, candidates], axis=1, kind="stable")
        return candidates[rows, order]

    def _vote(self, neighbour_idx: np.ndarray) -> np.ndarray:
        assert self.y_train is not None
        labels = self.y_train[neighbour_idx]
        out = np.empty(len(labels), dtype=np.int64)
        for i, row in enumerate(labels):
            # Counter preserves insertion order, and the row is nearest-first, so a tie
            # resolves to the class of the nearest neighbour -- matching the original.
            out[i] = Counter(row.tolist()).most_common(1)[0][0]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("ScratchKNN.predict called before fit()")

        Q = as_float32(X)
        predictions = np.empty(len(Q), dtype=np.int64)
        for start in range(0, len(Q), CHUNK_SIZE):
            chunk = Q[start : start + CHUNK_SIZE]
            predictions[start : start + len(chunk)] = self._vote(self._neighbours(chunk))
        return predictions

    def score_phishing(self, X: np.ndarray) -> np.ndarray:
        """Share of the k neighbours that are phishing.

        A vote fraction, not a calibrated probability. It is monotone in the evidence and
        that is all the threshold slider needs; presenting it as a calibrated posterior
        would overstate what k=20 nearest neighbours can tell you.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("ScratchKNN.score_phishing called before fit()")

        Q = as_float32(X)
        scores = np.empty(len(Q), dtype=np.float64)
        for start in range(0, len(Q), CHUNK_SIZE):
            chunk = Q[start : start + CHUNK_SIZE]
            neighbours = self._neighbours(chunk)
            labels = self.y_train[neighbours]
            scores[start : start + len(chunk)] = (labels == PHISHING_LABEL).mean(axis=1)
        return scores


class NaiveScratchKNN(ScratchKNN):
    """The original row-at-a-time loop, kept as the reference for the vectorised version.

    Used only by the equivalence test. It is far too slow to serve with -- which is the
    entire reason the vectorised implementation exists -- but a fast implementation with
    no slow one to check it against is just an assertion.
    """

    name = "KNN (from scratch, naive loop)"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("NaiveScratchKNN.predict called before fit()")

        Q = as_float32(X)
        predictions = []
        for row in Q:
            match self.metric:
                case "euclidean":
                    distances = np.sqrt(np.sum(np.square(row - self.X_train), axis=1))
                case "manhattan":
                    distances = np.sum(np.abs(row - self.X_train), axis=1)
                case "minkowski":
                    distances = np.sum(np.abs(row - self.X_train) ** self.p, axis=1) ** (
                        1 / self.p
                    )
                case _:
                    raise ValueError("Invalid distance metric.")

            neighbours = np.argsort(distances, kind="stable")[: self.k]
            classes = Counter(self.y_train[neighbours].tolist())
            predictions.append(classes.most_common(1)[0][0])
        return np.asarray(predictions, dtype=np.int64)
