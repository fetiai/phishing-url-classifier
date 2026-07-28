"""Metrics, oriented around phishing as the positive class.

WHY ACCURACY IS NOT THE HEADLINE
================================

The corpus is 92.48% legitimate. A model that answers "legitimate" to everything, always,
scores 0.9248 accuracy while catching zero phishing URLs. Every accuracy figure in this
project is therefore reported against that baseline, because on its own the number does
not distinguish a working detector from a constant.

Class 0 is phishing and is the positive class. Phishing recall -- the share of phishing
URLs actually caught -- leads, because missing a phishing page is the failure that hurts
someone and a false alarm is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from phishguard.schema import LEGITIMATE_LABEL, MAJORITY_BASELINE_ACCURACY, PHISHING_LABEL


@dataclass(frozen=True)
class ClassificationMetrics:
    """Confusion counts and the rates derived from them.

    The confusion matrix is oriented rows = true (0, 1), columns = predicted (0, 1),
    consistently everywhere in this codebase.
    """

    accuracy: float
    baseline_accuracy: float
    lift_over_baseline: float

    phishing_precision: float
    phishing_recall: float
    phishing_f1: float

    legitimate_precision: float
    legitimate_recall: float
    legitimate_f1: float

    confusion_matrix: list[list[int]]
    support_phishing: int
    support_legitimate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = int(((y_true == PHISHING_LABEL) & (y_pred == PHISHING_LABEL)).sum())
    fn = int(((y_true == PHISHING_LABEL) & (y_pred == LEGITIMATE_LABEL)).sum())
    fp = int(((y_true == LEGITIMATE_LABEL) & (y_pred == PHISHING_LABEL)).sum())
    tn = int(((y_true == LEGITIMATE_LABEL) & (y_pred == LEGITIMATE_LABEL)).sum())

    accuracy = _safe_ratio(tp + tn, len(y_true))

    phishing_precision = _safe_ratio(tp, tp + fp)
    phishing_recall = _safe_ratio(tp, tp + fn)
    phishing_f1 = _safe_ratio(
        2 * phishing_precision * phishing_recall, phishing_precision + phishing_recall
    )

    legitimate_precision = _safe_ratio(tn, tn + fn)
    legitimate_recall = _safe_ratio(tn, tn + fp)
    legitimate_f1 = _safe_ratio(
        2 * legitimate_precision * legitimate_recall,
        legitimate_precision + legitimate_recall,
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        baseline_accuracy=MAJORITY_BASELINE_ACCURACY,
        lift_over_baseline=accuracy - MAJORITY_BASELINE_ACCURACY,
        phishing_precision=phishing_precision,
        phishing_recall=phishing_recall,
        phishing_f1=phishing_f1,
        legitimate_precision=legitimate_precision,
        legitimate_recall=legitimate_recall,
        legitimate_f1=legitimate_f1,
        confusion_matrix=[[tp, fn], [fp, tn]],
        support_phishing=tp + fn,
        support_legitimate=fp + tn,
    )


def precision_recall_curve_phishing(
    y_true: np.ndarray, scores: np.ndarray, n_points: int = 101
) -> dict[str, list[float]]:
    """Precision and recall against a sweep of phishing-score thresholds."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    thresholds = np.linspace(0.0, 1.0, n_points)
    precisions, recalls = [], []
    for threshold in thresholds:
        predicted = np.where(scores >= threshold, PHISHING_LABEL, LEGITIMATE_LABEL)
        metrics = compute_metrics(y_true, predicted)
        precisions.append(metrics.phishing_precision)
        recalls.append(metrics.phishing_recall)

    return {
        "thresholds": thresholds.tolist(),
        "precision": precisions,
        "recall": recalls,
    }


def average_precision_phishing(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the phishing precision-recall curve.

    PR-AUC rather than ROC-AUC: at 7.5% positives, ROC-AUC is dominated by the large
    negative class and looks flattering even when precision is poor.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(-scores, kind="stable")
    labels = (y_true[order] == PHISHING_LABEL).astype(np.int64)

    cumulative_tp = np.cumsum(labels)
    ranks = np.arange(1, len(labels) + 1)
    precision_at_k = cumulative_tp / ranks

    total_positives = int(labels.sum())
    if total_positives == 0:
        return 0.0
    return float((precision_at_k * labels).sum() / total_positives)


def disagreement_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Share of rows on which two models differ.

    Reported for each scratch/reference pair. A from-scratch implementation is only
    evidence of understanding if somebody measured how far it lands from the reference.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return float((a != b).mean()) if len(a) else 0.0
