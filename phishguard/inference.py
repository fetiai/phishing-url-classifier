"""Single-URL classification, end to end.

THE ABSTENTION RULE
===================

When the fetch fails, 28 of 49 features are imputed -- and imputation fills them from
majority-class statistics, which means from *legitimate* pages, because the corpus is
92.5% legitimate. So a failed fetch does not produce a neutral prediction. It produces one
biased toward "legitimate", which is precisely the wrong direction for a phishing
detector: the model would be most reassuring exactly when it knows least.

This matters more than it sounds, because a failed fetch is the common case for the URLs
this system is meant to judge. The dataset was crawled in 2023-24 and phishing domains are
short-lived, so a phishing URL from it is usually dead, parked, or on registrar hold today.
"Could not reach it" correlates with "phishing" in reality and with "legitimate" in the
imputed features.

Below the coverage threshold the system therefore returns no verdict at all. Saying "not
enough evidence" is a worse product demo and a more honest answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from phishguard import schema
from phishguard.artifacts import ArtifactBundle
from phishguard.config import APP, FETCH, AppConfig, FetchConfig
from phishguard.features.extract import build_record, coverage, finalise_provenance
from phishguard.fetch import client
from phishguard.fetch.client import FetchResult
from phishguard.preprocess.transformer import Preprocessor


@dataclass
class ModelVerdict:
    key: str
    name: str
    family: str
    is_scratch: bool
    label: int
    phishing_score: float

    @property
    def verdict(self) -> str:
        return "phishing" if self.label == schema.PHISHING_LABEL else "legitimate"


@dataclass
class InferenceResult:
    url: str
    fetch: FetchResult
    verdicts: list[ModelVerdict]
    provenance: dict[str, str]
    features: dict[str, float]
    raw_values: dict[str, Any]
    coverage_scraped: int
    coverage_total: int
    coverage_ratio: float
    abstained: bool
    abstain_reason: str = ""
    cascade_fallbacks: dict[str, int] = field(default_factory=dict)

    @property
    def fetch_state(self) -> str:
        return self.fetch.state

    @property
    def consensus(self) -> str | None:
        """The verdict when both algorithm families agree, otherwise None.

        Grouped by family, not by model. The scratch pair and the reference pair are two
        implementations of the same two algorithms, so counting four votes would weight
        each algorithm twice and dress correlated agreement up as consensus.
        """
        if self.abstained:
            return None
        by_family: dict[str, set[str]] = {}
        for verdict in self.verdicts:
            by_family.setdefault(verdict.family, set()).add(verdict.verdict)
        decided = {
            family: next(iter(values)) for family, values in by_family.items() if len(values) == 1
        }
        if len(decided) != len(by_family):
            return None
        answers = set(decided.values())
        return answers.pop() if len(answers) == 1 else None


def classify_url(
    url: str,
    bundle: ArtifactBundle,
    *,
    fetch_config: FetchConfig | None = None,
    app_config: AppConfig | None = None,
    fetch_result: FetchResult | None = None,
) -> InferenceResult:
    """Fetch, extract, transform, and run all four models."""
    fetch_config = fetch_config or FETCH
    app_config = app_config or APP

    fetch = fetch_result if fetch_result is not None else client.get(url, fetch_config)

    robots_allowed: bool | None = None
    if fetch.ok and fetch_config.robots:
        robots_allowed = client.get_robots(fetch.final_url or url, fetch_config)

    record, provenance = build_record(
        url,
        fetch,
        ref_scope=app_config.ref_scope,
        robots_allowed=robots_allowed,
        demoted=bundle.demoted,
    )
    provenance = finalise_provenance(record, provenance)
    raw_values = {
        name: record.iloc[0][name] for name in schema.FEATURE_ORDER if name in record.columns
    }

    preprocessor = Preprocessor(bundle.stats)
    matrix = preprocessor.transform_matrix(record)
    features = {name: float(matrix[0][i]) for i, name in enumerate(schema.FEATURE_ORDER)}

    scraped, total, ratio = coverage(provenance)

    verdicts = [
        ModelVerdict(
            key=key,
            name=model.name,
            family=model.family,
            is_scratch=model.is_scratch,
            label=int(model.predict(matrix)[0]),
            phishing_score=float(model.score_phishing(matrix)[0]),
        )
        for key, model in bundle.models.items()
    ]

    abstained = ratio < app_config.coverage_min_ratio
    reason = ""
    if abstained:
        reason = (
            f"Only {scraped} of {total} page features could be read from this URL "
            f"({ratio:.0%}, below the {app_config.coverage_min_ratio:.0%} threshold). "
            "The rest would be filled from the training distribution, which is 92.5% "
            "legitimate pages -- so a prediction here would lean toward 'legitimate' "
            "because of the missing evidence rather than because of the URL."
        )

    return InferenceResult(
        url=url,
        fetch=fetch,
        verdicts=verdicts,
        provenance=provenance,
        features=features,
        raw_values=raw_values,
        coverage_scraped=scraped,
        coverage_total=total,
        coverage_ratio=ratio,
        abstained=abstained,
        abstain_reason=reason,
        cascade_fallbacks=dict(preprocessor.fallback_counts),
    )


def classify_frame(
    frame: pd.DataFrame, bundle: ArtifactBundle
) -> tuple[pd.DataFrame, np.ndarray]:
    """Classify a batch already in the raw feature schema, without fetching anything.

    The batch path for uploaded CSVs. It reuses the same preprocessor and the same models,
    so a row scored here and the same row scored one at a time agree exactly.
    """
    missing = set(schema.FEATURE_ORDER) - set(frame.columns)
    if missing:
        raise ValueError(f"uploaded data is missing feature columns: {sorted(missing)}")

    prepared = frame.copy()
    for column in schema.INTERMEDIATE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = np.nan

    preprocessor = Preprocessor(bundle.stats)
    matrix = preprocessor.transform_matrix(prepared)

    out = pd.DataFrame(index=frame.index)
    for key, model in bundle.models.items():
        out[f"{key}_label"] = model.predict(matrix)
        out[f"{key}_phishing_score"] = model.score_phishing(matrix)
    return out, matrix
