"""Turn a URL and a fetch result into one raw row.

The row is emitted in the *raw CSV schema shape* -- the same shape a row of the training
file has -- rather than as a 49-column feature vector. That is the point: the record then
passes through the identical transform as a training batch, with no special case anywhere
in the pipeline. A dedicated single-row path would be one more place for the serving
computation to drift from the training one.

The builder never guesses and never substitutes a default. Anything it could not determine
is NaN. Deciding what a NaN becomes is the preprocessor's job, and mixing the two would
destroy the record of which values were real -- which is the only thing that makes the
provenance display honest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phishguard import schema
from phishguard.config import RefScope
from phishguard.features.html_features import (
    empty_html_features,
    extract_html_features,
    parse_html,
)
from phishguard.fetch.client import FetchResult


def build_record(
    url: str,
    fetch: FetchResult,
    *,
    ref_scope: RefScope | None = None,
    robots_allowed: bool | None = None,
    demoted: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a 1-row raw-schema frame and the per-feature provenance map."""
    row: dict[str, Any] = dict.fromkeys(schema.WORKING_COLUMNS, np.nan)
    provenance: dict[str, str] = {}

    # The URL is the one thing always known. Every URL-derived feature is left NaN here
    # and computed by the fill chain inside transform, so single-URL inference and
    # training run the same code over the same inputs.
    row["URL"] = url
    for name in schema.URL_ONLY_FEATURES:
        provenance[name] = schema.PROVENANCE_URL

    if fetch.ok and fetch.html:
        soup = parse_html(fetch.html)
        page = extract_html_features(
            fetch.html,
            soup,
            fetch.final_url or url,
            list(fetch.redirect_chain),
            ref_scope=ref_scope,
            robots_allowed=robots_allowed,
            demoted=demoted,
        )
    else:
        page = empty_html_features(demoted)

    row["Title"] = page.get("Title")

    for name in schema.HTML_FEATURES:
        value = page.get(name)
        if name in demoted:
            provenance[name] = schema.PROVENANCE_DEMOTED
            continue
        if value is None:
            # Left NaN; transform will impute and relabel it.
            provenance[name] = schema.PROVENANCE_IMPUTED
        else:
            row[name] = value
            provenance[name] = schema.PROVENANCE_SCRAPED

    # The three title-dependent hybrids have URL-side implementations but need a Title,
    # so their provenance follows whether the fetch produced one.
    for name in schema.TITLE_HYBRID_FEATURES:
        provenance[name] = (
            schema.PROVENANCE_SCRAPED if row["Title"] is not None else schema.PROVENANCE_IMPUTED
        )

    frame = pd.DataFrame([row], columns=list(schema.WORKING_COLUMNS))
    return frame, provenance


def finalise_provenance(
    frame_before_transform: pd.DataFrame, provenance: dict[str, str]
) -> dict[str, str]:
    """Relabel as imputed anything that was still missing when transform took over.

    build_record emits url, scraped and demoted; this fills in imputed. Splitting it this
    way keeps the extractor from having to know what the imputer will do.
    """
    resolved = dict(provenance)
    row = frame_before_transform.iloc[0]
    have_url = pd.notna(row.get("URL"))
    have_title = pd.notna(row.get("Title"))

    for name in schema.FEATURE_ORDER:
        if resolved.get(name) == schema.PROVENANCE_DEMOTED:
            continue

        # Some features are deliberately left blank by the record builder: the fill chain
        # inside transform computes them, which is what makes single-URL inference run the
        # same code as training. For those, blank means "not computed yet", not "unknown",
        # and the honest label is the source they will be computed *from*.
        #
        # Getting this wrong is not cosmetic. Marking a value "imputed" when it was in
        # fact derived from real evidence understates what the system measured, and the
        # coverage meter -- which drives the abstention rule -- is built from these labels.
        if name in schema.URL_ONLY_FEATURES and have_url:
            resolved[name] = schema.PROVENANCE_URL
            continue

        # The three title-dependent hybrids have URL-side implementations but consume a
        # Title that only exists once the page has been fetched and parsed. With a title
        # they are derived from scraped evidence; without one they genuinely are imputed.
        if name in schema.TITLE_HYBRID_FEATURES:
            resolved[name] = (
                schema.PROVENANCE_SCRAPED if have_title else schema.PROVENANCE_IMPUTED
            )
            continue

        if name in frame_before_transform.columns and pd.isna(row[name]):
            resolved[name] = schema.PROVENANCE_IMPUTED

    return resolved


def coverage(provenance: dict[str, str]) -> tuple[int, int, float]:
    """How much of the obtainable page evidence this fetch actually produced.

    Two decisions about the denominator, both load-bearing.

    It is the page features, not all 49. The URL-derived features are always available, so
    including them would inflate coverage on a page that was never fetched at all --
    reporting 21/49 as "43% covered" when the page evidence is zero.

    It excludes **demoted** features. Those are permanently unavailable by policy, whatever
    happens on the wire, so counting them would conflate two different facts: "this fetch
    did not get much" and "we can never measure this". The abstention rule exists to detect
    the first. Counting demotions in the denominator would also make the threshold move
    every time the agreement gate is re-run -- and with enough demotions the service would
    abstain on every request no matter how well the fetch went, which is not caution, just
    breakage.

    How many features are permanently demoted is reported separately, so the reduced
    evidence base stays visible rather than being hidden by this choice.
    """
    obtainable = [
        name
        for name in schema.HTML_FEATURES
        if provenance.get(name) != schema.PROVENANCE_DEMOTED
    ]
    total = len(obtainable)
    scraped = sum(
        1 for name in obtainable if provenance.get(name) == schema.PROVENANCE_SCRAPED
    )
    return scraped, total, (scraped / total if total else 0.0)
