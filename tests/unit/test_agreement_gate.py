"""Tier 6a -- the agreement gate, offline against the committed fixtures.

Zero network. The fixture corpus was captured once through the production guarded client
and committed, so this measurement is a regression baseline rather than a re-measurement
of the live internet: the same bytes produce the same numbers on every run, and a change
here means the extractor changed, not that a website did.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from phishguard import schema
from phishguard.features.html_features import extract_html_features, parse_html

FIXTURE_DIR = Path("tests/fixtures/html")
EXPECTED = Path("tests/fixtures/expected.parquet")

pytestmark = pytest.mark.skipif(
    not EXPECTED.exists(), reason="no fixture corpus; run scripts/capture_fixtures.py"
)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return pd.read_parquet(EXPECTED)


def _read(sha1: str) -> str:
    return gzip.decompress((FIXTURE_DIR / f"{sha1}.html.gz").read_bytes()).decode(
        "utf-8", errors="replace"
    )


def test_every_recorded_fixture_is_present(corpus):
    missing = [s for s in corpus["sha1"] if not (FIXTURE_DIR / f"{s}.html.gz").exists()]
    assert not missing, f"{len(missing)} fixture(s) recorded but absent from the corpus"


def test_no_orphan_fixtures(corpus):
    """A fixture with no expected values contributes nothing and inflates the corpus."""
    recorded = {f"{s}.html.gz" for s in corpus["sha1"]}
    orphans = [p.name for p in FIXTURE_DIR.iterdir() if p.name not in recorded]
    assert not orphans, f"fixtures with no recorded expectations: {orphans}"


def test_corpus_is_label_stratified(corpus):
    """Both classes must be present, because they fail in completely different ways."""
    labels = set(corpus["label"].unique())
    assert labels == {schema.PHISHING_LABEL, schema.LEGITIMATE_LABEL}


def test_extraction_runs_over_every_fixture_without_raising(corpus):
    """Real pages, not hand-written markup. This is where malformed markup actually lives."""
    for record in corpus.itertuples():
        html = _read(record.sha1)
        features = extract_html_features(html, parse_html(html), record.final_url, [])
        assert set(schema.HTML_FEATURES) <= set(features), record.url


def test_extraction_is_deterministic(corpus):
    """Same bytes, same numbers. Without this the gate would drift on its own."""
    record = corpus.iloc[0]
    html = _read(record["sha1"])
    first = extract_html_features(html, parse_html(html), record["final_url"], [])
    second = extract_html_features(html, parse_html(html), record["final_url"], [])
    assert first == second


def test_reference_scope_is_reachable_from_configuration(corpus):
    """Which definition is right is settled by measurement, so both must be selectable."""
    record = corpus.iloc[0]
    html = _read(record["sha1"])
    soup = parse_html(html)

    anchors = extract_html_features(html, soup, record["final_url"], [], ref_scope="anchor")
    resources = extract_html_features(
        html, soup, record["final_url"], [], ref_scope="all_resources"
    )
    for name in ("NoOfSelfRef", "NoOfExternalRef", "NoOfEmptyRef"):
        assert name in anchors and name in resources


def test_demotion_is_enforced_over_real_pages(corpus):
    """Demotion has to hold against pages that would happily supply a value."""
    demoted = frozenset({"NoOfImage", "NoOfJS"})
    for record in corpus.head(10).itertuples():
        html = _read(record.sha1)
        features = extract_html_features(
            html, parse_html(html), record.final_url, [], demoted=demoted
        )
        for name in demoted:
            assert features[name] is None, record.url


def test_the_corpus_records_the_link_rot_problem(corpus):
    """Not a code test -- a finding, pinned so it does not get forgotten.

    Far fewer phishing pages could be captured than legitimate ones, because those domains
    are short-lived and the dataset was crawled in 2023-24. This is why the gate is judged
    on legitimate URLs, and why the application abstains rather than predicting when a
    fetch fails: unreachable correlates with phishing in reality and with legitimate in
    the imputed features.
    """
    phishing = int((corpus["label"] == schema.PHISHING_LABEL).sum())
    legitimate = int((corpus["label"] == schema.LEGITIMATE_LABEL).sum())
    assert phishing < legitimate, (
        "the corpus was sampled evenly by label; if phishing pages now capture as often "
        "as legitimate ones, the link-rot assumption behind the abstention rule and the "
        "gate's legitimate-only judgement deserves rechecking"
    )
