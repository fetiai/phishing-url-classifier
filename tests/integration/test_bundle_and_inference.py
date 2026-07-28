"""Tier 5 and Tier 8 -- the trained bundle and the inference path.

These need a real artifact bundle, so they are marked and skipped when none is present
rather than silently passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phishguard import schema
from phishguard.artifacts import BundleError, load_bundle, sha256_file
from phishguard.config import AppConfig, FetchConfig
from phishguard.fetch.client import FetchResult
from phishguard.fetch.safety import FetchOutcome
from phishguard.inference import classify_frame, classify_url
from phishguard.preprocess.transformer import Preprocessor
from phishguard.selftest import run_golden

ARTIFACTS = Path("artifacts/v1")

pytestmark = pytest.mark.artifacts


@pytest.fixture(scope="module")
def bundle():
    if not (ARTIFACTS / "manifest.json").exists():
        pytest.skip("no artifact bundle; run `python -m phishguard.train`")
    return load_bundle(ARTIFACTS)


# --- the bundle itself -------------------------------------------------------


def test_every_recorded_file_exists_and_matches_its_hash(bundle):
    """A bundle that does not match its manifest is a bundle nobody can trace to a
    training run."""
    for relative, record in bundle.manifest["files"].items():
        path = ARTIFACTS / relative
        assert path.exists(), relative
        assert sha256_file(path) == record["sha256"], relative


def test_bundle_contains_everything_the_application_reads(bundle):
    # manifest.json is deliberately absent from its own file list: it cannot record a
    # hash of a document that contains that hash.
    expected = {
        "fitted_stats.json",
        "metrics.json",
        "golden_row.json",
        "feature_reference_stats.json",
        "dataset_profile.json",
        "dataset_sample.parquet",
        "eval_predictions.parquet",
        "extraction_agreement.json",
        "models/knn_scratch.npz",
        "models/knn_sklearn.joblib",
        "models/nb_scratch.json",
        "models/nb_sklearn.joblib",
    }
    present = set(bundle.manifest["files"])
    assert expected <= present, f"bundle is missing: {sorted(expected - present)}"


def test_a_tampered_file_is_detected(bundle, tmp_path):
    """The verification has to actually verify."""
    import shutil

    copy = tmp_path / "v1"
    shutil.copytree(ARTIFACTS, copy)
    target = copy / "metrics.json"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(BundleError, match="does not match its recorded hash"):
        load_bundle(copy)


def test_golden_row_selftest_passes(bundle):
    assert run_golden(ARTIFACTS) == []


def test_feature_order_is_the_frozen_one(bundle):
    assert tuple(bundle.stats.feature_order) == schema.FEATURE_ORDER


def test_scaler_scope_survived_serialisation(bundle):
    assert tuple(bundle.stats.scaler.columns) == schema.NUMERICAL_COLUMNS
    assert len(bundle.stats.scaler.columns) == 30


def test_nb_drop_reasons_are_recorded(bundle):
    assert set(bundle.stats.nb_drop) == set(bundle.stats.nb_drop_reasons)


# --- Tier 5: the metric locks ------------------------------------------------

#: Copied from the original's stored output. The legacy profile reconstructs its
#: configuration -- including standardising each split by its own statistics -- so
#: reproducing these is what demonstrates the port did not change the arithmetic.
RECORDED = {
    "knn_sklearn": {"accuracy": 0.98643, "phishing_recall": 0.857},
    "knn_scratch": {"accuracy": 0.98070},
}


def test_legacy_profile_reproduces_the_recorded_numbers(bundle):
    legacy = bundle.metrics["profiles"].get("legacy")
    if legacy is None:
        pytest.skip("bundle was built without the legacy comparison")

    for key, expected in RECORDED.items():
        actual = legacy["models"][key]
        for metric, value in expected.items():
            assert actual[metric] == pytest.approx(value, abs=0.005), (
                f"legacy {key}.{metric}: expected ~{value}, got {actual[metric]}"
            )


def test_legacy_is_flagged_as_leaky_and_corrected_is_not(bundle):
    profiles = bundle.metrics["profiles"]
    assert profiles["corrected"]["leaky"] is False
    if "legacy" in profiles:
        assert profiles["legacy"]["leaky"] is True


def test_removing_the_leak_lowers_phishing_recall(bundle):
    """Records the size of the effect rather than assuming it.

    The leaked configuration standardised the validation split by its own statistics,
    which is information the model cannot have at serving time. The gap between the two
    profiles is how much of the reported performance came from that.
    """
    profiles = bundle.metrics["profiles"]
    if "legacy" not in profiles:
        pytest.skip("bundle was built without the legacy comparison")

    corrected = profiles["corrected"]["models"]["knn_sklearn"]["phishing_recall"]
    leaked = profiles["legacy"]["models"]["knn_sklearn"]["phishing_recall"]
    assert leaked > corrected, (
        "the leaked configuration is expected to score better; if it does not, either the "
        "reconstruction is wrong or the leak was not where we think it was"
    )


def test_every_accuracy_is_reported_against_the_baseline(bundle):
    """Accuracy without the 0.9248 baseline beside it is not interpretable."""
    assert bundle.metrics["baseline_accuracy"] == schema.MAJORITY_BASELINE_ACCURACY
    for profile in bundle.metrics["profiles"].values():
        for key, record in profile["models"].items():
            if key.startswith("_"):
                continue
            assert record["baseline_accuracy"] == schema.MAJORITY_BASELINE_ACCURACY


def test_models_beat_the_constant_predictor(bundle):
    for key, record in bundle.metrics["profiles"]["corrected"]["models"].items():
        if key.startswith("_"):
            continue
        assert record["accuracy"] > schema.MAJORITY_BASELINE_ACCURACY, key
        assert record["phishing_recall"] > 0.0, f"{key} catches no phishing at all"


def test_scratch_models_track_their_references(bundle):
    parity = bundle.metrics["profiles"]["corrected"]["models"]["_parity"]
    assert parity["knn"] < 0.05, f"scratch KNN diverges on {parity['knn']:.2%} of rows"
    assert parity["naive_bayes"] < 0.10, (
        f"scratch NB diverges on {parity['naive_bayes']:.2%} of rows"
    )


def test_no_test_split_metric_is_claimed(bundle):
    """The held-out file has no labels, so any test score would have been invented."""
    text = json.dumps(bundle.metrics).lower()
    assert "test_accuracy" not in text
    assert "n_test" not in text


# --- Tier 8: inference -------------------------------------------------------

FIXTURE_HTML = """<!doctype html><html><head><title>Acme Bank Sign In</title>
<meta name="description" content="Sign in to your account">
<meta name="viewport" content="width=device-width">
<link rel="icon" href="/favicon.ico"><link rel="stylesheet" href="/a.css">
</head><body>
<form action="/login"><input type="password" name="p"><input type="hidden" name="t">
<button>Sign in</button></form>
<a href="/help">Help</a><a href="https://facebook.com/acme">Facebook</a>
<img src="/logo.png"><script>var a=1;</script>
<footer>© 2024 Acme Bank. All rights reserved.</footer>
</body></html>"""


def _fixture_fetch(url: str = "https://acme.example.com/login") -> FetchResult:
    return FetchResult(
        outcome=FetchOutcome.OK,
        url=url,
        final_url=url,
        status_code=200,
        html=FIXTURE_HTML,
        content_type="text/html",
    )


def test_single_url_returns_four_verdicts_and_full_provenance(bundle):
    """A blank provenance cell is a bug: provenance is the mechanism by which the
    application is honest about what it measured versus what it filled in."""
    result = classify_url(
        "https://acme.example.com/login",
        bundle,
        fetch_result=_fixture_fetch(),
        fetch_config=FetchConfig(enabled=True, robots=False),
    )

    assert len(result.verdicts) == 4
    assert {v.family for v in result.verdicts} == {"knn", "naive_bayes"}

    assert set(result.provenance) >= set(schema.FEATURE_ORDER)
    for name in schema.FEATURE_ORDER:
        assert result.provenance[name] in schema.PROVENANCE_VALUES, name

    for verdict in result.verdicts:
        assert verdict.label in (0, 1)
        assert 0.0 <= verdict.phishing_score <= 1.0


def test_url_features_are_labelled_as_coming_from_the_url(bundle):
    result = classify_url(
        "https://acme.example.com/login",
        bundle,
        fetch_result=_fixture_fetch(),
        fetch_config=FetchConfig(enabled=True, robots=False),
    )
    for name in schema.URL_ONLY_FEATURES:
        assert result.provenance[name] == schema.PROVENANCE_URL, name


def test_successful_fetch_yields_scraped_page_features(bundle):
    result = classify_url(
        "https://acme.example.com/login",
        bundle,
        fetch_result=_fixture_fetch(),
        fetch_config=FetchConfig(enabled=True, robots=False),
    )
    scraped = [
        n for n in schema.HTML_FEATURES if result.provenance[n] == schema.PROVENANCE_SCRAPED
    ]
    assert len(scraped) >= 20, f"only {len(scraped)} page features were read from the fixture"
    assert result.fetch_state == "scraped"
    assert not result.abstained


@pytest.mark.parametrize(
    "outcome,expected_state",
    [
        (FetchOutcome.TIMEOUT, "unreachable"),
        (FetchOutcome.DNS_FAIL, "unreachable"),
        (FetchOutcome.BLOCKED_BY_POLICY, "unreachable"),
        (FetchOutcome.CHALLENGE_DETECTED, "challenged"),
        (FetchOutcome.DISABLED, "unreachable"),
    ],
)
def test_failed_fetch_abstains_rather_than_guessing(bundle, outcome, expected_state):
    """The core honesty property.

    With no page, 28 of 49 features are imputed from a corpus that is 92.5% legitimate,
    so a prediction here would lean legitimate because of the missing evidence rather
    than because of the URL.
    """
    result = classify_url(
        "https://dead.example.com/",
        bundle,
        fetch_result=FetchResult(outcome=outcome, url="https://dead.example.com/"),
        fetch_config=FetchConfig(enabled=True, robots=False),
    )

    assert result.fetch_state == expected_state
    assert result.coverage_ratio == 0.0
    assert result.abstained
    assert result.consensus is None
    assert "evidence" in result.abstain_reason.lower()

    for name in schema.HTML_FEATURES:
        assert result.provenance[name] in {
            schema.PROVENANCE_IMPUTED,
            schema.PROVENANCE_DEMOTED,
        }


def test_abstention_threshold_is_configurable(bundle):
    permissive = AppConfig(coverage_min_ratio=0.0)
    result = classify_url(
        "https://dead.example.com/",
        bundle,
        fetch_result=FetchResult(outcome=FetchOutcome.TIMEOUT, url="https://dead.example.com/"),
        fetch_config=FetchConfig(enabled=True, robots=False),
        app_config=permissive,
    )
    assert not result.abstained


def test_coverage_denominator_is_the_page_features_only(bundle):
    """Using all 49 would report 43% coverage for a page that was never fetched, because
    the URL features are always present."""
    result = classify_url(
        "https://dead.example.com/",
        bundle,
        fetch_result=FetchResult(outcome=FetchOutcome.TIMEOUT, url="https://dead.example.com/"),
        fetch_config=FetchConfig(enabled=True, robots=False),
    )
    assert result.coverage_total == len(schema.HTML_FEATURES) == 25
    assert result.coverage_scraped == 0


def test_single_url_matches_a_one_row_batch(bundle):
    """The invariance guarantee, end to end through the real models."""
    result = classify_url(
        "https://acme.example.com/login",
        bundle,
        fetch_result=_fixture_fetch(),
        fetch_config=FetchConfig(enabled=True, robots=False),
    )

    from phishguard.features.extract import build_record

    record, _ = build_record("https://acme.example.com/login", _fixture_fetch())
    matrix = Preprocessor(bundle.stats).transform_matrix(record)

    for i, name in enumerate(schema.FEATURE_ORDER):
        assert result.features[name] == pytest.approx(float(matrix[0][i]), abs=0)


def test_batch_scoring_agrees_with_single_row_scoring(bundle):
    sample = pd.read_parquet(ARTIFACTS / "dataset_sample.parquet").head(40)
    predictions, _ = classify_frame(sample, bundle)

    for i in range(0, 40, 7):
        single, _ = classify_frame(sample.iloc[[i]], bundle)
        for column in predictions.columns:
            assert single[column].iloc[0] == pytest.approx(
                predictions[column].iloc[i], abs=1e-12
            ), column


def test_batch_rejects_a_frame_missing_feature_columns(bundle):
    sample = pd.read_parquet(ARTIFACTS / "dataset_sample.parquet").head(5)
    with pytest.raises(ValueError, match="missing feature columns"):
        classify_frame(sample.drop(columns=["NoOfImage"]), bundle)


def test_demoted_features_are_never_scraped(bundle):
    """Demotion has to be enforced at extraction, not merely displayed."""
    from phishguard.features.extract import build_record

    demoted = frozenset({"NoOfImage", "NoOfCSS"})
    _record, provenance = build_record(
        "https://acme.example.com/login", _fixture_fetch(), demoted=demoted
    )
    for name in demoted:
        assert provenance[name] == schema.PROVENANCE_DEMOTED


def test_transformed_vector_is_finite_for_every_fetch_state(bundle):
    for outcome in (FetchOutcome.OK, FetchOutcome.TIMEOUT, FetchOutcome.DISABLED):
        fetch = _fixture_fetch() if outcome is FetchOutcome.OK else FetchResult(outcome, "u")
        result = classify_url(
            "https://acme.example.com/login",
            bundle,
            fetch_result=fetch,
            fetch_config=FetchConfig(enabled=True, robots=False),
        )
        values = np.array(list(result.features.values()))
        assert np.isfinite(values).all(), outcome
