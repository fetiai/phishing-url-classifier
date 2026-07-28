"""Tier 1 -- the ported feature functions against the frozen originals.

The comparison is exact. No tolerance, no ``check_dtype=False``, no ``allclose``. These
functions define what the training data means, so a discrepancy of any size is a
discrepancy in the feature distribution the models were fitted on -- and nothing
downstream would reveal it, because the metrics would simply be describing something
else.

The one intended difference between the two implementations is that the ported versions
take fitted statistics as arguments rather than computing them from their input frame. The
tests below supply those statistics from the same frame the original would have used, so
what is under test is the claim that injection changed the plumbing and not the arithmetic.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from phishguard.features import url_features as ported
from tests import legacy_reference as legacy

# Every fill_* the ported module is required to expose. The two entries the original also
# defines -- fill_numerical_columns and fill_categorical_columns -- are imputation stages
# and live in the preprocessing package, not here.
EXPECTED_FILL_FUNCTIONS = {
    "fill_url_length",
    "fill_domain",
    "fill_domain_length",
    "fill_is_domain_ip",
    "fill_tld",
    "fill_char_continuation_rate",
    "fill_tld_legitimate_prob",
    "fill_url_char_prob",
    "fill_tld_length",
    "fill_no_of_subdomains",
    "fill_has_obfuscation",
    "fill_no_of_obfuscated_characters",
    "fill_obfuscation_ratio",
    "fill_no_of_letters_in_url",
    "fill_letter_ratio_in_url",
    "fill_no_of_digits_in_url",
    "fill_digits_ratio_in_url",
    "fill_no_of_equals_in_url",
    "fill_no_of_qmark_in_url",
    "fill_no_of_ampersand_in_url",
    "fill_no_of_special_chars_in_url",
    "fill_specials_ratio_in_url",
    "fill_is_https",
    "fill_has_title",
    "fill_domain_title_match_score",
    "fill_url_title_match_score",
}

PREPROCESSING_STAGES = {"fill_numerical_columns", "fill_categorical_columns"}


def _fill_names(module) -> set[str]:
    return {n for n in dir(module) if n.startswith("fill_")}


def test_no_function_was_silently_dropped():
    """The meta-test: a port that forgets a function passes every other test here."""
    assert _fill_names(ported) == EXPECTED_FILL_FUNCTIONS
    assert _fill_names(legacy) == EXPECTED_FILL_FUNCTIONS | PREPROCESSING_STAGES


def test_expected_count_is_26():
    assert len(EXPECTED_FILL_FUNCTIONS) == 26


# --- per-function equivalence ------------------------------------------------
#
# Each case runs the legacy function and the ported function over independent copies of
# the same frame and compares the column each one writes.

SIMPLE_CASES = [
    ("fill_url_length", "URLLength", {"url_col": "URL", "url_length_col": "URLLength"}),
    ("fill_domain", "Domain", {"url_col": "URL", "domain_col": "Domain"}),
    (
        "fill_domain_length",
        "DomainLength",
        {"domain_col": "Domain", "domain_length_col": "DomainLength"},
    ),
    (
        "fill_is_domain_ip",
        "IsDomainIP",
        {"domain_col": "Domain", "is_domain_ip_col": "IsDomainIP"},
    ),
    ("fill_tld", "TLD", {"domain_col": "Domain", "tld_col": "TLD"}),
    (
        "fill_char_continuation_rate",
        "CharContinuationRate",
        {"url_col": "URL", "char_rate_col": "CharContinuationRate"},
    ),
    ("fill_tld_length", "TLDLength", {"tld_col": "TLD", "tld_length_col": "TLDLength"}),
    (
        "fill_no_of_subdomains",
        "NoOfSubDomain",
        {"domain_col": "Domain", "subdomain_col": "NoOfSubDomain"},
    ),
    (
        "fill_has_obfuscation",
        "HasObfuscation",
        {"url_col": "URL", "obfuscation_col": "HasObfuscation"},
    ),
    (
        "fill_no_of_obfuscated_characters",
        "NoOfObfuscatedChar",
        {"url_col": "URL", "obf_char_col": "NoOfObfuscatedChar"},
    ),
    (
        "fill_obfuscation_ratio",
        "ObfuscationRatio",
        {
            "obf_char_col": "NoOfObfuscatedChar",
            "url_length_col": "URLLength",
            "obf_ratio_col": "ObfuscationRatio",
        },
    ),
    (
        "fill_no_of_letters_in_url",
        "NoOfLettersInURL",
        {"url_col": "URL", "letters_col": "NoOfLettersInURL"},
    ),
    (
        "fill_letter_ratio_in_url",
        "LetterRatioInURL",
        {
            "letters_col": "NoOfLettersInURL",
            "url_length_col": "URLLength",
            "ratio_col": "LetterRatioInURL",
        },
    ),
    (
        "fill_no_of_digits_in_url",
        "NoOfDegitsInURL",
        {"url_col": "URL", "digits_col": "NoOfDegitsInURL"},
    ),
    (
        "fill_digits_ratio_in_url",
        "DegitRatioInURL",
        {"url_col": "URL", "ratio_col": "DegitRatioInURL"},
    ),
    (
        "fill_no_of_equals_in_url",
        "NoOfEqualsInURL",
        {"url_col": "URL", "equals_col": "NoOfEqualsInURL"},
    ),
    (
        "fill_no_of_qmark_in_url",
        "NoOfQMarkInURL",
        {"url_col": "URL", "qmark_col": "NoOfQMarkInURL"},
    ),
    (
        "fill_no_of_ampersand_in_url",
        "NoOfAmpersandInURL",
        {"url_col": "URL", "ampersand_col": "NoOfAmpersandInURL"},
    ),
    (
        "fill_no_of_special_chars_in_url",
        "NoOfOtherSpecialCharsInURL",
        {"url_col": "URL", "special_chars_col": "NoOfOtherSpecialCharsInURL"},
    ),
    (
        "fill_specials_ratio_in_url",
        "SpacialCharRatioInURL",
        {"url_col": "URL", "ratio_col": "SpacialCharRatioInURL"},
    ),
    ("fill_is_https", "IsHTTPS", {"url_col": "URL", "https_col": "IsHTTPS"}),
    ("fill_has_title", "HasTitle", {"title_col": "Title", "has_title_col": "HasTitle"}),
    (
        "fill_domain_title_match_score",
        "DomainTitleMatchScore",
        {
            "title_col": "Title",
            "domain_col": "Domain",
            "score_col": "DomainTitleMatchScore",
        },
    ),
    (
        "fill_url_title_match_score",
        "URLTitleMatchScore",
        {"title_col": "Title", "url_col": "URL", "score_col": "URLTitleMatchScore"},
    ),
]


@pytest.mark.parametrize("func_name,column,kwargs", SIMPLE_CASES, ids=[c[0] for c in SIMPLE_CASES])
def test_ported_function_matches_original(raw_X, func_name, column, kwargs):
    legacy_out = getattr(legacy, func_name)(raw_X.copy(), **kwargs)
    ported_out = getattr(ported, func_name)(raw_X.copy(), **kwargs)
    assert_series_equal(legacy_out[column], ported_out[column], check_exact=True)


def test_url_char_prob_matches(raw_X):
    """calculate_char_prob is already a separate function in the original, so this pair
    should agree without any injection argument."""
    legacy_prob = legacy.calculate_char_prob(raw_X.copy(), "URL")
    ported_prob = ported.calculate_char_prob(raw_X.copy(), "URL")
    assert legacy_prob == ported_prob

    legacy_out = legacy.fill_url_char_prob(raw_X.copy(), char_prob=legacy_prob)
    ported_out = ported.fill_url_char_prob(raw_X.copy(), char_prob=ported_prob)
    assert_series_equal(
        legacy_out["URLCharProb"], ported_out["URLCharProb"], check_exact=True
    )


def test_tld_legitimate_prob_matches_when_statistics_are_supplied(raw_X):
    """The injection case.

    The original derives its per-TLD means and its global fill value from the frame it is
    given. The port takes both as arguments. Supplying them from that same frame is the
    exact claim being tested: the plumbing changed, the arithmetic did not.
    """
    frame = raw_X.copy()
    frame = ported.fill_domain(frame)
    frame = ported.fill_is_domain_ip(frame)
    frame = ported.fill_tld(frame)

    tld_prob_mean, global_fill = ported.compute_tld_prob_statistics(frame)

    legacy_out = legacy.fill_tld_legitimate_prob(frame.copy())
    ported_out = ported.fill_tld_legitimate_prob(
        frame.copy(), tld_prob_mean=tld_prob_mean, global_fill_value=global_fill
    )
    assert_series_equal(
        legacy_out["TLDLegitimateProb"],
        ported_out["TLDLegitimateProb"],
        check_exact=True,
    )


def test_lifted_statistics_match_the_originals_inline_computation(raw_X):
    """compute_tld_prob_statistics must reproduce what the original computed inline,
    including the abs(skew) > 1 branch -- which is deliberately a different threshold from
    the numeric imputer's abs(skew) > 3."""
    frame = raw_X.copy()
    frame = ported.fill_domain(frame)
    frame = ported.fill_tld(frame)

    tld_prob_mean, global_fill = ported.compute_tld_prob_statistics(frame)

    skewness = frame["TLDLegitimateProb"].skew()
    expected = (
        frame["TLDLegitimateProb"].median()
        if (skewness > 1 or skewness < -1)
        else frame["TLDLegitimateProb"].mean()
    )
    assert global_fill == pytest.approx(float(expected), abs=0, rel=0)
    assert_series_equal(
        tld_prob_mean,
        frame.groupby("TLD")["TLDLegitimateProb"].mean(),
        check_exact=True,
    )


def test_full_chain_matches_end_to_end(raw_X):
    """The aggregate check: run every function in pipeline order through both
    implementations and compare the whole frame at once."""

    def run(mod, frame: pd.DataFrame) -> pd.DataFrame:
        frame = mod.fill_url_length(frame)
        frame = mod.fill_domain(frame)
        frame = mod.fill_domain_length(frame)
        frame = mod.fill_is_domain_ip(frame)
        frame = mod.fill_tld(frame)
        frame = mod.fill_char_continuation_rate(frame)
        if mod is ported:
            mean, fill = ported.compute_tld_prob_statistics(frame)
            frame = mod.fill_tld_legitimate_prob(
                frame, tld_prob_mean=mean, global_fill_value=fill
            )
        else:
            frame = mod.fill_tld_legitimate_prob(frame)
        frame = mod.fill_url_char_prob(frame, char_prob=mod.calculate_char_prob(frame, "URL"))
        frame = mod.fill_tld_length(frame)
        frame = mod.fill_no_of_subdomains(frame)
        frame = mod.fill_has_obfuscation(frame)
        frame = mod.fill_no_of_obfuscated_characters(frame)
        frame = mod.fill_obfuscation_ratio(frame)
        frame = mod.fill_no_of_letters_in_url(frame)
        frame = mod.fill_letter_ratio_in_url(frame)
        frame = mod.fill_no_of_digits_in_url(frame)
        frame = mod.fill_digits_ratio_in_url(frame)
        frame = mod.fill_no_of_equals_in_url(frame)
        frame = mod.fill_no_of_qmark_in_url(frame)
        frame = mod.fill_no_of_ampersand_in_url(frame)
        frame = mod.fill_no_of_special_chars_in_url(frame)
        frame = mod.fill_specials_ratio_in_url(frame)
        frame = mod.fill_is_https(frame)
        frame = mod.fill_has_title(frame)
        frame = mod.fill_domain_title_match_score(frame)
        return mod.fill_url_title_match_score(frame)

    assert_frame_equal(run(legacy, raw_X.copy()), run(ported, raw_X.copy()), check_exact=True)


# --- row-level twins ---------------------------------------------------------


def test_twins_agree_with_the_dataframe_functions(raw_X):
    """The twins are a convenience for single-URL inference, not a second implementation.
    Where they disagree with the DataFrame function, the twin is what is wrong."""
    frame = ported.fill_domain(raw_X.copy())
    frame = ported.fill_tld(frame)

    urls = frame["URL"]

    checks = [
        (ported.url_length, urls, "URLLength", ported.fill_url_length),
        (ported.char_continuation_rate, urls, "CharContinuationRate", ported.fill_char_continuation_rate),
        (ported.no_of_letters, urls, "NoOfLettersInURL", ported.fill_no_of_letters_in_url),
        (ported.no_of_digits, urls, "NoOfDegitsInURL", ported.fill_no_of_digits_in_url),
        (ported.no_of_equals, urls, "NoOfEqualsInURL", ported.fill_no_of_equals_in_url),
        (ported.no_of_qmark, urls, "NoOfQMarkInURL", ported.fill_no_of_qmark_in_url),
        (ported.no_of_ampersand, urls, "NoOfAmpersandInURL", ported.fill_no_of_ampersand_in_url),
        (ported.is_https, urls, "IsHTTPS", ported.fill_is_https),
    ]

    for twin, series, column, df_func in checks:
        blank = raw_X.copy()
        blank[column] = pd.NA
        computed = df_func(blank)[column]
        for i in range(0, len(series), 37):  # stride, to keep the test quick
            value = series.iloc[i]
            expected = computed.iloc[i]
            actual = twin(value if pd.notnull(value) else None)
            if pd.isnull(expected):
                assert actual is None or pd.isnull(actual), (column, i, value)
            else:
                assert float(actual) == float(expected), (column, i, value)


def test_domain_and_tld_twins(raw_X):
    for url in raw_X["URL"].dropna().head(200):
        domain = ported.domain_of(url)
        assert domain == ported.fill_domain(pd.DataFrame({"URL": [url], "Domain": [pd.NA]}))[
            "Domain"
        ].iloc[0]
        assert ported.tld_of(domain) == ported.fill_tld(
            pd.DataFrame({"Domain": [domain], "TLD": [pd.NA]})
        )["TLD"].iloc[0]


def test_reversed_string_rule_fires_on_bare_domains():
    """Documents preserved defect: rule 4 reverses the URL and tests a pattern that is
    invariant under reversal, so it matches every scheme-less bare domain. Preserved on
    purpose -- this is what produced the training labels."""
    assert ported.has_obfuscation("example.com") == 1
    assert ported.detect_advanced_obfuscation("example.com") == 1
    # A URL with a scheme contains '/' and ':', so rule 4 cannot match it.
    assert ported.detect_advanced_obfuscation("https://x.co/a") == 0
