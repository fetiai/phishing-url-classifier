"""The frozen feature contract.

Every column name, every column group, and the exact ordering of the 49 model features
live here and nowhere else. Nothing in the codebase may hardcode a column list; it imports
one of these constants instead.

Why this module is written defensively, with an assertion after every group: the original
notebook derived its column groups from whatever DataFrame happened to be in scope. The
scaler consequently touched only the 30 numeric columns and left the 19 binaries raw --
not by decision, but because ``numerical_columns`` was a notebook global that closed over
the training frame. Every recorded metric depends on that accident. Making the groups
explicit and asserting their sizes at import time means a typo fails immediately and
loudly, rather than silently changing what the models were fitted on.

Dataset column typos are real and preserved verbatim: ``NoOfDegitsInURL``,
``DegitRatioInURL``, ``SpacialCharRatioInURL``.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Raw CSV schema
# ---------------------------------------------------------------------------

#: The 56 columns of the raw CSV, in file order.
RAW_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "FILENAME",
    "URL",
    "URLLength",
    "Domain",
    "DomainLength",
    "IsDomainIP",
    "TLD",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "Title",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
    "label",
)

#: Dropped before modelling: identifiers and the target.
DROPPED_IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = ("id", "FILENAME")
TARGET_COLUMN: Final[str] = "label"

#: Text columns used to derive features and then dropped. They are not model inputs.
INTERMEDIATE_COLUMNS: Final[tuple[str, ...]] = ("URL", "Domain", "TLD", "Title")

#: The 53 columns the notebook's ``X_train`` carried, in file order.
WORKING_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c in RAW_COLUMNS if c not in DROPPED_IDENTIFIER_COLUMNS and c != TARGET_COLUMN
)

# ---------------------------------------------------------------------------
# The 49 model features
# ---------------------------------------------------------------------------

#: The frozen model input, in order. This ordering is the wire format between training
#: and serving: the matrix handed to every model has these columns, in exactly this
#: sequence. Reordering it silently changes what each model coefficient refers to.
FEATURE_ORDER: Final[tuple[str, ...]] = tuple(
    c for c in WORKING_COLUMNS if c not in INTERMEDIATE_COLUMNS
)

# ---------------------------------------------------------------------------
# Column groups, reproduced exactly as the notebook defined them
# ---------------------------------------------------------------------------

#: Notebook cell 12. Includes the four text intermediates, which is why it has 23
#: members while the model-facing filtered list has 19.
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "IsDomainIP",
    "HasObfuscation",
    "IsHTTPS",
    "HasTitle",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "HasDescription",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "Domain",
    "URL",
    "TLD",
    "Title",
    "HasExternalFormSubmit",
    "NoOfSelfRedirect",
    "NoOfURLRedirect",
)

#: Notebook cell 12: the same list without the four text intermediates. These are the
#: columns the scaler must pass through bit-identically.
CATEGORICAL_COLUMNS_FILTERED: Final[tuple[str, ...]] = tuple(
    c for c in CATEGORICAL_COLUMNS if c not in INTERMEDIATE_COLUMNS
)

#: Notebook cell 104. Order is load-bearing, not cosmetic: the imputation cascade appends
#: each just-filled column to the group-by key before moving to the next one, so column
#: *n* is imputed conditioned on columns 1..n-1. Reordering this list changes the fill
#: values it produces.
CATEGORICAL_COLUMNS_TO_FILL: Final[tuple[str, ...]] = (
    "IsDomainIP",
    "IsHTTPS",
    "HasTitle",
    "IsResponsive",
    "Pay",
    "HasHiddenFields",
    "Robots",
    "Crypto",
    "HasDescription",
    "Bank",
    "HasExternalFormSubmit",
    "HasFavicon",
    "HasSubmitButton",
    "HasPasswordField",
    "NoOfSelfRedirect",
    "HasCopyrightInfo",
    "NoOfURLRedirect",
    "HasSocialNet",
)

#: The initial group-by key for the cascade above (notebook cell 104).
CATEGORICAL_FILL_INITIAL_GROUP_BY: Final[tuple[str, ...]] = ("HasObfuscation",)

#: Notebook cell 13: everything in the working set that is not categorical. This is the
#: exact set the scaler touches, and no other.
NUMERICAL_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c in WORKING_COLUMNS if c not in CATEGORICAL_COLUMNS
)

#: Notebook cell 14.
CONTINUOUS_COLUMNS: Final[tuple[str, ...]] = (
    "CharContinuationRate",
    "ObfuscationRatio",
    "DegitRatioInURL",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "TLDLegitimateProb",
    "URLCharProb",
    "LetterRatioInURL",
    "SpacialCharRatioInURL",
)

#: Notebook cell 15.
DISCRETE_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c in NUMERICAL_COLUMNS if c not in CONTINUOUS_COLUMNS
)

# ---------------------------------------------------------------------------
# The 21 / 3 / 25 split by what a feature needs in order to be computed
# ---------------------------------------------------------------------------

#: Derivable from the URL string alone. Available even when the fetch fails.
URL_ONLY_FEATURES: Final[tuple[str, ...]] = (
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
)

#: Have a URL-side implementation, but consume a Title that only exists once the page has
#: been fetched and parsed. With no successful fetch these fall through to imputation.
TITLE_HYBRID_FEATURES: Final[tuple[str, ...]] = (
    "HasTitle",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
)

#: Require the fetched page. Slightly more than half the model's inputs -- which is why
#: single-URL inference fetches at all, and why it must abstain when the fetch fails.
HTML_FEATURES: Final[tuple[str, ...]] = tuple(
    c for c in FEATURE_ORDER if c not in URL_ONLY_FEATURES and c not in TITLE_HYBRID_FEATURES
)

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

#: Class 0 is phishing and is the positive class throughout this codebase. Class 1 is
#: legitimate. Every confusion matrix is oriented rows = true (0, 1), cols = pred (0, 1).
PHISHING_LABEL: Final[int] = 0
LEGITIMATE_LABEL: Final[int] = 1

#: A constant "always legitimate" predictor scores this on the full dataset. Any accuracy
#: figure is meaningless unless read against it.
MAJORITY_BASELINE_ACCURACY: Final[float] = 0.9248

#: Provenance values recorded per feature for a single inference.
PROVENANCE_URL: Final[str] = "url"
PROVENANCE_SCRAPED: Final[str] = "scraped"
PROVENANCE_IMPUTED: Final[str] = "imputed"
PROVENANCE_DEMOTED: Final[str] = "demoted"
PROVENANCE_VALUES: Final[frozenset[str]] = frozenset(
    {PROVENANCE_URL, PROVENANCE_SCRAPED, PROVENANCE_IMPUTED, PROVENANCE_DEMOTED}
)

# ---------------------------------------------------------------------------
# Import-time assertions. A typo above fails here, not three phases downstream.
# ---------------------------------------------------------------------------

assert len(RAW_COLUMNS) == 56, len(RAW_COLUMNS)
assert len(set(RAW_COLUMNS)) == 56, "duplicate column name in RAW_COLUMNS"
assert len(WORKING_COLUMNS) == 53, len(WORKING_COLUMNS)
assert len(FEATURE_ORDER) == 49, len(FEATURE_ORDER)

assert len(CATEGORICAL_COLUMNS) == 23, len(CATEGORICAL_COLUMNS)
assert len(CATEGORICAL_COLUMNS_FILTERED) == 19, len(CATEGORICAL_COLUMNS_FILTERED)
assert len(CATEGORICAL_COLUMNS_TO_FILL) == 18, len(CATEGORICAL_COLUMNS_TO_FILL)
assert len(NUMERICAL_COLUMNS) == 30, len(NUMERICAL_COLUMNS)
assert len(CONTINUOUS_COLUMNS) == 9, len(CONTINUOUS_COLUMNS)
assert len(DISCRETE_COLUMNS) == 21, len(DISCRETE_COLUMNS)

assert len(URL_ONLY_FEATURES) == 21, len(URL_ONLY_FEATURES)
assert len(TITLE_HYBRID_FEATURES) == 3, len(TITLE_HYBRID_FEATURES)
assert len(HTML_FEATURES) == 25, len(HTML_FEATURES)

# The scaler's scope and the pass-through set must exactly partition the 49 features.
assert set(NUMERICAL_COLUMNS) | set(CATEGORICAL_COLUMNS_FILTERED) == set(FEATURE_ORDER)
assert not (set(NUMERICAL_COLUMNS) & set(CATEGORICAL_COLUMNS_FILTERED))

# The three-way availability split must also partition the 49, with no overlap.
assert (
    set(URL_ONLY_FEATURES) | set(TITLE_HYBRID_FEATURES) | set(HTML_FEATURES)
    == set(FEATURE_ORDER)
)
assert not (set(URL_ONLY_FEATURES) & set(TITLE_HYBRID_FEATURES))
assert not (set(URL_ONLY_FEATURES) & set(HTML_FEATURES))
assert not (set(TITLE_HYBRID_FEATURES) & set(HTML_FEATURES))

assert set(CONTINUOUS_COLUMNS) <= set(NUMERICAL_COLUMNS)
assert set(CATEGORICAL_COLUMNS_TO_FILL) <= set(CATEGORICAL_COLUMNS_FILTERED)
assert set(CATEGORICAL_FILL_INITIAL_GROUP_BY) <= set(CATEGORICAL_COLUMNS_FILTERED)

# The cascade fills every categorical except the one it initially groups by.
assert set(CATEGORICAL_COLUMNS_FILTERED) - set(CATEGORICAL_COLUMNS_TO_FILL) == set(
    CATEGORICAL_FILL_INITIAL_GROUP_BY
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "CATEGORICAL_COLUMNS_FILTERED",
    "CATEGORICAL_COLUMNS_TO_FILL",
    "CATEGORICAL_FILL_INITIAL_GROUP_BY",
    "CONTINUOUS_COLUMNS",
    "DISCRETE_COLUMNS",
    "DROPPED_IDENTIFIER_COLUMNS",
    "FEATURE_ORDER",
    "HTML_FEATURES",
    "INTERMEDIATE_COLUMNS",
    "LEGITIMATE_LABEL",
    "MAJORITY_BASELINE_ACCURACY",
    "NUMERICAL_COLUMNS",
    "PHISHING_LABEL",
    "PROVENANCE_DEMOTED",
    "PROVENANCE_IMPUTED",
    "PROVENANCE_SCRAPED",
    "PROVENANCE_URL",
    "PROVENANCE_VALUES",
    "RAW_COLUMNS",
    "TARGET_COLUMN",
    "TITLE_HYBRID_FEATURES",
    "URL_ONLY_FEATURES",
    "WORKING_COLUMNS",
]
