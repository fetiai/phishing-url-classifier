"""The frozen feature contract, checked against the data rather than against itself.

schema.py asserts its own group sizes at import time. These tests do something the module
cannot: they read the actual CSV header and confirm the frozen lists still describe the
file on disk. A schema that is internally consistent but no longer matches the data would
pass every assertion in the module and still be wrong.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from phishguard import schema

TRAIN_CSV = Path("data/raw/train.csv")
TEST_CSV = Path("data/raw/test.csv")


def _header(path: Path) -> list[str]:
    if not path.exists():
        pytest.skip(f"{path} not present")
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def test_raw_columns_match_the_training_csv_header():
    assert _header(TRAIN_CSV) == list(schema.RAW_COLUMNS)


def test_test_csv_is_unlabeled():
    """There is no true test metric available for this project, only the validation split.

    The held-out file carries no label column, so any 'test accuracy' would have to have
    been invented. Asserting it here keeps that from being claimed later by accident.
    """
    header = _header(TEST_CSV)
    assert schema.TARGET_COLUMN not in header
    assert set(schema.RAW_COLUMNS) - set(header) == {schema.TARGET_COLUMN}


def test_feature_order_is_the_header_minus_identifiers_target_and_intermediates():
    header = _header(TRAIN_CSV)
    expected = [
        c
        for c in header
        if c
        not in {
            *schema.DROPPED_IDENTIFIER_COLUMNS,
            schema.TARGET_COLUMN,
            *schema.INTERMEDIATE_COLUMNS,
        }
    ]
    assert list(schema.FEATURE_ORDER) == expected


def test_feature_order_preserves_file_order():
    header = _header(TRAIN_CSV)
    positions = [header.index(c) for c in schema.FEATURE_ORDER]
    assert positions == sorted(positions)


def test_dataset_typos_are_preserved_verbatim():
    """These three names are misspelled in the source data. Correcting them would silently
    break every lookup against the CSV."""
    for typo in ("NoOfDegitsInURL", "DegitRatioInURL", "SpacialCharRatioInURL"):
        assert typo in schema.FEATURE_ORDER


def test_group_sizes():
    assert len(schema.FEATURE_ORDER) == 49
    assert len(schema.CATEGORICAL_COLUMNS) == 23
    assert len(schema.CATEGORICAL_COLUMNS_FILTERED) == 19
    assert len(schema.CATEGORICAL_COLUMNS_TO_FILL) == 18
    assert len(schema.NUMERICAL_COLUMNS) == 30
    assert len(schema.CONTINUOUS_COLUMNS) == 9
    assert len(schema.DISCRETE_COLUMNS) == 21
    assert len(schema.URL_ONLY_FEATURES) == 21
    assert len(schema.TITLE_HYBRID_FEATURES) == 3
    assert len(schema.HTML_FEATURES) == 25


def test_availability_split_partitions_the_features():
    """21 URL-only + 3 title-dependent + 25 page-derived. The 28 that need a fetch are why
    the system must abstain when it cannot get one."""
    assert (
        set(schema.URL_ONLY_FEATURES)
        | set(schema.TITLE_HYBRID_FEATURES)
        | set(schema.HTML_FEATURES)
    ) == set(schema.FEATURE_ORDER)
    assert len(schema.TITLE_HYBRID_FEATURES) + len(schema.HTML_FEATURES) == 28


def test_scaler_scope_and_passthrough_partition_the_features():
    assert set(schema.NUMERICAL_COLUMNS) | set(schema.CATEGORICAL_COLUMNS_FILTERED) == set(
        schema.FEATURE_ORDER
    )
    assert not (set(schema.NUMERICAL_COLUMNS) & set(schema.CATEGORICAL_COLUMNS_FILTERED))


def test_cascade_covers_every_categorical_except_the_initial_group_key():
    assert set(schema.CATEGORICAL_COLUMNS_FILTERED) - set(
        schema.CATEGORICAL_COLUMNS_TO_FILL
    ) == set(schema.CATEGORICAL_FILL_INITIAL_GROUP_BY)


def test_phishing_is_the_positive_class():
    """Class 0 is phishing. Reversing this silently inverts every reported recall."""
    assert schema.PHISHING_LABEL == 0
    assert schema.LEGITIMATE_LABEL == 1
