from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phishguard import schema

FIXTURES = Path(__file__).parent / "fixtures"
RAW_SAMPLE = FIXTURES / "raw_sample_2000.parquet"


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """2,000 committed rows in the raw 56-column schema.

    Over-samples the missing-URL and missing-Domain branches on purpose: those are where
    a port is most likely to drift and least likely to be caught by a uniform draw.
    """
    if not RAW_SAMPLE.exists():
        pytest.skip(f"{RAW_SAMPLE} not built; run scripts/sample_raw.py")
    return pd.read_parquet(RAW_SAMPLE)


@pytest.fixture(scope="session")
def raw_X(raw_sample: pd.DataFrame) -> pd.DataFrame:
    return raw_sample.drop(columns=[*schema.DROPPED_IDENTIFIER_COLUMNS, schema.TARGET_COLUMN])


@pytest.fixture(scope="session")
def raw_y(raw_sample: pd.DataFrame) -> pd.Series:
    return raw_sample[schema.TARGET_COLUMN]
