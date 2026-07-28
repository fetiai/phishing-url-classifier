"""Tier 8 -- every page actually renders.

These run each page through Streamlit's own script runner and assert no exception was
raised. That distinction matters: Streamlit answers HTTP 200 on ``/`` and on
``/_stcore/health`` whether or not the script inside it ran, because the server is healthy
even when the app is broken. A status-code check is evidence that something is listening,
not that anything works -- which is exactly how a page that raised ImportError on every
request passed a green health check.

Anything asserting only on a status code belongs in the same category and should be
treated with the same suspicion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ARTIFACTS = Path("artifacts/v1")
APP_DIR = Path("app")

PAGES = [
    "app/Home.py",
    "app/pages/1_Single_URL.py",
    "app/pages/2_Batch_CSV.py",
    "app/pages/3_Model_Evaluation.py",
    "app/pages/4_Dataset_Explorer.py",
    "app/pages/5_Methodology.py",
]

pytestmark = pytest.mark.artifacts


@pytest.fixture(scope="module")
def app_test_cls():
    return pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.mark.parametrize("page", PAGES, ids=[Path(p).stem for p in PAGES])
def test_page_runs_without_raising(app_test_cls, page):
    if not (ARTIFACTS / "manifest.json").exists():
        pytest.skip("no artifact bundle; run `python -m phishguard.train`")

    app = app_test_cls.from_file(page, default_timeout=120)
    app.run()

    assert not app.exception, (
        f"{page} raised: "
        + "; ".join(f"{e.type}: {e.value}" for e in app.exception)
    )


def test_every_page_file_is_covered():
    """A page added without a smoke test is a page nobody checks."""
    on_disk = {str(p) for p in [Path("app/Home.py"), *sorted(Path("app/pages").glob("*.py"))]}
    assert on_disk == set(PAGES), (
        f"page files and tested pages disagree: {sorted(on_disk ^ set(PAGES))}"
    )


def test_home_renders_the_scope_disclaimer(app_test_cls):
    """The framing is load-bearing, not decoration: this system must never present itself
    as a security product."""
    if not (ARTIFACTS / "manifest.json").exists():
        pytest.skip("no artifact bundle")

    app = app_test_cls.from_file("app/Home.py", default_timeout=120).run()
    text = " ".join(str(element.value) for element in app.info) + " ".join(
        str(element.value) for element in app.warning
    )
    assert "not a security product" in text.lower()


def test_evaluation_page_shows_the_baseline(app_test_cls):
    """Accuracy without the constant-predictor baseline beside it is not interpretable,
    so the page must not be able to lose it silently."""
    if not (ARTIFACTS / "manifest.json").exists():
        pytest.skip("no artifact bundle")

    app = app_test_cls.from_file("app/pages/3_Model_Evaluation.py", default_timeout=120).run()
    rendered = " ".join(str(e.value) for e in app.info) + " ".join(
        str(e.value) for e in app.caption
    )
    assert "0.9248" in rendered
