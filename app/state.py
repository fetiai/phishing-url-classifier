"""Shared application state and caching.

CACHE POLICY
============

``@st.cache_resource`` for the artifact bundle: one instance per process, shared across
every session, treated as immutable. The models and the 10,000-row KNN reference set are
the largest things in memory, and giving each visitor a copy would multiply resident
memory by the session count on a 2 GB box.

``@st.cache_data`` for anything derived and copyable -- extraction results, chart frames.
Each caller gets its own copy, so a page mutating a DataFrame cannot corrupt another
session's view of it.

``st.session_state`` holds only small values: the last URL, the last result, the rate
limiter's token bucket. Never a model, never a large frame.

Note the boundary: this package imports streamlit, and the library package never does. The
library also never has to know it is being served from a UI.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from phishguard.artifacts import ArtifactBundle, BundleError, load_bundle
from phishguard.config import APP, FETCH

RATE_BUCKET_KEY = "_fetch_tokens"
RATE_STAMP_KEY = "_fetch_window_start"


@st.cache_resource(show_spinner="Loading models...")
def get_bundle() -> ArtifactBundle | None:
    """Load and verify the artifact bundle once per process.

    Returns None rather than raising when there is no bundle, so the pages can render a
    useful explanation instead of a stack trace. Every page checks.
    """
    try:
        return load_bundle(APP.artifacts_dir, verify=True)
    except (BundleError, FileNotFoundError, OSError):
        return None


def require_bundle() -> ArtifactBundle:
    """Render the no-bundle explanation and stop, or return the bundle."""
    bundle = get_bundle()
    if bundle is None:
        st.error("No artifact bundle is loaded.")
        st.markdown(
            "This application only ever *loads* models; it never trains at request time. "
            "Build a bundle first:\n\n"
            "```bash\npython -m phishguard.train --profile corrected\n```\n\n"
            f"It is expected at `{APP.artifacts_dir}`."
        )
        st.stop()
    return bundle


@st.cache_data(ttl=900, show_spinner=False)
def load_dataset_sample() -> pd.DataFrame | None:
    path = APP.artifacts_dir / "dataset_sample.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=900, show_spinner=False)
def load_eval_predictions() -> pd.DataFrame | None:
    path = APP.artifacts_dir / "eval_predictions.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(ttl=900, show_spinner=False)
def load_dataset_profile() -> dict[str, Any]:
    path = APP.artifacts_dir / "dataset_profile.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def take_fetch_token() -> tuple[bool, int]:
    """Per-session token bucket. Returns (allowed, remaining).

    Rate limiting is per session and the reverse proxy limits per IP; neither alone is
    enough. This one keeps a single visitor from using the service as a scanning proxy.
    """
    now = time.monotonic()
    start = st.session_state.get(RATE_STAMP_KEY)

    if start is None or now - start > FETCH.rate_window_s:
        st.session_state[RATE_STAMP_KEY] = now
        st.session_state[RATE_BUCKET_KEY] = FETCH.rate_per_session

    tokens = st.session_state.get(RATE_BUCKET_KEY, FETCH.rate_per_session)
    if tokens <= 0:
        return False, 0

    st.session_state[RATE_BUCKET_KEY] = tokens - 1
    return True, tokens - 1


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def scope_disclaimer() -> None:
    """The framing every page carries.

    Not boilerplate: the system is trained on a static 2023-24 crawl, has no threat
    intelligence and no blocklist, and knows nothing about any campaign newer than its
    training data. Presenting it as a security tool would be the misrepresentation the
    whole project exists to avoid.
    """
    st.info(
        "**This is a coursework reimplementation, not a security product.** It is trained "
        "on a static 2023-24 dataset, has no threat intelligence and no blocklist, and "
        "knows nothing about any campaign newer than its training data. Do not use it to "
        "decide whether a link is safe.",
        icon=":material/warning:",
    )


def fetch_disabled_notice() -> bool:
    """Render the degraded state when fetching is off. True when disabled."""
    if FETCH.enabled:
        return False
    st.warning(
        "**Page fetching is currently disabled.** Only the 21 URL-derived features are "
        "available; the 28 that describe the page will be imputed. Predictions are "
        "correspondingly weaker, and the coverage meter will show why.",
        icon=":material/cloud_off:",
    )
    return True
