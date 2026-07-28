"""Landing page."""

from __future__ import annotations

import streamlit as st

from app.state import (
    fetch_disabled_notice,
    get_bundle,
    page_header,
    scope_disclaimer,
)
from phishguard.config import APP, FETCH
from phishguard.schema import MAJORITY_BASELINE_ACCURACY

st.set_page_config(
    page_title="PhishGuard",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header(
    "PhishGuard",
    "KNN and Gaussian Naive Bayes phishing URL classification, from scratch and from scikit-learn",
)

scope_disclaimer()
fetch_disabled_notice()

bundle = get_bundle()

st.subheader("Check a URL")
with st.form("quick_check"):
    url = st.text_input(
        "URL",
        placeholder="https://example.com/login",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Analyse", type="primary")

if submitted and url.strip():
    # Hand off rather than predicting here: page 1 owns the fetch, the coverage meter and
    # the provenance table, and duplicating that logic would let the two drift.
    st.session_state["pending_url"] = url.strip()
    st.switch_page("pages/1_Single_URL.py")

st.divider()

# --- the models -------------------------------------------------------------

st.subheader("Two algorithms, four implementations")
st.markdown(
    "Each algorithm is implemented twice: once from scratch, as the assignment required, "
    "and once with scikit-learn. **The pair is not two opinions.** The from-scratch model "
    "exists to be checked against its reference, and the number that matters is how often "
    "they disagree -- shown below. Averaging a model with its own reference would be "
    "double-counting, which is why there is no aggregate vote anywhere in this interface."
)

if bundle is None:
    st.warning(
        "No artifact bundle is loaded, so no measured numbers can be shown. Build one "
        "with `python -m phishguard.train --profile corrected`."
    )
else:
    corrected = bundle.metrics.get("profiles", {}).get("corrected", {}).get("models", {})
    parity = corrected.get("_parity", {})

    left, right = st.columns(2)
    for column, family, label in (
        (left, "knn", "k-Nearest Neighbours"),
        (right, "naive_bayes", "Gaussian Naive Bayes"),
    ):
        with column, st.container(border=True):
            st.markdown(f"### {label}")
            delta = parity.get(family)
            if delta is not None:
                st.caption(
                    f"The two implementations disagree on **{delta:.3%}** of validation rows."
                )
            for key, record in corrected.items():
                if key.startswith("_") or record.get("family") != family:
                    continue
                tag = " · educational reimplementation" if record["is_scratch"] else ""
                st.markdown(f"**{record['name']}**{tag}")
                a, b = st.columns(2)
                a.metric(
                    "Phishing recall",
                    f"{record['phishing_recall']:.3f}",
                    help="Share of phishing URLs actually caught. This is the headline.",
                )
                b.metric(
                    "Accuracy",
                    f"{record['accuracy']:.4f}",
                    delta=f"{record['accuracy'] - MAJORITY_BASELINE_ACCURACY:+.4f} vs baseline",
                    help=(
                        f"A constant 'legitimate' predictor scores "
                        f"{MAJORITY_BASELINE_ACCURACY} on this corpus."
                    ),
                )

    st.caption(
        f"Read every accuracy against the constant-predictor baseline of "
        f"**{MAJORITY_BASELINE_ACCURACY}**. Answering 'legitimate' to everything scores "
        "that while catching no phishing at all, so accuracy alone cannot tell a working "
        "detector from a constant. Phishing recall can."
    )

st.divider()

# --- how it works -----------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("How a URL is judged")
    st.markdown(
        """
1. The URL is validated and its address vetted, then the page is fetched under an
   SSRF guard with transfer caps and a redirect limit.
2. **21** features come from the URL string itself. **28** describe the page and need
   that fetch to have succeeded.
3. Anything unavailable is imputed from statistics learned at training time -- never
   recomputed from the input.
4. All four models score the result, and the per-feature provenance is shown.
"""
    )

with right:
    st.subheader("When it refuses to answer")
    st.markdown(
        f"""
Below **{APP.coverage_min_ratio:.0%}** page-feature coverage, no verdict is given.

Imputation fills missing features from the training distribution, which is
**{MAJORITY_BASELINE_ACCURACY:.2%} legitimate**. A failed fetch therefore does not produce
a neutral prediction -- it produces one biased toward *legitimate*, which is the wrong
direction for a phishing detector.

That is not a rare edge case. Phishing domains are short-lived, so a phishing URL from a
2023-24 crawl is usually dead today. "Could not reach it" correlates with phishing in
reality and with legitimate in the imputed features.
"""
    )

st.divider()

with st.expander("Configuration in effect"):
    st.markdown(
        f"""
| Setting | Value |
|---|---|
| Page fetching | `{FETCH.enabled}` |
| robots.txt fetched for the Robots feature | `{FETCH.robots}` |
| Reference-count scope | `{APP.ref_scope}` |
| Coverage threshold for a verdict | `{APP.coverage_min_ratio:.0%}` |
| Fetch budget | {FETCH.total_timeout_s:.0f}s wall clock, {FETCH.max_redirects} redirects, {FETCH.max_bytes // 1024} KiB |
| Per-session fetch limit | {FETCH.rate_per_session} per {FETCH.rate_window_s // 60} minutes |
"""
    )
    if bundle is not None:
        st.caption(
            f"Bundle {bundle.manifest.get('git_sha', 'unknown')[:12]} · "
            f"trained on {bundle.stats.n_train_rows:,} rows · "
            f"{len(bundle.stats.feature_order)} features"
        )
