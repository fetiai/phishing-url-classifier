"""Single-URL prediction, with the feature inspector as a tab rather than a page.

A separate inspector page would re-run the whole pipeline and re-fetch the URL, which
would be slow, would burn a rate-limit token, and -- because the page may have changed
between the two requests -- could show an explanation that does not correspond to the
prediction above it. The tab reads the result already in session state.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state import (
    fetch_disabled_notice,
    page_header,
    require_bundle,
    scope_disclaimer,
    take_fetch_token,
)
from phishguard import schema
from phishguard.config import APP, FETCH
from phishguard.inference import classify_url

st.set_page_config(page_title="Single URL · PhishGuard", page_icon=":material/link:", layout="wide")

page_header("Single URL", "Fetch, extract, and classify one address")

bundle = require_bundle()
scope_disclaimer()
fetch_disabled = fetch_disabled_notice()

default_url = st.session_state.pop("pending_url", "")

with st.form("single_url"):
    url = st.text_input("URL to analyse", value=default_url, placeholder="https://example.com/login")
    submitted = st.form_submit_button("Analyse", type="primary")

if submitted and url.strip():
    allowed, remaining = take_fetch_token()
    if not allowed:
        st.error(
            f"Fetch limit reached for this session ({FETCH.rate_per_session} per "
            f"{FETCH.rate_window_s // 60} minutes). This keeps the service from being "
            "used as a scanning proxy."
        )
    else:
        with st.spinner("Fetching and classifying..."):
            st.session_state["last_result"] = classify_url(url.strip(), bundle)
        st.caption(f"{remaining} fetches remaining in this session.")

result = st.session_state.get("last_result")

if result is None:
    st.info("Enter a URL above to analyse it.")
    st.stop()

# --- fetch state ------------------------------------------------------------

state_render = {
    "scraped": (
        st.success,
        ":material/check_circle:",
        "Page fetched and parsed.",
    ),
    "challenged": (
        st.warning,
        ":material/robot:",
        "The server returned a bot challenge or interstitial rather than the page. "
        "The HTML describes the challenge, not the site, so its features are not used.",
    ),
    "unreachable": (
        st.error,
        ":material/link_off:",
        "The page could not be fetched. All 28 page-derived features are imputed.",
    ),
}
renderer, icon, message = state_render[result.fetch_state]
detail = f" ({result.fetch.reason})" if result.fetch.reason else ""
renderer(f"**{result.fetch_state.title()}**{detail} — {message}", icon=icon)

if result.fetch.final_url and result.fetch.final_url != result.url:
    st.caption(f"Followed {len(result.fetch.redirect_chain)} redirect(s) to `{result.fetch.final_url}`")

# --- coverage ---------------------------------------------------------------

coverage_col, _ = st.columns([2, 3])
with coverage_col:
    st.progress(
        result.coverage_ratio,
        text=(
            f"Page-feature coverage: {result.coverage_scraped}/{result.coverage_total} "
            f"({result.coverage_ratio:.0%})"
        ),
    )

# --- verdicts or abstention -------------------------------------------------

if result.abstained:
    st.warning("### Not enough evidence to answer", icon=":material/help:")
    st.markdown(result.abstain_reason)
    with st.expander("Show the predictions anyway (they are not trustworthy here)"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": v.name,
                        "Verdict": v.verdict,
                        "Phishing score": round(v.phishing_score, 4),
                    }
                    for v in result.verdicts
                ]
            ),
            hide_index=True,
            width="stretch",
        )
else:
    consensus = result.consensus
    if consensus == "phishing":
        st.error("### Both algorithms say: phishing", icon=":material/dangerous:")
    elif consensus == "legitimate":
        st.success("### Both algorithms say: legitimate", icon=":material/verified:")
    else:
        st.warning("### The two algorithms disagree", icon=":material/help:")

    columns = st.columns(2)
    for column, family, label in (
        (columns[0], "knn", "k-Nearest Neighbours"),
        (columns[1], "naive_bayes", "Gaussian Naive Bayes"),
    ):
        with column, st.container(border=True):
            st.markdown(f"**{label}**")
            for verdict in result.verdicts:
                if verdict.family != family:
                    continue
                suffix = " (from scratch)" if verdict.is_scratch else " (scikit-learn)"
                st.metric(
                    f"{'Phishing' if verdict.verdict == 'phishing' else 'Legitimate'}{suffix}",
                    f"{verdict.phishing_score:.3f}",
                    help="Probability of phishing. Class 0 is the positive class.",
                )

# --- tabs -------------------------------------------------------------------

inspector, provenance_tab, diagnostics = st.tabs(
    ["Feature inspector", "Provenance", "Diagnostics"]
)

reference = bundle.feature_reference

with inspector:
    st.caption(
        "Every feature, its value after preprocessing, and how far it sits from each "
        "class's training mean. Contribution is the per-feature log-likelihood ratio from "
        "the from-scratch Naive Bayes -- positive favours phishing."
    )

    nb = bundle.nb_scratch
    rows = []
    for i, name in enumerate(schema.FEATURE_ORDER):
        value = result.features[name]
        stats = reference.get(name, {})
        contribution = 0.0
        if nb.theta_ is not None and nb.sigma_ is not None:
            import numpy as np

            log_p = [
                -0.5 * (np.log(2 * np.pi) + 2 * np.log(nb.sigma_[k][i]))
                - 0.5 * ((value - nb.theta_[k][i]) / nb.sigma_[k][i]) ** 2
                for k in (0, 1)
            ]
            contribution = float(log_p[0] - log_p[1])

        rows.append(
            {
                "Feature": name,
                "Source": result.provenance.get(name, "?"),
                "Value": round(value, 4),
                "Median (train)": round(stats.get("median", float("nan")), 4) if stats else None,
                "z vs phishing": (
                    round((value - stats["mean_phishing"]) / (stats["std_phishing"] or 1), 2)
                    if stats
                    else None
                ),
                "z vs legitimate": (
                    round((value - stats["mean_legitimate"]) / (stats["std_legitimate"] or 1), 2)
                    if stats
                    else None
                ),
                "Contribution": round(contribution, 3),
            }
        )

    frame = pd.DataFrame(rows).sort_values("Contribution", key=abs, ascending=False)
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Contribution": st.column_config.NumberColumn(
                help="Positive favours phishing, negative favours legitimate."
            )
        },
    )

with provenance_tab:
    counts = pd.Series(
        [result.provenance.get(n, "?") for n in schema.FEATURE_ORDER]
    ).value_counts()

    cols = st.columns(4)
    labels = {
        schema.PROVENANCE_URL: ("From the URL", "Computed from the address string alone."),
        schema.PROVENANCE_SCRAPED: ("Scraped", "Read from the fetched page."),
        schema.PROVENANCE_IMPUTED: (
            "Imputed",
            "Filled from training statistics because it could not be measured.",
        ),
        schema.PROVENANCE_DEMOTED: (
            "Demoted",
            "Failed its agreement gate; never extracted, always imputed.",
        ),
    }
    for column, (key, (label, help_text)) in zip(cols, labels.items()):
        column.metric(label, int(counts.get(key, 0)), help=help_text)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature": name,
                    "Source": result.provenance.get(name, "?"),
                    "Raw value": result.raw_values.get(name),
                    "After preprocessing": round(result.features[name], 4),
                }
                for name in schema.FEATURE_ORDER
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    if bundle.demoted:
        st.caption(f"Demoted features: {', '.join(sorted(bundle.demoted))}")

with diagnostics:
    st.markdown(
        f"""
| | |
|---|---|
| Requested URL | `{result.url}` |
| Final URL | `{result.fetch.final_url or "—"}` |
| Outcome | `{result.fetch.outcome.value}` |
| HTTP status | `{result.fetch.status_code or "—"}` |
| Content type | `{result.fetch.content_type or "—"}` |
| Bytes read | {result.fetch.bytes_read:,} |
| Elapsed | {result.fetch.elapsed_s:.2f}s |
| Redirects | {len(result.fetch.redirect_chain)} |
| Reference scope | `{APP.ref_scope}` |
"""
    )
    if result.cascade_fallbacks:
        st.caption(
            "Imputation steps that fell back to a global mode because this row's "
            "combination of values was never seen in training:"
        )
        st.json(result.cascade_fallbacks)

st.divider()
st.caption(
    "Fetching a URL causes this server to make an HTTP request to it. Requests are "
    "rate-limited, capped in size and time, follow at most a few redirects, and can only "
    "reach public internet addresses. No page content is stored."
)
