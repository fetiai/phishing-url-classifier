"""Batch classification.

Feature-CSV mode is the default and fetches nothing. URL-list mode is opt-in and hard
capped, because a batch fetcher accepting an arbitrary list is an open proxy with extra
steps -- one request turning into hundreds of outbound connections chosen by the caller.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.state import fetch_disabled_notice, page_header, require_bundle, take_fetch_token
from phishguard import schema
from phishguard.config import APP, FETCH
from phishguard.inference import classify_frame, classify_url

st.set_page_config(page_title="Batch · PhishGuard", page_icon=":material/table:", layout="wide")

page_header("Batch classification", "Score many rows at once")

bundle = require_bundle()
fetch_disabled = fetch_disabled_notice()

mode = st.radio(
    "Input",
    ["Feature CSV (no fetching)", "URL list (fetches every URL)"],
    horizontal=True,
    help=(
        "Feature CSV scores rows that already carry the dataset's columns. URL list "
        "fetches each address, which is slower and strictly rate-limited."
    ),
)

# --- feature CSV ------------------------------------------------------------

if mode.startswith("Feature"):
    st.markdown(
        f"Upload a CSV carrying the {len(schema.FEATURE_ORDER)} feature columns. "
        f"Missing values are fine — they are imputed exactly as in training. "
        f"Up to {APP.batch_max_rows:,} rows."
    )

    with st.expander("Required columns"):
        st.code(", ".join(schema.FEATURE_ORDER), language=None)

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        frame = pd.read_csv(uploaded, low_memory=False)

        if len(frame) > APP.batch_max_rows:
            st.warning(
                f"File has {len(frame):,} rows; scoring the first {APP.batch_max_rows:,}."
            )
            frame = frame.head(APP.batch_max_rows)

        missing = set(schema.FEATURE_ORDER) - set(frame.columns)
        if missing:
            st.error(
                f"{len(missing)} required column(s) are absent: {', '.join(sorted(missing))}"
            )
            st.stop()

        with st.spinner(f"Scoring {len(frame):,} rows..."):
            predictions, _ = classify_frame(frame, bundle)

        output = pd.concat([frame.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)

        st.success(f"Scored {len(output):,} rows.")

        summary = pd.DataFrame(
            [
                {
                    "Model": bundle.models[key].name,
                    "Flagged as phishing": int(
                        (predictions[f"{key}_label"] == schema.PHISHING_LABEL).sum()
                    ),
                    "Share": f"{(predictions[f'{key}_label'] == schema.PHISHING_LABEL).mean():.2%}",
                }
                for key in bundle.models
            ]
        )
        st.dataframe(summary, hide_index=True, width="stretch")

        st.dataframe(output.head(200), width="stretch")
        if len(output) > 200:
            st.caption(f"Showing the first 200 of {len(output):,} rows. Download for the rest.")

        buffer = io.StringIO()
        output.to_csv(buffer, index=False)
        st.download_button(
            "Download results",
            buffer.getvalue(),
            file_name="phishguard_predictions.csv",
            mime="text/csv",
            type="primary",
        )

# --- URL list ---------------------------------------------------------------

else:
    if fetch_disabled:
        st.stop()

    st.markdown(
        f"One URL per line, at most **{FETCH.batch_max_urls}**. Each is fetched under the "
        "same guard and the same per-session limit as a single lookup."
    )

    text = st.text_area("URLs", height=180, placeholder="https://example.com\nhttps://example.org")
    go = st.button("Analyse all", type="primary")

    if go and text.strip():
        urls = [line.strip() for line in text.splitlines() if line.strip()]

        if len(urls) > FETCH.batch_max_urls:
            st.warning(
                f"{len(urls)} URLs submitted; analysing the first {FETCH.batch_max_urls}. "
                "The cap is what keeps this from being usable as a scanning proxy."
            )
            urls = urls[: FETCH.batch_max_urls]

        rows = []
        progress = st.progress(0.0, text="Starting...")
        for i, url in enumerate(urls, start=1):
            allowed, _ = take_fetch_token()
            if not allowed:
                st.error(
                    f"Session fetch limit reached after {i - 1} URL(s). "
                    f"Remaining URLs were not fetched."
                )
                break

            progress.progress(i / len(urls), text=f"{i}/{len(urls)}  {url[:60]}")
            result = classify_url(url, bundle)

            row = {
                "URL": url,
                "Fetch": result.fetch_state,
                "Coverage": f"{result.coverage_ratio:.0%}",
                "Verdict": "insufficient evidence" if result.abstained else (result.consensus or "split"),
            }
            for verdict in result.verdicts:
                row[verdict.key] = round(verdict.phishing_score, 3)
            rows.append(row)

        progress.empty()

        if rows:
            output = pd.DataFrame(rows)
            st.dataframe(output, hide_index=True, width="stretch")

            abstained = int((output["Verdict"] == "insufficient evidence").sum())
            if abstained:
                st.caption(
                    f"{abstained} of {len(output)} URLs could not be judged: too little of "
                    "the page could be read, and imputing the rest would bias the answer "
                    "toward 'legitimate'."
                )

            buffer = io.StringIO()
            output.to_csv(buffer, index=False)
            st.download_button(
                "Download results",
                buffer.getvalue(),
                file_name="phishguard_url_predictions.csv",
                mime="text/csv",
            )
