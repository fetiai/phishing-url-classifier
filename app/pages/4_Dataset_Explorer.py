"""Dataset explorer, served from a stratified sample rather than the raw CSV."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state import load_dataset_profile, load_dataset_sample, page_header, require_bundle
from phishguard import schema

st.set_page_config(
    page_title="Dataset · PhishGuard", page_icon=":material/dataset:", layout="wide"
)

page_header("Dataset explorer", "PhishUSIIL phishing URL dataset, crawled 2023-24")

require_bundle()

sample = load_dataset_sample()
profile = load_dataset_profile()

if sample is None:
    st.info("No dataset sample in the bundle. Rebuild it with `python -m phishguard.train`.")
    st.stop()

st.caption(
    f"Charts are drawn from a stratified {len(sample):,}-row sample stored in the bundle, "
    f"so the application never opens the 29 MB training file. Counts below are for the "
    f"full {profile.get('n_rows', 0):,} rows."
)

# --- class balance ----------------------------------------------------------

st.subheader("Class balance")

counts = profile.get("class_counts", {})
balance = profile.get("class_balance", {})

a, b, c = st.columns(3)
a.metric("Legitimate (class 1)", f"{counts.get('1', 0):,}", f"{balance.get('1', 0):.2%}")
b.metric("Phishing (class 0)", f"{counts.get('0', 0):,}", f"{balance.get('0', 0):.2%}")
c.metric(
    "Constant-predictor accuracy",
    f"{schema.MAJORITY_BASELINE_ACCURACY}",
    help="What 'always legitimate' scores. Every accuracy figure must be read against it.",
)

st.warning(
    "The imbalance is the reason phishing recall leads every report in this application, "
    "and the reason a failed page fetch biases predictions toward *legitimate*: imputation "
    "fills from the majority class.",
    icon=":material/balance:",
)

st.divider()

# --- missingness ------------------------------------------------------------

st.subheader("Missing values")
st.markdown(
    "The missing values in this dataset are **deliberate**. Between 30% and 50% of each "
    "column was removed as an exercise — including 31% of `URL` and 50% of `Domain`, the "
    "two columns most other features are derived from. This is why the imputation cascade "
    "exists and why it has to run identically at training and serving time."
)

nan_rate = profile.get("nan_rate", {})
if nan_rate:
    frame = (
        pd.DataFrame({"column": list(nan_rate), "missing": list(nan_rate.values())})
        .sort_values("missing", ascending=False)
        .set_index("column")
    )
    st.bar_chart(frame, height=420)

st.divider()

# --- distributions ----------------------------------------------------------

st.subheader("Feature distributions by class")

feature = st.selectbox(
    "Feature",
    [f for f in schema.FEATURE_ORDER if f in sample.columns],
    index=0,
)

if feature:
    frame = sample[[feature, "label"]].dropna()
    if frame.empty:
        st.info("Every value of this feature is missing in the sample.")
    else:
        phishing = frame[frame["label"] == schema.PHISHING_LABEL][feature]
        legitimate = frame[frame["label"] == schema.LEGITIMATE_LABEL][feature]

        a, b = st.columns(2)
        a.metric("Mean, phishing", f"{phishing.mean():.4g}" if len(phishing) else "—")
        b.metric("Mean, legitimate", f"{legitimate.mean():.4g}" if len(legitimate) else "—")

        if pd.api.types.is_numeric_dtype(frame[feature]):
            binned = pd.cut(frame[feature], bins=30)
            chart = (
                frame.assign(bin=binned.apply(lambda x: x.mid if pd.notna(x) else None))
                .groupby(["bin", "label"], observed=True)
                .size()
                .unstack(fill_value=0)
            )
            chart.columns = [
                "phishing" if c == schema.PHISHING_LABEL else "legitimate" for c in chart.columns
            ]
            st.bar_chart(chart, height=360)
        else:
            st.dataframe(frame[feature].value_counts().head(30), width="stretch")

st.divider()

with st.expander("Feature groups"):
    st.markdown(
        f"""
| Group | Count | Meaning |
|---|---|---|
| From the URL | {len(schema.URL_ONLY_FEATURES)} | Available even when the page cannot be fetched |
| Title-dependent | {len(schema.TITLE_HYBRID_FEATURES)} | Have URL-side code but need the page title |
| From the page | {len(schema.HTML_FEATURES)} | Require a successful fetch |
| **Total** | **{len(schema.FEATURE_ORDER)}** | |

The dataset's column names contain three misspellings — `NoOfDegitsInURL`,
`DegitRatioInURL`, `SpacialCharRatioInURL` — which are preserved exactly, because
correcting them would break every lookup against the source data.
"""
    )
