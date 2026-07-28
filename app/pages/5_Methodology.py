"""The page that makes the project honest.

It carries the defect ledger, the provenance of every ported function, and the measured
extraction agreement. If any page here is uncomfortable to read, it should be this one.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state import page_header, require_bundle
from phishguard import schema
from phishguard.config import APP

st.set_page_config(
    page_title="Methodology · PhishGuard", page_icon=":material/science:", layout="wide"
)

page_header("Methodology", "What was inherited, what was wrong with it, and what was changed")

bundle = require_bundle()

st.markdown(
    """
This system was rebuilt from a coursework notebook that saved no model of any kind —
everything it computed died with its kernel. Rebuilding it meant reading its code closely,
and that surfaced defects serious enough that its reported numbers cannot be taken at face
value. They are published here rather than quietly corrected.
"""
)

# --- defect ledger ----------------------------------------------------------

st.subheader("Defect ledger")

DEFECTS = [
    (
        "Validation data scaled by its own statistics",
        "Each split was standardised with a scaler fitted on that split. The scaling "
        "behind every reported number therefore does not exist at serving time and "
        "cannot be reconstructed — there is no batch to take a mean over when someone "
        "submits one URL.",
        "A real fit/transform split. Statistics are learned once and persisted as plain "
        "numbers; the serving path can only apply them.",
    ),
    (
        "fit() did nothing",
        "The preprocessing object's fit() was `return self`. Every statistic — character "
        "frequencies, per-TLD means, imputation values, clip bounds — was recomputed "
        "inside transform() from whatever batch arrived. One row in meant statistics "
        "derived from that single row.",
        "transform() computes no statistic. A test transforms rows individually and as a "
        "batch and requires bitwise-identical output.",
    ),
    (
        "Naive Bayes multiplied ~49 densities",
        "The product underflows to zero on rows far from both class means. Once both "
        "classes reach zero, argmax returns class 0 unconditionally — the model reports "
        "phishing because the arithmetic collapsed, not because of the evidence.",
        "Log space. Same decision rule, no floor to hit.",
    ),
    (
        "KNN looped over rows in Python",
        "5m43s to classify the validation split, which makes interactive use impossible.",
        "Vectorised, with the selected neighbours re-sorted by distance so tie-breaking "
        "still matches the original.",
    ),
    (
        "The Naive Bayes drop list was discarded",
        "Columns were dropped from the model and the list was never recorded, so nothing "
        "downstream could say which features a served model had been fitted on.",
        "Persisted in the bundle with a reason per column.",
    ),
    (
        "Reference set and labels were misaligned by luck",
        "A 10,000-row reference set was paired with a full-length label vector. It "
        "happened to work only because row order aligned.",
        "The reference set and its labels are sliced together.",
    ),
    (
        "Test predictions ran on raw data",
        "The held-out file was scored without preprocessing.",
        "One code path; there is no way to reach a model without passing through "
        "transform.",
    ),
    (
        "Helpers closed over notebook globals",
        "The clipping and validation helpers read module-level column lists, which is why "
        "the scaler silently touched only 30 of the 49 columns.",
        "Column lists are frozen in the schema, asserted at import, and passed explicitly.",
    ),
    (
        "Dead code in transform",
        "Two statistics were computed at the top of transform and never read.",
        "Deleted; the same names now exist as fitted inputs that are genuinely consumed.",
    ),
    (
        "Metric cells referenced the wrong variables",
        "The Naive Bayes evaluation block printed KNN results.",
        "Metrics are computed once, in one place, and written to the bundle.",
    ),
    (
        "Data loaded from remote links",
        "Training read from a Google Drive URL and a personal file host — mutable, "
        "unversioned inputs unavailable to anyone re-running it.",
        "Training reads the committed CSV and nothing else.",
    ),
    (
        "SMOTE computed and never applied",
        "Resampling was demonstrated and then not used, while the headline accuracy was "
        "presented as though the imbalance had been addressed.",
        "Not applied. The imbalance is reported instead, and phishing recall leads.",
    ),
]

for i, (title, problem, fix) in enumerate(DEFECTS, start=1):
    with st.expander(f"{i}. {title}"):
        st.markdown(f"**What was wrong.** {problem}")
        st.markdown(f"**What changed.** {fix}")

st.divider()

# --- extraction agreement ---------------------------------------------------

st.subheader("Are the page features extracted correctly?")

st.warning(
    "**The honest answer is that this is the project's largest unknown.** The dataset's "
    "own extraction code is not published, and neither is the crawled HTML its rows were "
    "computed from. The 25 page-feature rules in this system are reconstructions inferred "
    "from the feature names. They may not match how the training data was produced, and "
    "reading them cannot establish whether they do.",
    icon=":material/warning:",
)

agreement = bundle.agreement or {}

if agreement.get("status") == "not_run":
    st.info(
        "The agreement harness has not been run against this bundle. No feature has been "
        "demoted — and none has been shown to be trustworthy either. Until it runs, treat "
        "every page-derived feature as unverified."
    )
elif agreement.get("features"):
    rows = [
        {
            "Feature": name,
            "Type": record.get("type"),
            "Metric": record.get("metric"),
            "Value": record.get("value"),
            "Gate": record.get("gate"),
            "Result": "pass" if record.get("passed") else "DEMOTED",
            "Sample": record.get("n"),
        }
        for name, record in agreement["features"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.caption(
        agreement.get("caveat")
        or (
            "Agreement is judged on legitimate URLs. On phishing URLs the measurement "
            "mostly reflects link rot rather than extractor correctness: those domains "
            "are largely dead, so what gets fetched is a registrar page rather than the "
            "page the dataset saw."
        )
    )

if bundle.demoted:
    st.error(
        f"**Demoted features ({len(bundle.demoted)}):** {', '.join(sorted(bundle.demoted))}. "
        "These are never extracted and are always imputed. They are labelled as such in "
        "the feature inspector.",
        icon=":material/block:",
    )

st.divider()

# --- port provenance --------------------------------------------------------

st.subheader("What was ported, and what is new")

st.dataframe(
    pd.DataFrame(
        [
            {
                "Component": "26 URL feature functions",
                "Status": "Ported verbatim",
                "Checked by": "Bit-identical comparison against a frozen copy of the originals",
            },
            {
                "Component": "Imputation and scaling",
                "Status": "Rewritten as fit/transform",
                "Checked by": "Row/batch invariance, bitwise",
            },
            {
                "Component": "25 page feature extractors",
                "Status": "New; no prior art exists",
                "Checked by": "Agreement gate against the dataset's own values",
            },
            {
                "Component": "KNN and Naive Bayes, from scratch",
                "Status": "Rewritten (vectorised, log space)",
                "Checked by": "Equivalence against the naive forms and against scikit-learn",
            },
            {
                "Component": "Guarded fetch client",
                "Status": "New",
                "Checked by": "SSRF rejection table, including redirect and rebinding cases",
            },
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.markdown(
    "The 26 ported functions preserve their original behaviour exactly, **including their "
    "quirks**. One obfuscation rule reverses the URL and tests a pattern that is "
    "unchanged by reversal, so it fires on any bare domain — almost certainly not what was "
    "intended. It is preserved anyway, because those functions define what the training "
    "data means, and changing one would silently shift the distribution the models were "
    "fitted on without any metric revealing it."
)

st.divider()

# --- known limitations ------------------------------------------------------

st.subheader("Known limitations")

st.markdown(
    f"""
**The distance metric is not principled.** KNN runs over 30 standardised dimensions and
19 raw 0/1 indicators, because the original's scaler only ever reached the numeric columns
— a consequence of variable scoping rather than a decision. It is kept deliberately, since
changing it would invalidate every comparison against the recorded numbers, but the
resulting weighting is an accident and Euclidean distance over the mixture has no
particular justification.

**There is no test score.** The held-out file that ships with the dataset carries no
labels. Only validation numbers exist, and none are presented as test results.

**Coverage below {APP.coverage_min_ratio:.0%} means no answer.** Imputation fills from a
corpus that is {schema.MAJORITY_BASELINE_ACCURACY:.2%} legitimate, so a failed fetch
biases toward *legitimate* — the wrong direction for a phishing detector. Since phishing
domains are short-lived, failed fetches are the common case for exactly the URLs this
system is meant to catch.

**The models know nothing after 2024.** No threat intelligence, no blocklist, no knowledge
of any campaign newer than the crawl.
"""
)
