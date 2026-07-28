"""Measured results, with the leak distinction stated rather than buried."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.state import load_eval_predictions, page_header, require_bundle
from phishguard.evaluate import compute_metrics
from phishguard.schema import MAJORITY_BASELINE_ACCURACY, PHISHING_LABEL

st.set_page_config(
    page_title="Evaluation · PhishGuard", page_icon=":material/analytics:", layout="wide"
)

page_header("Model evaluation", "Measured on the held-out validation split")

bundle = require_bundle()
metrics = bundle.metrics
profiles = metrics.get("profiles", {})

st.info(
    f"**Accuracy is not the headline here.** A model that answers 'legitimate' to "
    f"everything scores **{MAJORITY_BASELINE_ACCURACY}** on this corpus while catching no "
    f"phishing at all. Every accuracy below is shown against that baseline, and phishing "
    f"recall — the share of phishing URLs actually caught — leads.",
    icon=":material/info:",
)

st.caption(
    f"Validation split: {metrics.get('n_validation', 0):,} rows. "
    f"The held-out file that ships with the dataset carries no labels, so there is no "
    f"test score to report and none is invented."
)

# --- as-corrected vs as-recorded --------------------------------------------


def metrics_table(models: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": record["name"],
                "Phishing recall": record["phishing_recall"],
                "Phishing precision": record["phishing_precision"],
                "Phishing F1": record["phishing_f1"],
                "PR-AUC": record["average_precision_phishing"],
                "Accuracy": record["accuracy"],
                "vs baseline": record["accuracy"] - MAJORITY_BASELINE_ACCURACY,
            }
            for key, record in models.items()
            if not key.startswith("_")
        ]
    )


corrected = profiles.get("corrected", {})
legacy = profiles.get("legacy", {})

st.subheader("As corrected — canonical")
st.caption(corrected.get("note", ""))
if corrected:
    st.dataframe(
        metrics_table(corrected["models"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Phishing recall": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0
            ),
            "Accuracy": st.column_config.NumberColumn(format="%.5f"),
            "vs baseline": st.column_config.NumberColumn(format="%+.5f"),
        },
    )

if legacy:
    st.subheader("As recorded — provenance only, and leaky")
    st.warning(
        legacy.get("note", ""),
        icon=":material/warning:",
    )
    st.dataframe(
        metrics_table(legacy["models"]),
        hide_index=True,
        width="stretch",
        column_config={"Accuracy": st.column_config.NumberColumn(format="%.5f")},
    )
    st.caption(
        "These two tables differ by one identified cause — the fit/transform fix — and "
        "not by a bundle of simultaneous changes. That is the only reason comparing them "
        "means anything."
    )

st.divider()

# --- confusion matrices -----------------------------------------------------

st.subheader("Confusion matrices")
st.caption("Rows are the true class, columns the predicted one. Phishing is class 0.")

if corrected:
    columns = st.columns(2)
    for i, record in enumerate(
        [v for k, v in corrected["models"].items() if not k.startswith("_")]
    ):
        matrix = np.array(record["confusion_matrix"])
        with columns[i % 2], st.container(border=True):
            st.markdown(f"**{record['name']}**")
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["true phishing", "true legitimate"],
                    columns=["pred phishing", "pred legitimate"],
                ),
                width="stretch",
            )
            missed = int(matrix[0][1])
            st.caption(
                f"{missed:,} phishing URLs classified as legitimate "
                f"({missed / max(matrix[0].sum(), 1):.1%} of all phishing)."
            )

st.divider()

# --- threshold sweep --------------------------------------------------------

st.subheader("Threshold sweep")
predictions = load_eval_predictions()

if predictions is None:
    st.info("No stored predictions in the bundle, so the sweep cannot be drawn.")
else:
    model_keys = [c[: -len("_score")] for c in predictions.columns if c.endswith("_score")]
    chosen = st.selectbox(
        "Model",
        model_keys,
        format_func=lambda k: bundle.models[k].name if k in bundle.models else k,
    )
    threshold = st.slider(
        "Phishing score threshold",
        0.0,
        1.0,
        0.5,
        0.01,
        help="Label a URL phishing when its score reaches this value.",
    )

    y_true = predictions["y_true"].to_numpy()
    scores = predictions[f"{chosen}_score"].to_numpy()
    y_pred = np.where(scores >= threshold, PHISHING_LABEL, 1 - PHISHING_LABEL)
    at_threshold = compute_metrics(y_true, y_pred)

    a, b, c, d = st.columns(4)
    a.metric("Phishing recall", f"{at_threshold.phishing_recall:.3f}")
    b.metric("Phishing precision", f"{at_threshold.phishing_precision:.3f}")
    c.metric("Phishing F1", f"{at_threshold.phishing_f1:.3f}")
    d.metric(
        "Accuracy",
        f"{at_threshold.accuracy:.4f}",
        delta=f"{at_threshold.accuracy - MAJORITY_BASELINE_ACCURACY:+.4f}",
    )

    sweep = []
    for t in np.linspace(0.0, 1.0, 51):
        m = compute_metrics(y_true, np.where(scores >= t, PHISHING_LABEL, 1 - PHISHING_LABEL))
        sweep.append(
            {"threshold": t, "phishing recall": m.phishing_recall, "phishing precision": m.phishing_precision}
        )
    st.line_chart(pd.DataFrame(sweep).set_index("threshold"))

    st.caption(
        "Lowering the threshold catches more phishing at the cost of more false alarms. "
        "Where to put it is a product decision about which error is worse, not a "
        "statistical one."
    )

st.divider()

# --- parity -----------------------------------------------------------------

st.subheader("From-scratch versus reference")
parity = corrected.get("models", {}).get("_parity", {}) if corrected else {}
if parity:
    columns = st.columns(len(parity))
    for column, (family, rate) in zip(columns, parity.items()):
        column.metric(
            f"{family.replace('_', ' ').title()} disagreement",
            f"{rate:.3%}",
            help="Share of validation rows where the two implementations differ.",
        )
    st.caption(
        "A from-scratch implementation is only evidence of understanding if somebody "
        "measured how far it lands from its reference. These are the two implementations "
        "of each algorithm disagreeing — not four independent opinions."
    )
