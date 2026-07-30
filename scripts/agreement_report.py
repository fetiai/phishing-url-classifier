"""Measure whether the page-feature extractors match the dataset's own values.

This is the answer to the project's largest open question. The 25 page-feature rules are
reconstructions inferred from feature names -- the dataset's extraction code is not
published and neither is the HTML corpus its rows were computed from. Reading the rules
cannot establish whether they match. Measuring can.

WHAT THE NUMBERS DO AND DO NOT MEAN
===================================

Agreement is judged on **legitimate** URLs only. On phishing URLs the comparison mostly
measures link rot: the dataset was crawled in 2023-24 and phishing domains are
short-lived, so what gets fetched today is usually a registrar parking page rather than
the page the dataset saw. A low score there says the domain died, not that the extractor
is wrong. Both splits are reported; only the legitimate one gates.

A feature that misses its gate is **demoted**: its extractor returns None permanently, its
value is always imputed, and the interface labels it as not reliably extractable. Demotion
is deliberately blunt. A feature we cannot show to be right is worse than a missing one,
because it enters the model as evidence.

Run offline against the committed fixtures (no network) or live (marked, nightly).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from phishguard import schema
from phishguard.config import RefScope
from phishguard.features.html_features import extract_html_features, parse_html

FIXTURE_DIR = Path("tests/fixtures/html")
EXPECTED = Path("tests/fixtures/expected.parquet")

#: Binary features must agree this often, and count features must correlate at least this
#: well, on legitimate URLs.
BINARY_AGREEMENT_GATE = 0.85
COUNT_SPEARMAN_GATE = 0.70

BINARY_FEATURES = tuple(f for f in schema.HTML_FEATURES if f in schema.CATEGORICAL_COLUMNS_FILTERED)
COUNT_FEATURES = tuple(f for f in schema.HTML_FEATURES if f not in BINARY_FEATURES)


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement corrected for what chance alone would produce.

    Reported alongside the raw rate because a feature that is 95% zeros in both the
    dataset and our extraction scores 0.95 agreement by predicting nothing at all.
    """
    if len(a) == 0:
        return float("nan")
    observed = float((a == b).mean())
    expected = sum(
        float((a == value).mean()) * float((b == value).mean())
        for value in np.union1d(np.unique(a), np.unique(b))
    )
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def score_feature(
    name: str, expected: pd.Series, actual: pd.Series
) -> dict[str, Any]:
    mask = expected.notna() & actual.notna()
    exp = expected[mask].to_numpy(dtype=np.float64)
    act = actual[mask].to_numpy(dtype=np.float64)

    if len(exp) < 5:
        return {
            "type": "binary" if name in BINARY_FEATURES else "count",
            "metric": "insufficient_data",
            "value": None,
            "gate": None,
            "passed": False,
            "n": int(len(exp)),
            "note": "too few comparable rows to judge; demoted by default",
        }

    if name in BINARY_FEATURES:
        rate = float((exp == act).mean())
        return {
            "type": "binary",
            "metric": "agreement_rate",
            "value": round(rate, 4),
            "cohens_kappa": round(cohens_kappa(exp, act), 4),
            "gate": BINARY_AGREEMENT_GATE,
            "passed": rate >= BINARY_AGREEMENT_GATE,
            "n": int(len(exp)),
        }

    if np.std(exp) == 0 or np.std(act) == 0:
        rho = float("nan")
    else:
        rho = float(scipy_stats.spearmanr(exp, act).statistic)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(exp != 0, act / exp, np.nan)

    return {
        "type": "count",
        "metric": "spearman_rho",
        "value": None if np.isnan(rho) else round(rho, 4),
        "median_abs_error": round(float(np.median(np.abs(act - exp))), 4),
        "median_ratio": (
            None if np.all(np.isnan(ratios)) else round(float(np.nanmedian(ratios)), 4)
        ),
        "gate": COUNT_SPEARMAN_GATE,
        "passed": bool(not np.isnan(rho) and rho >= COUNT_SPEARMAN_GATE),
        "n": int(len(exp)),
    }


def extract_from_fixtures(expected: pd.DataFrame, ref_scope: RefScope) -> pd.DataFrame:
    rows = []
    for record in expected.itertuples():
        path = FIXTURE_DIR / f"{record.sha1}.html.gz"
        if not path.exists():
            continue
        html = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        features = extract_html_features(
            html,
            parse_html(html),
            record.final_url,
            [],
            ref_scope=ref_scope,
        )
        rows.append({"sha1": record.sha1, **{k: v for k, v in features.items() if k != "Title"}})
    return pd.DataFrame(rows)


def run(ref_scope: RefScope) -> dict[str, Any]:
    if not EXPECTED.exists():
        raise SystemExit(
            f"no fixture corpus at {EXPECTED}. Capture one first:\n"
            f"    python scripts/capture_fixtures.py"
        )

    expected = pd.read_parquet(EXPECTED)
    actual = extract_from_fixtures(expected, ref_scope)

    if actual.empty:
        raise SystemExit("no fixtures could be read; the corpus appears to be empty")

    merged = expected.merge(actual, on="sha1", suffixes=("_expected", "_actual"))
    legitimate = merged[merged["label"] == schema.LEGITIMATE_LABEL]
    phishing = merged[merged["label"] == schema.PHISHING_LABEL]

    features: dict[str, Any] = {}
    for name in schema.HTML_FEATURES:
        exp_col, act_col = f"{name}_expected", f"{name}_actual"
        if exp_col not in merged or act_col not in merged:
            continue

        result = score_feature(name, legitimate[exp_col], legitimate[act_col])
        if len(phishing):
            result["phishing_split"] = score_feature(
                name, phishing[exp_col], phishing[act_col]
            )
        features[name] = result

    demoted = sorted(name for name, record in features.items() if not record["passed"])

    return {
        "status": "measured",
        "ref_scope": ref_scope,
        "n_fixtures": int(len(merged)),
        "n_legitimate": int(len(legitimate)),
        "n_phishing": int(len(phishing)),
        "gates": {
            "binary_agreement_rate": BINARY_AGREEMENT_GATE,
            "count_spearman_rho": COUNT_SPEARMAN_GATE,
        },
        "judged_on": "legitimate URLs only",
        "caveat": (
            "Agreement on phishing URLs mostly measures link rot rather than extractor "
            "correctness. Those domains are largely dead, so what is fetched today is a "
            "registrar or parking page, not the page the dataset was built from. The "
            "phishing split is reported for transparency and does not gate."
        ),
        "features": features,
        "demoted": demoted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use committed fixtures (default)")
    parser.add_argument(
        "--ref-scope",
        choices=["anchor", "all_resources", "both"],
        default="both",
        help="which reference-count definition to score; 'both' picks the winner",
    )
    parser.add_argument("--out", type=Path, default=Path("extraction_agreement.json"))
    args = parser.parse_args()

    if args.ref_scope == "both":
        candidates = {scope: run(scope) for scope in ("anchor", "all_resources")}

        def ref_score(report: dict[str, Any]) -> float:
            values = [
                report["features"][name]["value"] or 0.0
                for name in ("NoOfSelfRef", "NoOfExternalRef", "NoOfEmptyRef")
                if name in report["features"]
            ]
            return float(np.mean(values)) if values else 0.0

        winner = max(candidates, key=lambda s: ref_score(candidates[s]))
        report = candidates[winner]
        report["ref_scope_comparison"] = {
            scope: ref_score(candidates[scope]) for scope in candidates
        }
        report["ref_scope_note"] = (
            f"Scored both definitions on the three reference-count features and kept "
            f"{winner!r}. This is a measured choice, not a preference."
        )
    else:
        report = run(args.ref_scope)  # type: ignore[arg-type]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"scored {report['n_fixtures']} fixtures ({report['n_legitimate']} legitimate)")
    print(f"reference scope: {report['ref_scope']}")
    for name, record in sorted(report["features"].items()):
        mark = "pass" if record["passed"] else "DEMOTE"
        print(f"  {mark:7} {name:24} {record['metric']}={record['value']}")
    print(f"\ndemoted: {', '.join(report['demoted']) or 'none'}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
