"""Start-up self-test.

Run inside the container at boot and in CI against the built image. It answers one
question: are the artifacts that shipped the artifacts that were tested, and does the
installed dependency set still produce the same numbers?

The check is the golden row -- one real record, its expected 49-vector, and its expected
prediction from each of the four models -- pushed through the whole pipeline and compared
exactly. A container that boots with a bundle producing different numbers is worse than
one that refuses to boot: it serves predictions that nobody can trace to a training run,
and nothing about its behaviour looks wrong from outside.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phishguard import schema
from phishguard.artifacts import BundleError, load_bundle
from phishguard.config import APP
from phishguard.preprocess.transformer import Preprocessor


def run_golden(artifacts_dir: Path) -> list[str]:
    """Return a list of failures; empty means the bundle is trustworthy."""
    failures: list[str] = []

    bundle = load_bundle(artifacts_dir, verify=True)

    if len(bundle.stats.feature_order) != 49:
        failures.append(f"feature_order has {len(bundle.stats.feature_order)} entries, expected 49")
    if tuple(bundle.stats.feature_order) != schema.FEATURE_ORDER:
        failures.append("bundle feature order does not match this build's frozen order")
    if tuple(bundle.stats.scaler.columns) != tuple(bundle.stats.numerical_columns):
        failures.append("scaler column set does not match the recorded numeric columns")

    golden = bundle.golden_row
    if not golden:
        failures.append("bundle contains no golden row; cannot verify end-to-end behaviour")
        return failures

    record = pd.DataFrame([golden["raw"]])
    for column in schema.WORKING_COLUMNS:
        if column not in record.columns:
            record[column] = np.nan
    record = record[list(schema.WORKING_COLUMNS)]

    actual_vector = Preprocessor(bundle.stats).transform_matrix(record)[0]
    expected_vector = np.asarray(golden["vector"], dtype=np.float32)

    if not np.array_equal(actual_vector, expected_vector):
        differing = np.nonzero(actual_vector != expected_vector)[0]
        detail = ", ".join(
            f"{schema.FEATURE_ORDER[i]}: expected {expected_vector[i]!r}, got {actual_vector[i]!r}"
            for i in differing[:5]
        )
        failures.append(f"golden row vector differs in {len(differing)} feature(s): {detail}")

    for key, model in bundle.models.items():
        expected_label = golden["predictions"].get(key)
        actual_label = int(model.predict(actual_vector.reshape(1, -1))[0])
        if expected_label is not None and actual_label != expected_label:
            failures.append(
                f"{key}: golden row predicted {actual_label}, bundle recorded {expected_label}"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", action="store_true", help="verify the golden row")
    parser.add_argument("--artifacts", type=Path, default=APP.artifacts_dir)
    args = parser.parse_args()

    try:
        failures = run_golden(args.artifacts)
    except (BundleError, FileNotFoundError) as exc:
        print(f"SELFTEST FAILED: {exc}", file=sys.stderr)
        return 1

    if failures:
        print("SELFTEST FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"selftest OK: bundle at {args.artifacts} reproduces the golden row exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
